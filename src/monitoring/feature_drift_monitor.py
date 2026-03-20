# src/monitoring/feature_drift_monitor.py
# Monitor de feature drift usando Evidently AI.
#
# Compara a distribuição das features da competência atual
# de scoring com a distribuição de referência (período de treino).
#
# Referência: distribuição das features no período de treino
#   (competências até TRAIN_END definido em conf/)
# Atual: features da competência mais recente scored
#
# Métricas calculadas por feature (FEATURE_COLS):
#   - PSI (Population Stability Index) — estabilidade de distribuição
#   - drift_detected: bool — se drift foi detectado para a feature
#
# Saídas:
#   - Tabela: monitoring_feature_drift (append por competencia)
#   - Artefato MLflow: feature_drift_report.html
#   - Artefato MLflow: feature_drift_summary.json
#   - Print no terminal com resumo por feature
#
# Thresholds:
#   PSI >= 0.20 → drift moderado (warn)
#   PSI >= 0.25 → drift severo (flag para retraining consideration)
#   Proporção de features com drift >= 0.30 → alerta geral

import json
import os
import tempfile
from datetime import date

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from evidently import Report

try:
    from evidently.presets import DataDriftPreset
except ImportError:
    from evidently.metric_preset import DataDriftPreset
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from cli import FEATURE_COLS

# ── constantes ────────────────────────────────────────────────────
PSI_WARN_THRESHOLD = 0.20  # drift moderado
PSI_SEVERE_THRESHOLD = 0.25  # drift severo
DRIFT_SHARE_THRESHOLD = 0.30  # % de features com drift → alerta geral

# amostra máxima para referência (eficiência em dados históricos grandes)
_MAX_REF_ROWS = 50_000


# ── carregamento de dados ─────────────────────────────────────────
def _carregar_referencia(
    spark: SparkSession, table_gold_features: str, train_end: str
) -> pd.DataFrame:
    """
    Carrega dados de referência (período de treino) da feature table.
    Usa competências <= train_end com target não nulo.
    Amostra até 50.000 linhas para eficiência.
    Retorna pandas DataFrame com FEATURE_COLS.
    """
    df = (
        spark.table(table_gold_features)
        .filter(F.col("competencia") <= train_end)
        .filter(F.col("target_alta_pressao").isNotNull())
        .select(*FEATURE_COLS)
        .dropna()
    )

    total = df.count()
    if total == 0:
        raise ValueError(
            f"Nenhum dado de referência encontrado em {table_gold_features} "
            f"para competencias <= {train_end}."
        )

    if total > _MAX_REF_ROWS:
        frac = _MAX_REF_ROWS / total
        df = df.sample(fraction=frac, seed=42)

    print(f"  Referência: {min(total, _MAX_REF_ROWS):,} linhas (de {total:,} disponíveis)")
    return df.toPandas()


def _carregar_atual(
    spark: SparkSession, table_gold_features: str, table_gold_scoring: str
) -> tuple[pd.DataFrame, str]:
    """
    Carrega features da competência mais recente em gold_pressure_scoring.
    Retorna (pandas DataFrame com FEATURE_COLS, competencia).
    """
    # identifica a competência mais recente scored
    ultima = (
        spark.table(table_gold_scoring)
        .select(F.max("competencia").alias("max_comp"))
        .collect()[0]["max_comp"]
    )

    if ultima is None:
        raise ValueError(f"Nenhuma competência encontrada em {table_gold_scoring}.")

    # busca features na gold_features para essa competência
    df = (
        spark.table(table_gold_features)
        .filter(F.col("competencia") == ultima)
        .select(*FEATURE_COLS)
        .dropna()
    )

    n = df.count()
    if n == 0:
        raise ValueError(
            f"Nenhuma feature encontrada em {table_gold_features} para competencia={ultima}."
        )

    print(f"  Atual: {n:,} linhas — competencia {ultima}")
    return df.toPandas(), ultima


# ── cálculo de drift ─────────────────────────────────────────────
def _calcular_drift(
    ref_df: pd.DataFrame, cur_df: pd.DataFrame, tmpdir: str
) -> tuple[dict, str | None]:
    """
    Executa Evidently Report com DataDriftPreset e extrai PSI por feature.
    Compatível com Evidently 0.7.x — usa o Snapshot retornado por run().
    """
    report = Report([DataDriftPreset(method="psi")])

    # run() retorna Snapshot com os resultados — não ignorar o retorno
    snapshot = report.run(
        reference_data=ref_df[FEATURE_COLS],
        current_data=cur_df[FEATURE_COLS],
    )

    # salva HTML via snapshot.save_html()
    html_path = os.path.join(tmpdir, "feature_drift_report.html")
    try:
        snapshot.save_html(html_path)
    except Exception as e:
        print(f"  ⚠ HTML não salvo: {e}")
        html_path = None

    # extrai métricas via snapshot.json()
    # estrutura real 0.7.x: uma entrada por feature com metric_name="ValueDrift(...)"
    # e value = PSI score como float direto
    import json as _json

    data = _json.loads(snapshot.json())
    drift_por_feature = {}

    for metric in data.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {})
        value = metric.get("value")

        # ignora DriftedColumnsCount (sumário geral) e outras métricas
        if "ValueDrift" not in metric_name:
            continue

        col = config.get("column", "")
        if col not in FEATURE_COLS:
            continue

        # value é float direto = PSI score
        psi_score = float(value) if isinstance(value, (int, float)) else 0.0

        # drift detectado se PSI >= threshold (default 0.1 no Evidently)
        threshold = float(config.get("threshold", 0.1))
        drift_detected = psi_score >= threshold

        drift_por_feature[col] = {
            "drift_detected": drift_detected,
            "drift_score": psi_score,
            "stattest": config.get("method", "psi"),
        }

    # preenche features não encontradas com zeros
    for feat in FEATURE_COLS:
        if feat not in drift_por_feature:
            drift_por_feature[feat] = {
                "drift_detected": False,
                "drift_score": 0.0,
                "stattest": "psi",
            }

    n_extraidas = sum(1 for v in drift_por_feature.values() if v["drift_score"] > 0)
    print(f"  Métricas extraídas: {n_extraidas}/{len(FEATURE_COLS)} features com PSI > 0")

    return drift_por_feature, html_path


# ── gráfico resumo ────────────────────────────────────────────────
def _plot_drift_summary(drift_results: dict, competencia: str, tmpdir: str) -> None:
    """
    Gera gráfico de barras horizontais com PSI score por feature,
    ordenado do maior para o menor. Salva em tmpdir.
    """
    features = list(drift_results.keys())
    scores = [drift_results[f]["drift_score"] for f in features]

    # ordena do maior para o menor
    pares = sorted(zip(scores, features, strict=False), reverse=True)
    scores_ord = [p[0] for p in pares]
    features_ord = [p[1] for p in pares]

    # cores por nível de severidade
    def _cor(psi: float) -> str:
        if psi >= PSI_SEVERE_THRESHOLD:
            return "#d62728"  # vermelho
        if psi >= PSI_WARN_THRESHOLD:
            return "#ff7f0e"  # laranja
        if psi >= 0.10:
            return "#ffbb78"  # amarelo
        return "#2ca02c"  # verde

    cores = [_cor(s) for s in scores_ord]

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(features_ord, scores_ord, color=cores)

    # linhas de threshold
    ax.axvline(
        PSI_WARN_THRESHOLD,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        label=f"Warn (PSI={PSI_WARN_THRESHOLD})",
    )
    ax.axvline(
        PSI_SEVERE_THRESHOLD,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Severo (PSI={PSI_SEVERE_THRESHOLD})",
    )

    # anotações de valor
    for bar, score in zip(bars, scores_ord, strict=False):
        ax.text(
            score + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("PSI Score")
    ax.set_title(f"Feature Drift PSI — Competência {competencia}")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()

    chart_path = os.path.join(tmpdir, "feature_drift_chart.png")
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── função principal ──────────────────────────────────────────────
def monitorar_drift(spark: SparkSession, args) -> dict:
    """
    Fluxo completo de detecção de feature drift.

    1. Carrega referência (treino) e atual (última competência scored)
    2. Calcula drift com Evidently PSI
    3. Determina status: ok / warn / alerta
    4. Loga no MLflow (métricas + artefatos)
    5. Grava em monitoring_feature_drift (append)
    6. Retorna dict com resultado

    Status:
      ok     → < 30% features com drift (PSI >= 0.20)
      warn   → >= 30% features com drift moderado
      alerta → >= 30% features com drift severo (PSI >= 0.25)
    """
    catalog = args.catalog
    schema = args.schema
    mlflow_experiment = args.mlflow_experiment
    table_gold_features = args.table_gold_features
    table_gold_scoring = args.table_gold_scoring
    train_end = args.train_end
    drift_seasonal_features = [s.strip() for s in args.drift_seasonal_features.split(",")]

    table_drift = f"{catalog}.{schema}.monitoring_feature_drift"

    mlflow.set_experiment(experiment_name=mlflow_experiment)

    print("\n── Feature Drift Monitor ──────────────────────────────────")

    ref_df = _carregar_referencia(spark, table_gold_features, train_end)
    cur_df, competencia_atual = _carregar_atual(spark, table_gold_features, table_gold_scoring)

    print(f"\n  Calculando drift PSI ({len(FEATURE_COLS)} features)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        drift_results, html_path = _calcular_drift(ref_df, cur_df, tmpdir)
        _plot_drift_summary(drift_results, competencia_atual, tmpdir)

        # ── classifica severidade ────────────────────────────────
        features_drift_severo = [
            f for f, r in drift_results.items() if r["drift_score"] >= PSI_SEVERE_THRESHOLD
        ]
        # features sazonais têm drift esperado quando comparando
        # um único mês com o histórico anual completo — excluir do share
        features_drift_moderado = [
            f
            for f, r in drift_results.items()
            if r["drift_score"] >= PSI_WARN_THRESHOLD and f not in drift_seasonal_features
        ]
        n_total = len(drift_results)
        n_severo = len(features_drift_severo)
        n_moderado = len(features_drift_moderado)
        drift_share = n_moderado / n_total if n_total > 0 else 0.0
        severe_share = n_severo / n_total if n_total > 0 else 0.0

        if severe_share >= DRIFT_SHARE_THRESHOLD:
            status = "alerta"
        elif drift_share >= DRIFT_SHARE_THRESHOLD:
            status = "warn"
        else:
            status = "ok"

        # ── print resumo ─────────────────────────────────────────
        print(f"\n  {'Feature':<30} {'PSI':>8}  Status")
        print("  " + "─" * 52)
        for feat in sorted(
            drift_results, key=lambda f: drift_results[f]["drift_score"], reverse=True
        ):
            r = drift_results[feat]
            psi = r["drift_score"]
            if psi >= PSI_SEVERE_THRESHOLD:
                flag = "DRIFT"
            elif psi >= PSI_WARN_THRESHOLD:
                flag = "WARN"
            else:
                flag = "OK"
            print(f"  {feat:<30} {psi:>8.4f}  {flag}")

        print("  " + "─" * 52)
        print(f"  Features com drift moderado : {n_moderado}/{n_total} ({drift_share:.0%})")
        print(f"  Features com drift severo   : {n_severo}/{n_total} ({severe_share:.0%})")
        print(f"  (features sazonais excluídas do drift_share: {drift_seasonal_features})")
        print(f"  Status geral: {status.upper()}")

        # ── salva summary JSON ───────────────────────────────────
        summary = {
            "competencia_atual": competencia_atual,
            "competencia_referencia_end": train_end,
            "status": status,
            "drift_share": drift_share,
            "n_features_drift": n_moderado,
            "n_features_total": n_total,
            "features_com_drift": features_drift_moderado,
            "features_drift_severo": features_drift_severo,
            "monitor_date": str(date.today()),
        }
        json_path = os.path.join(tmpdir, "feature_drift_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # ── loga no MLflow ───────────────────────────────────────
        with mlflow.start_run(run_name=f"feature_drift_{competencia_atual}"):
            mlflow.set_tags(
                {
                    "monitor_type": "feature_drift",
                    "competencia_atual": competencia_atual,
                    "competencia_referencia_end": train_end,
                    "status": status,
                }
            )

            # métrica agregada
            mlflow.log_metrics(
                {
                    "drift_share": drift_share,
                    "n_features_drift": float(n_moderado),
                    "n_features_total": float(n_total),
                    "severe_share": severe_share,
                }
            )

            # PSI por feature
            for feat, r in drift_results.items():
                mlflow.log_metric(f"psi_{feat}", r["drift_score"])

            # artefatos
            if html_path and os.path.exists(html_path):
                mlflow.log_artifact(html_path)
                print("  ✓ feature_drift_report.html logado")
            mlflow.log_artifact(os.path.join(tmpdir, "feature_drift_chart.png"))
            mlflow.log_artifact(json_path)

            run_id = mlflow.active_run().info.run_id

        # ── grava tabela de monitoramento ────────────────────────
        monitor_date = str(date.today())
        registros = []
        for feat, r in drift_results.items():
            psi = r["drift_score"]
            if psi >= PSI_SEVERE_THRESHOLD:
                severity = "severo"
            elif psi >= PSI_WARN_THRESHOLD:
                severity = "moderado"
            else:
                severity = "ok"

            registros.append(
                {
                    "monitor_date": monitor_date,
                    "competencia_atual": competencia_atual,
                    "competencia_referencia_end": train_end,
                    "feature_name": feat,
                    "drift_score": psi,
                    "drift_detected": r["drift_detected"],
                    "stattest_name": r["stattest"],
                    "severity": severity,
                    "n_reference": len(ref_df),
                    "n_current": len(cur_df),
                }
            )

        drift_pdf = pd.DataFrame(registros)
        drift_sdf = spark.createDataFrame(drift_pdf)

        drift_sdf.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
            table_drift
        )

        print(f"\n  ✓ {len(registros)} registros gravados em {table_drift}")

    return {
        "status": status,
        "drift_share": drift_share,
        "n_features_drift": n_moderado,
        "n_features_total": n_total,
        "competencia_atual": competencia_atual,
        "features_com_drift": features_drift_moderado,
        "run_id": run_id,
    }


# ── entrypoint ────────────────────────────────────────────────────
if __name__ == "__main__":
    from cli import build_parser

    p = build_parser("Monitor de feature drift — Radar de Pressão Assistencial")
    args, _ = p.parse_known_args()
    spark = SparkSession.builder.getOrCreate()
    resultado = monitorar_drift(spark, args)
    print(f"\nStatus: {resultado['status']}")
    print(f"Features com drift: {resultado['n_features_drift']}/{resultado['n_features_total']}")
    if resultado["features_com_drift"]:
        print(f"Features: {resultado['features_com_drift']}")
