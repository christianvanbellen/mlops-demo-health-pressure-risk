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
import sys
import tempfile
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from config import (
    CATALOG,
    FEATURE_COLS,
    MLFLOW_EXPERIMENT,
    SCHEMA,
    TABLE_GOLD_FEATURES,
    TABLE_GOLD_SCORING,
    TRAIN_END,
)

# ── constantes ────────────────────────────────────────────────────
TABLE_DRIFT = f"{CATALOG}.{SCHEMA}.monitoring_feature_drift"

PSI_WARN_THRESHOLD = 0.20  # drift moderado
PSI_SEVERE_THRESHOLD = 0.25  # drift severo
DRIFT_SHARE_THRESHOLD = 0.30  # % de features com drift → alerta geral

# amostra máxima para referência (eficiência em dados históricos grandes)
_MAX_REF_ROWS = 50_000


# ── carregamento de dados ─────────────────────────────────────────
def _carregar_referencia(spark: SparkSession) -> pd.DataFrame:
    """
    Carrega dados de referência (período de treino) da feature table.
    Usa competências <= TRAIN_END com target não nulo.
    Amostra até 50.000 linhas para eficiência.
    Retorna pandas DataFrame com FEATURE_COLS.
    """
    df = (
        spark.table(TABLE_GOLD_FEATURES)
        .filter(F.col("competencia") <= TRAIN_END)
        .filter(F.col("target_alta_pressao").isNotNull())
        .select(*FEATURE_COLS)
        .dropna()
    )

    total = df.count()
    if total == 0:
        raise ValueError(
            f"Nenhum dado de referência encontrado em {TABLE_GOLD_FEATURES} "
            f"para competencias <= {TRAIN_END}."
        )

    if total > _MAX_REF_ROWS:
        frac = _MAX_REF_ROWS / total
        df = df.sample(fraction=frac, seed=42)

    print(f"  Referência: {min(total, _MAX_REF_ROWS):,} linhas (de {total:,} disponíveis)")
    return df.toPandas()


def _carregar_atual(spark: SparkSession) -> tuple[pd.DataFrame, str]:
    """
    Carrega features da competência mais recente em gold_pressure_scoring.
    Retorna (pandas DataFrame com FEATURE_COLS, competencia).
    """
    # identifica a competência mais recente scored
    ultima = (
        spark.table(TABLE_GOLD_SCORING)
        .select(F.max("competencia").alias("max_comp"))
        .collect()[0]["max_comp"]
    )

    if ultima is None:
        raise ValueError(f"Nenhuma competência encontrada em {TABLE_GOLD_SCORING}.")

    # busca features na gold_features para essa competência
    df = (
        spark.table(TABLE_GOLD_FEATURES)
        .filter(F.col("competencia") == ultima)
        .select(*FEATURE_COLS)
        .dropna()
    )

    n = df.count()
    if n == 0:
        raise ValueError(
            f"Nenhuma feature encontrada em {TABLE_GOLD_FEATURES} para competencia={ultima}."
        )

    print(f"  Atual: {n:,} linhas — competencia {ultima}")
    return df.toPandas(), ultima


# ── cálculo de drift ─────────────────────────────────────────────
def _calcular_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, tmpdir: str) -> dict:
    """
    Executa o Evidently Report com DataDriftPreset(method="psi")
    comparando ref_df (referência) com cur_df (atual).

    Retorna dict com resultado por feature:
      {
        "feature_name": {
          "drift_detected": bool,
          "drift_score": float,   # PSI score
          "stattest": str,        # método usado
        },
        ...
      }

    Salva feature_drift_report.html no tmpdir para logar no MLflow.
    """
    report = Report([DataDriftPreset(method="psi")])
    report.run(
        reference_data=ref_df[FEATURE_COLS],
        current_data=cur_df[FEATURE_COLS],
    )

    # salva HTML para artefato MLflow
    html_path = os.path.join(tmpdir, "feature_drift_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(report.get_html())

    # extrai métricas por feature — tenta múltiplas estruturas entre versões
    result = report.as_dict()
    drift_por_feature = {}

    for metric in result.get("metrics", []):
        metric_id = metric.get("metric", "")

        # estrutura 1: DataDriftTable (agrupa todas as colunas)
        if "DataDriftTable" in metric_id:
            cols = metric.get("result", {}).get("drift_by_columns", {})
            for col, stats in cols.items():
                drift_por_feature[col] = {
                    "drift_detected": bool(stats.get("drift_detected", False)),
                    "drift_score": float(stats.get("drift_score", 0.0)),
                    "stattest": str(stats.get("stattest", "psi")),
                }

        # estrutura 2: ColumnDriftMetric (uma entrada por coluna)
        elif "ColumnDriftMetric" in metric_id:
            col = metric.get("result", {}).get("column_name", "")
            stat = metric.get("result", {})
            if col:
                drift_por_feature[col] = {
                    "drift_detected": bool(stat.get("drift_detected", False)),
                    "drift_score": float(stat.get("drift_score", 0.0)),
                    "stattest": str(stat.get("stattest_name", "psi")),
                }

    # fallback: estrutura não reconhecida — loga dump parcial para debug
    if not drift_por_feature:
        import json as _json

        print("  ⚠ Estrutura do Report não reconhecida. Dump parcial:")
        print(_json.dumps(result.get("metrics", [])[:2], indent=2, default=str))

    return drift_por_feature


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
def monitorar_drift(spark: SparkSession, experiment_path: str | None = None) -> dict:
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
    exp_path = experiment_path or MLFLOW_EXPERIMENT
    mlflow.set_experiment(exp_path)

    print("\n── Feature Drift Monitor ──────────────────────────────────")

    ref_df = _carregar_referencia(spark)
    cur_df, competencia_atual = _carregar_atual(spark)

    print(f"\n  Calculando drift PSI ({len(FEATURE_COLS)} features)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        drift_results = _calcular_drift(ref_df, cur_df, tmpdir)
        _plot_drift_summary(drift_results, competencia_atual, tmpdir)

        # ── classifica severidade ────────────────────────────────
        features_drift_severo = [
            f for f, r in drift_results.items() if r["drift_score"] >= PSI_SEVERE_THRESHOLD
        ]
        features_drift_moderado = [
            f for f, r in drift_results.items() if r["drift_score"] >= PSI_WARN_THRESHOLD
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
        print(f"  Status geral: {status.upper()}")

        # ── salva summary JSON ───────────────────────────────────
        summary = {
            "competencia_atual": competencia_atual,
            "competencia_referencia_end": TRAIN_END,
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
                    "competencia_referencia_end": TRAIN_END,
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
            mlflow.log_artifact(os.path.join(tmpdir, "feature_drift_report.html"))
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
                    "competencia_referencia_end": TRAIN_END,
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
            TABLE_DRIFT
        )

        print(f"\n  ✓ {len(registros)} registros gravados em {TABLE_DRIFT}")

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
    spark = SparkSession.builder.getOrCreate()
    resultado = monitorar_drift(spark)
    print(f"\nStatus: {resultado['status']}")
    print(f"Features com drift: {resultado['n_features_drift']}/{resultado['n_features_total']}")
    if resultado["features_com_drift"]:
        print(f"Features: {resultado['features_com_drift']}")
