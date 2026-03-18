# src/monitoring/performance_monitor.py
# Monitor de performance do modelo em produção
#
# - Monitora Precision@K(15%) e AUC-PR ao longo das competências
# - Dois modos: backtesting histórico + produção (quando T+1 consolidar)
# - Score de T é comparado com target realizado de T+1
# - Critério de target válido: consolidado OU estabilizando,
#   sem forward fill de capacity
# - Threshold de alerta: Precision@K < 0.55 → trigger de retraining
# - Grava em monitoring_performance e loga no MLflow
# - Scores históricos simulados via @champion para backtesting

import os
import tempfile
from datetime import date

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA = "dev_christian_van_bellen"

TABLE_FEATURES = f"{CATALOG}.{SCHEMA}.gold_pressure_features"
TABLE_SCORING = f"{CATALOG}.{SCHEMA}.gold_pressure_scoring"
TABLE_MONITOR = f"{CATALOG}.{SCHEMA}.monitoring_performance"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.pressure_risk_classifier"

TARGET_COL = "target_alta_pressao"

FEATURE_COLS = [
    "casos_por_leito",
    "casos_por_leito_lag1",
    "casos_por_leito_lag2",
    "casos_por_leito_lag3",
    "casos_por_leito_ma2",
    "casos_por_leito_ma3",
    "casos_srag_lag1",
    "casos_srag_lag2",
    "obitos_por_leito",
    "uti_por_leito_uti",
    "share_idosos",
    "growth_mom",
    "growth_3m",
    "acceleration",
    "rolling_std_3m",
    "leitos_totais",
    "leitos_uti",
    "num_hospitais",
    "mes",
    "quarter",
    "is_semester1",
    "is_rainy_season",
]

# critérios de qualidade para target válido no monitor
VALID_CONSOLIDATION = ["consolidado", "estabilizando"]

# threshold de alerta — Precision@K abaixo disso dispara retraining
PRECISION_K_THRESHOLD = 0.55


# ── simulação histórica ──────────────────────────────────────────
def _simular_scores_historicos(spark: SparkSession) -> DataFrame:
    """
    Para competências onde não existe score gravado em gold_pressure_scoring
    mas existe target válido no Gold, simula o que o @champion teria previsto.

    Isso permite backtesting mesmo sem histórico de scores gravados.

    Lógica:
      1. Carrega o modelo @champion.
      2. Para cada competência com target válido que NÃO está em
         gold_pressure_scoring, aplica o modelo.
      3. Retorna DataFrame com mesmo schema de gold_pressure_scoring +
         coluna simulated=True.
    """
    # competências já scored
    scored_comps = spark.table(TABLE_SCORING).select("competencia").distinct()

    # competências com target válido ainda não scored
    gold = spark.table(TABLE_FEATURES)
    para_simular = (
        gold.filter(F.col(TARGET_COL).isNotNull())
        .filter(F.col("srag_consolidation_flag").isin(VALID_CONSOLIDATION))
        .filter(F.col("capacity_is_forward_fill") == False)
        .join(scored_comps, on="competencia", how="left_anti")
        .select("competencia")
        .distinct()
        .orderBy("competencia")
    )

    comps = [r["competencia"] for r in para_simular.collect()]

    if not comps:
        print("  Nenhuma competência histórica para simular — tudo já scored.")
        return spark.createDataFrame([], schema=None)

    n_exibir = min(5, len(comps))
    sufixo = "..." if len(comps) > 5 else ""
    print(
        f"  Simulando scores para {len(comps)} competências históricas: {comps[:n_exibir]}{sufixo}"
    )

    # carrega champion
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion_run_id = mv.run_id
    champion_version = mv.version

    import mlflow.lightgbm
    import mlflow.spark

    run = client.get_run(champion_run_id)
    path = f"{run.info.artifact_uri}/model"
    mtype = run.data.params.get("model_type", "").lower()

    if "lightgbm" in mtype:
        model = mlflow.lightgbm.load_model(path)

        def _predict(df_spark):
            df_pd = df_spark.toPandas()
            df_pd["risk_score"] = model.predict(df_pd[FEATURE_COLS]).round(6)
            return df_spark.sparkSession.createDataFrame(df_pd)
    else:
        from pyspark.ml.functions import vector_to_array

        model = mlflow.spark.load_model(path)

        def _predict(df_spark):
            preds = model.transform(df_spark)
            return preds.withColumn(
                "risk_score",
                F.round(
                    vector_to_array(F.col("probability")).getItem(1).cast("double"),
                    6,
                ),
            )

    partes = []
    for comp in comps:
        df_comp = gold.filter(F.col("competencia") == comp).filter(F.col(TARGET_COL).isNotNull())
        for col in FEATURE_COLS:
            df_comp = df_comp.withColumn(col, F.col(col).cast("double"))
        df_comp = df_comp.dropna(subset=FEATURE_COLS)

        df_scored = _predict(df_comp)
        df_scored = (
            df_scored.withColumn("model_alias", F.lit("champion"))
            .withColumn("model_version", F.lit(champion_version))
            .withColumn("model_type", F.lit(mtype))
            .withColumn("score_date", F.lit(str(date.today())))
            .withColumn("simulated", F.lit(True))
        )
        partes.append(df_scored)

    from functools import reduce

    return reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), partes)


# ── métricas por competência ─────────────────────────────────────
def _calcular_metricas_por_competencia(spark: SparkSession) -> pd.DataFrame:
    """
    Para cada competência T onde existe score (real ou simulado) E target de T+1:
      - join score(T) com target(T+1) por municipio_id
      - calcula Precision@K(15%), Recall@K(15%), AUC-PR, AUC-ROC
      - retorna DataFrame pandas com uma linha por competência
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    gold = spark.table(TABLE_FEATURES)
    scored = spark.table(TABLE_SCORING).withColumn("simulated", F.lit(False))

    # inclui scores simulados para competências históricas
    simulados = _simular_scores_historicos(spark)
    if simulados.count() > 0:
        scored = scored.unionByName(simulados, allowMissingColumns=True)

    # para cada competência scored, o target real é na competência seguinte
    scored = scored.withColumn(
        "competencia_target",
        F.date_format(
            F.add_months(
                F.to_date(
                    F.concat(F.col("competencia"), F.lit("01")),
                    "yyyyMMdd",
                ),
                1,
            ),
            "yyyyMM",
        ),
    )

    # target real da competência seguinte
    target_real = (
        gold.filter(F.col(TARGET_COL).isNotNull())
        .filter(F.col("srag_consolidation_flag").isin(VALID_CONSOLIDATION))
        .filter(F.col("capacity_is_forward_fill") == False)
        .select(
            "municipio_id",
            "competencia",
            TARGET_COL,
            "srag_consolidation_flag",
        )
        .withColumnRenamed("competencia", "competencia_target")
        .withColumnRenamed(TARGET_COL, "target_realizado")
    )

    # join score × target
    df_eval = scored.join(
        target_real,
        on=["municipio_id", "competencia_target"],
        how="inner",
    )

    competencias = [
        r["competencia"]
        for r in df_eval.select("competencia").distinct().orderBy("competencia").collect()
    ]
    print(f"  Competências avaliáveis: {len(competencias)}")

    resultados = []
    for comp in competencias:
        df_c = df_eval.filter(F.col("competencia") == comp).toPandas()
        if len(df_c) < 10:
            continue

        scores = df_c["risk_score"].values
        labels = df_c["target_realizado"].values
        k_ref = max(1, int(len(df_c) * 0.15))

        # Precision@K e Recall@K
        ordem = np.argsort(scores)[::-1]
        top_k = labels[ordem][:k_ref]
        tp_k = int(top_k.sum())
        prec_k = tp_k / k_ref
        rec_k = tp_k / max(1, int(labels.sum()))

        # AUC
        try:
            auc_roc = float(roc_auc_score(labels, scores))
            auc_pr = float(average_precision_score(labels, scores))
        except Exception:
            auc_roc = None
            auc_pr = None

        # garante que pegamos só o primeiro valor escalar,
        # mesmo se a coluna tiver nome duplicado após o join
        col_vals = df_c["srag_consolidation_flag"]
        if hasattr(col_vals, "iloc"):
            val = col_vals.iloc[0]
            consolidation = str(val.iloc[0]) if hasattr(val, "iloc") else str(val)
        else:
            consolidation = str(col_vals)
        simulated = bool(df_c.get("simulated", pd.Series([True])).iloc[0])

        resultados.append(
            {
                "competencia": comp,
                "n_municipios": len(df_c),
                "n_positivos": int(labels.sum()),
                "k_ref": k_ref,
                "precision_at_k": round(prec_k, 4),
                "recall_at_k": round(rec_k, 4),
                "auc_roc": round(auc_roc, 4) if auc_roc is not None else None,
                "auc_pr": round(auc_pr, 4) if auc_pr is not None else None,
                "consolidation_flag": consolidation,
                "simulated": simulated,
                "monitor_date": str(date.today()),
            }
        )

    return pd.DataFrame(resultados)


# ── artefatos ────────────────────────────────────────────────────
def _plot_performance_timeline(
    df_metricas: pd.DataFrame,
    versao_champion: str = "?",
):
    """
    Figura com 3 subplots:
      1. Precision@K(15%) ao longo das competências
      2. AUC-PR ao longo das competências
      3. Volume de municípios avaliados por competência (barras)
    Diferencia simulado (backtesting) de real (produção).
    Salva como performance_timeline.png e loga no MLflow.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    df = df_metricas.sort_values("competencia").reset_index(drop=True)
    n = len(df)
    mask_sim = df["simulated"] == True
    mask_real = df["simulated"] == False
    idx_sim = [i for i, s in enumerate(df["simulated"]) if s]
    idx_real = [i for i, s in enumerate(df["simulated"]) if not s]

    # labels do eixo x: AAAAMM → AAAA-MM
    x_labels = [f"{c[:4]}-{c[4:]}" for c in df["competencia"]]
    tick_step = max(1, n // 12 * 3 // 3)  # a cada 3 competências no mínimo
    tick_step = max(3, tick_step)
    tick_idxs = list(range(0, n, tick_step))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ── subplot 1 — Precision@K ────────────────────────────────────
    thr1 = PRECISION_K_THRESHOLD

    # faixas de fundo ok / alerta
    ax1.axhspan(thr1, 1.0, color="green", alpha=0.08, label="Zona ok")
    ax1.axhspan(0.0, thr1, color="red", alpha=0.08, label="Zona alerta")

    if mask_sim.any():
        ax1.plot(
            idx_sim,
            df["precision_at_k"][mask_sim].values,
            "o--",
            color="steelblue",
            alpha=0.7,
            markersize=4,
            label="Simulado (backtesting)",
        )
    if mask_real.any():
        ax1.plot(
            idx_real,
            df["precision_at_k"][mask_real].values,
            "o-",
            color="darkorange",
            linewidth=2,
            markersize=6,
            label="Real (produção)",
        )

    # anotações de valor em cada ponto
    for i, v in enumerate(df["precision_at_k"]):
        if pd.notna(v):
            ax1.annotate(
                f"{v:.2f}",
                xy=(i, v),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=7,
                ha="center",
            )

    ax1.axhline(
        y=thr1, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Threshold de alerta"
    )
    ax1.set_title("Precision@K(15%) — Backtesting Histórico", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Precision@K", fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=9, loc="lower left")

    # ── subplot 2 — AUC-PR ────────────────────────────────────────
    thr2 = 0.55

    ax2.axhspan(thr2, 1.0, color="green", alpha=0.08)
    ax2.axhspan(0.0, thr2, color="red", alpha=0.08)

    if mask_sim.any():
        ax2.plot(
            idx_sim,
            df["auc_pr"][mask_sim].values,
            "o--",
            color="steelblue",
            alpha=0.7,
            markersize=4,
        )
    if mask_real.any():
        ax2.plot(
            idx_real,
            df["auc_pr"][mask_real].values,
            "o-",
            color="darkorange",
            linewidth=2,
            markersize=6,
        )

    for i, v in enumerate(df["auc_pr"]):
        if pd.notna(v):
            ax2.annotate(
                f"{v:.2f}",
                xy=(i, v),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=7,
                ha="center",
            )

    ax2.axhline(y=thr2, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_title("AUC-PR ao longo do tempo", fontsize=13, fontweight="bold")
    ax2.set_ylabel("AUC-PR", fontsize=10)
    ax2.set_ylim(0, 1.0)

    # ── subplot 3 — Volume de municípios ──────────────────────────
    mediana_n = df["n_municipios"].median()

    bar_colors = ["darkorange" if not s else "lightsteelblue" for s in df["simulated"]]
    ax3.bar(range(n), df["n_municipios"], color=bar_colors, alpha=0.85)
    ax3.axhline(
        y=mediana_n,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=f"Mediana ({int(mediana_n):,})",
    )

    # anota n apenas nas últimas 6 competências
    for i in range(max(0, n - 6), n):
        v = df["n_municipios"].iloc[i]
        ax3.annotate(
            f"{v:,}", xy=(i, v), xytext=(0, 4), textcoords="offset points", fontsize=7, ha="center"
        )

    ax3.set_title("Municípios Avaliados por Competência", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Municípios", fontsize=10)
    ax3.set_ylim(0, df["n_municipios"].max() * 1.15)
    ax3.legend(fontsize=9)

    # ── eixo x compartilhado ───────────────────────────────────────
    ax3.set_xticks(tick_idxs)
    ax3.set_xticklabels([x_labels[i] for i in tick_idxs], rotation=45, fontsize=8)
    ax3.set_xlabel("Competência", fontsize=10)

    # linha vertical separando última simulada da primeira real
    if mask_sim.any() and mask_real.any():
        ultimo_sim = max(idx_sim)
        primeiro_real = min(idx_real)
        sep_x = (ultimo_sim + primeiro_real) / 2
        for ax in (ax1, ax2, ax3):
            ax.axvline(x=sep_x, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)

    # título geral
    fig.suptitle(
        f"Monitor de Performance — @champion v{versao_champion}\n"
        f"Avaliado em {date.today()}  |  {n} competências  |  "
        f"Threshold Precision@K={PRECISION_K_THRESHOLD}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "performance_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    os.remove(path)
    print("  ✓ performance_timeline.png logado")


# ── função principal ─────────────────────────────────────────────
def monitorar(spark: SparkSession, experiment_path: str = None) -> dict:
    """
    Calcula métricas de performance por competência, detecta degradação
    e grava resultados em monitoring_performance.

    Retorna dict com status, trigger_retraining e métricas recentes.
    """
    experiment_path = (
        experiment_path or "/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr"
    )
    mlflow.set_experiment(experiment_path)

    with mlflow.start_run(run_name=f"performance_monitor_{date.today()}") as run:
        print("\n── Performance Monitor ──")
        print(f"  Data: {date.today()}")
        print(f"  Threshold Precision@K: {PRECISION_K_THRESHOLD}")

        df_metricas = _calcular_metricas_por_competencia(spark)

        if df_metricas.empty:
            print("  ⚠ Nenhuma competência avaliável ainda.")
            return {"status": "sem_dados", "trigger_retraining": False}

        # métricas das últimas 3 competências (reais ou simuladas)
        ultimas = df_metricas.sort_values("competencia").tail(3)
        prec_recente = float(ultimas["precision_at_k"].mean())
        auc_recente = (
            float(ultimas["auc_pr"].dropna().mean()) if ultimas["auc_pr"].notna().any() else 0.0
        )

        print("\n── Métricas recentes (últimas 3 competências) ──")
        print(
            ultimas[
                [
                    "competencia",
                    "precision_at_k",
                    "auc_pr",
                    "n_municipios",
                    "consolidation_flag",
                    "simulated",
                ]
            ].to_string(index=False)
        )

        print(f"\n  Precision@K médio: {prec_recente:.4f}")
        print(f"  AUC-PR médio:      {auc_recente:.4f}")
        print(f"  Threshold:         {PRECISION_K_THRESHOLD}")

        # decisão de trigger
        trigger = prec_recente < PRECISION_K_THRESHOLD
        status = "alerta_retraining" if trigger else "ok"

        print(f"\n  Status: {status}")
        if trigger:
            print(
                f"  ⚠ Precision@K ({prec_recente:.4f}) abaixo do threshold ({PRECISION_K_THRESHOLD})"
            )
            print("  → Retraining recomendado")
        else:
            print("  ✓ Performance dentro do esperado")

        # loga métricas no MLflow
        mlflow.log_metrics(
            {
                "precision_at_k_recente": prec_recente,
                "auc_pr_recente": auc_recente,
                "n_competencias_avaliadas": float(len(df_metricas)),
                "trigger_retraining": float(trigger),
            }
        )
        mlflow.set_tag("monitor_status", status)

        # artefatos
        client_mv = MlflowClient()
        champion_version = client_mv.get_model_version_by_alias(MODEL_NAME, "champion").version
        _plot_performance_timeline(df_metricas, versao_champion=champion_version)

        # grava resultado no TABLE_MONITOR (append)
        df_spark = spark.createDataFrame(df_metricas)
        spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE_MONITOR} USING DELTA")
        (
            df_spark.write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(TABLE_MONITOR)
        )
        print(f"  ✓ Resultados gravados em {TABLE_MONITOR}")

        resultado = {
            "status": status,
            "trigger_retraining": trigger,
            "precision_at_k": prec_recente,
            "auc_pr": auc_recente,
            "competencias": len(df_metricas),
            "run_id": run.info.run_id,
        }
        print(f"\n✓ Monitor run ID: {run.info.run_id}")
        return resultado


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    resultado = monitorar(spark)
    print(f"\nTrigger retraining: {resultado['trigger_retraining']}")
