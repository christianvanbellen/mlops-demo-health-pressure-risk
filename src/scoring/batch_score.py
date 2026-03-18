# src/scoring/batch_score.py
# Batch scoring semanal de risco de pressão assistencial
#
# - Aplica @champion sobre a competência mais recente elegível para scoring
# - Política de confiança: capacity real + SRAG consolidado ou estabilizando
#   (data_quality_score >= 0.5; competências "recentes" < 45 dias são excluídas)
# - Modo A/B canary: 20% challenger / 80% champion se --ab ativo
# - Roteamento determinístico por municipio_id (MD5 hash % 100)
#   → mesmo município sempre no mesmo modelo durante o período A/B
# - Grava em gold_pressure_scoring com mode=append (histórico)
# - risk_class baseado em percentis p85/p70 do score desta competência (v2)
# - Para ativar A/B: passar ab_test=True ou flag --ab no CLI

import hashlib
from datetime import datetime

import mlflow.lightgbm
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA = "dev_christian_van_bellen"

TABLE_FEATURES = f"{CATALOG}.{SCHEMA}.gold_pressure_features"
TABLE_SCORING = f"{CATALOG}.{SCHEMA}.gold_pressure_scoring"
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

# A/B canary: 20% dos municípios vai para @challenger se existir
AB_CHALLENGER_PCT = 0.20

# threshold mínimo de data_quality_score para uma competência ser scored
SCORING_MIN_QUALITY = 0.5


# ── alias e modelo ───────────────────────────────────────────────
def _inferir_model_type(run_id: str) -> str:
    """
    Infere o tipo de modelo a partir dos parâmetros do run no MLflow.
    Retorna 'lightgbm' ou 'spark'.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    model_type = run.data.params.get("model_type", "").lower()
    if "lightgbm" in model_type:
        return "lightgbm"
    return "spark"


def _get_artifact_path(run_id: str) -> str:
    """
    Retorna o path dbfs:/ do artefato — necessário porque runs:/ usa
    /dbfs/tmp internamente, que está inacessível neste workspace.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    return f"{run.info.artifact_uri}/model"


def _get_model_info(alias: str) -> dict | None:
    """
    Retorna metadados da versão associada ao alias no registry.
    Retorna None se o alias não existir.
    """
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
        return {
            "version": mv.version,
            "run_id": mv.run_id,
            "alias": alias,
            "model_type": _inferir_model_type(mv.run_id),
        }
    except Exception:
        return None


def _carregar_modelo(model_info: dict, spark: SparkSession) -> tuple:
    """
    Carrega o modelo correto com base no model_type.
    Retorna (model, model_type).
    """
    run_id = model_info["run_id"]
    model_type = model_info["model_type"]
    path = _get_artifact_path(run_id)

    if model_type == "lightgbm":
        return mlflow.lightgbm.load_model(path), "lightgbm"
    return mlflow.spark.load_model(path), "spark"


# ── roteamento A/B ───────────────────────────────────────────────
def _ab_route(municipio_id: str, challenger_exists: bool) -> str:
    """
    Roteamento determinístico por municipio_id.
    O mesmo município sempre vai para o mesmo modelo durante o período
    de A/B — evita inconsistência de scores entre competências.
    Retorna 'champion' ou 'challenger'.
    """
    if not challenger_exists:
        return "champion"
    hash_val = int(hashlib.md5(municipio_id.encode()).hexdigest(), 16)
    if (hash_val % 100) < (AB_CHALLENGER_PCT * 100):
        return "challenger"
    return "champion"


def _ab_route_col(challenger_exists: bool):
    """
    Versão Spark de _ab_route — retorna uma expressão de coluna.
    Usa MD5 dos primeiros 8 hex chars convertido para inteiro (base 16→10),
    depois % 100 para decidir o grupo.
    """
    if not challenger_exists:
        return F.lit("champion")

    hash_int = F.conv(
        F.substring(F.md5(F.col("municipio_id")), 1, 8),
        16,
        10,
    ).cast("long")

    return F.when(
        (hash_int % 100) < int(AB_CHALLENGER_PCT * 100),
        F.lit("challenger"),
    ).otherwise(F.lit("champion"))


# ── dados e features ─────────────────────────────────────────────
def _get_competencia_scoring(spark: SparkSession) -> str:
    """
    Retorna a competência mais recente adequada para scoring
    seguindo a política de confiança de dados:

      1. target null (mês sem t+1 ainda)
      2. capacity_is_forward_fill = False (capacity real)
      3. srag_consolidation_flag != 'recente' (SRAG suficientemente consolidado)
      4. data_quality_score >= SCORING_MIN_QUALITY

    Política de scoring:
      consolidado   (>= 90 dias): score confiável
      estabilizando (>= 45 dias): score aceitável
      recente       (< 45 dias):  excluído — dados muito parciais

    Loga as competências disponíveis e o motivo da escolha.
    """
    df = spark.table(TABLE_FEATURES)

    # todas as competências candidatas ao scoring
    candidatas = (
        df.filter(F.col(TARGET_COL).isNull())
        .groupBy("competencia")
        .agg(
            F.first("capacity_is_forward_fill").alias("forward_fill"),
            F.first("srag_consolidation_flag").alias("consolidation"),
            F.first("data_quality_score").alias("quality_score"),
            F.count("*").alias("municipios"),
        )
        .orderBy(F.col("competencia").desc())
    )

    print("\n── Competências candidatas ao scoring ──")
    candidatas.show(truncate=False)

    # aplica política de confiança
    elegivel = (
        candidatas.filter(F.col("forward_fill") == False)
        .filter(F.col("consolidation") != "recente")
        .filter(F.col("quality_score") >= SCORING_MIN_QUALITY)
        .agg(F.max("competencia").alias("max_comp"))
        .collect()[0]["max_comp"]
    )

    if elegivel is None:
        raise ValueError(
            "Nenhuma competência elegível para scoring encontrada.\n"
            "Critérios: capacity real + SRAG consolidado ou estabilizando.\n"
            "Verifique se gold_pressure_features está atualizado."
        )

    print(f"\n  ✓ Competência selecionada: {elegivel}")
    return elegivel


def _preparar_features(spark: SparkSession, competencia: str) -> DataFrame:
    """
    Lê features da competência alvo, casteando para double.
    Mantém colunas de contexto além das features do modelo.
    """
    df = spark.table(TABLE_FEATURES)
    df = df.filter((F.col("competencia") == competencia) & F.col(TARGET_COL).isNull())

    for col in FEATURE_COLS:
        df = df.withColumn(col, F.col(col).cast("double"))

    colunas_contexto = [
        "municipio_id",
        "municipio_nome",
        "uf",
        "regiao",
        "competencia",
    ]
    colunas_qualidade = [
        "srag_consolidation_flag",
        "data_quality_score",
        "capacity_is_forward_fill",
    ]
    # FEATURE_COLS já inclui leitos_totais e leitos_uti — não duplicar
    todas_colunas = colunas_contexto + FEATURE_COLS + colunas_qualidade
    return df.select(todas_colunas)


# ── scoring ──────────────────────────────────────────────────────
def _aplicar_score_lgbm(model, df_spark: DataFrame, model_info: dict) -> DataFrame:
    """
    Aplica modelo LightGBM e devolve DataFrame Spark com risk_score.
    Preserva TODAS as colunas originais do df_spark.
    """
    # converte tudo para pandas preservando todas as colunas
    df_pd = df_spark.toPandas()

    # aplica score só nas feature cols
    scores = model.predict(df_pd[FEATURE_COLS])

    # adiciona risk_score ao pandas df completo
    df_pd["risk_score"] = scores.round(6).astype(float)

    # reconverte para Spark preservando todas as colunas
    return df_spark.sparkSession.createDataFrame(df_pd)


def _aplicar_score_spark(model, df_spark: DataFrame, model_info: dict) -> DataFrame:
    """
    Aplica modelo Spark ML Pipeline e extrai probabilidade da classe positiva.
    """
    preds = model.transform(df_spark)
    preds = preds.withColumn(
        "risk_score",
        F.round(
            vector_to_array(F.col("probability")).getItem(1).cast("double"),
            6,
        ),
    )
    # remove colunas internas do pipeline Spark ML
    for c in ["features", "scaled_features", "rawPrediction", "probability", "prediction"]:
        if c in preds.columns:
            preds = preds.drop(c)
    return preds


def _classificar_risco(df: DataFrame) -> DataFrame:
    """
    Classifica municípios em alto/moderado/baixo com base nos
    percentis 85 e 70 do risk_score desta competência.

    Garante que ~15% dos municípios ficam em alto risco e ~15%
    em moderado — ranking estável e comparável entre competências.

    Percentis calculados sobre o conjunto scored (não histórico),
    coerente com o target_definition_version=v2 que também usa p85.

    risk_threshold_version = "v2"
    """
    p85, p70 = df.approxQuantile("risk_score", [0.85, 0.70], 0.01)

    print(f"  Thresholds de risco — p85={p85:.6f}  p70={p70:.6f}")

    return (
        df.withColumn(
            "risk_class",
            F.when(F.col("risk_score") >= p85, F.lit("alto"))
            .when(F.col("risk_score") >= p70, F.lit("moderado"))
            .otherwise(F.lit("baixo")),
        )
        .withColumn("risk_threshold_version", F.lit("v2"))
        .withColumn("risk_p85", F.lit(float(p85)))
        .withColumn("risk_p70", F.lit(float(p70)))
    )


# ── função principal ─────────────────────────────────────────────
def score(spark: SparkSession, ab_test: bool = False) -> str:
    """
    Aplica o modelo @champion (e @challenger se ab_test=True) sobre a
    competência mais recente disponível para scoring.
    Grava resultado em gold_pressure_scoring (append).

    Parâmetro ab_test:
      False → usa só @champion (modo normal)
      True  → ativa roteamento 20/80 se @challenger existir
    """
    print("\n── Batch Score ──")
    print(f"  Modo A/B: {ab_test}")
    print(f"  Timestamp: {datetime.now()}")

    # ── aliases disponíveis ────────────────────────────────────────
    champion_info = _get_model_info("champion")
    challenger_info = _get_model_info("challenger") if ab_test else None

    if champion_info is None:
        raise ValueError(
            "Nenhum modelo com alias @champion encontrado no registry. "
            "Execute evaluate.py para promover um champion."
        )

    print(f"  Champion  : v{champion_info['version']} ({champion_info['model_type']})")
    if challenger_info:
        print(f"  Challenger: v{challenger_info['version']} ({challenger_info['model_type']})")
    else:
        print("  Challenger: não existe (modo normal)")

    # ── competência e features ─────────────────────────────────────
    competencia = _get_competencia_scoring(spark)
    df_features = _preparar_features(spark, competencia)
    n_municipios = df_features.count()
    print(f"  Municípios a scorar: {n_municipios:,}")

    # ── roteamento A/B ─────────────────────────────────────────────
    challenger_exists = challenger_info is not None
    df_features = df_features.withColumn(
        "modelo_destinado",
        _ab_route_col(challenger_exists),
    )

    df_champion = df_features.filter(F.col("modelo_destinado") == "champion")
    df_challenger = (
        df_features.filter(F.col("modelo_destinado") == "challenger") if challenger_exists else None
    )

    # ── score grupo champion ───────────────────────────────────────
    champ_model, champ_type = _carregar_modelo(champion_info, spark)

    if champ_type == "lightgbm":
        df_scored_champ = _aplicar_score_lgbm(champ_model, df_champion, champion_info)
    else:
        df_scored_champ = _aplicar_score_spark(champ_model, df_champion, champion_info)

    df_scored_champ = (
        df_scored_champ.withColumn("model_alias", F.lit("champion"))
        .withColumn("model_version", F.lit(champion_info["version"]))
        .withColumn("model_type", F.lit(champion_info["model_type"]))
    )

    # ── score grupo challenger (se A/B ativo) ──────────────────────
    if df_challenger is not None:
        chall_model, chall_type = _carregar_modelo(challenger_info, spark)

        if chall_type == "lightgbm":
            df_scored_chall = _aplicar_score_lgbm(chall_model, df_challenger, challenger_info)
        else:
            df_scored_chall = _aplicar_score_spark(chall_model, df_challenger, challenger_info)

        df_scored_chall = (
            df_scored_chall.withColumn("model_alias", F.lit("challenger"))
            .withColumn("model_version", F.lit(challenger_info["version"]))
            .withColumn("model_type", F.lit(challenger_info["model_type"]))
        )
        df_scored = df_scored_champ.unionByName(df_scored_chall)
    else:
        df_scored = df_scored_champ

    # ── classificação e metadados ──────────────────────────────────
    df_scored = _classificar_risco(df_scored)

    score_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_scored = (
        df_scored.withColumn("score_date", F.lit(score_date))
        .withColumn("competencia", F.lit(competencia))
        .drop("modelo_destinado")
    )

    # ── schema de saída ────────────────────────────────────────────
    colunas_output = [
        "municipio_id",
        "municipio_nome",
        "uf",
        "regiao",
        "competencia",
        "score_date",
        "risk_score",
        "risk_class",
        "risk_threshold_version",
        "risk_p85",
        "risk_p70",
        "model_alias",
        "model_version",
        "model_type",
        "leitos_totais",
        "leitos_uti",
        "casos_por_leito",
        "casos_por_leito_lag1",
        "growth_mom",
        "rolling_std_3m",
        "srag_consolidation_flag",
        "data_quality_score",
        "capacity_is_forward_fill",
    ]
    df_output = df_scored.select(colunas_output)

    # ── grava (append — histórico de scores) ──────────────────────
    spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE_SCORING} USING DELTA")
    (
        df_output.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(TABLE_SCORING)
    )

    n_scored = df_output.count()
    print(f"\n✓ Score gravado em {TABLE_SCORING}")
    print(f"  Competência: {competencia}")
    print(f"  Municípios scored: {n_scored:,}")

    # ── resumo ─────────────────────────────────────────────────────
    print("\n── Distribuição de risco ──")
    (
        df_output.groupBy("risk_class", "model_alias")
        .count()
        .orderBy("risk_class", "model_alias")
        .show()
    )

    print("\n── Top 10 municípios em alto risco ──")
    (
        df_output.filter(F.col("risk_class") == "alto")
        .orderBy(F.col("risk_score").desc())
        .select("municipio_id", "municipio_nome", "uf", "risk_score", "risk_class", "model_alias")
        .show(10, truncate=False)
    )

    return competencia


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    spark = SparkSession.builder.getOrCreate()
    ab_test = "--ab" in sys.argv
    score(spark, ab_test=ab_test)
