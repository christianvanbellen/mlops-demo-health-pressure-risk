# src/transforms/gold_pressure_features.py
# Transform Silver → Gold — Feature store de pressão assistencial
#
# Fontes:
#   silver_srag_municipio_semana    → demanda (casos SRAG hospitalizados)
#   silver_capacity_municipio_mes   → oferta (leitos e estabelecimentos)
# Destino:
#   gold_pressure_features
#
# Grain de saída: municipio_id × semana_epidemiologica
#
# Join key entre fontes:
#   srag.competencia == cap.competencia   (ambas derivam AAAAMM do mesmo período)
#   + municipio_id
#
# Pressure Score — versão v1 (pressure_formula_version = "v1"):
#   demand         = casos_srag_ma2 + 0.5 * casos_obito_lag1
#   capacity       = leitos_totais  + 2.0 * leitos_uti
#   pressure_score = demand / (capacity + 1) + 0.3 * growth_wow
#   Pesos heurísticos α=0.5, β=2.0, γ=0.3 — versionados para rastreabilidade.
#
# Target — versão v1 (target_definition_version = "v1"):
#   target_high_pressure = 1 se pressure_score(t+1) >= percentil_85_nacional(t+1)
#   O percentil é calculado sobre todos os municípios por semana (não por município).
#   Semanas sem t+1 (últimas semanas de cada município) → target = null.
#   ESSAS LINHAS SÃO MANTIDAS — são usadas no scoring ao vivo (semana corrente).
#
# Aviso de atraso de notificação:
#   As últimas 2 semanas de cada ano podem estar incompletas na bronze_srag.
#   O modelo de treino deve filtrar por semanas suficientemente antigas.
#   Esta tabela mantém todas as semanas para não perder o dado de scoring.

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from databricks.feature_engineering import FeatureEngineeringClient

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA  = "dev_christian_van_bellen"

TABLE_SRC_SRAG = f"{CATALOG}.{SCHEMA}.silver_srag_municipio_semana"
TABLE_SRC_CAP  = f"{CATALOG}.{SCHEMA}.silver_capacity_municipio_mes"
TABLE_DST      = f"{CATALOG}.{SCHEMA}.gold_pressure_features"

# Windows reutilizados em múltiplas funções
MUNICIPIO_W  = Window.partitionBy("municipio_id").orderBy("semana_epidemiologica")
MUNICIPIO_W2 = MUNICIPIO_W.rowsBetween(-2, -1)   # 2 semanas anteriores (para MA2)
MUNICIPIO_W4 = MUNICIPIO_W.rowsBetween(-4, -1)   # 4 semanas anteriores (para MA4, std)


# ── funções ─────────────────────────────────────────────────────
def _join_srag_capacity(srag, cap):
    """
    LEFT JOIN de srag com cap em municipio_id + competencia.
    Capacidade nula após o join (município sem dados naquele mês) → 0.
    Prefixamos as colunas de cap para evitar ambiguidade no join.
    """
    cap_renamed = cap.select(
        F.col("municipio_id")         .alias("_cap_municipio_id"),
        F.col("competencia")          .alias("_cap_competencia"),
        F.col("municipio_nome"),
        F.col("uf")                   .alias("_cap_uf"),
        F.col("regiao"),
        F.col("num_estabelecimentos"),
        F.col("num_hospitais"),
        F.col("leitos_totais"),
        F.col("leitos_sus"),
        F.col("leitos_uti"),
        F.col("leitos_uti_sus"),
        F.col("leitos_uti_adulto"),
        F.col("leitos_uti_pediatrico"),
        F.col("leitos_uti_neonatal"),
    )

    df = srag.join(
        cap_renamed,
        (srag["municipio_id"] == cap_renamed["_cap_municipio_id"]) &
        (srag["competencia"]  == cap_renamed["_cap_competencia"]),
        how="left",
    ).drop("_cap_municipio_id", "_cap_competencia")

    # uf: usa o da srag, cai para o da cap se nulo
    df = df.withColumn("uf", F.coalesce(F.col("uf"), F.col("_cap_uf"))).drop("_cap_uf")

    # campos de capacidade: nulo → 0 para não contaminar o pressure score
    for col in [
        "num_estabelecimentos", "num_hospitais",
        "leitos_totais", "leitos_sus", "leitos_uti", "leitos_uti_sus",
        "leitos_uti_adulto", "leitos_uti_pediatrico", "leitos_uti_neonatal",
    ]:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))

    sem_capacity = df.filter(F.col("leitos_totais") == 0).count()
    print(f"  Municípios/semanas sem capacity após join: {sem_capacity:,}  (leitos_totais = 0)")

    return df


def _filtrar_sem_hospital(df):
    """
    Remove município-semanas sem capacidade hospitalar cadastrada
    (leitos_totais = 0 após o join com capacity).
    Esses municípios não têm hospital próprio — casos SRAG registrados
    provavelmente foram atendidos em outro município.
    Mantê-los distorce o pressure_score e o percentil do target.
    """
    total = df.count()
    sem_leitos = df.filter(F.col("leitos_totais") == 0).count()
    print(f"\n── Filtro sem hospital ──")
    print(f"  Removidos (leitos_totais = 0): {sem_leitos:,}  ({sem_leitos/total*100:.1f}%)")
    print(f"  Municípios sem hospital: ~3,637 (23% do total)")
    df = df.filter(F.col("leitos_totais") > 0)
    print(f"  Mantidos: {df.count():,}")
    return df


def _lags_e_medias_moveis(df):
    """
    Lags de casos_srag e casos_obito, e médias móveis MA2/MA4.
    Primeiras semanas de cada município ficam nulas → coalesce com 0.
    """
    df = (
        df
        .withColumn("casos_srag_lag1",  F.coalesce(F.lag("casos_srag",  1).over(MUNICIPIO_W), F.lit(0)))
        .withColumn("casos_srag_lag2",  F.coalesce(F.lag("casos_srag",  2).over(MUNICIPIO_W), F.lit(0)))
        .withColumn("casos_obito_lag1", F.coalesce(F.lag("casos_obito", 1).over(MUNICIPIO_W), F.lit(0)))
        # MA2: média das 2 semanas anteriores (rowsBetween -2, -1)
        .withColumn("casos_srag_ma2",   F.coalesce(F.avg("casos_srag").over(MUNICIPIO_W2), F.col("casos_srag_lag1").cast("double")))
        # MA4: média das 4 semanas anteriores (rowsBetween -4, -1)
        .withColumn("casos_srag_ma4",   F.coalesce(F.avg("casos_srag").over(MUNICIPIO_W4), F.col("casos_srag_lag1").cast("double")))
    )
    return df


def _features_dinamica(df):
    """
    Features de tendência e aceleração da demanda.
    growth_wow: crescimento semana a semana (denominador + 1 para evitar divisão por 0).
    """
    df = (
        df
        .withColumn(
            "growth_wow",
            F.round((F.col("casos_srag") - F.col("casos_srag_lag1")) / (F.col("casos_srag_lag1") + 1), 4),
        )
        .withColumn(
            "growth_2w",
            F.round((F.col("casos_srag") - F.col("casos_srag_lag2")) / (F.col("casos_srag_lag2") + 1), 4),
        )
    )
    # acceleration precisa do lag de growth_wow — calculado após derivar growth_wow
    df = df.withColumn(
        "acceleration",
        F.round(F.col("growth_wow") - F.coalesce(F.lag("growth_wow", 1).over(MUNICIPIO_W), F.lit(0.0)), 4),
    )
    df = df.withColumn(
        "rolling_std_4w",
        F.round(F.coalesce(F.stddev("casos_srag").over(MUNICIPIO_W4), F.lit(0.0)), 4),
    )
    return df


def _features_pressao_relativa(df):
    """
    Ratios de demanda por capacidade instalada — todos com lag1 para evitar data leakage.
    """
    casos_uti_lag1 = F.coalesce(F.lag("casos_uti", 1).over(MUNICIPIO_W), F.lit(0))

    df = (
        df
        .withColumn("casos_uti_lag1", casos_uti_lag1)
        .withColumn(
            "srag_per_bed_lag1",
            F.round(F.col("casos_srag_lag1") / (F.col("leitos_totais") + 1), 4),
        )
        .withColumn(
            "srag_per_icu_lag1",
            F.round(F.col("casos_srag_lag1") / (F.col("leitos_uti") + 1), 4),
        )
        .withColumn(
            "severe_per_icu_lag1",
            F.round(F.col("casos_uti_lag1") / (F.col("leitos_uti") + 1), 4),
        )
        .withColumn(
            "deaths_per_icu_lag1",
            F.round(F.col("casos_obito_lag1") / (F.col("leitos_uti") + 1), 4),
        )
    )
    return df


def _calcular_pressure_score(df):
    """
    Pressure Score v1:
      demand         = MA2(casos_srag) + 0.5 * casos_obito_lag1
      capacity       = leitos_totais   + 2.0 * leitos_uti
      pressure_score = demand / (capacity + 1) + 0.3 * growth_wow
    Pesos: α=0.5 (mortes), β=2.0 (UTI), γ=0.3 (tendência).
    """
    demand   = F.col("casos_srag_ma2") + F.lit(0.5) * F.coalesce(F.col("casos_obito_lag1"), F.lit(0.0))
    capacity = F.col("leitos_totais").cast("double") + F.lit(2.0) * F.col("leitos_uti").cast("double")

    df = (
        df
        .withColumn(
            "pressure_score",
            F.round(demand / (capacity + 1) + F.lit(0.3) * F.col("growth_wow"), 6),
        )
        .withColumn("pressure_formula_version", F.lit("v1"))
    )
    return df


def _calcular_target(df):
    """
    Target v1: binário de alta pressão na semana seguinte (t+1).
      1 se pressure_score(t+1) >= percentil_85_nacional da semana t+1
      null se não há semana t+1 (últimas semanas — mantidas para scoring)
    """
    # passo 1: pressure_score da próxima semana para cada município
    df = df.withColumn(
        "pressure_score_next",
        F.lead("pressure_score", 1).over(MUNICIPIO_W),
    )

    # passo 2: percentil 85 nacional por semana_epidemiologica
    p85 = (
        df.groupBy("semana_epidemiologica")
        .agg(
            F.percentile_approx("pressure_score_next", 0.85).alias("_p85_nacional"),
        )
    )

    # passo 3: join do p85 de volta ao df principal
    df = df.join(p85, on="semana_epidemiologica", how="left")

    # passo 4: target binário (null onde pressure_score_next é null → sem t+1)
    df = (
        df
        .withColumn(
            "target_high_pressure",
            F.when(
                F.col("pressure_score_next").isNotNull(),
                F.when(F.col("pressure_score_next") >= F.col("_p85_nacional"), F.lit(1))
                 .otherwise(F.lit(0)),
            ),
            # quando pressure_score_next é null, o when externo retorna null implicitamente
        )
        .withColumn("target_definition_version", F.lit("v1"))
        .drop("pressure_score_next", "_p85_nacional")
    )
    return df


def _features_sazonais(df):
    """Extrai features de sazonalidade a partir de semana_epidemiologica (AAAA-WW)."""
    df = (
        df
        .withColumn("ano",          F.col("semana_epidemiologica").substr(1, 4).cast("integer"))
        .withColumn("epi_week",     F.col("semana_epidemiologica").substr(6, 2).cast("integer"))
        .withColumn("quarter",      F.ceil(F.col("epi_week") / 13).cast("integer"))
        .withColumn("is_semester1", F.when(F.col("epi_week") <= 26, F.lit(1)).otherwise(F.lit(0)))
    )
    return df


def _validar_e_filtrar(df):
    """
    Loga e filtra anomalias de dados.
    NÃO remove semanas com target nulo — essas são usadas para scoring ao vivo.
    """
    total = df.count()
    print(f"\n── Validação de qualidade ──")
    print(f"  Total antes da filtragem: {total:,}")

    # informativo: semanas sem target (útimas semanas por município, sem t+1)
    sem_target = df.filter(F.col("target_high_pressure").isNull()).count()
    print(f"  Semanas sem target (scoring only, mantidas): {sem_target:,}")

    # regra 1: municipio_id não nulo
    sem_municipio = df.filter(F.col("municipio_id").isNull()).count()
    if sem_municipio:
        print(f"  ⚠ Descartados por municipio_id nulo: {sem_municipio:,}")
    df = df.filter(F.col("municipio_id").isNotNull())

    # regra 2: pressure_score não nulo
    sem_score = df.filter(F.col("pressure_score").isNull()).count()
    if sem_score:
        print(f"  ⚠ Descartados por pressure_score nulo: {sem_score:,}")
    df = df.filter(F.col("pressure_score").isNotNull())

    # regra 3: target_high_pressure deve ser 0, 1 ou null — filtrar valores corrompidos
    target_invalido = df.filter(
        F.col("target_high_pressure").isNotNull() &
        ~F.col("target_high_pressure").isin(0, 1)
    ).count()
    if target_invalido:
        print(f"  ⚠ Descartados por target_high_pressure fora de {{0, 1}}: {target_invalido:,}")
    df = df.filter(
        F.col("target_high_pressure").isNull() |
        F.col("target_high_pressure").isin(0, 1)
    )

    total_apos = df.count()
    print(f"  Total após filtragem: {total_apos:,}  (descartados: {total - total_apos:,})")
    return df


def _adicionar_metadados(df):
    """Adiciona colunas de rastreabilidade do processamento."""
    return (
        df
        .withColumn("_processed_at", F.current_timestamp())
        .withColumn("_source_srag",  F.lit(TABLE_SRC_SRAG))
        .withColumn("_source_cap",   F.lit(TABLE_SRC_CAP))
    )


def transformar(spark: SparkSession):
    """
    Executa o pipeline completo Silver → Gold de features de pressão assistencial.
    Grava via Databricks Feature Engineering Client para registrar lineage e
    permitir point-in-time lookups no treinamento.
    """
    fe = FeatureEngineeringClient()

    print("Lendo fontes silver ...")
    srag = spark.table(TABLE_SRC_SRAG)
    cap  = spark.table(TABLE_SRC_CAP)
    print(f"  srag: {srag.count():,} linhas | cap: {cap.count():,} linhas")

    df = _join_srag_capacity(srag, cap)
    df = _filtrar_sem_hospital(df)
    df = _lags_e_medias_moveis(df)
    df = _features_dinamica(df)
    df = _features_pressao_relativa(df)
    df = _calcular_pressure_score(df)
    df = _calcular_target(df)
    df = _features_sazonais(df)
    df = _validar_e_filtrar(df)
    df = _adicionar_metadados(df)

    # verifica se a feature table já existe
    try:
        fe.get_table(TABLE_DST)
        table_exists = True
    except Exception:
        table_exists = False

    if not table_exists:
        print(f"\nCriando feature table {TABLE_DST} ...")
        fe.create_table(
            name=TABLE_DST,
            primary_keys=["municipio_id", "semana_epidemiologica"],
            df=df,
            description=(
                "Feature store de pressão assistencial respiratória. "
                "Grain: municipio_id x semana_epidemiologica. "
                "pressure_formula_version=v1, target_definition_version=v1."
            ),
        )
    else:
        print(f"\nAtualizando feature table {TABLE_DST} ...")
        fe.write_table(
            name=TABLE_DST,
            df=df,
            mode="overwrite",
        )

    print(f"✓ Feature table {TABLE_DST} atualizada.")


def show_summary(spark: SparkSession):
    """
    Imprime estatísticas básicas da gold para validação pós-execução:
    - totais gerais
    - distribuição do target por ano
    - estatísticas do pressure_score
    - amostra de 5 linhas
    """
    print(f"\n── Resumo da tabela {TABLE_DST} ──")
    df = spark.table(TABLE_DST)

    print(f"  Total de linhas: {df.count():,}")
    print(f"  Municípios distintos: {df.select('municipio_id').distinct().count():,}")
    print(f"  Semanas distintas:    {df.select('semana_epidemiologica').distinct().count():,}")

    print("\n  Distribuição do target por ano (% de alta pressão):")
    (
        df.filter(F.col("target_high_pressure").isNotNull())
        .withColumn("ano", F.col("semana_epidemiologica").substr(1, 4))
        .groupBy("ano")
        .agg(
            F.count("*")                                             .alias("total"),
            F.sum("target_high_pressure")                           .alias("positivos"),
            F.round(F.avg("target_high_pressure") * 100, 2)         .alias("pct_positivos"),
        )
        .orderBy("ano")
        .show(truncate=False)
    )

    print("\n  Estatísticas do pressure_score:")
    df.select(
        F.round(F.min("pressure_score"),    6).alias("min"),
        F.round(F.max("pressure_score"),    6).alias("max"),
        F.round(F.avg("pressure_score"),    6).alias("mean"),
        F.round(F.stddev("pressure_score"), 6).alias("stddev"),
    ).show(truncate=False)

    print("\n  Amostra (5 linhas):")
    df.select(
        "municipio_id", "semana_epidemiologica",
        "pressure_score", "target_high_pressure",
        "casos_srag", "leitos_totais", "leitos_uti",
    ).show(5, truncate=False)


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    transformar(spark)
