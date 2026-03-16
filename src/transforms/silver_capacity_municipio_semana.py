# src/transforms/silver_capacity_municipio_semana.py
# Transform Bronze → Silver — Capacidade hospitalar por município × semana epidemiológica
#
# Fonte  : ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos
# Destino: ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_semana
#
# Grain de saída: municipio_id (CO_IBGE, 6 dígitos) × semana_epidemiologica (AAAA-WW)
#
# Lógica de semana epidemiológica:
#   A bronze usa competência mensal (COMP = AAAAMM). Cada mês cobre ~4 semanas
#   epidemiológicas, mas para o MVP optamos por representar cada competência pela
#   semana do seu primeiro dia (date(AAAA, MM, 01)).
#   Isso garante uma granularidade temporal sem ambiguidade no join com a silver_srag,
#   que opera em semanas. A limitação — uma competência mapear para uma única semana
#   em vez de 4 — é aceitável para o MVP e deve ser revisada na Fase 2 se o modelo
#   precisar de variação intra-mensal de capacidade.
#
# Decisão de filtragem:
#   Registros com municipio_id nulo ou leitos_totais < 0 são removidos, mas
#   a contagem de descartados é sempre logada. A remoção silenciosa é evitada
#   para facilitar auditoria e detecção de problemas na bronze.

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA  = "dev_christian_van_bellen"

TABLE_SRC = f"{CATALOG}.{SCHEMA}.bronze_hospitais_leitos"
TABLE_DST = f"{CATALOG}.{SCHEMA}.silver_capacity_municipio_semana"

# Colunas numéricas da bronze (chegam como string) — nulos e "" viram 0
COLUNAS_NUMERICAS = [
    "LEITOS_EXISTENTES",
    "LEITOS_SUS",
    "UTI_TOTAL_EXIST",
    "UTI_TOTAL_SUS",
    "UTI_ADULTO_EXIST",
    "UTI_PEDIATRICO_EXIST",
    "UTI_NEONATAL_EXIST",
]


# ── funções ─────────────────────────────────────────────────────
def _cast_numerico(df):
    """
    Converte colunas numéricas de string para integer.
    Strings vazias ou nulas viram 0 para não perder registros válidos na agregação.
    """
    for col in COLUNAS_NUMERICAS:
        df = df.withColumn(
            col,
            F.when(F.trim(F.col(col)) == "", F.lit(0))
            .otherwise(F.col(col).cast("integer"))
            .cast("integer")  # garante int mesmo se o when retornar nulo
        )
        # nulos remanescentes (ex: strings não numéricas) → 0
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    return df


def _derivar_semana_epidemiologica(df):
    """
    Converte COMP (AAAAMM) para data do primeiro dia do mês,
    depois deriva ano_epi, semana_epi e semana_epidemiologica (AAAA-WW).
    """
    df = df.withColumn(
        "_data_ref",
        F.to_date(F.concat_ws("-", F.col("COMP").substr(1, 4), F.col("COMP").substr(5, 2), F.lit("01")), "yyyy-MM-dd"),
    )
    df = df.withColumn("_semana_num", F.weekofyear(F.col("_data_ref")))
    df = df.withColumn("_ano_epi",    F.year(F.col("_data_ref")))
    # formata AAAA-WW com zero-padding na semana
    df = df.withColumn(
        "semana_epidemiologica",
        F.concat(
            F.col("_ano_epi").cast("string"),
            F.lit("-"),
            F.lpad(F.col("_semana_num").cast("string"), 2, "0"),
        ),
    )
    return df.drop("_data_ref", "_semana_num", "_ano_epi")


def _flag_hospital(df):
    """Marca 1 se DS_TIPO_UNIDADE contém 'HOSPITAL' (case insensitive)."""
    return df.withColumn(
        "is_hospital",
        F.when(F.upper(F.col("DS_TIPO_UNIDADE")).contains("HOSPITAL"), F.lit(1)).otherwise(F.lit(0)),
    )


def _agregar(df):
    """Agrega por municipio_id × semana_epidemiologica."""
    return (
        df.groupBy("CO_IBGE", "semana_epidemiologica")
        .agg(
            F.first("MUNICIPIO",  ignorenulls=True).alias("municipio_nome"),
            F.first("UF",         ignorenulls=True).alias("uf"),
            F.first("REGIAO",     ignorenulls=True).alias("regiao"),
            F.first("COMP",       ignorenulls=True).alias("competencia_ref"),
            F.countDistinct("CNES")                .alias("num_estabelecimentos"),
            F.sum("is_hospital")                   .alias("num_hospitais"),
            F.sum("LEITOS_EXISTENTES")             .alias("leitos_totais"),
            F.sum("LEITOS_SUS")                    .alias("leitos_sus"),
            F.sum("UTI_TOTAL_EXIST")               .alias("leitos_uti"),
            F.sum("UTI_TOTAL_SUS")                 .alias("leitos_uti_sus"),
            F.sum("UTI_ADULTO_EXIST")              .alias("leitos_uti_adulto"),
            F.sum("UTI_PEDIATRICO_EXIST")          .alias("leitos_uti_pediatrico"),
            F.sum("UTI_NEONATAL_EXIST")            .alias("leitos_uti_neonatal"),
        )
        .withColumnRenamed("CO_IBGE", "municipio_id")
    )


def _validar_e_filtrar(df):
    """
    Aplica regras de qualidade mínima e loga quantas linhas cada regra descartou.
    Retorna o DataFrame limpo.
    """
    total = df.count()
    print(f"\n── Validação de qualidade ──")
    print(f"  Total antes da filtragem: {total:,}")

    # regra 1: municipio_id não nulo
    sem_municipio = df.filter(F.col("municipio_id").isNull()).count()
    if sem_municipio:
        print(f"  ⚠ Descartados por municipio_id nulo: {sem_municipio:,}")
    df = df.filter(F.col("municipio_id").isNotNull())

    # regra 2: leitos_totais >= 0 (cast falho geraria negativo, mas coalesce já garante 0)
    leitos_negativos = df.filter(F.col("leitos_totais") < 0).count()
    if leitos_negativos:
        print(f"  ⚠ Descartados por leitos_totais < 0: {leitos_negativos:,}")
    df = df.filter(F.col("leitos_totais") >= 0)

    total_apos = df.count()
    print(f"  Total após filtragem: {total_apos:,}  (descartados: {total - total_apos:,})")
    return df


def _adicionar_metadados(df):
    """Adiciona colunas de rastreabilidade do processamento."""
    return (
        df
        .withColumn("_processed_at",  F.current_timestamp())
        .withColumn("_source_table",  F.lit("bronze_hospitais_leitos"))
    )


def transformar(spark: SparkSession):
    """
    Executa o pipeline completo Bronze → Silver de capacidade hospitalar.
    """
    print(f"Lendo {TABLE_SRC} ...")
    df = spark.table(TABLE_SRC)
    print(f"  Registros na bronze: {df.count():,}")

    df = _cast_numerico(df)
    df = _flag_hospital(df)
    df = _derivar_semana_epidemiologica(df)
    df = _agregar(df)
    df = _validar_e_filtrar(df)
    df = _adicionar_metadados(df)

    print(f"\nGravando em {TABLE_DST} ...")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE_DST} USING DELTA")
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .saveAsTable(TABLE_DST)
    )
    print(f"✓ {TABLE_DST} atualizada.")


def show_summary(spark: SparkSession):
    """
    Imprime estatísticas básicas da silver para validação pós-execução:
    - contagem de linhas por semana_epidemiologica
    - ranges de leitos_totais e leitos_uti
    """
    print(f"\n── Resumo da tabela {TABLE_DST} ──")
    df = spark.table(TABLE_DST)

    print(f"  Total de linhas: {df.count():,}")
    print(f"  Municípios distintos: {df.select('municipio_id').distinct().count():,}")
    print(f"  Semanas distintas: {df.select('semana_epidemiologica').distinct().count():,}")

    print("\n  Leitos por semana (amostra):")
    (
        df.groupBy("semana_epidemiologica")
        .agg(
            F.count("municipio_id")     .alias("municipios"),
            F.sum("leitos_totais")      .alias("leitos_totais_br"),
            F.sum("leitos_uti")         .alias("leitos_uti_br"),
        )
        .orderBy("semana_epidemiologica")
        .show(20, truncate=False)
    )


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    transformar(spark)
