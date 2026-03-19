# src/transforms/silver_capacity_municipio_mes.py
# Transform Bronze → Silver — Capacidade hospitalar por município × competência mensal
#
# Fonte  : ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos
# Destino: ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_mes
#
# Grain de saída: municipio_id (CO_IBGE, 6 dígitos) × competencia (AAAAMM)
#
# Por que granularidade mensal (e não semanal):
#   A fonte de capacidade (CNES / Hospitais e Leitos) é publicada mensalmente.
#   Cada arquivo representa a fotografia da capacidade instalada naquele mês.
#   Tentar derivar semanas a partir de COMP introduziria ambiguidade (qual das ~4
#   semanas do mês representa o mês inteiro?) sem ganho real de informação.
#   A silver mantém o grain natural da fonte: um registro por município por mês.
#
# Join downstream (gold_features):
#   Para cada semana epidemiológica W, o gold faz:
#     competencia = AAAAMM  onde  date(AAAA, MM, 01) <= data_semana_W <= último dia do mês
#   Isso propaga a capacidade mensal para todas as semanas contidas naquele mês,
#   sem duplicar nem perder dados.
#
# Decisão de filtragem:
#   Registros com municipio_id nulo ou leitos_totais < 0 são removidos, mas
#   a contagem de descartados é sempre logada. A remoção silenciosa é evitada
#   para facilitar auditoria e detecção de problemas na bronze.

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config import CATALOG, SCHEMA, TABLE_BRONZE_HOSPITAIS_LEITOS, TABLE_SILVER_CAPACITY
from quality.checks import checks_silver_capacity
from quality.runner import run_checks

# ── configuração ────────────────────────────────────────────────

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
def _resolver_municipio_id(df):
    """
    Preenche CO_IBGE nos anos 2023/2024 (onde a coluna é ausente na fonte)
    via lookup construído a partir dos registros 2025/2026 (que têm CO_IBGE).

    Estratégia:
      lookup = registros com CO_IBGE não nulo
               agrupados por upper(trim(MUNICIPIO)) → primeiro CO_IBGE distinto
      Para registros com CO_IBGE nulo, join pelo nome normalizado do município.
    """
    nulos_antes = df.filter(F.col("CO_IBGE").isNull()).count()
    print("\n── Resolução de municipio_id ──")
    print(f"  Registros com CO_IBGE nulo antes do lookup: {nulos_antes:,}")

    if nulos_antes == 0:
        print("  Nenhum registro a resolver — pulando lookup.")
        return df

    # lookup: nome normalizado → CO_IBGE (apenas de registros com código já conhecido)
    lookup = (
        df.filter(F.col("CO_IBGE").isNotNull())
        .select(
            F.upper(F.trim(F.col("MUNICIPIO"))).alias("municipio_key"),
            F.col("CO_IBGE").alias("_co_ibge_lookup"),
        )
        .dropDuplicates(["municipio_key"])
    )

    # adiciona chave normalizada ao df principal para o join
    df = df.withColumn("_municipio_key", F.upper(F.trim(F.col("MUNICIPIO"))))

    # join pelo nome normalizado com o lookup
    df = df.join(
        lookup.withColumnRenamed("municipio_key", "_municipio_key"),
        on="_municipio_key",
        how="left",
    )

    # preenche CO_IBGE onde era nulo com o valor encontrado no lookup
    df = df.withColumn(
        "CO_IBGE",
        F.coalesce(F.col("CO_IBGE"), F.col("_co_ibge_lookup")),
    ).drop("_municipio_key", "_co_ibge_lookup")

    nulos_resolvidos = nulos_antes - df.filter(F.col("CO_IBGE").isNull()).count()
    nulos_restantes = df.filter(F.col("CO_IBGE").isNull()).count()
    print(f"  Resolvidos pelo lookup: {nulos_resolvidos:,}")
    if nulos_restantes:
        print(f"  ⚠ Continuam nulos após lookup (município não encontrado): {nulos_restantes:,}")
    else:
        print("  ✓ Todos os registros resolvidos.")
    return df


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
            .cast("integer"),  # garante int mesmo se o when retornar nulo
        )
        # nulos remanescentes (ex: strings não numéricas) → 0
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    return df


def _flag_hospital(df):
    """Marca 1 se DS_TIPO_UNIDADE contém 'HOSPITAL' (case insensitive)."""
    return df.withColumn(
        "is_hospital",
        F.when(F.upper(F.col("DS_TIPO_UNIDADE")).contains("HOSPITAL"), F.lit(1)).otherwise(
            F.lit(0)
        ),
    )


def _agregar(df):
    """Agrega por municipio_id × competencia (AAAAMM)."""
    return (
        df.groupBy("CO_IBGE", "COMP")
        .agg(
            F.first("MUNICIPIO", ignorenulls=True).alias("municipio_nome"),
            F.first("UF", ignorenulls=True).alias("uf"),
            F.first("REGIAO", ignorenulls=True).alias("regiao"),
            F.countDistinct("CNES").alias("num_estabelecimentos"),
            F.sum("is_hospital").alias("num_hospitais"),
            F.sum("LEITOS_EXISTENTES").alias("leitos_totais"),
            F.sum("LEITOS_SUS").alias("leitos_sus"),
            F.sum("UTI_TOTAL_EXIST").alias("leitos_uti"),
            F.sum("UTI_TOTAL_SUS").alias("leitos_uti_sus"),
            F.sum("UTI_ADULTO_EXIST").alias("leitos_uti_adulto"),
            F.sum("UTI_PEDIATRICO_EXIST").alias("leitos_uti_pediatrico"),
            F.sum("UTI_NEONATAL_EXIST").alias("leitos_uti_neonatal"),
        )
        .withColumnRenamed("CO_IBGE", "municipio_id")
        .withColumnRenamed("COMP", "competencia")
    )


def _validar_e_filtrar(df):
    """
    Aplica regras de qualidade mínima e loga quantas linhas cada regra descartou.
    Retorna o DataFrame limpo.
    """
    total = df.count()
    print("\n── Validação de qualidade ──")
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
    return df.withColumn("_processed_at", F.current_timestamp()).withColumn(
        "_source_table", F.lit("bronze_hospitais_leitos")
    )


def transformar(spark: SparkSession):
    """
    Executa o pipeline completo Bronze → Silver de capacidade hospitalar.
    """
    print(f"Lendo {TABLE_BRONZE_HOSPITAIS_LEITOS} ...")
    df = spark.table(TABLE_BRONZE_HOSPITAIS_LEITOS)
    print(f"  Registros na bronze: {df.count():,}")

    df = _resolver_municipio_id(df)
    df = _cast_numerico(df)
    df = _flag_hospital(df)
    df = _agregar(df)
    df = _validar_e_filtrar(df)
    df = _adicionar_metadados(df)

    df = run_checks(
        spark,
        df,
        checks=checks_silver_capacity(),
        table_name=TABLE_SILVER_CAPACITY,
        quarantine_table=f"{CATALOG}.{SCHEMA}.quarantine_silver_capacity",
    )

    print(f"\nGravando em {TABLE_SILVER_CAPACITY} ...")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE_SILVER_CAPACITY} USING DELTA")
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .saveAsTable(TABLE_SILVER_CAPACITY)
    )
    print(f"✓ {TABLE_SILVER_CAPACITY} atualizada.")


def show_summary(spark: SparkSession):
    """
    Imprime estatísticas básicas da silver para validação pós-execução:
    - contagem de linhas por competencia
    - totais nacionais de leitos e UTI
    """
    print(f"\n── Resumo da tabela {TABLE_SILVER_CAPACITY} ──")
    df = spark.table(TABLE_SILVER_CAPACITY)

    print(f"  Total de linhas: {df.count():,}")
    print(f"  Municípios distintos: {df.select('municipio_id').distinct().count():,}")
    print(f"  Competências distintas: {df.select('competencia').distinct().count():,}")

    print("\n  Leitos por competência (amostra):")
    (
        df.groupBy("competencia")
        .agg(
            F.count("municipio_id").alias("municipios"),
            F.sum("leitos_totais").alias("leitos_totais_br"),
            F.sum("leitos_uti").alias("leitos_uti_br"),
        )
        .orderBy("competencia")
        .show(20, truncate=False)
    )


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    transformar(spark)
