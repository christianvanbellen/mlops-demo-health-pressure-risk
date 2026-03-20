# src/transforms/silver_srag_municipio_semana.py
# Transform Bronze → Silver — Casos SRAG hospitalizados por município × semana epidemiológica
#
# Fonte  : ds_dev_db.dev_christian_van_bellen.bronze_srag
# Destino: ds_dev_db.dev_christian_van_bellen.silver_srag_municipio_semana
#
# Grain de saída: municipio_id (CO_MUN_RES, 6 dígitos) × semana_epidemiologica (AAAA-WW)
#
# Decodificação dos campos categóricos da bronze:
#   HOSPITAL : 1=sim, 2=não, 9=ignorado, NULL
#              → filtro: só HOSPITAL="1" (hospitalizados)
#   UTI      : 1=sim, 2=não, 9=ignorado, NULL, e 1 registro corrompido com texto
#              → UTI_flag = 1 apenas para "1" (trata texto como não-UTI)
#   EVOLUCAO : 1=cura, 2=óbito, 3=óbito outras causas, 9=ignorado, NULL
#              → OBITO_flag = 1 para "2" ou "3"
#
# ⚠ Atraso de notificação:
#   As últimas 2 semanas de cada ano podem estar incompletas no momento da ingestão.
#   Semanas recentes devem ser excluídas do treino e reprocessadas nas ingestões
#   subsequentes. Consulte CLAUDE.md para a política de reprocessamento.
#
# Derivação de competencia:
#   A partir de _ano_arquivo + SEM_PRI, calcula a data aproximada da semana
#   (primeiro dia do ano + (SEM_PRI - 1) * 7 dias) e extrai o mês (AAAAMM).
#   Isso permite o join downstream com silver_capacity_municipio_mes:
#     silver_srag.competencia == silver_capacity.competencia
#   Um caso notificado na semana 5 de 2025 → data_ref ≈ 2025-01-29 → competencia "202501"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from quality.checks import checks_silver_srag
from quality.runner import run_checks

# ── configuração ────────────────────────────────────────────────


# ── funções ─────────────────────────────────────────────────────
def _filtrar_hospitalizados(df):
    """
    Mantém apenas casos hospitalizados (HOSPITAL = "1").
    Loga quantos registros foram descartados.
    """
    total = df.count()
    df_filtrado = df.filter(F.col("HOSPITAL") == "1")
    descartados = total - df_filtrado.count()
    print("\n── Filtro HOSPITAL = 1 ──")
    print(f"  Total na bronze: {total:,}")
    print(f"  Descartados (HOSPITAL != '1' ou nulo): {descartados:,}")
    print(f"  Mantidos: {df_filtrado.count():,}")
    return df_filtrado


def _filtrar_sem_pri_valido(df):
    """
    Remove registros onde SEM_PRI é nulo ou não conversível para inteiro.
    Loga quantos foram descartados.
    """
    invalidos = df.filter(
        F.col("SEM_PRI").isNull() | F.col("SEM_PRI").cast("integer").isNull()
    ).count()
    if invalidos:
        print(f"  ⚠ Descartados por SEM_PRI nulo ou não numérico: {invalidos:,}")
    df = df.filter(F.col("SEM_PRI").isNotNull() & F.col("SEM_PRI").cast("integer").isNotNull())
    print(f"  Mantidos após filtro SEM_PRI: {df.count():,}")
    return df


def _derivar_semana_e_competencia(df):
    """
    Deriva semana_epidemiologica (AAAA-WW) e competencia (AAAAMM) a partir de
    _ano_arquivo (int) e SEM_PRI (string).
    """
    # semana_epidemiologica: AAAA-WW com zero-padding
    df = df.withColumn(
        "semana_epidemiologica",
        F.concat(
            F.col("_ano_arquivo").cast("string"),
            F.lit("-"),
            F.lpad(F.col("SEM_PRI").cast("integer").cast("string"), 2, "0"),
        ),
    )

    # data aproximada da semana: 1º de janeiro do ano + (semana - 1) * 7 dias
    df = df.withColumn(
        "_data_ref",
        F.date_add(
            F.to_date(
                F.concat(F.col("_ano_arquivo").cast("string"), F.lit("-01-01")),
                "yyyy-MM-dd",
            ),
            (F.col("SEM_PRI").cast("integer") - 1) * 7,
        ),
    )

    # competencia = AAAAMM do mês em que a semana cai
    df = df.withColumn("competencia", F.date_format(F.col("_data_ref"), "yyyyMM"))

    return df.drop("_data_ref")


def _derivar_flags(df):
    """
    Deriva flags binárias dos campos categóricos para facilitar a agregação.
    Trata valores corrompidos como 0 (ausência do evento).
    """
    # UTI: só "1" conta — texto corrompido, "2", "9", NULL → 0
    df = df.withColumn(
        "UTI_flag",
        F.when(F.col("UTI") == "1", F.lit(1)).otherwise(F.lit(0)),
    )

    # Óbito: EVOLUCAO "2" (óbito) ou "3" (óbito por outras causas)
    df = df.withColumn(
        "OBITO_flag",
        F.when(F.col("EVOLUCAO").isin("2", "3"), F.lit(1)).otherwise(F.lit(0)),
    )

    # Idoso: idade >= 60 anos (NU_IDADE_N já em anos na bronze)
    df = df.withColumn(
        "IDOSO_flag",
        F.when(F.col("NU_IDADE_N").cast("integer") >= 60, F.lit(1)).otherwise(F.lit(0)),
    )

    # Pediátrico: idade < 12 anos
    df = df.withColumn(
        "PEDIATRICO_flag",
        F.when(F.col("NU_IDADE_N").cast("integer") < 12, F.lit(1)).otherwise(F.lit(0)),
    )

    return df


def _agregar(df):
    """Agrega por municipio_id × semana_epidemiologica."""
    df_agg = (
        df.groupBy("CO_MUN_RES", "semana_epidemiologica")
        .agg(
            F.first("competencia", ignorenulls=True).alias("competencia"),
            F.first("SG_UF_NOT", ignorenulls=True).alias("uf"),
            F.count("*").alias("casos_srag"),
            F.sum("UTI_flag").alias("casos_uti"),
            F.sum("OBITO_flag").alias("casos_obito"),
            F.sum("IDOSO_flag").alias("casos_idosos"),
            F.sum("PEDIATRICO_flag").alias("casos_pediatricos"),
        )
        .withColumnRenamed("CO_MUN_RES", "municipio_id")
    )

    # percentuais derivados — casos_srag >= 1 garantido pelo filtro de qualidade posterior
    df_agg = df_agg.withColumn(
        "pct_uti",
        F.round(F.col("casos_uti") / F.col("casos_srag"), 4),
    )
    df_agg = df_agg.withColumn(
        "pct_obito",
        F.round(F.col("casos_obito") / F.col("casos_srag"), 4),
    )

    return df_agg


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

    # regra 2: casos_srag >= 1 (não deve ocorrer após count(*), mas garante consistência)
    sem_casos = df.filter(F.col("casos_srag") < 1).count()
    if sem_casos:
        print(f"  ⚠ Descartados por casos_srag < 1: {sem_casos:,}")
    df = df.filter(F.col("casos_srag") >= 1)

    # regra 3: semana_epidemiologica no formato AAAA-WW (4 dígitos, hífen, 2 dígitos)
    formato_invalido = df.filter(~F.col("semana_epidemiologica").rlike(r"^\d{4}-\d{2}$")).count()
    if formato_invalido:
        print(
            f"  ⚠ Descartados por semana_epidemiologica fora do formato AAAA-WW: {formato_invalido:,}"
        )
    df = df.filter(F.col("semana_epidemiologica").rlike(r"^\d{4}-\d{2}$"))

    total_apos = df.count()
    print(f"  Total após filtragem: {total_apos:,}  (descartados: {total - total_apos:,})")
    return df


def _adicionar_metadados(df):
    """Adiciona colunas de rastreabilidade do processamento."""
    return df.withColumn("_processed_at", F.current_timestamp()).withColumn(
        "_source_table", F.lit("bronze_srag")
    )


def transformar(spark: SparkSession, args):
    """
    Executa o pipeline completo Bronze → Silver de casos SRAG por município × semana.
    """
    catalog = args.catalog
    schema = args.schema
    table_bronze_srag = args.table_bronze_srag
    table_silver_srag = args.table_silver_srag

    print(f"Lendo {table_bronze_srag} ...")
    df = spark.table(table_bronze_srag)

    df = _filtrar_hospitalizados(df)
    df = _filtrar_sem_pri_valido(df)
    df = _derivar_semana_e_competencia(df)
    df = _derivar_flags(df)
    df = _agregar(df)
    df = _validar_e_filtrar(df)
    df = _adicionar_metadados(df)

    df = run_checks(
        spark,
        df,
        checks=checks_silver_srag(),
        table_name=table_silver_srag,
        quarantine_table=f"{catalog}.{schema}.quarantine_silver_srag",
    )

    print(f"\nGravando em {table_silver_srag} ...")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table_silver_srag} USING DELTA")
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .saveAsTable(table_silver_srag)
    )
    print(f"✓ {table_silver_srag} atualizada.")


def show_summary(spark: SparkSession, args):
    """
    Imprime estatísticas básicas da silver para validação pós-execução:
    - total de linhas, municípios e semanas distintos
    - casos por ano (primeiros 4 caracteres de semana_epidemiologica)
    - top 5 municípios por total de casos
    """
    table_silver_srag = args.table_silver_srag
    print(f"\n── Resumo da tabela {table_silver_srag} ──")
    df = spark.table(table_silver_srag)

    print(f"  Total de linhas: {df.count():,}")
    print(f"  Municípios distintos: {df.select('municipio_id').distinct().count():,}")
    print(f"  Semanas distintas: {df.select('semana_epidemiologica').distinct().count():,}")

    print("\n  Casos SRAG por ano:")
    (
        df.withColumn("ano", F.col("semana_epidemiologica").substr(1, 4))
        .groupBy("ano")
        .agg(
            F.sum("casos_srag").alias("casos_srag_total"),
            F.sum("casos_uti").alias("casos_uti_total"),
            F.sum("casos_obito").alias("casos_obito_total"),
        )
        .orderBy("ano")
        .show(truncate=False)
    )

    print("\n  Top 5 municípios por total de casos (todos os anos):")
    (
        df.groupBy("municipio_id", "uf")
        .agg(F.sum("casos_srag").alias("casos_srag_total"))
        .orderBy(F.col("casos_srag_total").desc())
        .show(5, truncate=False)
    )


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    from cli import build_parser

    p = build_parser("Transform Bronze → Silver — SRAG por município × semana")
    args, _ = p.parse_known_args()

    spark = SparkSession.builder.getOrCreate()
    transformar(spark, args)
