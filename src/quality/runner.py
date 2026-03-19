# src/quality/runner.py
# Helper para executar checks DQX e gravar linhas inválidas em quarentena.
#
# Uso:
#   from quality.runner import run_checks
#   df = run_checks(spark, df, checks_bronze_srag(), TABLE_BRONZE_SRAG,
#                  f"{CATALOG}.{SCHEMA}.quarantine_bronze_srag")
#
# Retorna o DataFrame filtrado (somente linhas válidas).
# Linhas que falharam em checks de criticality="error" são gravadas na
# tabela de quarentena em modo append para auditoria posterior.

from datetime import datetime

from databricks.labs.dqx.engine import DQEngine
from pyspark.sql import DataFrame, SparkSession


def run_checks(
    spark: SparkSession,
    df: DataFrame,
    checks: list[dict],
    table_name: str,
    quarantine_table: str,
) -> DataFrame:
    """
    Aplica os checks DQX ao DataFrame e retorna apenas as linhas válidas.

    Linhas que falharam em checks de criticality="error" são gravadas na
    tabela de quarentena com metadados de rastreabilidade.

    Args:
        spark:            SparkSession ativa.
        df:               DataFrame a ser validado.
        checks:           Lista de checks DQX (saída de checks_*.py).
        table_name:       Nome da tabela de destino (usado nos logs).
        quarantine_table: Nome completo da tabela de quarentena Delta.

    Returns:
        DataFrame com somente as linhas que passaram em todos os checks "error".
    """
    engine = DQEngine(spark)
    valid_df, quarantine_df = engine.apply_checks_by_metadata_and_split(df, checks)

    qtd_validas = valid_df.count()
    qtd_quarentena = quarantine_df.count()

    print(f"  DQX [{table_name}]: {qtd_validas:,} válidas | {qtd_quarentena:,} em quarentena")

    if qtd_quarentena > 0:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        quarantine_df = quarantine_df.withColumn(
            "_quarantine_ts",
            __import__("pyspark.sql.functions", fromlist=["lit"]).lit(ts),
        ).withColumn(
            "_quarantine_target",
            __import__("pyspark.sql.functions", fromlist=["lit"]).lit(table_name),
        )

        quarantine_df.write.format("delta").mode("append").option(
            "mergeSchema", "true"
        ).saveAsTable(quarantine_table)

        print(f"  ⚠  {qtd_quarentena:,} linha(s) gravada(s) em {quarantine_table}")

    return valid_df
