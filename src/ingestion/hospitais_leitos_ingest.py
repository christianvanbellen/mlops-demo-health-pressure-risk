# src/ingestion/hospitais_leitos_ingest.py
# Ingestão Bronze — Hospitais e Leitos
# Fonte: https://dadosabertos.saude.gov.br/dataset/hospitais-e-leitos

import requests
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA  = "dev_christian_van_bellen"
TABLE   = f"{CATALOG}.{SCHEMA}.bronze_hospitais_leitos"

PAGINA_FONTE = "https://dadosabertos.saude.gov.br/dataset/hospitais-e-leitos"

# URL base estável — nomenclatura não muda entre atualizações
URL_BASE = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/Leitos_SUS/Leitos_{ano}.csv"

ANOS = {
    2022: {"formato": "csv", "is_live": False},
    2023: {"formato": "csv", "is_live": False},
    2024: {"formato": "csv", "is_live": False},
}

COLUNAS_ESSENCIAIS = [
    "COMP",
    "REGIAO",
    "UF",
    "MUNICIPIO",           # nome do município — sem código IBGE nesta fonte
    "CNES",
    "NOME_ESTABELECIMENTO",
    "TP_GESTAO",
    "CO_TIPO_UNIDADE",
    "DS_TIPO_UNIDADE",
    "NATUREZA_JURIDICA",
    "DESC_NATUREZA_JURIDICA",
    "LEITOS_EXISTENTES",
    "LEITOS_SUS",
    "UTI_TOTAL_EXIST",
    "UTI_TOTAL_SUS",
    "UTI_ADULTO_EXIST",
    "UTI_ADULTO_SUS",
    "UTI_PEDIATRICO_EXIST",
    "UTI_PEDIATRICO_SUS",
    "UTI_NEONATAL_EXIST",
    "UTI_NEONATAL_SUS",
    # Nota: esta fonte não contém código IBGE do município (apenas nome).
    # O join com municipio_id (IBGE 6 dígitos) será feito na camada silver,
    # via lookup na tabela silver_dim_municipio (UF + nome normalizado → municipio_id).
]

MIN_LINHAS_VALIDAS = 1_000

# ── funções ─────────────────────────────────────────────────────
def scrape_urls() -> dict:
    """
    Monta as URLs diretamente a partir da lista de anos.
    A URL desta fonte é estável (Leitos_{ANO}.csv) e não muda entre atualizações,
    portanto não é necessário scraping. Mantém a assinatura para consistência
    com srag_ingest.py.
    Retorna dict {ano: {csv: url}}.
    """
    return {
        ano: {"csv": URL_BASE.format(ano=ano)}
        for ano in ANOS
    }


def baixar_arquivo(url: str, ano: int, fmt: str) -> str:
    """Baixa o arquivo para o Volume de landing e retorna o caminho local."""
    caminho = f"/Volumes/ds_dev_db/dev_christian_van_bellen/landing/hospitais_leitos_{ano}.{fmt}"
    print(f"  Baixando {ano} ({fmt}) de {url} ...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(caminho, "wb") as f:
        f.write(r.content)
    print(f"  ✓ {len(r.content)/1e6:.1f} MB salvo em {caminho}")
    return caminho


def ler_e_enriquecer(spark: SparkSession, caminho: str, ano: int, url: str, is_live: bool):
    """Lê o arquivo, seleciona colunas essenciais e adiciona metadados de ingestão."""
    df = (
        spark.read
        .option("header", "true")
        .option("sep", ",")
        .option("encoding", "latin1")
        .option("inferSchema", "false")
        .csv(caminho)
    )

    # casteia tudo para string — bronze é sempre string
    df = df.select([F.col(c).cast("string").alias(c) for c in df.columns])

    # validação mínima — arquivo corrompido ou URL errada retornam poucos registros
    n_linhas = df.count()
    if n_linhas < MIN_LINHAS_VALIDAS:
        raise ValueError(
            f"Arquivo de {ano} tem apenas {n_linhas:,} linhas — "
            f"esperado >= {MIN_LINHAS_VALIDAS:,}. "
            f"Verifique se o arquivo está corrompido ou se a URL está correta: {url}"
        )

    # seleciona colunas disponíveis
    colunas_presentes = [c for c in COLUNAS_ESSENCIAIS if c in df.columns]
    colunas_faltando  = [c for c in COLUNAS_ESSENCIAIS if c not in df.columns]
    if colunas_faltando:
        print(f"  ⚠ Colunas ausentes em {ano}: {colunas_faltando}")

    df = df.select(colunas_presentes)

    # metadados de ingestão
    df = (
        df
        .withColumn("_ano_arquivo",   F.lit(ano))
        .withColumn("_is_live",       F.lit(is_live))
        .withColumn("_snapshot_date", F.lit(datetime.today().strftime("%Y-%m-%d")))
        .withColumn("_source_url",    F.lit(url))
        .withColumn("_ingestion_ts",  F.current_timestamp())
    )
    return df


def gravar_bronze(spark: SparkSession, apenas_live: bool = False):
    """
    Orquestra montagem de URLs + download + gravação na tabela Bronze.
    apenas_live=True não reprocessa nenhum ano (todos são estáticos).
    apenas_live=False reprocessa todos os anos (carga inicial ou reprocessamento).
    """
    print("Montando URLs da fonte Hospitais e Leitos...")
    urls_disponiveis = scrape_urls()
    print(f"URLs mapeadas: {list(urls_disponiveis.keys())}")

    spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE} USING DELTA")

    for ano, config in ANOS.items():
        if apenas_live and not config["is_live"]:
            print(f"\n── {ano}: pulando (congelado) ──")
            continue

        fmt = config["formato"]
        url = urls_disponiveis.get(ano, {}).get(fmt)

        if not url:
            print(f"\n── {ano}: URL não encontrada para formato {fmt} ⚠")
            continue

        print(f"\n── Hospitais e Leitos {ano} (congelado) ──")
        caminho = baixar_arquivo(url, ano, fmt)
        df      = ler_e_enriquecer(spark, caminho, ano, url, config["is_live"])

        print(f"  Linhas lidas: {df.count():,}")

        # remove partição do ano antes de reescrever (permite reprocessamento seguro)
        try:
            spark.sql(f"DELETE FROM {TABLE} WHERE CAST(_ano_arquivo AS INT) = {ano}")
        except Exception as e:
            print(f"  ⚠ DELETE ignorado ({type(e).__name__}: {e}) — tabela vazia ou predicado sem resultado.")
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(TABLE)
        print(f"  ✓ Gravado em {TABLE}")

    print("\n✓ Ingestão Hospitais e Leitos concluída.")


def show_summary(spark: SparkSession):
    """
    Consulta bronze_hospitais_leitos e imprime:
    - contagem de linhas por _ano_arquivo e UF
    - range de COMP (competência) min/max por ano
    Útil para validar a ingestão após execução.
    """
    print(f"\n── Resumo da tabela {TABLE} ──")
    df = spark.table(TABLE)

    resumo = (
        df.groupBy("_ano_arquivo", "UF")
        .agg(
            F.count("*").alias("total_linhas"),
            F.min("COMP").alias("comp_min"),
            F.max("COMP").alias("comp_max"),
        )
        .orderBy("_ano_arquivo", "UF")
    )

    resumo.show(truncate=False)


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    spark       = SparkSession.builder.getOrCreate()
    apenas_live = "--live" in sys.argv
    gravar_bronze(spark, apenas_live=apenas_live)
