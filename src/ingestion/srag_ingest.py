# src/ingestion/srag_ingest.py
# Ingestão Bronze — SRAG / SIVEP-Gripe
# Fonte: https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026

import os
import re
import sys
from datetime import datetime

import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CATALOG, LANDING_PATH, SCHEMA, TABLE_BRONZE_SRAG
from quality.checks import checks_bronze_srag
from quality.runner import run_checks

# ── configuração ────────────────────────────────────────────────
PAGINA_FONTE = "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026"

# URLs de fallback — atualizadas em 2026-03
# Usar quando o scraping da página falhar
URLS_FALLBACK = {
    2023: {
        "csv": "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2023/INFLUD23-01-04-2024.csv"
    },
    2024: {
        "csv": "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2024/INFLUD24-30-12-2024.csv"
    },
    2025: {
        "parquet": "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2025/INFLUD25-24-02-2025.parquet"
    },
    2026: {
        "parquet": "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2026/INFLUD26-03-03-2026.parquet"
    },
}

ANOS = {
    2023: {"formato": "csv", "is_live": False},
    2024: {"formato": "csv", "is_live": False},
    2025: {"formato": "parquet", "is_live": True},
    2026: {"formato": "parquet", "is_live": True},
}

COLUNAS_ESSENCIAIS = [
    "DT_NOTIFIC",
    "DT_SIN_PRI",
    "CO_MUN_RES",
    "CO_MUN_NOT",
    "SG_UF_NOT",
    "NU_IDADE_N",
    "CS_SEXO",
    "HOSPITAL",
    "UTI",
    "EVOLUCAO",
    "CLASSI_FIN",
    "SEM_NOT",
    "SEM_PRI",
]

MIN_LINHAS_VALIDAS = 1_000


# ── funções ─────────────────────────────────────────────────────
def scrape_urls() -> dict:
    """
    Busca as URLs atuais de cada ano direto na página do OpenDataSUS.
    Retorna dict {ano: {csv: url, parquet: url}}.
    Evita depender de URLs hardcoded que mudam a cada atualização semanal.
    Em caso de falha no scraping, usa URLS_FALLBACK e loga o erro.
    """
    try:
        r = requests.get(PAGINA_FONTE, timeout=15)
        r.raise_for_status()
        links = re.findall(
            r"https://s3\.sa-east-1\.amazonaws\.com/ckan\.saude\.gov\.br/SRAG/\d{4}/INFLUD\d{2}-[\d-]+\.\w+",
            r.text,
        )
        resultado = {}
        for link in set(links):
            partes = link.split("/")
            ano = int(partes[-2])
            fmt = link.split(".")[-1]
            if ano not in resultado:
                resultado[ano] = {}
            resultado[ano][fmt] = link

        if not resultado:
            raise ValueError(
                "Nenhuma URL encontrada na página — possível mudança de estrutura HTML."
            )

        return resultado

    except Exception as e:
        print(f"  ⚠ Falha no scraping da página ({type(e).__name__}: {e})")
        print("  → Usando URLs de fallback hardcoded (atualizadas em 2026-03).")
        return URLS_FALLBACK


def baixar_arquivo(url: str, ano: int, fmt: str) -> str:
    """Baixa o arquivo para /tmp e retorna o caminho local."""
    caminho = f"{LANDING_PATH}/srag_{ano}.{fmt}"
    print(f"  Baixando {ano} ({fmt}) de {url} ...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(caminho, "wb") as f:
        f.write(r.content)
    print(f"  ✓ {len(r.content) / 1e6:.1f} MB salvo em {caminho}")
    return caminho


def ler_e_enriquecer(spark: SparkSession, caminho: str, ano: int, url: str, is_live: bool):
    """Lê o arquivo, seleciona colunas essenciais e adiciona metadados de ingestão."""
    fmt = caminho.split(".")[-1]

    if fmt == "csv":
        df = (
            spark.read.option("header", "true")
            .option("sep", ";")
            .option("encoding", "latin1")
            .option("inferSchema", "false")
            .csv(caminho)
        )
    else:
        df = spark.read.parquet(caminho)

    # casteia tudo para string — bronze é sempre string, igual ao CSV
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
    colunas_faltando = [c for c in COLUNAS_ESSENCIAIS if c not in df.columns]
    if colunas_faltando:
        print(f"  ⚠ Colunas ausentes em {ano}: {colunas_faltando}")

    df = df.select(colunas_presentes)

    # metadados de ingestão
    df = (
        df.withColumn("_ano_arquivo", F.lit(ano))
        .withColumn("_is_live", F.lit(is_live))
        .withColumn("_snapshot_date", F.lit(datetime.today().strftime("%Y-%m-%d")))
        .withColumn("_source_url", F.lit(url))
        .withColumn("_ingestion_ts", F.current_timestamp())
    )
    return df


def gravar_bronze(spark: SparkSession, apenas_live: bool = False):
    """
    Orquestra scraping de URLs + download + gravação na tabela Bronze.
    apenas_live=True reprocessa só 2025 e 2026 (execução semanal).
    apenas_live=False reprocessa todos os anos (carga inicial).
    """
    print("Buscando URLs atuais na página do OpenDataSUS...")
    urls_disponiveis = scrape_urls()
    print(f"URLs encontradas: {list(urls_disponiveis.keys())}")

    spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE_BRONZE_SRAG} USING DELTA")

    for ano, config in ANOS.items():
        if apenas_live and not config["is_live"]:
            print(f"\n── {ano}: pulando (congelado) ──")
            continue

        fmt = config["formato"]
        url = urls_disponiveis.get(ano, {}).get(fmt)

        if not url:
            print(f"\n── {ano}: URL não encontrada para formato {fmt} ⚠")
            continue

        print(f"\n── SRAG {ano} ({'live' if config['is_live'] else 'congelado'}) ──")
        caminho = baixar_arquivo(url, ano, fmt)
        df = ler_e_enriquecer(spark, caminho, ano, url, config["is_live"])

        print(f"  Linhas lidas: {df.count():,}")

        df = run_checks(
            spark,
            df,
            checks=checks_bronze_srag(),
            table_name=TABLE_BRONZE_SRAG,
            quarantine_table=f"{CATALOG}.{SCHEMA}.quarantine_bronze_srag",
        )

        # remove partição do ano antes de reescrever (permite reprocessamento seguro)
        try:
            spark.sql(f"DELETE FROM {TABLE_BRONZE_SRAG} WHERE CAST(_ano_arquivo AS INT) = {ano}")
        except Exception as e:
            print(
                f"  ⚠ DELETE ignorado ({type(e).__name__}: {e}) — tabela vazia ou predicado sem resultado."
            )
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
            TABLE_BRONZE_SRAG
        )
        print(f"  ✓ Gravado em {TABLE_BRONZE_SRAG}")

    print("\n✓ Ingestão SRAG concluída.")


def show_summary(spark: SparkSession):
    """
    Consulta bronze_srag e imprime:
    - contagem de linhas por _ano_arquivo
    - range de DT_NOTIFIC por ano
    Útil para validar a ingestão após execução.
    """
    print(f"\n── Resumo da tabela {TABLE_BRONZE_SRAG} ──")
    df = spark.table(TABLE_BRONZE_SRAG)

    resumo = (
        df.groupBy("_ano_arquivo")
        .agg(
            F.count("*").alias("total_linhas"),
            F.min("DT_NOTIFIC").alias("dt_notific_min"),
            F.max("DT_NOTIFIC").alias("dt_notific_max"),
        )
        .orderBy("_ano_arquivo")
    )

    resumo.show(truncate=False)


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    spark = SparkSession.builder.getOrCreate()
    apenas_live = "--live" in sys.argv
    gravar_bronze(spark, apenas_live=apenas_live)
