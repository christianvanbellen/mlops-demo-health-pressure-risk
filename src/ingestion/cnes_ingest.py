# src/ingestion/cnes_ingest.py
# Ingestão Bronze — CNES Estabelecimentos (DATASUS)
#
# Fonte:
#   Primária : FTP DATASUS — ftp://ftp.datasus.gov.br/cnes/
#   Fallback  : página de downloads do CNES — https://cnes.datasus.gov.br/pages/downloads/arquivosBaseDados.jsp
#   Arquivo   : ESTABDADOS_XXXXXX.csv  (XXXXXX = AAAAMM da competência mensal)
#   Encoding  : latin1 / separador ";"
#
# Colunas selecionadas e renomeadas para snake_case português:
#   CO_UNIDADE          → cnes_id
#   CO_MUNICIPIO_GESTOR → municipio_id
#   NO_FANTASIA         → nome_estabelecimento
#   TP_UNIDADE          → tipo_estabelecimento
#   TP_GESTAO           → gestao
#   TP_PFPJ             → natureza_juridica
#   CO_ESTADO_GESTOR    → uf
#
# Idempotência:
#   Para cada ano processado, o script apaga os registros do mesmo ano com
#   DELETE WHERE _ano_arquivo = {ano} antes de reescrever via append.
#   Isso permite reprocessar qualquer ano sem duplicar dados.

import re
import requests
from datetime import datetime
from ftplib import FTP, error_perm
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA  = "dev_christian_van_bellen"
TABLE   = f"{CATALOG}.{SCHEMA}.bronze_cnes_estabelecimentos"

FTP_HOST     = "ftp.datasus.gov.br"
FTP_DIR      = "/cnes/"
PAGINA_FONTE = "https://cnes.datasus.gov.br/pages/downloads/arquivosBaseDados.jsp"

# Competências de fallback — uma por ano (a mais recente disponível em 2026-03).
# Usar quando a listagem FTP falhar.
# Atualize estas entradas sempre que fizer um reprocessamento manual.
COMPETENCIAS_FALLBACK = {
    2023: "202312",
    2024: "202412",
    2025: "202512",
    2026: "202602",
}

ANOS = {
    2023: {"is_live": False},
    2024: {"is_live": False},
    2025: {"is_live": True},
    2026: {"is_live": True},
}

# Mapeamento: nome original no CSV → nome destino no bronze
COLUNAS_MAPA = {
    "CO_UNIDADE":          "cnes_id",
    "CO_MUNICIPIO_GESTOR": "municipio_id",
    "NO_FANTASIA":         "nome_estabelecimento",
    "TP_UNIDADE":          "tipo_estabelecimento",
    "TP_GESTAO":           "gestao",
    "TP_PFPJ":             "natureza_juridica",
    "CO_ESTADO_GESTOR":    "uf",
}

MIN_LINHAS_VALIDAS = 1_000

# ── funções ─────────────────────────────────────────────────────
def _nome_arquivo(competencia: str) -> str:
    """Retorna o nome do arquivo CSV para uma competência AAAAMM."""
    return f"ESTABDADOS_{competencia}.csv"


def _listar_competencias_ftp() -> dict:
    """
    Conecta ao FTP do DATASUS, lista os arquivos ESTABDADOS_XXXXXX.csv
    e retorna {ano: competencia_mais_recente}.
    Lança exceção se a conexão falhar — o chamador trata com fallback.
    """
    ftp = FTP(FTP_HOST, timeout=30)
    ftp.login()
    ftp.cwd(FTP_DIR)
    arquivos = ftp.nlst()
    ftp.quit()

    padrao    = re.compile(r"ESTABDADOS_(\d{6})\.csv", re.IGNORECASE)
    resultado = {}
    for nome in arquivos:
        m = padrao.match(nome)
        if not m:
            continue
        comp = m.group(1)
        ano  = int(comp[:4])
        # mantém a competência mais recente de cada ano
        if ano not in resultado or comp > resultado[ano]:
            resultado[ano] = comp

    return resultado


def scrape_urls() -> dict:
    """
    Tenta listar competências via FTP e devolve {ano: competencia}.
    Em caso de falha (FTP bloqueado, timeout), usa COMPETENCIAS_FALLBACK.
    Mantém a mesma assinatura dos outros scripts de ingestão do projeto.
    """
    try:
        resultado = _listar_competencias_ftp()
        if not resultado:
            raise ValueError("Nenhum arquivo ESTABDADOS encontrado no FTP.")
        print(f"  ✓ FTP: competências encontradas = {resultado}")
        return resultado
    except Exception as e:
        print(f"  ⚠ Falha na listagem FTP ({type(e).__name__}: {e})")
        print(f"  → Usando competências de fallback hardcoded (atualizadas em 2026-03).")
        return COMPETENCIAS_FALLBACK


def baixar_arquivo(competencia: str) -> tuple:
    """
    Baixa ESTABDADOS_{competencia}.csv do FTP para o Volume de landing.
    Tenta ftplib primeiro; se falhar, tenta ftp:// via requests.
    Retorna (caminho_local, url_usada).
    """
    import os
    LANDING = "/Volumes/ds_dev_db/dev_christian_van_bellen/landing"
    nome    = _nome_arquivo(competencia)
    caminho = f"{LANDING}/cnes_{competencia}.csv"
    url_ftp = f"ftp://{FTP_HOST}{FTP_DIR}{nome}"

    # tentativa 1: ftplib (mais confiável em ambientes sem proxy HTTP)
    try:
        print(f"  Baixando {competencia} via ftplib de {FTP_HOST}{FTP_DIR}{nome} ...")
        ftp = FTP(FTP_HOST, timeout=300)
        ftp.login()
        ftp.cwd(FTP_DIR)
        with open(caminho, "wb") as f:
            ftp.retrbinary(f"RETR {nome}", f.write)
        ftp.quit()
        print(f"  ✓ {os.path.getsize(caminho)/1e6:.1f} MB salvo em {caminho}")
        return caminho, url_ftp

    except Exception as e:
        print(f"  ⚠ ftplib falhou ({type(e).__name__}: {e}) — tentando requests...")

    # tentativa 2: requests com ftp://
    r = requests.get(url_ftp, timeout=300)
    r.raise_for_status()
    with open(caminho, "wb") as f:
        f.write(r.content)
    print(f"  ✓ {len(r.content)/1e6:.1f} MB salvo em {caminho} (via requests ftp://)")
    return caminho, url_ftp


def ler_e_enriquecer(spark: SparkSession, caminho: str, ano: int, url: str, is_live: bool, competencia: str):
    """Lê o CSV, renomeia colunas e adiciona metadados de ingestão."""
    df = (
        spark.read
        .option("header", "true")
        .option("sep", ";")
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
            f"Arquivo de {competencia} tem apenas {n_linhas:,} linhas — "
            f"esperado >= {MIN_LINHAS_VALIDAS:,}. "
            f"Verifique se o arquivo está corrompido ou se a URL está correta: {url}"
        )

    # seleciona e renomeia colunas disponíveis
    colunas_presentes = {orig: dest for orig, dest in COLUNAS_MAPA.items() if orig in df.columns}
    colunas_faltando  = [orig for orig in COLUNAS_MAPA if orig not in df.columns]
    if colunas_faltando:
        print(f"  ⚠ Colunas ausentes em {competencia}: {colunas_faltando}")

    df = df.select([F.col(orig).alias(dest) for orig, dest in colunas_presentes.items()])

    # metadados de ingestão
    df = (
        df
        .withColumn("_ano_arquivo",   F.lit(ano))
        .withColumn("_competencia",   F.lit(competencia))      # AAAAMM — útil no silver para join temporal de capacidade
        .withColumn("_is_live",       F.lit(is_live))
        .withColumn("_snapshot_date", F.lit(datetime.today().strftime("%Y-%m-%d")))
        .withColumn("_source_url",    F.lit(url))
        .withColumn("_ingestion_ts",  F.current_timestamp())
    )
    return df


def gravar_bronze(spark: SparkSession, apenas_live: bool = False):
    """
    Orquestra listagem de competências + download + gravação na tabela Bronze.
    apenas_live=True reprocessa só anos marcados como is_live (execução semanal).
    apenas_live=False reprocessa todos os anos (carga inicial ou reprocessamento completo).
    """
    print("Buscando competências disponíveis no FTP do DATASUS...")
    competencias_disponiveis = scrape_urls()
    print(f"Competências mapeadas: {competencias_disponiveis}")

    spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE} USING DELTA")

    for ano, config in ANOS.items():
        if apenas_live and not config["is_live"]:
            print(f"\n── {ano}: pulando (congelado) ──")
            continue

        competencia = competencias_disponiveis.get(ano)
        if not competencia:
            print(f"\n── {ano}: competência não encontrada ⚠")
            continue

        status = "live" if config["is_live"] else "congelado"
        print(f"\n── CNES Estabelecimentos {ano} (competência {competencia}, {status}) ──")
        caminho, url = baixar_arquivo(competencia)
        df           = ler_e_enriquecer(spark, caminho, ano, url, config["is_live"], competencia)

        print(f"  Linhas lidas: {df.count():,}")

        # remove partição do ano antes de reescrever (permite reprocessamento seguro)
        try:
            spark.sql(f"DELETE FROM {TABLE} WHERE CAST(_ano_arquivo AS INT) = {ano}")
        except Exception as e:
            print(f"  ⚠ DELETE ignorado ({type(e).__name__}: {e}) — tabela vazia ou predicado sem resultado.")
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(TABLE)
        print(f"  ✓ Gravado em {TABLE}")

    print("\n✓ Ingestão CNES Estabelecimentos concluída.")


def show_summary(spark: SparkSession):
    """
    Consulta bronze_cnes_estabelecimentos e imprime:
    - contagem de linhas por _ano_arquivo e uf
    - competência min/max por ano
    Útil para validar a ingestão após execução.
    """
    print(f"\n── Resumo da tabela {TABLE} ──")
    df = spark.table(TABLE)

    resumo = (
        df.groupBy("_ano_arquivo", "uf")
        .agg(
            F.count("*").alias("total_linhas"),
            F.min("_competencia").alias("competencia_min"),
            F.max("_competencia").alias("competencia_max"),
        )
        .orderBy("_ano_arquivo", "uf")
    )

    resumo.show(truncate=False)


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    spark       = SparkSession.builder.getOrCreate()
    apenas_live = "--live" in sys.argv
    gravar_bronze(spark, apenas_live=apenas_live)
