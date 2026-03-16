# src/ingestion/cnes_ingest.py
# Ingestão Bronze — CNES Estabelecimentos e Leitos (DATASUS)
#
# Fonte:
#   Primária : FTP DATASUS — ftp://ftp.datasus.gov.br/cnes/
#   Fallback  : página de downloads — https://cnes.datasus.gov.br/pages/downloads/arquivosBaseDados.jsp
#   Arquivo   : BASE_DE_DADOS_CNES_{AAAAMM}.ZIP  (competência mensal)
#
# Conteúdo do ZIP relevante para o projeto:
#   tbEstabelecimento{AAAAMM}.csv  → cadastro de estabelecimentos (~286 MB, sep=";", latin1)
#   tbLeito{AAAAMM}.csv            → leitos por estabelecimento (sep=";", latin1)
#
# Tabelas Bronze geradas:
#   bronze_cnes_estabelecimentos  ← tbEstabelecimento
#   bronze_cnes_leitos            ← tbLeito
#
# Colunas selecionadas e renomeadas — tbEstabelecimento:
#   CO_UNIDADE          → cnes_id
#   CO_MUNICIPIO_GESTOR → municipio_id
#   NO_FANTASIA         → nome_estabelecimento
#   TP_UNIDADE          → tipo_estabelecimento
#   TP_GESTAO           → gestao
#   TP_PFPJ             → natureza_juridica
#   CO_ESTADO_GESTOR    → uf
#
# Colunas selecionadas e renomeadas — tbLeito:
#   CO_UNIDADE          → cnes_id
#   CO_MUNICIPIO_GESTOR → municipio_id        (se existir)
#   CO_LEITO            → leito_id            (se existir)
#   QT_EXIST            → leitos_existentes
#   QT_SUS              → leitos_sus
#   TP_LEITO            → tipo_leito          (se existir)
#
# Idempotência:
#   Para cada ano processado, o script apaga os registros do mesmo ano com
#   DELETE WHERE _ano_arquivo = {ano} antes de reescrever via append.
#   Isso permite reprocessar qualquer ano sem duplicar dados.
#
# Limpeza:
#   O ZIP e os CSVs extraídos são deletados do Volume após a gravação
#   para economizar espaço no landing.

# ── NOTAS DE DESENVOLVIMENTO ──────────────────────────────────
#
# STATUS: pausado — não bloqueante para o MVP
#
# O que foi descoberto durante a validação (2026-03-16):
#
# 1. NOME REAL DO ARQUIVO NO FTP
#    Errado no script atual:  ESTABDADOS_{competencia}.csv
#    Correto:                 BASE_DE_DADOS_CNES_{competencia}.ZIP
#    Exemplo:                 BASE_DE_DADOS_CNES_202602.ZIP (~709MB)
#
# 2. CONTEÚDO DO ZIP
#    O ZIP contém ~100 CSVs. Os relevantes para o projeto são:
#    - tbEstabelecimento{competencia}.csv  (~286MB) → bronze_cnes_estabelecimentos
#    - tbLeito{competencia}.csv            (~0MB)   → bronze_cnes_leitos
#
# 3. PENDÊNCIAS ANTES DE RETOMAR
#    - Corrigir _nome_arquivo() e regex de _listar_competencias_ftp()
#      para o padrão BASE_DE_DADOS_CNES_{competencia}.ZIP
#    - Adicionar lógica de descompactação com zipfile
#    - Extrair só tbEstabelecimento e tbLeito do ZIP (ignorar os demais)
#    - Deletar ZIP e CSVs do Volume após ingestão (economiza espaço)
#    - Validar nomes reais das colunas de tbLeito antes de usar
#      COLUNAS_MAPA_LEITO (download anterior corrompeu o ZIP)
#    - Confirmar se CO_MUNICIPIO_GESTOR existe em tbLeito
#      (pode ser necessário join com tbEstabelecimento para obter municipio_id)
#
# 4. DECISÃO DE ESCOPO
#    Para o MVP, bronze_hospitais_leitos já cobre toda a capacidade necessária:
#    leitos_totais, leitos_complementares, leitos_uti, municipio_id.
#    O CNES de estabelecimentos entra na Fase 2 para enriquecer com:
#    num_estabelecimentos, num_hospitais, tipo_estabelecimento por município.
#
# ──────────────────────────────────────────────────────────────

import io
import os
import re
import requests
import zipfile
from datetime import datetime
from ftplib import FTP
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA  = "dev_christian_van_bellen"

TABLE_ESTAB = f"{CATALOG}.{SCHEMA}.bronze_cnes_estabelecimentos"
TABLE_LEITO = f"{CATALOG}.{SCHEMA}.bronze_cnes_leitos"

LANDING      = "/Volumes/ds_dev_db/dev_christian_van_bellen/landing"
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

# Mapeamento tbEstabelecimento: nome original → nome destino
COLUNAS_MAPA_ESTAB = {
    "CO_UNIDADE":          "cnes_id",
    "CO_MUNICIPIO_GESTOR": "municipio_id",
    "NO_FANTASIA":         "nome_estabelecimento",
    "TP_UNIDADE":          "tipo_estabelecimento",
    "TP_GESTAO":           "gestao",
    "TP_PFPJ":             "natureza_juridica",
    "CO_ESTADO_GESTOR":    "uf",
}

# Mapeamento tbLeito: nome original → nome destino
COLUNAS_MAPA_LEITO = {
    "CO_UNIDADE":          "cnes_id",
    "CO_MUNICIPIO_GESTOR": "municipio_id",        # se existir
    "CO_LEITO":            "leito_id",            # se existir
    "QT_EXIST":            "leitos_existentes",
    "QT_SUS":              "leitos_sus",
    "TP_LEITO":            "tipo_leito",          # se existir
}

MIN_LINHAS_VALIDAS = 1_000

# ── funções ─────────────────────────────────────────────────────
def _nome_zip(competencia: str) -> str:
    """Retorna o nome do arquivo ZIP para uma competência AAAAMM."""
    return f"BASE_DE_DADOS_CNES_{competencia}.ZIP"


def _listar_competencias_ftp() -> dict:
    """
    Conecta ao FTP do DATASUS, lista os arquivos BASE_DE_DADOS_CNES_XXXXXX.ZIP
    e retorna {ano: competencia_mais_recente}.
    Lança exceção se a conexão falhar — o chamador trata com fallback.
    """
    ftp = FTP(FTP_HOST, timeout=30)
    ftp.login()
    ftp.cwd(FTP_DIR)
    arquivos = ftp.nlst()
    ftp.quit()

    padrao    = re.compile(r"BASE_DE_DADOS_CNES_(\d{6})\.ZIP", re.IGNORECASE)
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
            raise ValueError("Nenhum arquivo BASE_DE_DADOS_CNES encontrado no FTP.")
        print(f"  ✓ FTP: competências encontradas = {resultado}")
        return resultado
    except Exception as e:
        print(f"  ⚠ Falha na listagem FTP ({type(e).__name__}: {e})")
        print(f"  → Usando competências de fallback hardcoded (atualizadas em 2026-03).")
        return COMPETENCIAS_FALLBACK


def baixar_e_extrair(competencia: str) -> tuple:
    """
    Baixa BASE_DE_DADOS_CNES_{competencia}.ZIP do FTP para o Volume de landing,
    extrai tbEstabelecimento{competencia}.csv e tbLeito{competencia}.csv,
    e deleta o ZIP imediatamente para economizar espaço.

    Tenta ftplib primeiro; se falhar, tenta ftp:// via requests.
    Retorna (caminho_estab, caminho_leito, url_usada).
    """
    nome_zip    = _nome_zip(competencia)
    caminho_zip = f"{LANDING}/cnes_{competencia}.zip"
    url_ftp     = f"ftp://{FTP_HOST}{FTP_DIR}{nome_zip}"

    # tentativa 1: ftplib (mais confiável em ambientes sem proxy HTTP)
    try:
        print(f"  Baixando {nome_zip} via ftplib de {FTP_HOST}{FTP_DIR} ...")
        ftp = FTP(FTP_HOST, timeout=300)
        ftp.login()
        ftp.cwd(FTP_DIR)
        with open(caminho_zip, "wb") as f:
            ftp.retrbinary(f"RETR {nome_zip}", f.write)
        ftp.quit()
        print(f"  ✓ {os.path.getsize(caminho_zip)/1e6:.1f} MB salvo em {caminho_zip}")

    except Exception as e:
        print(f"  ⚠ ftplib falhou ({type(e).__name__}: {e}) — tentando requests ftp://...")
        r = requests.get(url_ftp, timeout=300)
        r.raise_for_status()
        with open(caminho_zip, "wb") as f:
            f.write(r.content)
        print(f"  ✓ {len(r.content)/1e6:.1f} MB salvo em {caminho_zip} (via requests ftp://)")

    # extração dos dois CSVs relevantes
    nome_estab = f"tbEstabelecimento{competencia}.csv"
    nome_leito = f"tbLeito{competencia}.csv"
    caminho_estab = f"{LANDING}/cnes_estab_{competencia}.csv"
    caminho_leito = f"{LANDING}/cnes_leito_{competencia}.csv"

    with zipfile.ZipFile(caminho_zip) as zf:
        arquivos_zip = zf.namelist()
        print(f"  Arquivos no ZIP: {arquivos_zip}")

        # busca case-insensitive — nomes de arquivo variam entre competências
        mapa = {n.lower(): n for n in arquivos_zip}

        if nome_estab.lower() not in mapa:
            raise ValueError(f"Arquivo {nome_estab} não encontrado no ZIP. Disponíveis: {arquivos_zip}")
        with zf.open(mapa[nome_estab.lower()]) as src, open(caminho_estab, "wb") as dst:
            dst.write(src.read())
        print(f"  ✓ Extraído: {nome_estab}")

        if nome_leito.lower() not in mapa:
            raise ValueError(f"Arquivo {nome_leito} não encontrado no ZIP. Disponíveis: {arquivos_zip}")
        with zf.open(mapa[nome_leito.lower()]) as src, open(caminho_leito, "wb") as dst:
            dst.write(src.read())
        print(f"  ✓ Extraído: {nome_leito}")

    # deleta o ZIP — CSVs extraídos serão deletados após gravação
    os.remove(caminho_zip)
    print(f"  ✓ ZIP removido do landing")

    return caminho_estab, caminho_leito, url_ftp


def _adicionar_metadados(df, ano: int, competencia: str, is_live: bool, url: str):
    """Adiciona as colunas de metadados padrão do projeto."""
    return (
        df
        .withColumn("_ano_arquivo",   F.lit(ano))
        .withColumn("_competencia",   F.lit(competencia))      # AAAAMM — útil no silver para join temporal de capacidade
        .withColumn("_is_live",       F.lit(is_live))
        .withColumn("_snapshot_date", F.lit(datetime.today().strftime("%Y-%m-%d")))
        .withColumn("_source_url",    F.lit(url))
        .withColumn("_ingestion_ts",  F.current_timestamp())
    )


def _ler_csv(spark: SparkSession, caminho: str) -> object:
    """Lê um CSV do CNES (sep=";", latin1) e casteia tudo para string."""
    df = (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .option("encoding", "latin1")
        .option("inferSchema", "false")
        .csv(caminho)
    )
    # casteia tudo para string — bronze é sempre string
    return df.select([F.col(c).cast("string").alias(c) for c in df.columns])


def ler_e_enriquecer_estab(spark: SparkSession, caminho: str, ano: int, url: str, is_live: bool, competencia: str):
    """Lê tbEstabelecimento, renomeia colunas e adiciona metadados."""
    df = _ler_csv(spark, caminho)

    n_linhas = df.count()
    if n_linhas < MIN_LINHAS_VALIDAS:
        raise ValueError(
            f"tbEstabelecimento {competencia} tem apenas {n_linhas:,} linhas — "
            f"esperado >= {MIN_LINHAS_VALIDAS:,}. URL: {url}"
        )

    colunas_presentes = {orig: dest for orig, dest in COLUNAS_MAPA_ESTAB.items() if orig in df.columns}
    colunas_faltando  = [orig for orig in COLUNAS_MAPA_ESTAB if orig not in df.columns]
    if colunas_faltando:
        print(f"  ⚠ tbEstabelecimento — colunas ausentes em {competencia}: {colunas_faltando}")

    df = df.select([F.col(orig).alias(dest) for orig, dest in colunas_presentes.items()])
    return _adicionar_metadados(df, ano, competencia, is_live, url)


def ler_e_enriquecer_leito(spark: SparkSession, caminho: str, ano: int, url: str, is_live: bool, competencia: str):
    """Lê tbLeito, renomeia colunas e adiciona metadados."""
    df = _ler_csv(spark, caminho)

    n_linhas = df.count()
    if n_linhas < MIN_LINHAS_VALIDAS:
        raise ValueError(
            f"tbLeito {competencia} tem apenas {n_linhas:,} linhas — "
            f"esperado >= {MIN_LINHAS_VALIDAS:,}. URL: {url}"
        )

    colunas_presentes = {orig: dest for orig, dest in COLUNAS_MAPA_LEITO.items() if orig in df.columns}
    colunas_faltando  = [orig for orig in COLUNAS_MAPA_LEITO if orig not in df.columns]
    if colunas_faltando:
        print(f"  ⚠ tbLeito — colunas ausentes em {competencia}: {colunas_faltando}")

    df = df.select([F.col(orig).alias(dest) for orig, dest in colunas_presentes.items()])
    return _adicionar_metadados(df, ano, competencia, is_live, url)


def inspecionar_colunas(spark: SparkSession, competencia: str):
    """
    Baixa o ZIP da competência informada, extrai os dois CSVs relevantes,
    imprime o schema completo e as primeiras 5 linhas de cada um.
    Usado para validar os nomes reais das colunas antes da carga completa.
    NÃO grava nada no Bronze.
    """
    print(f"\n── Inspecionando colunas para competência {competencia} ──")
    caminho_estab, caminho_leito, _ = baixar_e_extrair(competencia)

    try:
        for label, caminho in [("tbEstabelecimento", caminho_estab), ("tbLeito", caminho_leito)]:
            print(f"\n{'─'*60}")
            print(f"  {label}{competencia}.csv")
            print(f"{'─'*60}")
            df = _ler_csv(spark, caminho)
            df.printSchema()
            print(f"  Primeiras 5 linhas:")
            df.show(5, truncate=False)
    finally:
        # limpa os CSVs extraídos — inspeção não deixa rastro no landing
        for caminho in (caminho_estab, caminho_leito):
            try:
                os.remove(caminho)
            except OSError:
                pass
        print(f"\n  ✓ CSVs de inspeção removidos do landing")


def _gravar_tabela(spark: SparkSession, df, table: str, ano: int):
    """Gravação idempotente: DELETE do ano + append."""
    try:
        spark.sql(f"DELETE FROM {table} WHERE CAST(_ano_arquivo AS INT) = {ano}")
    except Exception as e:
        print(f"  ⚠ DELETE ignorado ({type(e).__name__}: {e}) — tabela vazia ou predicado sem resultado.")
    df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table)
    print(f"  ✓ Gravado em {table}")


def gravar_bronze(spark: SparkSession, apenas_live: bool = False):
    """
    Orquestra listagem de competências + download + extração + gravação nas tabelas Bronze.
    apenas_live=True reprocessa só anos marcados como is_live (execução semanal).
    apenas_live=False reprocessa todos os anos (carga inicial ou reprocessamento completo).
    """
    print("Buscando competências disponíveis no FTP do DATASUS...")
    competencias_disponiveis = scrape_urls()
    print(f"Competências mapeadas: {competencias_disponiveis}")

    spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE_ESTAB} USING DELTA")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE_LEITO} USING DELTA")

    for ano, config in ANOS.items():
        if apenas_live and not config["is_live"]:
            print(f"\n── {ano}: pulando (congelado) ──")
            continue

        competencia = competencias_disponiveis.get(ano)
        if not competencia:
            print(f"\n── {ano}: competência não encontrada ⚠")
            continue

        status = "live" if config["is_live"] else "congelado"
        print(f"\n── CNES {ano} (competência {competencia}, {status}) ──")

        caminho_estab, caminho_leito, url = baixar_e_extrair(competencia)

        try:
            # ── tbEstabelecimento ──
            df_estab = ler_e_enriquecer_estab(spark, caminho_estab, ano, url, config["is_live"], competencia)
            print(f"  Estabelecimentos lidos: {df_estab.count():,}")
            _gravar_tabela(spark, df_estab, TABLE_ESTAB, ano)

            # ── tbLeito ──
            df_leito = ler_e_enriquecer_leito(spark, caminho_leito, ano, url, config["is_live"], competencia)
            print(f"  Leitos lidos: {df_leito.count():,}")
            _gravar_tabela(spark, df_leito, TABLE_LEITO, ano)

        finally:
            # limpa CSVs extraídos independentemente de erro — economiza espaço no landing
            for caminho in (caminho_estab, caminho_leito):
                try:
                    os.remove(caminho)
                except OSError:
                    pass
            print(f"  ✓ CSVs extraídos removidos do landing")

    print("\n✓ Ingestão CNES concluída.")


def show_summary(spark: SparkSession):
    """
    Consulta as duas tabelas Bronze e imprime contagens por ano/uf.
    Útil para validar a ingestão após execução.
    """
    for table, group_col in [(TABLE_ESTAB, "uf"), (TABLE_LEITO, "cnes_id")]:
        print(f"\n── Resumo da tabela {table} ──")
        df = spark.table(table)
        resumo = (
            df.groupBy("_ano_arquivo")
            .agg(
                F.count("*").alias("total_linhas"),
                F.min("_competencia").alias("competencia_min"),
                F.max("_competencia").alias("competencia_max"),
            )
            .orderBy("_ano_arquivo")
        )
        resumo.show(truncate=False)


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    spark       = SparkSession.builder.getOrCreate()
    apenas_live = "--live" in sys.argv
    gravar_bronze(spark, apenas_live=apenas_live)
