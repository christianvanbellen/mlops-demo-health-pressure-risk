# src/ingestion/hospitais_leitos_ingest.py
# Ingestão Bronze — Hospitais e Leitos
# Fonte: https://dadosabertos.saude.gov.br/dataset/hospitais-e-leitos

import io
import zipfile
from datetime import datetime

import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from quality.checks import checks_bronze_hospitais_leitos
from quality.runner import run_checks

# ── configuração ────────────────────────────────────────────────
PAGINA_FONTE = "https://dadosabertos.saude.gov.br/dataset/hospitais-e-leitos"

# URLs base por padrão de nomenclatura
# 2023–2024: CSV direto (Leitos_{ANO}.csv)
# 2025–2026: CSV compactado em zip (Leitos_csv_{ANO}.zip)
URL_BASE_CSV = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/Leitos_SUS/Leitos_{ano}.csv"
URL_BASE_ZIP = (
    "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/Leitos_SUS/Leitos_csv_{ano}.zip"
)

ANOS = {
    2023: {"formato": "csv", "sep": ",", "zip": False, "is_live": False},
    2024: {"formato": "csv", "sep": ",", "zip": False, "is_live": False},
    2025: {"formato": "csv", "sep": ";", "zip": True, "is_live": True},
    2026: {"formato": "csv", "sep": ";", "zip": True, "is_live": True},
}

COLUNAS_ESSENCIAIS = [
    "COMP",
    "REGIAO",
    "UF",
    "MUNICIPIO",  # nome do município — sem código IBGE nesta fonte
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
    "CO_IBGE",  # disponível em 2025/2026; ausente em 2023/2024 — tratado pelo filtro de colunas presentes
    # Nota: 2023/2024 não contém código IBGE (apenas nome do município).
    # O join com municipio_id (IBGE 6 dígitos) será feito na camada silver:
    # - 2025/2026: via CO_IBGE diretamente
    # - 2023/2024: via lookup silver_dim_municipio (UF + nome normalizado → municipio_id)
]

MIN_LINHAS_VALIDAS = 1_000


# ── funções ─────────────────────────────────────────────────────
def scrape_urls() -> dict:
    """
    Monta as URLs diretamente a partir da lista de anos.
    A URL desta fonte é estável e não muda entre atualizações,
    portanto não é necessário scraping. Mantém a assinatura para consistência
    com srag_ingest.py.
    Retorna dict {ano: {csv: url}}.
    """
    resultado = {}
    for ano, config in ANOS.items():
        if config["zip"]:
            resultado[ano] = {"csv": URL_BASE_ZIP.format(ano=ano)}
        else:
            resultado[ano] = {"csv": URL_BASE_CSV.format(ano=ano)}
    return resultado


def baixar_arquivo(url: str, ano: int, fmt: str, is_zip: bool, landing_path: str) -> str:
    """
    Baixa o arquivo para o Volume de landing e retorna o caminho do CSV local.
    Se is_zip=True: baixa o .zip e extrai o CSV interno para o Volume.
    """
    print(f"  Baixando {ano} ({'zip' if is_zip else fmt}) de {url} ...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    print(f"  ✓ {len(r.content) / 1e6:.1f} MB recebidos")

    if is_zip:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise ValueError(f"Nenhum CSV encontrado no zip de {ano}: {zf.namelist()}")
            nome_csv = csvs[0]
            caminho = f"{landing_path}/hospitais_leitos_{ano}.csv"
            with zf.open(nome_csv) as src, open(caminho, "wb") as dst:
                dst.write(src.read())
            print(f"  ✓ CSV extraído: {nome_csv} → {caminho}")
    else:
        caminho = f"{landing_path}/hospitais_leitos_{ano}.{fmt}"
        with open(caminho, "wb") as f:
            f.write(r.content)
        print(f"  ✓ Salvo em {caminho}")

    return caminho


def ler_e_enriquecer(
    spark: SparkSession, caminho: str, ano: int, url: str, is_live: bool, sep: str
):
    """Lê o arquivo, seleciona colunas essenciais e adiciona metadados de ingestão."""
    df = (
        spark.read.option("header", "true")
        .option("sep", sep)
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


def gravar_bronze(spark: SparkSession, args, apenas_live: bool = False):
    """
    Orquestra montagem de URLs + download + gravação na tabela Bronze.
    apenas_live=True reprocessa só 2025 e 2026 (execução semanal).
    apenas_live=False reprocessa todos os anos (carga inicial ou reprocessamento).
    """
    catalog = args.catalog
    schema = args.schema
    table_bronze_hospitais_leitos = args.table_bronze_hospitais_leitos
    landing_path = args.landing_path

    print("Montando URLs da fonte Hospitais e Leitos...")
    urls_disponiveis = scrape_urls()
    print(f"URLs mapeadas: {list(urls_disponiveis.keys())}")

    spark.sql(f"CREATE TABLE IF NOT EXISTS {table_bronze_hospitais_leitos} USING DELTA")

    for ano, config in ANOS.items():
        if apenas_live and not config["is_live"]:
            print(f"\n── {ano}: pulando (congelado) ──")
            continue

        fmt = config["formato"]
        url = urls_disponiveis.get(ano, {}).get(fmt)

        if not url:
            print(f"\n── {ano}: URL não encontrada para formato {fmt} ⚠")
            continue

        status = "live" if config["is_live"] else "congelado"
        print(f"\n── Hospitais e Leitos {ano} ({status}) ──")
        caminho = baixar_arquivo(url, ano, fmt, config["zip"], landing_path)
        df = ler_e_enriquecer(spark, caminho, ano, url, config["is_live"], config["sep"])

        print(f"  Linhas lidas: {df.count():,}")

        df = run_checks(
            spark,
            df,
            checks=checks_bronze_hospitais_leitos(),
            table_name=table_bronze_hospitais_leitos,
            quarantine_table=f"{catalog}.{schema}.quarantine_bronze_hospitais_leitos",
        )

        # remove partição do ano antes de reescrever (permite reprocessamento seguro)
        try:
            spark.sql(
                f"DELETE FROM {table_bronze_hospitais_leitos} WHERE CAST(_ano_arquivo AS INT) = {ano}"
            )
        except Exception as e:
            print(
                f"  ⚠ DELETE ignorado ({type(e).__name__}: {e}) — tabela vazia ou predicado sem resultado."
            )
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
            table_bronze_hospitais_leitos
        )
        print(f"  ✓ Gravado em {table_bronze_hospitais_leitos}")

    print("\n✓ Ingestão Hospitais e Leitos concluída.")


def show_summary(spark: SparkSession, args):
    """
    Consulta bronze_hospitais_leitos e imprime:
    - contagem de linhas por _ano_arquivo e UF
    - range de COMP (competência) min/max por ano
    Útil para validar a ingestão após execução.
    """
    table_bronze_hospitais_leitos = args.table_bronze_hospitais_leitos
    print(f"\n── Resumo da tabela {table_bronze_hospitais_leitos} ──")
    df = spark.table(table_bronze_hospitais_leitos)

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
    from cli import build_parser

    p = build_parser("Ingestão Bronze — Hospitais e Leitos")
    p.add_argument(
        "--live", action="store_true", default=False, help="Reprocessa apenas anos live (2025/2026)"
    )
    args, _ = p.parse_known_args()

    spark = SparkSession.builder.getOrCreate()
    gravar_bronze(spark, args, apenas_live=args.live)
