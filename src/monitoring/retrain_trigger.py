# src/monitoring/retrain_trigger.py
#
# Trigger automático de retraining baseado em degradação de performance.
#
# - Lê monitoring_performance para avaliar degradação
# - Regra principal: Precision@K < 0.55 por 2+ competências consecutivas
# - Regra secundária: queda > 15pp vs média das 6 competências anteriores
# - Dispara Job via Databricks Jobs API REST se trigger acionado
# - dry_run=True avalia sem disparar (para testes)
# - Loga decisão no MLflow com motivo e métricas de suporte
# - Job precisa existir com nome RETRAIN_JOB_NAME para disparo automático

import os
from datetime import date

import mlflow
import pandas as pd
import requests
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from config import (
    MIN_CONSECUTIVE_BELOW,
    MLFLOW_EXPERIMENT,
    PRECISION_K_THRESHOLD,
    RETRAIN_JOB_NAME,
)
from config import (  # noqa: E402
    TABLE_GOLD_MONITOR as TABLE_MONITOR,
)


# ── carregamento do histórico ────────────────────────────────────
def _carregar_historico_monitor(spark: SparkSession) -> pd.DataFrame:
    """
    Lê monitoring_performance e retorna o histórico ordenado
    por competencia desc, só com colunas relevantes para o trigger.
    Filtra duplicatas: se o monitor rodou múltiplas vezes para a
    mesma competência, pega o registro mais recente (max monitor_date).
    """
    df = spark.table(TABLE_MONITOR)

    # deduplicar por competencia — pega o monitor mais recente
    w = Window.partitionBy("competencia").orderBy(F.col("monitor_date").desc())
    df = df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")

    return (
        df.select(
            "competencia",
            "precision_at_k",
            "auc_pr",
            "n_municipios",
            "consolidation_flag",
            "simulated",
            "monitor_date",
        )
        .orderBy("competencia")
        .toPandas()
    )


# ── avaliação do trigger ─────────────────────────────────────────
def _avaliar_trigger(df_historico: pd.DataFrame) -> dict:
    """
    Avalia se o trigger de retraining deve ser acionado.

    Regras:
    1. DEGRADAÇÃO CONFIRMADA (trigger principal):
       Precision@K < threshold por MIN_CONSECUTIVE_BELOW (2) competências
       consecutivas mais recentes — exclui competências simuladas se
       houver dados reais disponíveis.

    2. QUEDA ABRUPTA (trigger secundário):
       Precision@K da competência mais recente caiu > 15pp em relação
       à média das 6 competências anteriores.

    Retorna dict com:
      trigger: bool
      reason: str explicando o motivo
      precision_at_k_recente: float
      n_consecutivas_abaixo: int
      competencias_avaliadas: lista das últimas N competencias
      detalhes: dict com métricas de suporte
    """
    df = df_historico.sort_values("competencia").copy()

    if len(df) < MIN_CONSECUTIVE_BELOW:
        return {
            "trigger": False,
            "reason": (
                f"Histórico insuficiente — {len(df)} competências, mínimo {MIN_CONSECUTIVE_BELOW}"
            ),
            "precision_at_k_recente": None,
            "n_consecutivas_abaixo": 0,
            "competencias_avaliadas": df["competencia"].tolist(),
        }

    # conta competências consecutivas abaixo do threshold
    # começando da mais recente
    ultimas = df.tail(10)  # avalia janela das últimas 10
    abaixo_threshold = (ultimas["precision_at_k"] < PRECISION_K_THRESHOLD).values[::-1]

    n_consecutivas = 0
    for v in abaixo_threshold:
        if v:
            n_consecutivas += 1
        else:
            break

    prec_recente = float(df["precision_at_k"].iloc[-1])
    comp_recente = df["competencia"].iloc[-1]

    # regra 1 — degradação confirmada
    if n_consecutivas >= MIN_CONSECUTIVE_BELOW:
        return {
            "trigger": True,
            "reason": (
                f"Degradação confirmada: Precision@K abaixo de {PRECISION_K_THRESHOLD} "
                f"por {n_consecutivas} competências consecutivas "
                f"(última: {comp_recente} = {prec_recente:.4f})"
            ),
            "precision_at_k_recente": prec_recente,
            "n_consecutivas_abaixo": n_consecutivas,
            "competencias_avaliadas": df["competencia"].tail(5).tolist(),
            "detalhes": {
                "trigger_type": "degradacao_confirmada",
                "threshold": PRECISION_K_THRESHOLD,
                "min_consecutivas": MIN_CONSECUTIVE_BELOW,
            },
        }

    # regra 2 — queda abrupta
    if len(df) >= 7:
        media_anterior = float(df["precision_at_k"].iloc[-7:-1].mean())
        queda = media_anterior - prec_recente
        if queda > 0.15:
            return {
                "trigger": True,
                "reason": (
                    f"Queda abrupta: Precision@K de {comp_recente} "
                    f"({prec_recente:.4f}) caiu {queda:.2f}pp "
                    f"vs média anterior ({media_anterior:.4f})"
                ),
                "precision_at_k_recente": prec_recente,
                "n_consecutivas_abaixo": n_consecutivas,
                "competencias_avaliadas": df["competencia"].tail(5).tolist(),
                "detalhes": {
                    "trigger_type": "queda_abrupta",
                    "media_anterior": media_anterior,
                    "queda_pp": queda,
                },
            }

    return {
        "trigger": False,
        "reason": (
            f"Performance estável. "
            f"Precision@K recente: {prec_recente:.4f} "
            f"({n_consecutivas} competência(s) abaixo do threshold)"
        ),
        "precision_at_k_recente": prec_recente,
        "n_consecutivas_abaixo": n_consecutivas,
        "competencias_avaliadas": df["competencia"].tail(5).tolist(),
    }


# ── utilitários Databricks API ───────────────────────────────────
def _get_credentials() -> tuple[str, str]:
    """
    Resolve host e token Databricks em ordem de prioridade:
      1. Variáveis de ambiente DATABRICKS_HOST / DATABRICKS_TOKEN
      2. dbutils (disponível dentro do workspace)
    Retorna (host, token) ou ("", "") se nenhum disponível.
    """
    host = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")

    if not host or not token:
        try:
            ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
            host = host or ctx.apiUrl().get()
            token = token or ctx.apiToken().get()
        except Exception:
            pass

    return host, token


def _get_job_id(job_name: str) -> str | None:
    """
    Busca o ID do Job pelo nome via Databricks Jobs API.
    Usa o token do ambiente (DATABRICKS_TOKEN) e o host
    (DATABRICKS_HOST) configurados no workspace.
    Retorna None se o Job não for encontrado.
    """
    host, token = _get_credentials()

    if not host or not token:
        print("  ⚠ Credenciais Databricks não encontradas (DATABRICKS_HOST/TOKEN)")
        return None

    url = f"https://{host.rstrip('/')}/api/2.1/jobs/list"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()

    jobs = resp.json().get("jobs", [])
    for job in jobs:
        if job.get("settings", {}).get("name") == job_name:
            return str(job["job_id"])

    return None


def _disparar_job(job_id: str, motivo: str) -> dict:
    """
    Dispara uma run do Job via Databricks Jobs API REST.
    Passa o motivo como parâmetro da run para rastreabilidade.
    Retorna dict com run_id e status.
    """
    host, token = _get_credentials()

    if not host or not token:
        raise RuntimeError("Credenciais Databricks não disponíveis para disparar o Job.")

    url = f"https://{host.rstrip('/')}/api/2.1/jobs/run-now"
    payload = {
        "job_id": int(job_id),
        "job_parameters": {
            "trigger_reason": motivo,
            "trigger_date": str(date.today()),
            "retrain_id": f"auto_{date.today().strftime('%Y%m%d')}",
        },
    }

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    run_id = resp.json().get("run_id")

    return {
        "job_id": job_id,
        "run_id": run_id,
        "status": "disparado",
        "trigger_reason": motivo,
        "trigger_date": str(date.today()),
    }


# ── função principal ─────────────────────────────────────────────
def verificar_e_disparar(spark: SparkSession, dry_run: bool = False) -> dict:
    """
    Função principal do trigger.

    dry_run=True: avalia a regra mas NÃO dispara o Job
                  (útil para testar a lógica sem side effects)
    dry_run=False: dispara o Job se a regra for acionada

    Fluxo:
      1. Carrega histórico do monitor
      2. Avalia regra de trigger
      3. Se trigger=True e dry_run=False: busca job_id e dispara
      4. Loga decisão no MLflow
      5. Retorna resultado com status e rastreabilidade
    """
    print("\n── Retrain Trigger ──")
    print(f"  Data:    {date.today()}")
    print(f"  Dry run: {dry_run}")

    # 1. carrega histórico
    df_historico = _carregar_historico_monitor(spark)
    print(f"  Competências no histórico: {len(df_historico)}")

    if df_historico.empty:
        print("  ⚠ Histórico vazio — monitor ainda não rodou")
        return {"status": "sem_historico", "trigger": False}

    # 2. avalia regra
    avaliacao = _avaliar_trigger(df_historico)

    print("\n── Avaliação ──")
    print(f"  Trigger:           {avaliacao['trigger']}")
    print(f"  Motivo:            {avaliacao['reason']}")
    print(f"  Precision@K rec.:  {avaliacao.get('precision_at_k_recente', 'N/A')}")
    print(f"  Consecutivas:      {avaliacao['n_consecutivas_abaixo']}")
    print(f"  Competências eval: {avaliacao['competencias_avaliadas']}")

    # 3. dispara Job se necessário
    job_result = None
    if avaliacao["trigger"]:
        if dry_run:
            print("\n  [DRY RUN] Trigger acionado — Job NÃO disparado")
            print(f"  Job que seria disparado: {RETRAIN_JOB_NAME}")
            job_result = {"status": "dry_run", "job_name": RETRAIN_JOB_NAME}
        else:
            print(f"\n  Buscando Job '{RETRAIN_JOB_NAME}' ...")
            job_id = _get_job_id(RETRAIN_JOB_NAME)

            if job_id is None:
                print(f"  ⚠ Job '{RETRAIN_JOB_NAME}' não encontrado no workspace")
                print("  → Trigger registrado mas Job não disparado")
                print(
                    "  → Crie o Job no Databricks com esse nome para habilitar o disparo automático"
                )
                job_result = {"status": "job_nao_encontrado", "job_name": RETRAIN_JOB_NAME}
            else:
                print(f"  Job encontrado (id={job_id}) — disparando ...")
                job_result = _disparar_job(job_id, avaliacao["reason"])
                print(f"  ✓ Job disparado — run_id={job_result['run_id']}")
    else:
        print("\n  ✓ Sem trigger — retraining não necessário")

    # 4. loga decisão no MLflow
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"retrain_trigger_{date.today()}") as run:
        mlflow.log_params(
            {
                "threshold": PRECISION_K_THRESHOLD,
                "min_consecutivas": MIN_CONSECUTIVE_BELOW,
                "dry_run": dry_run,
                "job_name": RETRAIN_JOB_NAME,
            }
        )
        mlflow.log_metrics(
            {
                "trigger_acionado": float(avaliacao["trigger"]),
                "n_consecutivas_abaixo": float(avaliacao["n_consecutivas_abaixo"]),
                "precision_at_k_recente": float(avaliacao.get("precision_at_k_recente") or 0),
            }
        )
        mlflow.set_tag("trigger_reason", avaliacao["reason"])
        mlflow.set_tag(
            "trigger_status",
            "acionado" if avaliacao["trigger"] else "nao_acionado",
        )

    # 5. retorna resultado
    resultado = {
        "status": "acionado" if avaliacao["trigger"] else "ok",
        "trigger": avaliacao["trigger"],
        "reason": avaliacao["reason"],
        "precision_at_k_recente": avaliacao.get("precision_at_k_recente"),
        "n_consecutivas_abaixo": avaliacao["n_consecutivas_abaixo"],
        "dry_run": dry_run,
        "job_result": job_result,
        "mlflow_run_id": run.info.run_id,
    }

    print(f"\n✓ Trigger run ID: {run.info.run_id}")
    return resultado


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    spark = SparkSession.builder.getOrCreate()
    dry_run = "--dry-run" in sys.argv
    resultado = verificar_e_disparar(spark, dry_run=dry_run)
    print(f"\nTrigger: {resultado['trigger']}")
    print(f"Motivo:  {resultado['reason']}")
