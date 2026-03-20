# src/training/evaluate.py
# Avaliação comparativa e promoção de modelos via lógica champion/challenger
#
# Fluxo:
#   1. Carrega e avalia os dois novos modelos (LR Baseline e LightGBM).
#   2. Se não existe @champion: promove o melhor automaticamente (first_deploy).
#   3. Se já existe @champion: avalia o champion atual nos mesmos dados.
#      a. Novo melhor > champion → registra como @challenger + aciona human gate.
#      b. Novo melhor <= champion → no_change, champion mantido.
#
# Critério: Precision@K(15% do val set)
#   K(15%) representa os municípios que um gestor monitoraria num ciclo mensal.
#   AUC-ROC mede discriminação geral; Precision@K mede utilidade operacional.
#
# Artefatos gerados:
#   - evaluation_report.json          : resumo estruturado para auditoria
#   - precision_at_k_comparison.png   : curvas Precision@K val e test

import json
import os
import tempfile
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.spark
import numpy as np
from mlflow.tracking import MlflowClient
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import average_precision_score, roc_auc_score

from cli import FEATURE_COLS, TARGET_COL


# ── aliases ─────────────────────────────────────────────────────
def _champion_existe(model_name: str) -> bool:
    """Retorna True se o alias @champion está atribuído no registry."""
    client = MlflowClient()
    try:
        client.get_registered_model_alias(model_name, "champion")
        return True
    except Exception:
        return False


def _get_version_por_run_id(run_id: str, model_name: str) -> str:
    """Encontra a versão registrada no registry para um dado run_id."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.run_id == run_id:
            return v.version
    raise ValueError(f"Versão não encontrada para run_id={run_id}")


def _registrar_alias(version: str, alias: str, model_name: str):
    """Atribui um alias (champion ou challenger) a uma versão no registry."""
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version,
    )
    print(f"  ✓ Alias @{alias} → versão {version}")


def _get_champion_run_id(model_name: str) -> str:
    """Retorna o run_id da versão atualmente marcada como @champion."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "champion")
    return mv.run_id


# ── dados e avaliação ────────────────────────────────────────────
def _carregar_dados(spark: SparkSession, table_features: str):
    """
    Lê gold_pressure_features, filtra linhas com target válido,
    casteia features para double e remove nulos.
    """
    df = spark.table(table_features)
    df = df.filter(F.col(TARGET_COL).isNotNull())
    df = df.filter(F.col("leitos_totais") >= 10)
    for col in FEATURE_COLS:
        df = df.withColumn(col, F.col(col).cast("double"))
    df = df.withColumn(TARGET_COL, F.col(TARGET_COL).cast("double"))
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    total = df.count()
    positivos = df.filter(F.col(TARGET_COL) == 1.0).count()
    print("\n── Dados carregados ──")
    print(f"  Total: {total:,}  |  Positivos: {positivos:,} ({positivos / total * 100:.1f}%)")
    return df


def _split_temporal(df, train_end: str, val_end: str, test_start: str):
    """Split temporal estrito por competencia (AAAAMM)."""
    train = df.filter(F.col("competencia") <= train_end)
    val = df.filter((F.col("competencia") > train_end) & (F.col("competencia") <= val_end))
    test = df.filter(F.col("competencia") > val_end)
    print("\n── Split temporal ──")
    print(f"  Treino  (<=  {train_end}): {train.count():,}")
    print(f"  Val     ({train_end[:4]}{int(train_end[4:]) + 1:02d}–{val_end}): {val.count():,}")
    print(f"  Teste   (>= {test_start}): {test.count():,}")
    return train, val, test


def _precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k_values: list) -> dict:
    """
    Calcula Precision@K e Recall@K ordenando os scores em ordem decrescente.
    K é o número de municípios alertados — reflete a capacidade operacional
    de monitoramento de um gestor num ciclo mensal.
    """
    n_positivos = int(labels.sum())
    ordem = np.argsort(scores)[::-1]
    labels_ord = labels[ordem]

    resultado = {}
    for k in k_values:
        k_clip = min(k, len(labels))
        top_k = labels_ord[:k_clip]
        tp_k = int(top_k.sum())
        resultado[k] = {
            "precision@k": tp_k / (k_clip + 1e-9),
            "recall@k": tp_k / (n_positivos + 1e-9),
            "tp@k": tp_k,
        }
    return resultado


def _get_artifact_path(run_id: str, artifact_subpath: str = "model") -> str:
    """
    Retorna o path dbfs:/ do artefato — necessário porque runs:/ usa
    /dbfs/tmp internamente, que está inacessível neste workspace.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    return f"{run.info.artifact_uri}/{artifact_subpath}"


def _carregar_modelo_lr(spark: SparkSession, run_id: str):
    """Carrega modelo LR Spark salvo no MLflow."""
    print(f"  Carregando LR (run_id={run_id}) ...")
    artifact_path = _get_artifact_path(run_id)
    return mlflow.spark.load_model(artifact_path)


def _carregar_modelo_lgbm(run_id: str):
    """Carrega modelo LightGBM salvo no MLflow."""
    print(f"  Carregando LightGBM (run_id={run_id}) ...")
    artifact_path = _get_artifact_path(run_id)
    return mlflow.lightgbm.load_model(artifact_path)


def _scores_lr(model, split_sp) -> tuple:
    """Aplica modelo LR e extrai probabilidades da classe positiva."""
    preds = model.transform(split_sp)
    scores = np.array(
        preds.withColumn("prob_pos", vector_to_array(F.col("probability")).getItem(1))
        .select("prob_pos")
        .rdd.map(lambda r: float(r[0]))
        .collect()
    )
    labels = np.array(preds.select(TARGET_COL).rdd.map(lambda r: float(r[0])).collect())
    return scores, labels


def _avaliar_modelo(
    scores: np.ndarray, labels: np.ndarray, split_name: str, model_name: str, k_ref: int
) -> dict:
    """
    Calcula AUC-ROC, AUC-PR e Precision/Recall@K para múltiplos K.
    k_ref corresponde a 15% do split, usada como critério de promoção.
    """
    auc_roc = roc_auc_score(labels, scores)
    auc_pr = average_precision_score(labels, scores)

    k_values = sorted(set([10, 20, 50, 100, 200, 500, k_ref]))
    pk_results = _precision_recall_at_k(scores, labels, k_values)

    print(
        f"  [{model_name} / {split_name}]  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}"
        f"  Prec@{k_ref}={pk_results[k_ref]['precision@k']:.4f}"
        f"  Rec@{k_ref}={pk_results[k_ref]['recall@k']:.4f}"
    )

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "precision_at_k_ref": pk_results[k_ref]["precision@k"],
        "recall_at_k_ref": pk_results[k_ref]["recall@k"],
        "precision_recall_at_k": pk_results,
    }


# ── artefatos ────────────────────────────────────────────────────
def _plot_comparacao(resultados: dict, k_ref_val: int, k_ref_test: int):
    """
    Curvas Precision@K para val e test dos dois novos modelos lado a lado.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    model_keys = ["baseline_logistic_regression", "lightgbm_gbt"]
    display_names = {"baseline_logistic_regression": "LR Baseline", "lightgbm_gbt": "LightGBM"}
    colors = {"baseline_logistic_regression": "steelblue", "lightgbm_gbt": "darkorange"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, split_key, k_ref, titulo in [
        (ax1, "val", k_ref_val, "Precision@K — Val Set"),
        (ax2, "test", k_ref_test, "Precision@K — Test Set"),
    ]:
        for mk in model_keys:
            pk_data = resultados[mk][split_key]["precision_recall_at_k"]
            k_vals = sorted(pk_data.keys())
            prec = [pk_data[k]["precision@k"] for k in k_vals]
            ax.plot(
                k_vals,
                prec,
                marker="o",
                markersize=4,
                linewidth=2,
                color=colors[mk],
                label=display_names[mk],
            )

        ax.axvline(x=k_ref, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.annotate(
            f"K ref (15%)\nK={k_ref:,}",
            xy=(k_ref, ax.get_ylim()[1] * 0.9),
            xytext=(k_ref + 5, ax.get_ylim()[1] * 0.88),
            fontsize=8,
            color="red",
        )
        ax.set_title(titulo, fontsize=13, fontweight="bold")
        ax.set_xlabel("K (municípios alertados)", fontsize=10)
        ax.set_ylabel("Precision@K", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.02)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "precision_at_k_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    os.remove(path)
    print("  ✓ precision_at_k_comparison.png logado")


def _gerar_relatorio(
    retrain_id: str,
    resultados_novos: dict,
    champion_info: dict | None,
    melhor_key: str,
    melhor_version: str,
    decision: str,
    human_gate: bool,
    k_ref_val: int,
    k_ref_test: int,
    next_steps: list,
    model_name: str = "",
):
    """
    Serializa o estado completo da avaliação em evaluation_report.json
    e loga no MLflow. Inclui decision, human_gate e instruções de next_steps.
    """
    melhor = resultados_novos[melhor_key]

    relatorio = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "retrain_id": retrain_id,
        "champion_exists": champion_info is not None,
        "decision": decision,
        "champion_alias": "@champion",
        "challenger_alias": "@challenger" if decision == "registered_challenger" else None,
        "criterion": "precision@k(15%) no val set",
        "k_ref_val": k_ref_val,
        "k_ref_test": k_ref_test,
        "human_gate_required": human_gate,
        "human_gate_instructions": (
            f"Acesse o MLflow Model Registry ({model_name}), "
            f"compare @champion e @challenger e promova manualmente "
            f"atribuindo o alias @champion à versão {melhor_version}."
            if human_gate
            else None
        ),
        "models": {
            "melhor_novo_modelo": {
                "run_id": melhor["run_id"],
                "model_type": melhor_key,
                "val_auc_roc": round(melhor["val"]["auc_roc"], 6),
                "val_auc_pr": round(melhor["val"]["auc_pr"], 6),
                "val_precision_at_k": round(melhor["val"]["precision_at_k_ref"], 6),
                "test_precision_at_k": round(melhor["test"]["precision_at_k_ref"], 6),
            },
            "champion_atual": (
                {
                    "run_id": champion_info["run_id"],
                    "val_precision_at_k": round(champion_info["val_precision_at_k"], 6),
                    "test_precision_at_k": round(champion_info["test_precision_at_k"], 6),
                }
                if champion_info is not None
                else None
            ),
        },
        "next_steps": next_steps,
    }

    print("\n── Relatório de avaliação ──")
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "evaluation_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    mlflow.log_artifact(path)
    os.remove(path)
    print("  ✓ evaluation_report.json logado")


# ── auxiliar ────────────────────────────────────────────────────
def _inferir_model_type(run_id: str) -> str:
    """
    Infere o tipo de modelo a partir dos parâmetros do run no MLflow.
    Retorna 'lightgbm' ou 'spark'.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    model_type = params.get("model_type", "").lower()
    if "lightgbm" in model_type:
        return "lightgbm"
    return "spark"


# ── função principal ─────────────────────────────────────────────
def avaliar(
    spark: SparkSession, args, lr_run_id: str, lgbm_run_id: str, retrain_id: str = None
) -> tuple:
    """
    Compara LR e LightGBM e decide entre first_deploy, registered_challenger
    ou no_change com base na lógica champion/challenger.

    Retorna (decision, melhor_run_id).
    """
    mlflow_experiment_name = args.mlflow_experiment
    model_name = args.model_name
    table_features = args.table_gold_features
    train_end = args.train_end
    val_end = args.val_end
    test_start = args.test_start

    retrain_id = retrain_id or str(uuid.uuid4())[:8]

    # --- DEBUG: inspecionar contexto MLflow injetado pelo Databricks ---
    print(f"DEBUG MLFLOW_EXPERIMENT_ID env: {repr(os.environ.get('MLFLOW_EXPERIMENT_ID'))}")
    print(f"DEBUG MLFLOW_EXPERIMENT_NAME env: {repr(os.environ.get('MLFLOW_EXPERIMENT_NAME'))}")
    print(f"DEBUG MLFLOW_RUN_ID env: {repr(os.environ.get('MLFLOW_RUN_ID'))}")
    print(
        f"DEBUG todas env vars MLFLOW: { {k: v for k, v in os.environ.items() if 'MLFLOW' in k} }"
    )

    # --- asserts: garantir que o valor passado via argparse está correto ---
    assert mlflow_experiment_name is not None, "mlflow_experiment não pode ser None"
    assert mlflow_experiment_name != "None", (
        f"mlflow_experiment resolveu para string 'None': {mlflow_experiment_name!r}"
    )
    assert mlflow_experiment_name.startswith("/"), (
        f"mlflow_experiment deve ser um path absoluto, recebeu: {mlflow_experiment_name!r}"
    )

    # --- fix: limpar env vars injetadas pelo Databricks antes de set_experiment ---
    removed_id = os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    removed_name = os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
    print(f"DEBUG removido MLFLOW_EXPERIMENT_ID: {repr(removed_id)}")
    print(f"DEBUG removido MLFLOW_EXPERIMENT_NAME: {repr(removed_name)}")

    client = MlflowClient()
    exp = client.get_experiment_by_name(mlflow_experiment_name)
    if exp is None:
        experiment_id = client.create_experiment(mlflow_experiment_name)
        print(f"DEBUG experimento criado: {mlflow_experiment_name!r} id={experiment_id!r}")
    else:
        experiment_id = exp.experiment_id
        print(f"DEBUG experimento encontrado: {mlflow_experiment_name!r} id={experiment_id!r}")

    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"DEBUG mlflow.set_experiment OK com experiment_id={experiment_id!r}")
    # --- fim DEBUG ---

    with mlflow.start_run(run_name=f"evaluation_{retrain_id}") as run:
        mlflow.set_tag("retrain_id", retrain_id)

        # ── dados ──────────────────────────────────────────────────
        df = _carregar_dados(spark, table_features)
        train, val, test = _split_temporal(df, train_end, val_end, test_start)

        k_ref_val = int(val.count() * 0.15)
        k_ref_test = int(test.count() * 0.15)
        print(f"\nK referência — val: {k_ref_val:,}  |  test: {k_ref_test:,}")

        # pandas para LightGBM
        X_val = val.select(FEATURE_COLS).toPandas()
        y_val = val.select(TARGET_COL).toPandas()[TARGET_COL]
        X_test = test.select(FEATURE_COLS).toPandas()
        y_test = test.select(TARGET_COL).toPandas()[TARGET_COL]

        # ── carrega e avalia os dois novos modelos ─────────────────
        print("\nCarregando modelos do MLflow ...")
        lr_model = _carregar_modelo_lr(spark, lr_run_id)
        lgbm_model = _carregar_modelo_lgbm(lgbm_run_id)

        print("\n── Métricas dos novos modelos ──")
        lr_scores_val, lr_labels_val = _scores_lr(lr_model, val)
        lr_scores_test, lr_labels_test = _scores_lr(lr_model, test)

        resultados_novos = {
            "baseline_logistic_regression": {
                "run_id": lr_run_id,
                "model_type": "baseline_logistic_regression",
                "val": _avaliar_modelo(lr_scores_val, lr_labels_val, "val", "LR", k_ref_val),
                "test": _avaliar_modelo(lr_scores_test, lr_labels_test, "test", "LR", k_ref_test),
            },
            "lightgbm_gbt": {
                "run_id": lgbm_run_id,
                "model_type": "lightgbm_gbt",
                "val": _avaliar_modelo(
                    lgbm_model.predict(X_val), y_val.values, "val", "LGBM", k_ref_val
                ),
                "test": _avaliar_modelo(
                    lgbm_model.predict(X_test), y_test.values, "test", "LGBM", k_ref_test
                ),
            },
        }

        # melhor dos dois novos (por Precision@K no val)
        melhor_key = max(
            resultados_novos,
            key=lambda k: resultados_novos[k]["val"]["precision_at_k_ref"],
        )
        melhor = resultados_novos[melhor_key]
        melhor_version = _get_version_por_run_id(melhor["run_id"], model_name)

        # ── lógica champion/challenger ─────────────────────────────
        champion_existe = _champion_existe(model_name)

        if not champion_existe:
            # CASO 1: primeiro deploy — sem gate humano
            _registrar_alias(melhor_version, "champion", model_name)
            decision = "first_deploy"
            human_gate = False
            champion_info = None
            next_steps = [
                "Primeiro modelo promovido automaticamente como @champion.",
                "Nenhuma ação necessária.",
                f"Modelo: {melhor_key}  |  Versão: {melhor_version}",
            ]
            print(f"\n✓ FIRST DEPLOY — {melhor_key} v{melhor_version} promovido como @champion")

        else:
            # CASO 2: já existe champion — avalia-o nos mesmos dados
            champion_run_id = _get_champion_run_id(model_name)
            champion_model_type = _inferir_model_type(champion_run_id)

            print(f"\nCarregando champion atual (run_id={champion_run_id}) ...")
            if champion_model_type == "lightgbm":
                champ_model = _carregar_modelo_lgbm(champion_run_id)
                champ_scores_val = champ_model.predict(X_val)
                champ_scores_test = champ_model.predict(X_test)
                champ_labels_val = y_val.values
                champ_labels_test = y_test.values
            else:
                champ_model = _carregar_modelo_lr(spark, champion_run_id)
                champ_scores_val, champ_labels_val = _scores_lr(champ_model, val)
                champ_scores_test, champ_labels_test = _scores_lr(champ_model, test)

            print("\n── Métricas do champion atual ──")
            champ_metrics_val = _avaliar_modelo(
                champ_scores_val, champ_labels_val, "val", "CHAMPION", k_ref_val
            )
            champ_metrics_test = _avaliar_modelo(
                champ_scores_test, champ_labels_test, "test", "CHAMPION", k_ref_test
            )

            champ_prec = champ_metrics_val["precision_at_k_ref"]
            novo_prec = melhor["val"]["precision_at_k_ref"]
            delta = novo_prec - champ_prec

            champion_info = {
                "run_id": champion_run_id,
                "val_precision_at_k": champ_prec,
                "test_precision_at_k": champ_metrics_test["precision_at_k_ref"],
            }

            if delta > 0:
                # novo modelo supera o champion — registra como @challenger
                _registrar_alias(melhor_version, "challenger", model_name)
                decision = "registered_challenger"
                human_gate = True
                next_steps = [
                    f"Challenger registrado: {melhor_key} v{melhor_version}",
                    f"Ganho em Precision@K(val): +{delta:.4f} ({champ_prec:.4f} → {novo_prec:.4f})",
                    "Para promover o challenger:",
                    "  1. Abra o MLflow Model Registry",
                    f"  2. Acesse o modelo {model_name}",
                    "  3. Remova o alias @champion da versão atual",
                    f"  4. Atribua @champion à versão {melhor_version}",
                    "Para ativar A/B test (canary 20%):",
                    "  Mantenha ambos os aliases e configure AB_TEST=true no Job de scoring",
                ]
                print(f"\n✓ CHALLENGER REGISTRADO — {melhor_key} v{melhor_version}")
                print(f"  Champion atual: Prec@K={champ_prec:.4f}")
                print(f"  Challenger:     Prec@K={novo_prec:.4f}  (delta={delta:+.4f})")
                print("  → Human gate necessário. Ver evaluation_report.json para instruções.")
            else:
                # novo modelo não supera o champion
                decision = "no_change"
                human_gate = False
                next_steps = [
                    "Novo modelo não supera o champion atual.",
                    f"Champion mantido: Prec@K={champ_prec:.4f}",
                    f"Novo modelo:      Prec@K={novo_prec:.4f}  (delta={delta:+.4f})",
                    "Nenhuma ação necessária.",
                ]
                print("\n✓ NO CHANGE — champion mantido")
                print(f"  Champion: Prec@K={champ_prec:.4f}")
                print(f"  Novo:     Prec@K={novo_prec:.4f}  (delta={delta:+.4f})")

        # ── artefatos ──────────────────────────────────────────────
        print("\nGerando artefatos ...")
        _plot_comparacao(resultados_novos, k_ref_val, k_ref_test)
        _gerar_relatorio(
            retrain_id=retrain_id,
            resultados_novos=resultados_novos,
            champion_info=champion_info,
            melhor_key=melhor_key,
            melhor_version=melhor_version,
            decision=decision,
            human_gate=human_gate,
            k_ref_val=k_ref_val,
            k_ref_test=k_ref_test,
            next_steps=next_steps,
            model_name=model_name,
        )

        mlflow.set_tags(
            {
                "retrain_id": retrain_id,
                "decision": decision,
                "human_gate": str(human_gate),
                "champion_version": melhor_version if not champion_existe else "unchanged",
            }
        )
        mlflow.log_metrics(
            {
                "lr_val_auc_roc": resultados_novos["baseline_logistic_regression"]["val"][
                    "auc_roc"
                ],
                "lr_val_precision_at_k": resultados_novos["baseline_logistic_regression"]["val"][
                    "precision_at_k_ref"
                ],
                "lgbm_val_auc_roc": resultados_novos["lightgbm_gbt"]["val"]["auc_roc"],
                "lgbm_val_precision_at_k": resultados_novos["lightgbm_gbt"]["val"][
                    "precision_at_k_ref"
                ],
                "k_ref_val": float(k_ref_val),
            }
        )

        print(f"\n✓ Evaluation run ID: {run.info.run_id}")
        print(f"  Decision: {decision}")
        return decision, melhor["run_id"]


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    from cli import build_parser

    p = build_parser("Avaliação comparativa e promoção de modelos champion/challenger")
    args, _ = p.parse_known_args()
    spark = SparkSession.builder.getOrCreate()
    avaliar(spark, args, args.lr_run_id, args.lgbm_run_id)
