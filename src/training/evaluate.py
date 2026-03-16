# src/training/evaluate.py
# Avaliação comparativa e promoção do champion
#
# Compara LR Baseline vs LightGBM treinados em train_baseline_lr.py e train_gbt.py.
# Declara o champion e atribui o alias @champion no Unity Catalog Model Registry.
#
# Critério de promoção: Precision@K(15% do val set)
#   K(15%) representa os municípios que um gestor de saúde monitoraria
#   num ciclo mensal — alerta acionável, não apenas ranking.
#   AUC-ROC mede discriminação geral; Precision@K mede utilidade operacional.
#
# Artefatos gerados:
#   - evaluation_report.json      : resumo estruturado para auditoria
#   - precision_at_k_comparison.png : curvas Precision@K val e test
#
# Pré-requisito: ambos os modelos devem estar treinados e registrados no
#   MLflow antes de executar este script. Passar os run_ids como argumento.

import mlflow
import mlflow.lightgbm
import mlflow.spark
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, average_precision_score
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
import tempfile
import os
import json
from datetime import datetime

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA  = "dev_christian_van_bellen"

TABLE_FEATURES = f"{CATALOG}.{SCHEMA}.gold_pressure_features"
MODEL_NAME     = f"{CATALOG}.{SCHEMA}.pressure_risk_classifier"
EXPERIMENT     = "/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr"

TARGET_COL = "target_alta_pressao"

FEATURE_COLS = [
    "casos_por_leito",      "casos_por_leito_lag1", "casos_por_leito_lag2",
    "casos_por_leito_lag3", "casos_por_leito_ma2",  "casos_por_leito_ma3",
    "casos_srag_lag1",      "casos_srag_lag2",
    "obitos_por_leito",     "uti_por_leito_uti",    "share_idosos",
    "growth_mom",           "growth_3m",            "acceleration",
    "rolling_std_3m",       "leitos_totais",        "leitos_uti",
    "num_hospitais",        "mes",                  "quarter",
    "is_semester1",         "is_rainy_season",
]

TRAIN_END  = "202412"
VAL_END    = "202506"
TEST_START = "202507"


# ── funções ─────────────────────────────────────────────────────
def _carregar_dados(spark: SparkSession):
    """
    Lê gold_pressure_features, filtra linhas com target válido,
    casteia features para double e remove nulos.
    """
    df = spark.table(TABLE_FEATURES)
    df = df.filter(F.col(TARGET_COL).isNotNull())
    df = df.filter(F.col("leitos_totais") >= 10)
    for col in FEATURE_COLS:
        df = df.withColumn(col, F.col(col).cast("double"))
    df = df.withColumn(TARGET_COL, F.col(TARGET_COL).cast("double"))
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    total     = df.count()
    positivos = df.filter(F.col(TARGET_COL) == 1.0).count()
    print(f"\n── Dados carregados ──")
    print(f"  Total: {total:,}  |  Positivos: {positivos:,} ({positivos/total*100:.1f}%)")
    return df


def _split_temporal(df):
    """Split temporal estrito por competencia (AAAAMM)."""
    train = df.filter(F.col("competencia") <= TRAIN_END)
    val   = df.filter((F.col("competencia") > TRAIN_END) & (F.col("competencia") <= VAL_END))
    test  = df.filter(F.col("competencia") > VAL_END)
    print(f"\n── Split temporal ──")
    print(f"  Treino  (<=  {TRAIN_END}): {train.count():,}")
    print(f"  Val     ({TRAIN_END[:4]}{int(TRAIN_END[4:])+1:02d}–{VAL_END}): {val.count():,}")
    print(f"  Teste   (>= {TEST_START}): {test.count():,}")
    return train, val, test


def _precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k_values: list) -> dict:
    """
    Calcula Precision@K e Recall@K ordenando os scores em ordem decrescente.
    K é o número de municípios alertados — reflete a capacidade operacional
    de monitoramento de um gestor num ciclo mensal.
    """
    n_positivos = int(labels.sum())
    ordem       = np.argsort(scores)[::-1]
    labels_ord  = labels[ordem]

    resultado = {}
    for k in k_values:
        k_clip     = min(k, len(labels))
        top_k      = labels_ord[:k_clip]
        tp_k       = int(top_k.sum())
        resultado[k] = {
            "precision@k": tp_k / (k_clip + 1e-9),
            "recall@k":    tp_k / (n_positivos + 1e-9),
            "tp@k":        tp_k,
        }
    return resultado


def _carregar_modelo_lr(spark: SparkSession, run_id: str):
    """Carrega modelo LR Spark salvo no MLflow."""
    print(f"  Carregando LR (run_id={run_id}) ...")
    return mlflow.spark.load_model(f"runs:/{run_id}/model")


def _carregar_modelo_lgbm(run_id: str):
    """Carrega modelo LightGBM salvo no MLflow."""
    print(f"  Carregando LightGBM (run_id={run_id}) ...")
    return mlflow.lightgbm.load_model(f"runs:/{run_id}/model")


def _scores_lr(model, split_sp) -> tuple:
    """Aplica modelo LR e extrai probabilidades da classe positiva."""
    preds  = model.transform(split_sp)
    scores = np.array(
        preds.withColumn("prob_pos", vector_to_array(F.col("probability")).getItem(1))
        .select("prob_pos")
        .rdd.map(lambda r: float(r[0]))
        .collect()
    )
    labels = np.array(
        preds.select(TARGET_COL).rdd.map(lambda r: float(r[0])).collect()
    )
    return scores, labels


def _scores_lgbm(model, X_pd: pd.DataFrame, y_pd: pd.Series) -> tuple:
    """Extrai scores LightGBM e converte labels para numpy."""
    return model.predict(X_pd), y_pd.values


def _avaliar_modelo(scores: np.ndarray, labels: np.ndarray,
                    split_name: str, model_name: str, k_ref: int) -> dict:
    """
    Calcula AUC-ROC, AUC-PR e Precision/Recall@K para múltiplos K.
    k_ref corresponde a 15% do split, usada como critério de promoção.
    """
    auc_roc = roc_auc_score(labels, scores)
    auc_pr  = average_precision_score(labels, scores)

    k_values   = sorted(set([10, 20, 50, 100, 200, 500, k_ref]))
    pk_results = _precision_recall_at_k(scores, labels, k_values)

    print(f"  [{model_name} / {split_name}]  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}"
          f"  Prec@{k_ref}={pk_results[k_ref]['precision@k']:.4f}"
          f"  Rec@{k_ref}={pk_results[k_ref]['recall@k']:.4f}")

    return {
        "auc_roc":            auc_roc,
        "auc_pr":             auc_pr,
        "precision_at_k_ref": pk_results[k_ref]["precision@k"],
        "recall_at_k_ref":    pk_results[k_ref]["recall@k"],
        "precision_recall_at_k": pk_results,
    }


def _plot_comparacao(resultados: dict, k_ref_val: int, k_ref_test: int):
    """
    Curvas Precision@K para val e test dos dois modelos lado a lado.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    model_keys   = ["baseline_logistic_regression", "lightgbm_gbt"]
    display_names = {"baseline_logistic_regression": "LR Baseline",
                     "lightgbm_gbt":                "LightGBM"}
    colors        = {"baseline_logistic_regression": "steelblue",
                     "lightgbm_gbt":                "darkorange"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, split_key, k_ref, titulo in [
        (ax1, "val",  k_ref_val,  f"Precision@K — Val Set"),
        (ax2, "test", k_ref_test, f"Precision@K — Test Set"),
    ]:
        for mk in model_keys:
            pk_data = resultados[mk][split_key]["precision_recall_at_k"]
            k_vals  = sorted(pk_data.keys())
            prec    = [pk_data[k]["precision@k"] for k in k_vals]
            ax.plot(k_vals, prec, marker="o", markersize=4, linewidth=2,
                    color=colors[mk], label=display_names[mk])

        k_ref_local = k_ref_val if split_key == "val" else k_ref_test
        ax.axvline(x=k_ref_local, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.annotate(f"K ref (15%)\nK={k_ref_local:,}", xy=(k_ref_local, ax.get_ylim()[1] * 0.9),
                    xytext=(k_ref_local + 5, ax.get_ylim()[1] * 0.88),
                    fontsize=8, color="red")
        ax.set_title(titulo, fontsize=13, fontweight="bold")
        ax.set_xlabel("K (municípios alertados)", fontsize=10)
        ax.set_ylabel("Precision@K", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.02)

    tmpdir = tempfile.mkdtemp()
    path   = os.path.join(tmpdir, "precision_at_k_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    os.remove(path)
    print("  ✓ precision_at_k_comparison.png logado")


def _gerar_relatorio(resultados: dict, champion_name: str,
                     k_ref_val: int, k_ref_test: int):
    """
    Serializa métricas de comparação em evaluation_report.json e loga no MLflow.
    """
    def _resumo(res, k_ref):
        return {
            "run_id":              res["run_id"],
            "val_auc_roc":         round(res["val"]["auc_roc"], 6),
            "val_auc_pr":          round(res["val"]["auc_pr"], 6),
            "val_precision_at_k":  round(res["val"]["precision_at_k_ref"], 6),
            "val_recall_at_k":     round(res["val"]["recall_at_k_ref"], 6),
            "test_auc_roc":        round(res["test"]["auc_roc"], 6),
            "test_auc_pr":         round(res["test"]["auc_pr"], 6),
            "test_precision_at_k": round(res["test"]["precision_at_k_ref"], 6),
            "test_recall_at_k":    round(res["test"]["recall_at_k_ref"], 6),
        }

    relatorio = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "champion":        champion_name,
        "criterion":       f"precision@k({k_ref_val}) no val set",
        "k_ref_val":       k_ref_val,
        "k_ref_test":      k_ref_test,
        "models": {
            "baseline_logistic_regression": _resumo(
                resultados["baseline_logistic_regression"], k_ref_val
            ),
            "lightgbm_gbt": _resumo(
                resultados["lightgbm_gbt"], k_ref_val
            ),
        },
    }

    print("\n── Relatório de avaliação ──")
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))

    tmpdir = tempfile.mkdtemp()
    path   = os.path.join(tmpdir, "evaluation_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    mlflow.log_artifact(path)
    os.remove(path)
    print("  ✓ evaluation_report.json logado")


def _promover_champion(champion_run_id: str, champion_name: str):
    """
    Encontra a versão do modelo registrada no run e atribui alias @champion.
    """
    client   = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    champion_version = None
    for v in versions:
        if v.run_id == champion_run_id:
            champion_version = v.version
            break

    if champion_version is None:
        raise ValueError(
            f"Versão do modelo não encontrada para run_id={champion_run_id}. "
            f"Certifique-se de que o modelo foi registrado no run de treinamento."
        )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=champion_version,
    )

    print(f"\n✓ Champion promovido:")
    print(f"  Modelo  : {MODEL_NAME}")
    print(f"  Versão  : {champion_version}")
    print(f"  Run ID  : {champion_run_id}")
    print(f"  Alias   : @champion")


def avaliar(spark: SparkSession, lr_run_id: str, lgbm_run_id: str) -> tuple:
    """
    Compara LR e LightGBM, declara o champion por Precision@K(15% val)
    e atribui o alias @champion no Unity Catalog Model Registry.
    Retorna (champion_name, champion_run_id).
    """
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="model_evaluation") as run:

        # ── dados ──────────────────────────────────────────────────
        df             = _carregar_dados(spark)
        train, val, test = _split_temporal(df)

        k_ref_val  = int(val.count()  * 0.15)
        k_ref_test = int(test.count() * 0.15)
        print(f"\nK referência — val: {k_ref_val:,}  |  test: {k_ref_test:,}")

        # pandas para LightGBM
        X_val  = val.select(FEATURE_COLS).toPandas()
        y_val  = val.select(TARGET_COL).toPandas()[TARGET_COL]
        X_test = test.select(FEATURE_COLS).toPandas()
        y_test = test.select(TARGET_COL).toPandas()[TARGET_COL]

        # ── modelos ────────────────────────────────────────────────
        print("\nCarregando modelos do MLflow ...")
        lr_model   = _carregar_modelo_lr(spark, lr_run_id)
        lgbm_model = _carregar_modelo_lgbm(lgbm_run_id)

        # ── scores ─────────────────────────────────────────────────
        print("\nExtaindo scores ...")
        lr_scores_val,  lr_labels_val  = _scores_lr(lr_model, val)
        lr_scores_test, lr_labels_test = _scores_lr(lr_model, test)
        lgbm_scores_val  = lgbm_model.predict(X_val)
        lgbm_scores_test = lgbm_model.predict(X_test)

        # ── avaliação ──────────────────────────────────────────────
        print("\n── Métricas de avaliação ──")
        resultados = {
            "baseline_logistic_regression": {
                "run_id": lr_run_id,
                "val":  _avaliar_modelo(lr_scores_val,   lr_labels_val,       "val",  "LR",   k_ref_val),
                "test": _avaliar_modelo(lr_scores_test,  lr_labels_test,      "test", "LR",   k_ref_test),
            },
            "lightgbm_gbt": {
                "run_id": lgbm_run_id,
                "val":  _avaliar_modelo(lgbm_scores_val,  y_val.values,  "val",  "LGBM", k_ref_val),
                "test": _avaliar_modelo(lgbm_scores_test, y_test.values, "test", "LGBM", k_ref_test),
            },
        }

        print("\n── Comparação val ──")
        for model_key, res in resultados.items():
            v = res["val"]
            print(f"  {model_key:<35}  AUC-ROC={v['auc_roc']:.4f}  AUC-PR={v['auc_pr']:.4f}  Prec@K={v['precision_at_k_ref']:.4f}")

        print("\n── Comparação test ──")
        for model_key, res in resultados.items():
            t = res["test"]
            print(f"  {model_key:<35}  AUC-ROC={t['auc_roc']:.4f}  AUC-PR={t['auc_pr']:.4f}  Prec@K={t['precision_at_k_ref']:.4f}")

        # ── decisão do champion ────────────────────────────────────
        lr_prec   = resultados["baseline_logistic_regression"]["val"]["precision_at_k_ref"]
        lgbm_prec = resultados["lightgbm_gbt"]["val"]["precision_at_k_ref"]

        if lgbm_prec >= lr_prec:
            champion_name   = "lightgbm_gbt"
            champion_run_id = lgbm_run_id
        else:
            champion_name   = "baseline_logistic_regression"
            champion_run_id = lr_run_id

        print(f"\n✓ Champion declarado: {champion_name}")
        print(f"  Critério: Precision@K({k_ref_val}) no val")
        print(f"  LR:   {lr_prec:.4f}")
        print(f"  LGBM: {lgbm_prec:.4f}")

        # ── artefatos e métricas ───────────────────────────────────
        print("\nGerando artefatos ...")
        _plot_comparacao(resultados, k_ref_val, k_ref_test)
        _gerar_relatorio(resultados, champion_name, k_ref_val, k_ref_test)

        mlflow.log_metrics({
            "lr_val_auc_roc":          resultados["baseline_logistic_regression"]["val"]["auc_roc"],
            "lr_val_precision_at_k":   lr_prec,
            "lgbm_val_auc_roc":        resultados["lightgbm_gbt"]["val"]["auc_roc"],
            "lgbm_val_precision_at_k": lgbm_prec,
            "k_ref_val":               float(k_ref_val),
        })

        # ── promoção ───────────────────────────────────────────────
        _promover_champion(champion_run_id, champion_name)

        print(f"\n✓ Evaluation run ID: {run.info.run_id}")
        return champion_name, champion_run_id


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    # substituir pelos run_ids reais antes de executar
    LR_RUN_ID   = "SUBSTITUIR"
    LGBM_RUN_ID = "SUBSTITUIR"
    spark = SparkSession.builder.getOrCreate()
    avaliar(spark, LR_RUN_ID, LGBM_RUN_ID)
