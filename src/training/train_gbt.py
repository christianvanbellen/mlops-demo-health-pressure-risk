# src/training/train_gbt.py
# Modelo principal: LightGBM (classificação binária)
#
# Comparar com baseline LR no mesmo experimento MLflow — ambos usam o mesmo
# EXPERIMENT path para aparecerem lado a lado no UI de comparação.
#
# Decisões de design:
#     Compensa o desbalanceamento de ~15% de positivos sem oversample/undersample.
#   Early stopping em val (50 rounds) — evita overfitting sem tuning manual de rounds.
#   Feature importance por gain — mais estável que split count para features correlatas.
#   log_models=False no autolog — modelo logado manualmente com registered_model_name
#     para garantir o registro no Unity Catalog.
#
# Split temporal — NUNCA aleatório:
#   Treino : competencia <= 202412  (histórico 2023-2024)
#   Val    : 202501 – 202506        (primeiro semestre 2025, usado no early stopping)
#   Teste  : 202507+                (segundo semestre 2025 em diante)

import json
import os
import tempfile

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from cli import FEATURE_COLS, TARGET_COL


# ── funções ─────────────────────────────────────────────────────
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
    negativos = total - positivos
    print("\n── Dados carregados ──")
    print(
        f"  Total: {total:,}  |  Positivos: {positivos:,} ({positivos / total * 100:.1f}%)  |  Negativos: {negativos:,}"
    )
    return df


def _split_temporal(df, train_end: str, val_end: str, test_start: str):
    """
    Divide o dataset por competencia (AAAAMM) — split temporal estrito.
    """
    train = df.filter(F.col("competencia") <= train_end)
    val = df.filter((F.col("competencia") > train_end) & (F.col("competencia") <= val_end))
    test = df.filter(F.col("competencia") > val_end)

    print("\n── Split temporal ──")
    print(f"  Treino  (<=  {train_end}): {train.count():,}")
    print(f"  Val     ({train_end[:4]}{int(train_end[4:]) + 1:02d}–{val_end}): {val.count():,}")
    print(f"  Teste   (>= {test_start}): {test.count():,}")
    return train, val, test


def _para_pandas(train_sp, val_sp, test_sp):
    """
    Converte os splits Spark para pandas. A conversão é feita aqui,
    de uma vez, para evitar múltiplos .toPandas() dispersos no código.
    """
    print("\nConvertendo splits para pandas ...")
    X_train = train_sp.select(FEATURE_COLS).toPandas()
    y_train = train_sp.select(TARGET_COL).toPandas()[TARGET_COL]
    X_val = val_sp.select(FEATURE_COLS).toPandas()
    y_val = val_sp.select(TARGET_COL).toPandas()[TARGET_COL]
    X_test = test_sp.select(FEATURE_COLS).toPandas()
    y_test = test_sp.select(TARGET_COL).toPandas()[TARGET_COL]

    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def _treinar_lgbm(
    X_train, y_train, X_val, y_val, lgbm_params: dict, num_boost_round: int, early_stopping: int
):
    """Treina o LightGBM com early stopping no conjunto de validação."""
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    print("\nTreinando LightGBM ...")
    model = lgb.train(
        params=lgbm_params,
        train_set=train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stopping),
            lgb.log_evaluation(50),
        ],
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


def _avaliar_lgbm(model, X, y, split_name: str) -> dict:
    """
    Calcula AUC-ROC, AUC-PR e métricas de classificação a threshold=0.5.
    """
    probs = model.predict(X)
    preds_bin = (probs >= 0.5).astype(int)

    auc_roc = roc_auc_score(y, probs)
    auc_pr = average_precision_score(y, probs)

    tp = int(((preds_bin == 1) & (y == 1)).sum())
    fp = int(((preds_bin == 1) & (y == 0)).sum())
    fn = int(((preds_bin == 0) & (y == 1)).sum())
    tn = int(((preds_bin == 0) & (y == 0)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n── Métricas [{split_name}] ──")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"  AUC-PR    : {auc_pr:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _log_model_summary(
    model, X_train, y_train, lgbm_params: dict, num_boost_round: int, early_stopping: int
):
    """
    Gera model_summary.txt com feature importance (gain), parâmetros e
    estatísticas do treino.
    """
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importance(importance_type="gain"), strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    positivos = int(y_train.sum())
    total = len(y_train)

    lines = [
        "=== MODEL SUMMARY — LightGBM Baseline ===",
        "",
        f"Best iteration: {model.best_iteration}",
        "",
        "Feature Importance (gain, sorted descending):",
        f"{'Feature':<32} {'Importance (gain)':>18}",
        "─" * 52,
    ]
    for feat, imp in importances:
        lines.append(f"{feat:<32} {imp:>18.2f}")

    lines += [
        "",
        "LightGBM Parameters:",
    ]
    for k, v in lgbm_params.items():
        lines.append(f"  {k:<22}: {v}")

    lines += [
        "",
        f"num_boost_round  : {num_boost_round}",
        f"early_stopping   : {early_stopping}",
        "",
        f"Training set size: {total:,}",
        f"Positive rate:     {positivos / total * 100:.1f}%",
    ]

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "model_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    mlflow.log_artifact(path)
    os.remove(path)
    print("  ✓ model_summary.txt logado")


def _plot_roc_pr_curves(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Gera figura com ROC Curve e Precision-Recall Curve para os três splits.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    splits = [
        ("train", X_train, y_train, "steelblue", "--"),
        ("val", X_val, y_val, "darkorange", "-"),
        ("test", X_test, y_test, "forestgreen", "-"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    prevalencia = float(y_train.mean())

    for split_name, X, y, color, ls in splits:
        probs = model.predict(X)

        fpr, tpr, _ = roc_curve(y, probs)
        auc_roc = roc_auc_score(y, probs)
        ax1.plot(
            fpr,
            tpr,
            color=color,
            linestyle=ls,
            linewidth=2,
            label=f"{split_name.capitalize()} (AUC={auc_roc:.3f})",
        )

        prec, rec, _ = precision_recall_curve(y, probs)
        auc_pr = average_precision_score(y, probs)
        ax2.plot(
            rec,
            prec,
            color=color,
            linestyle=ls,
            linewidth=2,
            label=f"{split_name.capitalize()} (AUC-PR={auc_pr:.3f})",
        )

    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_title("ROC Curve — LightGBM", fontsize=13, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=10)
    ax1.set_ylabel("True Positive Rate", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])

    ax2.axhline(
        y=prevalencia,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Baseline (prevalência={prevalencia:.2f})",
    )
    ax2.set_title("Precision-Recall Curve — LightGBM", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=10)
    ax2.set_ylabel("Precision", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "roc_pr_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    os.remove(path)
    print("  ✓ roc_pr_curves.png logado")


def _plot_target_incidence_timeline(df_spark, train_end: str, val_end: str):
    """
    Incidência do target ao longo das competências com faixas de train/val/test.
    Idêntico ao baseline — recebe df Spark completo.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    incidencia = (
        df_spark.groupBy("competencia")
        .agg(F.round(F.avg(TARGET_COL) * 100, 2).alias("pct_positivos"))
        .orderBy("competencia")
        .toPandas()
    )
    comps = incidencia["competencia"].tolist()
    pcts = incidencia["pct_positivos"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))

    train_mask = [i for i, c in enumerate(comps) if c <= train_end]
    val_mask = [i for i, c in enumerate(comps) if train_end < c <= val_end]
    test_mask = [i for i, c in enumerate(comps) if c > val_end]

    for mask, color, label in [
        (train_mask, "steelblue", "Train"),
        (val_mask, "darkorange", "Val"),
        (test_mask, "forestgreen", "Test"),
    ]:
        if mask:
            ax.axvspan(mask[0] - 0.5, mask[-1] + 0.5, alpha=0.1, color=color, label=label)
            mid = (mask[0] + mask[-1]) / 2
            ax.annotate(
                label, xy=(mid, 27), ha="center", fontsize=9, color=color, fontweight="bold"
            )

    ax.plot(range(len(comps)), pcts, color="steelblue", linewidth=2, marker="o", markersize=4)
    ax.axhline(y=15, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Referência 15%")

    tick_pos = [i for i, c in enumerate(comps) if c.endswith("01")]
    tick_labels = [f"{c[:4]}-01" for c in comps if c.endswith("01")]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=9, rotation=30)

    ax.set_ylim(0, 30)
    ax.set_xlabel("Competência", fontsize=10)
    ax.set_ylabel("% Municípios em Alta Pressão", fontsize=10)
    ax.set_title("Target Incidence by Competencia", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "target_incidence_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    os.remove(path)
    print("  ✓ target_incidence_timeline.png logado")


def _plot_feature_importance(model):
    """
    Gráfico horizontal de barras com top 15 features por gain (importância estável).
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    importances = sorted(
        zip(FEATURE_COLS, model.feature_importance(importance_type="gain"), strict=False),
        key=lambda x: x[1],
        reverse=True,
    )[:15]
    feats, imps = zip(*importances, strict=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(feats)), imps, color="steelblue", edgecolor="white")

    for bar, imp in zip(bars, imps, strict=False):
        ax.text(
            bar.get_width() + max(imps) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{imp:.0f}",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importância (gain)", fontsize=10)
    ax.set_title("Feature Importance (Gain) — LightGBM", fontsize=13, fontweight="bold")

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    os.remove(path)
    print("  ✓ feature_importance.png logado")


def _plot_decile_analysis(model, X, y, split_name: str):
    """
    Divide scores de probabilidade em 10 decis e plota incidência do target
    e volume de registros por decil. Usa pandas em vez de PySpark.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    probs = model.predict(X)
    df_d = pd.DataFrame({"prob": probs, "target": y.values})
    df_d["decil"] = pd.qcut(probs, q=10, labels=False, duplicates="drop") + 1

    decil_agg = (
        df_d.groupby("decil")
        .agg(
            n=("target", "count"),
            positivos=("target", "sum"),
            pct_positivos=("target", lambda s: round(s.mean() * 100, 2)),
            score_medio=("prob", lambda s: round(s.mean() * 100, 2)),
        )
        .reset_index()
    )

    decis = decil_agg["decil"].tolist()
    pcts = decil_agg["pct_positivos"].tolist()
    volumes = decil_agg["n"].tolist()
    prevalencia = float(y.mean()) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(decis, pcts, color="steelblue", edgecolor="white", linewidth=0.5)
    ax1.axhline(
        y=prevalencia,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Prevalência ({prevalencia:.1f}%)",
    )
    for bar, pct in zip(bars1, pcts, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.set_title(
        f"Target Incidence by Score Decile [{split_name}]", fontsize=13, fontweight="bold"
    )
    ax1.set_xlabel("Decil", fontsize=10)
    ax1.set_ylabel("% Target = 1", fontsize=10)
    ax1.set_xticks(decis)
    ax1.legend(fontsize=9)

    bars2 = ax2.bar(decis, volumes, color="slategrey", edgecolor="white", linewidth=0.5)
    for bar, n in zip(bars2, volumes, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{n:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_title(f"Records by Score Decile [{split_name}]", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Decil", fontsize=10)
    ax2.set_ylabel("Registros", fontsize=10)
    ax2.set_xticks(decis)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, f"decile_analysis_{split_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    os.remove(path)
    print(f"  ✓ decile_analysis_{split_name}.png logado")


def treinar(spark: SparkSession, args) -> str:
    """
    Treina o modelo LightGBM, avalia nos três splits, gera artefatos visuais
    e registra no MLflow e Unity Catalog Model Registry.
    Retorna o run_id do MLflow.
    """
    experiment = args.mlflow_experiment
    model_name = args.model_name
    table_features = args.table_gold_features
    train_end = args.train_end
    val_end = args.val_end
    test_start = args.test_start
    lgbm_params = json.loads(args.lgbm_params_json)
    num_boost_round = args.num_boost_round
    early_stopping = args.early_stopping

    mlflow.set_experiment(experiment_name=experiment)
    mlflow.lightgbm.autolog(log_models=False)
    # log_models=False — modelo logado manualmente com registered_model_name
    # para garantir registro no Unity Catalog

    with mlflow.start_run(run_name="lightgbm_gbt") as run:
        mlflow.log_params(
            {
                "model_type": "LightGBM",
                "features": len(FEATURE_COLS),
                "num_boost_round": num_boost_round,
                "early_stopping": early_stopping,
                "train_end": train_end,
                "val_end": val_end,
                "test_start": test_start,
                "target": TARGET_COL,
                "target_version": "v2",
                "grain": "municipio_x_competencia",
                **lgbm_params,
            }
        )

        df_spark = _carregar_dados(spark, table_features)
        train_sp, val_sp, test_sp = _split_temporal(df_spark, train_end, val_end, test_start)
        X_train, y_train, X_val, y_val, X_test, y_test = _para_pandas(train_sp, val_sp, test_sp)

        model = _treinar_lgbm(
            X_train, y_train, X_val, y_val, lgbm_params, num_boost_round, early_stopping
        )
        mlflow.log_metric("best_iteration", model.best_iteration)

        # ── avaliação nos três splits ──────────────────────────────
        all_metrics = {}
        for split_name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            metrics = _avaliar_lgbm(model, X, y, split_name)
            for k, v in metrics.items():
                mlflow.log_metric(f"{split_name}_{k}", v)
            all_metrics[split_name] = metrics

        # ── artefatos visuais ──────────────────────────────────────
        print("\nGerando artefatos visuais ...")
        _log_model_summary(model, X_train, y_train, lgbm_params, num_boost_round, early_stopping)
        _plot_roc_pr_curves(model, X_train, y_train, X_val, y_val, X_test, y_test)
        _plot_target_incidence_timeline(df_spark, train_end, val_end)
        _plot_feature_importance(model)
        for split_name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            _plot_decile_analysis(model, X, y, split_name)

        # ── registro do modelo ────────────────────────────────────
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name,
        )

        print(f"\n✓ Run ID: {run.info.run_id}")
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Val  AUC-ROC: {all_metrics['val']['auc_roc']:.4f}")
        print(f"  Test AUC-ROC: {all_metrics['test']['auc_roc']:.4f}")
        return run.info.run_id


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    from cli import build_parser

    p = build_parser("Treino do modelo principal LightGBM")
    args, _ = p.parse_known_args()
    spark = SparkSession.builder.getOrCreate()
    treinar(spark, args)
