# src/training/train_baseline_lr.py
# Baseline — Logistic Regression com StandardScaler
#
# Objetivo: estabelecer uma linha de base interpretável antes de modelos GBT.
#           A LR converge rápido, expõe coeficientes e é facilmente auditável.
#
# Split temporal — NUNCA aleatório:
#   Treino : competencia <= 202412  (histórico 2023-2024)
#   Val    : 202501 – 202506        (primeiro semestre 2025)
#   Teste  : 202507+                (segundo semestre 2025 em diante)
#   Scoring ao vivo: 2026
#
# Target: target_alta_pressao (v2)
#   1 se casos_por_leito(t+1) >= percentil 85 nacional da competencia t+1
#   ~15% de positivos (desbalanceamento esperado — não corrigido no baseline)
#
# Rastreabilidade:
#   - Experiment: /mlops-demo/pressure-risk-baseline-lr
#   - Modelo registrado no Unity Catalog: pressure_risk_classifier

import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.spark
import numpy as np
from mlflow.models.signature import infer_signature
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import precision_recall_curve, roc_curve

from config import (
    FEATURE_COLS,
    MODEL_NAME,
    TARGET_COL,
    TEST_START,
    TRAIN_END,
    VAL_END,
)
from config import (
    MLFLOW_EXPERIMENT as EXPERIMENT,
)
from config import (  # noqa: E402
    TABLE_GOLD_FEATURES as TABLE_FEATURES,
)


# ── funções ─────────────────────────────────────────────────────
def _carregar_dados(spark: SparkSession):
    """
    Lê gold_pressure_features, filtra linhas com target válido,
    casteia features para double e remove nulos.
    """
    df = spark.table(TABLE_FEATURES)

    # apenas linhas com target definido (exclui último mês, usado para scoring)
    df = df.filter(F.col(TARGET_COL).isNotNull())

    # garantia extra de qualidade — mesmo critério do gold_features
    df = df.filter(F.col("leitos_totais") >= 10)

    # cast de todas as features para double (necessário para VectorAssembler)
    for col in FEATURE_COLS:
        df = df.withColumn(col, F.col(col).cast("double"))
    df = df.withColumn(TARGET_COL, F.col(TARGET_COL).cast("double"))

    # remove nulos nas colunas de modelo
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    total = df.count()
    positivos = df.filter(F.col(TARGET_COL) == 1.0).count()
    negativos = total - positivos
    print("\n── Dados carregados ──")
    print(
        f"  Total: {total:,}  |  Positivos: {positivos:,} ({positivos / total * 100:.1f}%)  |  Negativos: {negativos:,}"
    )
    return df


def _split_temporal(df):
    """
    Divide o dataset por competencia (AAAAMM) — split temporal estrito.
    Não há vazamento de informação futura no treino.
    """
    train = df.filter(F.col("competencia") <= TRAIN_END)
    val = df.filter((F.col("competencia") > TRAIN_END) & (F.col("competencia") <= VAL_END))
    test = df.filter(F.col("competencia") > VAL_END)

    print("\n── Split temporal ──")
    print(f"  Treino  (<=  {TRAIN_END}): {train.count():,}")
    print(f"  Val     ({TRAIN_END[:4]}{int(TRAIN_END[4:]) + 1:02d}–{VAL_END}): {val.count():,}")
    print(f"  Teste   (>= {TEST_START}): {test.count():,}")
    return train, val, test


def _avaliar(predictions, split_name: str) -> dict:
    """
    Calcula AUC-ROC, AUC-PR e métricas de classificação a threshold=0.5.
    Imprime tabela formatada e retorna dict para log no MLflow.
    """
    roc_eval = BinaryClassificationEvaluator(labelCol=TARGET_COL, metricName="areaUnderROC")
    pr_eval = BinaryClassificationEvaluator(labelCol=TARGET_COL, metricName="areaUnderPR")
    auc_roc = roc_eval.evaluate(predictions)
    auc_pr = pr_eval.evaluate(predictions)

    # métricas de classificação a threshold 0.5
    tp = predictions.filter((F.col("prediction") == 1.0) & (F.col(TARGET_COL) == 1.0)).count()
    fp = predictions.filter((F.col("prediction") == 1.0) & (F.col(TARGET_COL) == 0.0)).count()
    fn = predictions.filter((F.col("prediction") == 0.0) & (F.col(TARGET_COL) == 1.0)).count()
    tn = predictions.filter((F.col("prediction") == 0.0) & (F.col(TARGET_COL) == 0.0)).count()

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


def _log_model_summary(model, train):
    """
    Extrai coeficientes da LR, ordena por valor absoluto e salva como
    model_summary.txt para leitura fácil no MLflow UI.
    """
    lr_model = model.stages[-1]
    coefs = lr_model.coefficients.toArray()
    intercept = lr_model.intercept

    coef_pairs = sorted(
        zip(FEATURE_COLS, coefs, strict=False), key=lambda x: abs(x[1]), reverse=True
    )

    total = train.count()
    positivos = train.filter(F.col(TARGET_COL) == 1.0).count()

    lines = [
        "=== MODEL SUMMARY — Logistic Regression Baseline ===",
        "",
        f"Intercept: {intercept:.6f}",
        "",
        "Coefficients (sorted by absolute value):",
        f"{'Feature':<32} {'Coefficient':>12}",
        "─" * 46,
    ]
    for feat, coef in coef_pairs:
        sinal = "+" if coef >= 0 else ""
        lines.append(f"{feat:<32} {sinal}{coef:.6f}")

    lines += [
        "",
        "Hyperparameters:",
        "  regParam        : 0.01",
        "  elasticNetParam : 0.0",
        "  maxIter         : 100",
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


def _plot_roc_pr_curves(model, train, val, test):
    """
    Gera figura com ROC Curve e Precision-Recall Curve para os três splits.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    splits = [
        ("train", train, "steelblue", "--"),
        ("val", val, "darkorange", "-"),
        ("test", test, "forestgreen", "-"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # prevalência para baseline de PR
    total_labels = np.array(train.select(TARGET_COL).rdd.map(lambda r: float(r[0])).collect())
    prevalencia = total_labels.mean()

    for split_name, split_df, color, ls in splits:
        preds = model.transform(split_df)
        probs = np.array(preds.select("probability").rdd.map(lambda r: float(r[0][1])).collect())
        labels = np.array(preds.select(TARGET_COL).rdd.map(lambda r: float(r[0])).collect())

        # ROC
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_roc = np.trapz(tpr, fpr)
        ax1.plot(
            fpr,
            tpr,
            color=color,
            linestyle=ls,
            linewidth=2,
            label=f"{split_name.capitalize()} (AUC={auc_roc:.3f})",
        )

        # PR
        prec, rec, _ = precision_recall_curve(labels, probs)
        auc_pr = np.trapz(prec, rec[::-1])
        ax2.plot(
            rec,
            prec,
            color=color,
            linestyle=ls,
            linewidth=2,
            label=f"{split_name.capitalize()} (AUC-PR={auc_pr:.3f})",
        )

    # ROC — diagonal de referência
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_title("ROC Curve — LR Baseline", fontsize=13, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=10)
    ax1.set_ylabel("True Positive Rate", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])

    # PR — baseline de prevalência
    ax2.axhline(
        y=prevalencia,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Baseline (prevalência={prevalencia:.2f})",
    )
    ax2.set_title("Precision-Recall Curve — LR Baseline", fontsize=13, fontweight="bold")
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


def _plot_target_incidence_timeline(df):
    """
    Incidência do target ao longo das competências com faixas de train/val/test.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # agrega por competencia
    incidencia = (
        df.groupBy("competencia")
        .agg(F.round(F.avg(TARGET_COL) * 100, 2).alias("pct_positivos"))
        .orderBy("competencia")
        .toPandas()
    )
    comps = incidencia["competencia"].tolist()
    pcts = incidencia["pct_positivos"].tolist()
    x = range(len(comps))

    fig, ax = plt.subplots(figsize=(10, 5))

    # faixas de fundo
    train_mask = [i for i, c in enumerate(comps) if c <= TRAIN_END]
    val_mask = [i for i, c in enumerate(comps) if TRAIN_END < c <= VAL_END]
    test_mask = [i for i, c in enumerate(comps) if c > VAL_END]

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

    # série temporal
    ax.plot(x, pcts, color="steelblue", linewidth=2, marker="o", markersize=4)

    # linha de referência dos 15%
    ax.axhline(y=15, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Referência 15%")

    # labels do eixo x — só o primeiro mês de cada ano
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


def _plot_decile_analysis(model, df, split_name: str):
    """
    Divide scores de probabilidade em 10 decis e plota incidência do target
    e volume de registros por decil.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # passo 1: obter predictions com prob_positivo como coluna double
    preds = model.transform(df)
    preds = preds.withColumn(
        "prob_positivo",
        vector_to_array(F.col("probability")).getItem(1).cast("double"),
    )

    # passo 2: calcular quantis para definir os decis
    quantis = preds.approxQuantile("prob_positivo", [i / 10 for i in range(1, 11)], 0.01)

    # passo 3: atribuir decil a cada registro via F.when encadeado corretamente
    decil_expr = F.when(F.col("prob_positivo") <= quantis[0], F.lit(1))
    for d in range(1, 9):
        decil_expr = decil_expr.when(
            (F.col("prob_positivo") > quantis[d - 1]) & (F.col("prob_positivo") <= quantis[d]),
            F.lit(d + 1),
        )
    decil_expr = decil_expr.otherwise(F.lit(10))
    preds = preds.withColumn("decil", decil_expr)

    # passo 4: agregar por decil
    decil_df = (
        preds.groupBy("decil")
        .agg(
            F.count("*").alias("n"),
            F.sum(TARGET_COL).alias("positivos"),
            F.round(F.avg(TARGET_COL) * 100, 2).alias("pct_positivos"),
            F.round(F.avg("prob_positivo") * 100, 2).alias("score_medio"),
        )
        .orderBy("decil")
        .toPandas()
    )

    decis = decil_df["decil"].tolist()
    pcts = decil_df["pct_positivos"].tolist()
    volumes = decil_df["n"].tolist()
    prevalencia = preds.filter(F.col(TARGET_COL) == 1.0).count() / preds.count() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # subplot 1 — incidência por decil
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

    # subplot 2 — volume por decil
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


def treinar(spark: SparkSession) -> str:
    """
    Treina o modelo baseline de Logistic Regression, avalia nos três splits
    e registra o modelo no MLflow e no Unity Catalog Model Registry.
    Retorna o run_id do MLflow.
    """
    mlflow.set_experiment(experiment_name=EXPERIMENT)

    with mlflow.start_run(run_name="baseline_logistic_regression") as run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "features": len(FEATURE_COLS),
                "train_end": TRAIN_END,
                "val_end": VAL_END,
                "test_start": TEST_START,
                "regParam": 0.01,
                "elasticNetParam": 0.0,
                "maxIter": 100,
                "target": TARGET_COL,
                "target_version": "v2",
                "grain": "municipio_x_competencia",
            }
        )

        df = _carregar_dados(spark)
        train, val, test = _split_temporal(df)

        # ── pipeline ──────────────────────────────────────────────
        assembler = VectorAssembler(
            inputCols=FEATURE_COLS,
            outputCol="features_raw",
            handleInvalid="skip",
        )
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=True,
            withStd=True,
        )
        lr = LogisticRegression(
            featuresCol="features",
            labelCol=TARGET_COL,
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0,
            family="binomial",
        )
        pipeline = Pipeline(stages=[assembler, scaler, lr])

        print("\nTreinando pipeline ...")
        model = pipeline.fit(train)

        # ── avaliação nos três splits ──────────────────────────────
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            preds = model.transform(split_df)
            metrics = _avaliar(preds, split_name)
            for k, v in metrics.items():
                mlflow.log_metric(f"{split_name}_{k}", v)

        # ── artefatos visuais ─────────────────────────────────────
        print("\nGerando artefatos visuais ...")
        _log_model_summary(model, train)
        _plot_roc_pr_curves(model, train, val, test)
        _plot_target_incidence_timeline(df)
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            _plot_decile_analysis(model, split_df, split_name)

        # ── assinatura e registro do modelo ───────────────────────
        train_sample = train.select(FEATURE_COLS).limit(100).toPandas()
        pred_sample = model.transform(train.limit(100)).select("probability").toPandas()
        signature = infer_signature(train_sample, pred_sample)

        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        print(f"\n✓ Run ID: {run.info.run_id}")
        print(f"  Modelo registrado como: {MODEL_NAME}")
        return run.info.run_id


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    treinar(spark)
