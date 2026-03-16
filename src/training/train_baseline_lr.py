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

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from databricks.feature_engineering import FeatureEngineeringClient
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA  = "dev_christian_van_bellen"

TABLE_FEATURES = f"{CATALOG}.{SCHEMA}.gold_pressure_features"
MODEL_NAME     = f"{CATALOG}.{SCHEMA}.pressure_risk_classifier"
EXPERIMENT     = "/mlops-demo/pressure-risk-baseline-lr"

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

# Split temporal — não alterar sem versionar o experimento
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

    total     = df.count()
    positivos = df.filter(F.col(TARGET_COL) == 1.0).count()
    negativos = total - positivos
    print(f"\n── Dados carregados ──")
    print(f"  Total: {total:,}  |  Positivos: {positivos:,} ({positivos/total*100:.1f}%)  |  Negativos: {negativos:,}")
    return df


def _split_temporal(df):
    """
    Divide o dataset por competencia (AAAAMM) — split temporal estrito.
    Não há vazamento de informação futura no treino.
    """
    train = df.filter(F.col("competencia") <= TRAIN_END)
    val   = df.filter((F.col("competencia") > TRAIN_END) & (F.col("competencia") <= VAL_END))
    test  = df.filter(F.col("competencia") > VAL_END)

    print(f"\n── Split temporal ──")
    print(f"  Treino  (<=  {TRAIN_END}): {train.count():,}")
    print(f"  Val     ({TRAIN_END[:4]}{int(TRAIN_END[4:])+1:02d}–{VAL_END}): {val.count():,}")
    print(f"  Teste   (>= {TEST_START}): {test.count():,}")
    return train, val, test


def _avaliar(predictions, split_name: str) -> dict:
    """
    Calcula AUC-ROC, AUC-PR e métricas de classificação a threshold=0.5.
    Imprime tabela formatada e retorna dict para log no MLflow.
    """
    roc_eval = BinaryClassificationEvaluator(
        labelCol=TARGET_COL, metricName="areaUnderROC"
    )
    pr_eval = BinaryClassificationEvaluator(
        labelCol=TARGET_COL, metricName="areaUnderPR"
    )
    auc_roc = roc_eval.evaluate(predictions)
    auc_pr  = pr_eval.evaluate(predictions)

    # métricas de classificação a threshold 0.5
    tp = predictions.filter((F.col("prediction") == 1.0) & (F.col(TARGET_COL) == 1.0)).count()
    fp = predictions.filter((F.col("prediction") == 1.0) & (F.col(TARGET_COL) == 0.0)).count()
    fn = predictions.filter((F.col("prediction") == 0.0) & (F.col(TARGET_COL) == 1.0)).count()
    tn = predictions.filter((F.col("prediction") == 0.0) & (F.col(TARGET_COL) == 0.0)).count()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n── Métricas [{split_name}] ──")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"  AUC-PR    : {auc_pr:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    return {
        "auc_roc":   auc_roc,
        "auc_pr":    auc_pr,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        float(tp),
        "fp":        float(fp),
        "fn":        float(fn),
        "tn":        float(tn),
    }


def treinar(spark: SparkSession) -> str:
    """
    Treina o modelo baseline de Logistic Regression, avalia nos três splits
    e registra o modelo no MLflow e no Unity Catalog Model Registry.
    Retorna o run_id do MLflow.
    """
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="baseline_logistic_regression") as run:

        mlflow.log_params({
            "model_type":      "LogisticRegression",
            "features":        len(FEATURE_COLS),
            "train_end":       TRAIN_END,
            "val_end":         VAL_END,
            "test_start":      TEST_START,
            "regParam":        0.01,
            "elasticNetParam": 0.0,
            "maxIter":         100,
            "target":          TARGET_COL,
            "target_version":  "v2",
            "grain":           "municipio_x_competencia",
        })

        df              = _carregar_dados(spark)
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
            preds   = model.transform(split_df)
            metrics = _avaliar(preds, split_name)
            for k, v in metrics.items():
                mlflow.log_metric(f"{split_name}_{k}", v)

        # ── assinatura e registro do modelo ───────────────────────
        train_sample = train.select(FEATURE_COLS).limit(100).toPandas()
        pred_sample  = model.transform(train.limit(100)).select("probability").toPandas()
        signature    = infer_signature(train_sample, pred_sample)

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
