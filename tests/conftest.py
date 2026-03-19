import os
import sys
from unittest.mock import MagicMock

# ── Mock do pyspark ────────────────────────────────────
# Os módulos de src/ importam pyspark no nível do módulo.
# Para testes unitários locais (sem Spark), mockamos os
# módulos antes de qualquer import de src/.
# No cluster Databricks o pyspark real é usado.

PYSPARK_MOCKS = [
    "pyspark",
    "pyspark.sql",
    "pyspark.sql.functions",
    "pyspark.sql.window",
    "pyspark.sql.types",
    "pyspark.sql.dataframe",
    "pyspark.ml",
    "pyspark.ml.feature",
    "pyspark.ml.classification",
    "pyspark.ml.evaluation",
    "pyspark.ml.pipeline",
    "pyspark.ml.functions",
    "databricks.feature_engineering",
    "mlflow.spark",
    # DQX — não disponível fora do cluster Databricks
    "databricks.labs.dqx",
    "databricks.labs.dqx.engine",
    "databricks.labs.dqx.col_functions",
    "databricks.labs.dqx.row_checks",
    "databricks.sdk",
    # Evidently — não instalado no ambiente de testes unitários locais
    "evidently",
    "evidently.presets",
    "evidently.metric_preset",
]

for mod in PYSPARK_MOCKS:
    sys.modules[mod] = MagicMock()

# Garante que src/ está no PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Variáveis de ambiente necessárias para importar módulos que dependem de config.py
_DEV_VARS = {
    "CATALOG": "ds_dev_db",
    "SCHEMA": "dev_christian_van_bellen",
    "TABLE_BRONZE_SRAG": "ds_dev_db.dev_christian_van_bellen.bronze_srag",
    "TABLE_BRONZE_HOSPITAIS_LEITOS": "ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos",
    "TABLE_BRONZE_CNES": "ds_dev_db.dev_christian_van_bellen.bronze_cnes_estabelecimentos",
    "TABLE_SILVER_SRAG": "ds_dev_db.dev_christian_van_bellen.silver_srag_municipio_semana",
    "TABLE_SILVER_CAPACITY": "ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_mes",
    "TABLE_GOLD_FEATURES": "ds_dev_db.dev_christian_van_bellen.gold_pressure_features",
    "TABLE_GOLD_SCORING": "ds_dev_db.dev_christian_van_bellen.gold_pressure_scoring",
    "TABLE_GOLD_MONITOR": "ds_dev_db.dev_christian_van_bellen.monitoring_performance",
    "LANDING_PATH": "/Volumes/ds_dev_db/dev_christian_van_bellen/landing",
    "MLFLOW_EXPERIMENT": "/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr",
    "MODEL_NAME": "ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier",
    "RETRAIN_JOB_NAME": "job_health_pressure_retrain",
    "TRAIN_END": "202412",
    "VAL_END": "202506",
    "TEST_START": "202507",
    "TARGET_PERCENTILE": "0.85",
    "PRECISION_K_THRESHOLD": "0.55",
    "MIN_CONSECUTIVE_BELOW": "2",
    "SCORING_MIN_QUALITY": "0.5",
    "AB_CHALLENGER_PCT": "0.20",
    "DRIFT_SEASONAL_FEATURES": "mes,quarter,is_semester1,is_rainy_season",
    "LGBM_PARAMS_JSON": '{"objective":"binary","metric":"binary_logloss","boosting_type":"gbdt","num_leaves":63,"learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"min_child_samples":20,"reg_alpha":0.1,"reg_lambda":0.1,"verbose":-1}',
    "NUM_BOOST_ROUND": "500",
    "EARLY_STOPPING": "50",
}
for _k, _v in _DEV_VARS.items():
    os.environ.setdefault(_k, _v)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def sample_monitor_df():
    return pd.DataFrame({
        "competencia":        ["202301", "202302", "202303",
                               "202304", "202305", "202306"],
        "precision_at_k":     [0.60, 0.62, 0.59, 0.61, 0.63, 0.60],
        "auc_pr":             [0.65, 0.67, 0.64, 0.66, 0.68, 0.65],
        "n_municipios":       [3300] * 6,
        "consolidation_flag": ["consolidado"] * 6,
        "simulated":          [True] * 6,
        "monitor_date":       ["2026-03-17"] * 6,
    })
