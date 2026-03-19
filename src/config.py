"""
Módulo central de configuração do projeto.

Resolução em cascata para cada variável (ordem de prioridade):
  1. Spark conf  — spark.conf.get("health.pressure.<key>")
                   injetável via Spark Config no cluster ou new_cluster
  2. Env var     — os.environ.get("<KEY>")
                   injetável via variáveis de ambiente do cluster
  3. Default     — valor hardcodado passado na chamada (= valores de dev)

Se nenhuma camada resolver e não houver default → ValueError explícito.

Uso: from config import CATALOG, TABLE_BRONZE_SRAG, ...
"""

import json
import os


class ConfigLoader:
    """
    Resolve configurações em cascata:
    1. Spark conf  (spark.conf.get com prefixo health.pressure.<key>)
    2. Env var     (os.environ.get com key em uppercase)
    3. Default     (valor hardcodado passado na chamada)
    """

    def __init__(self):
        self._spark = None

    def _get_spark(self):
        try:
            from pyspark.sql import SparkSession

            return SparkSession.getActiveSession()
        except Exception:
            return None

    def get(self, key: str, dtype: type = str, default=None):
        """
        Parâmetros:
          key     — nome da variável (snake_case)
          dtype   — tipo de conversão: str, int, float, bool, list, dict
          default — valor fallback; se None e nenhuma camada resolver → erro
        """
        spark_key = f"health.pressure.{key}"
        env_key = key.upper()

        # 1. Spark conf
        # Valores do Spark conf são sempre strings; isinstance garante que
        # mocks (testes unitários) não passem para _cast acidentalmente.
        spark = self._get_spark()
        if spark is not None:
            val = spark.conf.get(spark_key, None)
            if isinstance(val, str):
                return self._cast(val, dtype, key)

        # 2. Env var
        val = os.environ.get(env_key)
        if val is not None:
            return self._cast(val, dtype, key)

        # 3. Default
        if default is not None:
            return self._cast(str(default) if not isinstance(default, (list, dict)) else default, dtype, key)

        raise ValueError(
            f"Config '{key}' não encontrada. "
            f"Tentadas: spark.conf '{spark_key}', env var '{env_key}'. "
            f"Configure uma das camadas ou passe default=."
        )

    def _cast(self, value, dtype: type, key: str):
        # list e dict chegam já convertidos quando vêm do default
        if isinstance(value, dtype):
            return value
        if not isinstance(value, str):
            value = str(value)
        try:
            if dtype is bool:
                return value.lower() in ("true", "1", "yes")
            if dtype is list:
                return [v.strip() for v in value.split(",")]
            if dtype is dict:
                return json.loads(value)
            return dtype(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Config '{key}': não foi possível converter '{value}' para {dtype.__name__}"
            ) from e


# ── instância global ──────────────────────────────────────────────
_cfg = ConfigLoader()

# ── Ambiente ─────────────────────────────────────────────────────
CATALOG = _cfg.get("catalog", str, "ds_dev_db")
SCHEMA  = _cfg.get("schema",  str, "dev_christian_van_bellen")

# ── Tabelas Bronze ────────────────────────────────────────────────
TABLE_BRONZE_SRAG = _cfg.get(
    "table_bronze_srag", str,
    "ds_dev_db.dev_christian_van_bellen.bronze_srag",
)
TABLE_BRONZE_HOSPITAIS_LEITOS = _cfg.get(
    "table_bronze_hospitais_leitos", str,
    "ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos",
)
TABLE_BRONZE_CNES = _cfg.get(
    "table_bronze_cnes", str,
    "ds_dev_db.dev_christian_van_bellen.bronze_cnes_estabelecimentos",
)

# ── Tabelas Silver ────────────────────────────────────────────────
TABLE_SILVER_SRAG = _cfg.get(
    "table_silver_srag", str,
    "ds_dev_db.dev_christian_van_bellen.silver_srag_municipio_semana",
)
TABLE_SILVER_CAPACITY = _cfg.get(
    "table_silver_capacity", str,
    "ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_mes",
)

# ── Tabelas Gold ──────────────────────────────────────────────────
TABLE_GOLD_FEATURES = _cfg.get(
    "table_gold_features", str,
    "ds_dev_db.dev_christian_van_bellen.gold_pressure_features",
)
TABLE_GOLD_SCORING = _cfg.get(
    "table_gold_scoring", str,
    "ds_dev_db.dev_christian_van_bellen.gold_pressure_scoring",
)
TABLE_GOLD_MONITOR = _cfg.get(
    "table_gold_monitor", str,
    "ds_dev_db.dev_christian_van_bellen.monitoring_performance",
)

# ── Volume de landing ─────────────────────────────────────────────
LANDING_PATH = _cfg.get(
    "landing_path", str,
    "/Volumes/ds_dev_db/dev_christian_van_bellen/landing",
)

# ── MLflow ────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = _cfg.get(
    "mlflow_experiment", str,
    "/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr",
)

# ── Model Registry ────────────────────────────────────────────────
MODEL_NAME = _cfg.get(
    "model_name", str,
    "ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier",
)
RETRAIN_JOB_NAME = _cfg.get("retrain_job_name", str, "job_health_pressure_retrain")

# ── Features ─────────────────────────────────────────────────────
# Constantes do domínio do modelo — não configuráveis externamente
FEATURE_COLS = [
    "casos_por_leito",
    "casos_por_leito_lag1",
    "casos_por_leito_lag2",
    "casos_por_leito_lag3",
    "casos_por_leito_ma2",
    "casos_por_leito_ma3",
    "casos_srag_lag1",
    "casos_srag_lag2",
    "obitos_por_leito",
    "uti_por_leito_uti",
    "share_idosos",
    "growth_mom",
    "growth_3m",
    "acceleration",
    "rolling_std_3m",
    "leitos_totais",
    "leitos_uti",
    "num_hospitais",
    "mes",
    "quarter",
    "is_semester1",
    "is_rainy_season",
]
TARGET_COL = "target_alta_pressao"

# ── Split temporal ────────────────────────────────────────────────
TRAIN_END  = _cfg.get("train_end",  str, "202412")
VAL_END    = _cfg.get("val_end",    str, "202506")
TEST_START = _cfg.get("test_start", str, "202507")

# ── Thresholds operacionais ───────────────────────────────────────
TARGET_PERCENTILE     = _cfg.get("target_percentile",     float, 0.85)
PRECISION_K_THRESHOLD = _cfg.get("precision_k_threshold", float, 0.55)
MIN_CONSECUTIVE_BELOW = _cfg.get("min_consecutive_below", int,   2)
SCORING_MIN_QUALITY   = _cfg.get("scoring_min_quality",   float, 0.5)
AB_CHALLENGER_PCT     = _cfg.get("ab_challenger_pct",     float, 0.20)
DRIFT_SEASONAL_FEATURES = _cfg.get(
    "drift_seasonal_features", list,
    ["mes", "quarter", "is_semester1", "is_rainy_season"],
)

# ── Parâmetros LightGBM ───────────────────────────────────────────
LGBM_PARAMS = _cfg.get(
    "lgbm_params_json", dict,
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    },
)
NUM_BOOST_ROUND = _cfg.get("num_boost_round", int, 500)
EARLY_STOPPING  = _cfg.get("early_stopping",  int, 50)
