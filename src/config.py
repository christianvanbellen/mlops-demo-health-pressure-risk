"""
Módulo central de configuração do projeto.
Lê conf/{env}.yml e expõe todas as constantes para os demais módulos importarem.
Uso: from config import CATALOG, TABLE_BRONZE_SRAG, ...
"""

import os
from pathlib import Path

import yaml


def _load_config() -> dict:
    env = os.environ.get("APP_ENV", "dev")
    conf_dir = Path(__file__).parent.parent / "conf"
    path = conf_dir / f"{env}.yml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Set APP_ENV to a valid environment (dev, prod)."
        )
    with open(path) as f:
        return yaml.safe_load(f)


_cfg = _load_config()

# ── Ambiente ─────────────────────────────────────────────────────
CATALOG = _cfg["catalog"]
SCHEMA = _cfg["schema"]

# ── Tabelas Bronze ────────────────────────────────────────────────
TABLE_BRONZE_SRAG = _cfg["tables"]["bronze_srag"]
TABLE_BRONZE_HOSPITAIS_LEITOS = _cfg["tables"]["bronze_hospitais_leitos"]
TABLE_BRONZE_CNES = _cfg["tables"]["bronze_cnes"]

# ── Tabelas Silver ────────────────────────────────────────────────
TABLE_SILVER_SRAG = _cfg["tables"]["silver_srag"]
TABLE_SILVER_CAPACITY = _cfg["tables"]["silver_capacity"]

# ── Tabelas Gold ──────────────────────────────────────────────────
TABLE_GOLD_FEATURES = _cfg["tables"]["gold_features"]
TABLE_GOLD_SCORING = _cfg["tables"]["gold_scoring"]
TABLE_GOLD_MONITOR = _cfg["tables"]["gold_monitoring"]

# ── Volume de landing ─────────────────────────────────────────────
LANDING_PATH = _cfg.get("landing_path", f"/Volumes/{CATALOG}/{SCHEMA}/landing")

# ── MLflow ────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = _cfg["mlflow"]["experiment_path"]

# ── Model Registry ────────────────────────────────────────────────
MODEL_NAME = _cfg["model"]["name"]
RETRAIN_JOB_NAME = _cfg["model"]["retrain_job_name"]

# ── Features ─────────────────────────────────────────────────────
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
TRAIN_END = _cfg["training"]["train_end"]
VAL_END = _cfg["training"]["val_end"]
TEST_START = _cfg["training"]["test_start"]

# ── Thresholds operacionais ───────────────────────────────────────
TARGET_PERCENTILE = _cfg["thresholds"]["target_percentile"]
PRECISION_K_THRESHOLD = _cfg["thresholds"]["precision_k_threshold"]
MIN_CONSECUTIVE_BELOW = _cfg["thresholds"]["min_consecutive_below"]
SCORING_MIN_QUALITY = _cfg["thresholds"]["scoring_min_quality"]
AB_CHALLENGER_PCT = _cfg["thresholds"]["ab_challenger_pct"]
DRIFT_SEASONAL_FEATURES = _cfg["thresholds"].get(
    "drift_seasonal_features", ["mes", "quarter", "is_semester1", "is_rainy_season"]
)

# ── Parâmetros LightGBM ───────────────────────────────────────────
LGBM_PARAMS = _cfg["lgbm"]["params"]
NUM_BOOST_ROUND = _cfg["lgbm"]["num_boost_round"]
EARLY_STOPPING = _cfg["lgbm"]["early_stopping_rounds"]
