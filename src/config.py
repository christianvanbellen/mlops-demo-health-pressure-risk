"""
Módulo central de configuração do projeto.
Lê variáveis de ambiente e expõe todas as constantes para os demais módulos importarem.
Uso: from config import CATALOG, TABLE_BRONZE_SRAG, ...

Em desenvolvimento, defina as variáveis via scripts/setup-dev.sh (bash) ou
scripts/setup-dev.ps1 (PowerShell). Em produção, o DAB injeta automaticamente
via environment_variables nos jobs (resources/jobs.yml).
"""

import json
import os

# ── Ambiente ─────────────────────────────────────────────────────
CATALOG = os.environ["CATALOG"]
SCHEMA = os.environ["SCHEMA"]

# ── Tabelas Bronze ────────────────────────────────────────────────
TABLE_BRONZE_SRAG = os.environ["TABLE_BRONZE_SRAG"]
TABLE_BRONZE_HOSPITAIS_LEITOS = os.environ["TABLE_BRONZE_HOSPITAIS_LEITOS"]
TABLE_BRONZE_CNES = os.environ["TABLE_BRONZE_CNES"]

# ── Tabelas Silver ────────────────────────────────────────────────
TABLE_SILVER_SRAG = os.environ["TABLE_SILVER_SRAG"]
TABLE_SILVER_CAPACITY = os.environ["TABLE_SILVER_CAPACITY"]

# ── Tabelas Gold ──────────────────────────────────────────────────
TABLE_GOLD_FEATURES = os.environ["TABLE_GOLD_FEATURES"]
TABLE_GOLD_SCORING = os.environ["TABLE_GOLD_SCORING"]
TABLE_GOLD_MONITOR = os.environ["TABLE_GOLD_MONITOR"]

# ── Volume de landing ─────────────────────────────────────────────
LANDING_PATH = os.environ["LANDING_PATH"]

# ── MLflow ────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = os.environ["MLFLOW_EXPERIMENT"]

# ── Model Registry ────────────────────────────────────────────────
MODEL_NAME = os.environ["MODEL_NAME"]
RETRAIN_JOB_NAME = os.environ["RETRAIN_JOB_NAME"]

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
TRAIN_END = os.environ["TRAIN_END"]
VAL_END = os.environ["VAL_END"]
TEST_START = os.environ["TEST_START"]

# ── Thresholds operacionais ───────────────────────────────────────
TARGET_PERCENTILE = float(os.environ["TARGET_PERCENTILE"])
PRECISION_K_THRESHOLD = float(os.environ["PRECISION_K_THRESHOLD"])
MIN_CONSECUTIVE_BELOW = int(os.environ["MIN_CONSECUTIVE_BELOW"])
SCORING_MIN_QUALITY = float(os.environ["SCORING_MIN_QUALITY"])
AB_CHALLENGER_PCT = float(os.environ["AB_CHALLENGER_PCT"])
DRIFT_SEASONAL_FEATURES = os.environ["DRIFT_SEASONAL_FEATURES"].split(",")

# ── Parâmetros LightGBM ───────────────────────────────────────────
LGBM_PARAMS = json.loads(os.environ["LGBM_PARAMS_JSON"])
NUM_BOOST_ROUND = int(os.environ["NUM_BOOST_ROUND"])
EARLY_STOPPING = int(os.environ["EARLY_STOPPING"])
