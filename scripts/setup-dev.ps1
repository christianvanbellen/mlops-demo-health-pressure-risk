# scripts/setup-dev.ps1
# Sobrescreve as variáveis de ambiente de configuração para desenvolvimento local.
# Uso: . .\scripts\setup-dev.ps1  (dot-source para exportar para o shell corrente)
#
# Só é necessário quando você quiser sobrescrever os defaults do ConfigLoader
# (src/config.py). Em desenvolvimento normal os defaults já refletem o ambiente dev.
#
# Equivalente bash: scripts/setup-dev.sh

$env:CATALOG = "ds_dev_db"
$env:SCHEMA  = "dev_christian_van_bellen"

$env:TABLE_BRONZE_SRAG             = "ds_dev_db.dev_christian_van_bellen.bronze_srag"
$env:TABLE_BRONZE_HOSPITAIS_LEITOS = "ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos"
$env:TABLE_BRONZE_CNES             = "ds_dev_db.dev_christian_van_bellen.bronze_cnes_estabelecimentos"

$env:TABLE_SILVER_SRAG     = "ds_dev_db.dev_christian_van_bellen.silver_srag_municipio_semana"
$env:TABLE_SILVER_CAPACITY = "ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_mes"

$env:TABLE_GOLD_FEATURES = "ds_dev_db.dev_christian_van_bellen.gold_pressure_features"
$env:TABLE_GOLD_SCORING  = "ds_dev_db.dev_christian_van_bellen.gold_pressure_scoring"
$env:TABLE_GOLD_MONITOR  = "ds_dev_db.dev_christian_van_bellen.monitoring_performance"

$env:LANDING_PATH = "/Volumes/ds_dev_db/dev_christian_van_bellen/landing"

$env:MLFLOW_EXPERIMENT = "/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr"

$env:MODEL_NAME       = "ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier"
$env:RETRAIN_JOB_NAME = "job_health_pressure_retrain"

$env:TRAIN_END  = "202412"
$env:VAL_END    = "202506"
$env:TEST_START = "202507"

$env:TARGET_PERCENTILE      = "0.85"
$env:PRECISION_K_THRESHOLD  = "0.55"
$env:MIN_CONSECUTIVE_BELOW  = "2"
$env:SCORING_MIN_QUALITY    = "0.5"
$env:AB_CHALLENGER_PCT      = "0.20"
$env:DRIFT_SEASONAL_FEATURES = "mes,quarter,is_semester1,is_rainy_season"

$env:LGBM_PARAMS_JSON = '{"objective":"binary","metric":"binary_logloss","boosting_type":"gbdt","num_leaves":63,"learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"min_child_samples":20,"reg_alpha":0.1,"reg_lambda":0.1,"verbose":-1}'
$env:NUM_BOOST_ROUND  = "500"
$env:EARLY_STOPPING   = "50"

Write-Host "Variaveis de ambiente de desenvolvimento configuradas."
