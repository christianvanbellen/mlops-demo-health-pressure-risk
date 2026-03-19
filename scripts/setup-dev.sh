#!/usr/bin/env bash
# scripts/setup-dev.sh
# Define as variáveis de ambiente necessárias para rodar os módulos src/ localmente.
# Uso: source scripts/setup-dev.sh
#
# Equivalente PowerShell: scripts/setup-dev.ps1

export CATALOG="ds_dev_db"
export SCHEMA="dev_christian_van_bellen"

export TABLE_BRONZE_SRAG="ds_dev_db.dev_christian_van_bellen.bronze_srag"
export TABLE_BRONZE_HOSPITAIS_LEITOS="ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos"
export TABLE_BRONZE_CNES="ds_dev_db.dev_christian_van_bellen.bronze_cnes_estabelecimentos"

export TABLE_SILVER_SRAG="ds_dev_db.dev_christian_van_bellen.silver_srag_municipio_semana"
export TABLE_SILVER_CAPACITY="ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_mes"

export TABLE_GOLD_FEATURES="ds_dev_db.dev_christian_van_bellen.gold_pressure_features"
export TABLE_GOLD_SCORING="ds_dev_db.dev_christian_van_bellen.gold_pressure_scoring"
export TABLE_GOLD_MONITOR="ds_dev_db.dev_christian_van_bellen.monitoring_performance"

export LANDING_PATH="/Volumes/ds_dev_db/dev_christian_van_bellen/landing"

export MLFLOW_EXPERIMENT="/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr"

export MODEL_NAME="ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier"
export RETRAIN_JOB_NAME="job_health_pressure_retrain"

export TRAIN_END="202412"
export VAL_END="202506"
export TEST_START="202507"

export TARGET_PERCENTILE="0.85"
export PRECISION_K_THRESHOLD="0.55"
export MIN_CONSECUTIVE_BELOW="2"
export SCORING_MIN_QUALITY="0.5"
export AB_CHALLENGER_PCT="0.20"
export DRIFT_SEASONAL_FEATURES="mes,quarter,is_semester1,is_rainy_season"

export LGBM_PARAMS_JSON='{"objective":"binary","metric":"binary_logloss","boosting_type":"gbdt","num_leaves":63,"learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"min_child_samples":20,"reg_alpha":0.1,"reg_lambda":0.1,"verbose":-1}'
export NUM_BOOST_ROUND="500"
export EARLY_STOPPING="50"

echo "✓ Variáveis de ambiente de desenvolvimento configuradas."
