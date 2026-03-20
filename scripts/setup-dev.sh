#!/usr/bin/env bash
# scripts/setup-dev.sh
# Exemplo de como sobrescrever os defaults do cli.py via argumentos de linha de comando
# em desenvolvimento local.
#
# Os scripts src/ agora recebem configuração via argparse (src/cli.py).
# Os defaults já refletem o ambiente de desenvolvimento — não é necessário
# setar variáveis de ambiente para usar os scripts normalmente.
#
# Use este arquivo como referência dos argumentos disponíveis, ou copie
# os argumentos abaixo para a linha de comando ao executar um script.
#
# Equivalente PowerShell: scripts/setup-dev.ps1

# Exemplo: executar o scoring com configuração explícita
# python src/scoring/batch_score.py \
#   --catalog ds_dev_db \
#   --schema dev_christian_van_bellen \
#   --table_gold_features ds_dev_db.dev_christian_van_bellen.gold_pressure_features \
#   --table_gold_scoring ds_dev_db.dev_christian_van_bellen.gold_pressure_scoring \
#   --model_name ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier \
#   --mlflow_experiment "/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr" \
#   --scoring_min_quality 0.5 \
#   --ab_challenger_pct 0.20

# Defaults de referência (mesmos valores usados em src/cli.py):
# --catalog                  ds_dev_db
# --schema                   dev_christian_van_bellen
# --table_bronze_srag        ds_dev_db.dev_christian_van_bellen.bronze_srag
# --table_bronze_hospitais_leitos  ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos
# --table_bronze_cnes        ds_dev_db.dev_christian_van_bellen.bronze_cnes_estabelecimentos
# --table_silver_srag        ds_dev_db.dev_christian_van_bellen.silver_srag_municipio_semana
# --table_silver_capacity    ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_mes
# --table_gold_features      ds_dev_db.dev_christian_van_bellen.gold_pressure_features
# --table_gold_scoring       ds_dev_db.dev_christian_van_bellen.gold_pressure_scoring
# --table_gold_monitor       ds_dev_db.dev_christian_van_bellen.monitoring_performance
# --landing_path             /Volumes/ds_dev_db/dev_christian_van_bellen/landing
# --mlflow_experiment        /Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr
# --model_name               ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier
# --retrain_job_name         job_health_pressure_retrain
# --train_end                202412
# --val_end                  202506
# --test_start               202507
# --target_percentile        0.85
# --precision_k_threshold    0.55
# --min_consecutive_below    2
# --scoring_min_quality      0.5
# --ab_challenger_pct        0.20
# --drift_seasonal_features  mes,quarter,is_semester1,is_rainy_season
# --lgbm_params_json         '{"objective":"binary","metric":"binary_logloss",...}'
# --num_boost_round          500
# --early_stopping           50

echo "Referência de argumentos CLI disponível — ver src/cli.py para lista completa."
