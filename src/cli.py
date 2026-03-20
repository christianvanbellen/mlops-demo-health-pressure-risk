# src/cli.py
# Parser centralizado de argumentos de linha de comando.
#
# Uso nos scripts:
#   from cli import build_parser, FEATURE_COLS, TARGET_COL
#
#   p = build_parser("Descrição do script")
#   p.add_argument("--live", action="store_true", default=False)  # flags específicas do script
#   args, _ = p.parse_known_args()
#
# Todos os defaults replicam exatamente os valores de desenvolvimento anteriormente
# definidos em src/config.py. O build_parser() retorna um ArgumentParser pronto para
# uso com parse_known_args() — nunca use parse_args() para manter compatibilidade
# com flags adicionais do Databricks.

import argparse

# ── Constantes de domínio — não configuráveis via CLI ────────────────────────
# Mantidas como constantes de módulo porque não dependem de ambiente e não
# devem ser sobrescritas por operadores. São parte da definição do modelo.
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


# ── Parser compartilhado ──────────────────────────────────────────────────────
def build_parser(description: str = "") -> argparse.ArgumentParser:
    """
    Retorna um ArgumentParser com todos os argumentos configuráveis do projeto.

    Use parse_known_args() nos scripts (nunca parse_args()) para compatibilidade
    com argumentos extras injetados pelo runtime do Databricks.

    Flags específicas de script (--live, --ab, --dry-run) devem ser adicionadas
    pelo script após chamar build_parser(), antes de parse_known_args().

    Os defaults refletem o ambiente de desenvolvimento (ds_dev_db / dev_christian_van_bellen).
    Em produção, todos os valores são passados via parâmetros do job Databricks.
    """
    p = argparse.ArgumentParser(description=description)

    # ── Ambiente ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--catalog",
        type=str,
        default="ds_dev_db",
        help="Catálogo Unity Catalog (default: ds_dev_db)",
    )
    p.add_argument(
        "--schema",
        type=str,
        default="dev_christian_van_bellen",
        help="Schema Unity Catalog (default: dev_christian_van_bellen)",
    )

    # ── Tabelas Bronze ────────────────────────────────────────────────────────
    p.add_argument(
        "--table_bronze_srag",
        type=str,
        default="ds_dev_db.dev_christian_van_bellen.bronze_srag",
    )
    p.add_argument(
        "--table_bronze_hospitais_leitos",
        type=str,
        default="ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos",
    )
    p.add_argument(
        "--table_bronze_cnes",
        type=str,
        default="ds_dev_db.dev_christian_van_bellen.bronze_cnes_estabelecimentos",
    )

    # ── Tabelas Silver ────────────────────────────────────────────────────────
    p.add_argument(
        "--table_silver_srag",
        type=str,
        default="ds_dev_db.dev_christian_van_bellen.silver_srag_municipio_semana",
    )
    p.add_argument(
        "--table_silver_capacity",
        type=str,
        default="ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_mes",
    )

    # ── Tabelas Gold ──────────────────────────────────────────────────────────
    p.add_argument(
        "--table_gold_features",
        type=str,
        default="ds_dev_db.dev_christian_van_bellen.gold_pressure_features",
    )
    p.add_argument(
        "--table_gold_scoring",
        type=str,
        default="ds_dev_db.dev_christian_van_bellen.gold_pressure_scoring",
    )
    p.add_argument(
        "--table_gold_monitor",
        type=str,
        default="ds_dev_db.dev_christian_van_bellen.monitoring_performance",
    )

    # ── Volume de landing ─────────────────────────────────────────────────────
    p.add_argument(
        "--landing_path",
        type=str,
        default="/Volumes/ds_dev_db/dev_christian_van_bellen/landing",
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mlflow_experiment",
        type=str,
        default="/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr",
    )

    # ── Model Registry ────────────────────────────────────────────────────────
    p.add_argument(
        "--model_name",
        type=str,
        default="ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier",
    )
    p.add_argument(
        "--retrain_job_name",
        type=str,
        default="job_health_pressure_retrain",
    )

    # ── Split temporal ────────────────────────────────────────────────────────
    p.add_argument(
        "--train_end",
        type=str,
        default="202412",
        help="Competência de fim do período de treino (AAAAMM, inclusive)",
    )
    p.add_argument(
        "--val_end",
        type=str,
        default="202506",
        help="Competência de fim do período de validação (AAAAMM, inclusive)",
    )
    p.add_argument(
        "--test_start",
        type=str,
        default="202507",
        help="Competência de início do período de teste (AAAAMM, inclusive)",
    )

    # ── Thresholds operacionais ───────────────────────────────────────────────
    p.add_argument(
        "--target_percentile",
        type=float,
        default=0.85,
        help="Percentil do target de alta pressão (default: 0.85 = p85 nacional)",
    )
    p.add_argument(
        "--precision_k_threshold",
        type=float,
        default=0.55,
        help="Threshold de Precision@K para trigger de retraining (default: 0.55)",
    )
    p.add_argument(
        "--min_consecutive_below",
        type=int,
        default=2,
        help="Mínimo de competências consecutivas abaixo do threshold para trigger (default: 2)",
    )
    p.add_argument(
        "--scoring_min_quality",
        type=float,
        default=0.5,
        help="Score mínimo de qualidade dos dados para scoring (default: 0.5)",
    )
    p.add_argument(
        "--ab_challenger_pct",
        type=float,
        default=0.20,
        help="Proporção de municípios roteados para o challenger no A/B test (default: 0.20)",
    )

    # ── Features sazonais para drift ─────────────────────────────────────────
    # Passado como string CSV e convertido com: [s.strip() for s in args.drift_seasonal_features.split(",")]
    p.add_argument(
        "--drift_seasonal_features",
        type=str,
        default="mes,quarter,is_semester1,is_rainy_season",
        help="Features sazonais excluídas do drift_share (CSV, default: mes,quarter,is_semester1,is_rainy_season)",
    )

    # ── Parâmetros LightGBM ───────────────────────────────────────────────────
    # Passado como JSON string e convertido com: json.loads(args.lgbm_params_json)
    p.add_argument(
        "--lgbm_params_json",
        type=str,
        default=(
            '{"objective":"binary","metric":"binary_logloss","boosting_type":"gbdt",'
            '"num_leaves":63,"learning_rate":0.05,"feature_fraction":0.8,'
            '"bagging_fraction":0.8,"bagging_freq":5,"min_child_samples":20,'
            '"reg_alpha":0.1,"reg_lambda":0.1,"verbose":-1}'
        ),
        help="Parâmetros LightGBM em formato JSON (string)",
    )
    p.add_argument(
        "--num_boost_round",
        type=int,
        default=500,
        help="Número máximo de iterações LightGBM (default: 500)",
    )
    p.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Rounds sem melhoria para early stopping LightGBM (default: 50)",
    )

    # ── Run IDs para evaluate.py ──────────────────────────────────────────────
    p.add_argument(
        "--lr_run_id",
        type=str,
        default="",
        help="MLflow run_id do modelo Logistic Regression a avaliar",
    )
    p.add_argument(
        "--lgbm_run_id",
        type=str,
        default="",
        help="MLflow run_id do modelo LightGBM a avaliar",
    )

    return p
