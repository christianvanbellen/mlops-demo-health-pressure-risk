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
    "mlflow.spark",
    # DQX — não disponível fora do cluster Databricks
    "databricks.labs.dqx",
    "databricks.labs.dqx.engine",
    "databricks.labs.dqx.check_funcs",
    "databricks.labs.dqx.rule",
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
# src/cli.py define os defaults de desenvolvimento como valores default do argparse —
# não é necessário setar variáveis de ambiente para a suíte de testes unitários.

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
