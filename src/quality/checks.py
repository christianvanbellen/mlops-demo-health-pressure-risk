# src/quality/checks.py
# Suítes de checks DQX por camada da Medallion.
#
# Cada função retorna uma lista de dicionários no formato esperado pelo
# DQEngine.apply_checks_by_metadata_and_split():
#   [{"criticality": "error"|"warn", "check": <DQColRule|DQRowRule>}, ...]
#
# criticality="error"  → linha vai para tabela de quarentena (não gravada)
# criticality="warn"   → linha é gravada mas recebe flag de alerta
#
# Uso:
#   from quality.checks import checks_bronze_srag
#   checks = checks_bronze_srag()

from databricks.labs.dqx.col_functions import (
    is_not_null,
    is_not_null_and_not_empty,
    is_valid_date,
    value_is_in_list,
)
from databricks.labs.dqx.engine import DQColRule, DQRowRule


def checks_bronze_srag() -> list[dict]:
    """Checks para bronze_srag (SRAG/SIVEP-Gripe)."""
    return [
        # campos obrigatórios de chave
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="CO_MUN_RES",
                function=is_not_null_and_not_empty,
                name="co_mun_res_nao_nulo",
            ),
        },
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="DT_NOTIFIC",
                function=is_not_null_and_not_empty,
                name="dt_notific_nao_nulo",
            ),
        },
        # data de primeira sintoma válida (pode ser nula, mas se presente deve ser data)
        {
            "criticality": "warn",
            "check": DQColRule(
                col_name="DT_SIN_PRI",
                function=is_valid_date,
                name="dt_sin_pri_formato_valido",
            ),
        },
        # metadados de ingestão obrigatórios
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="_snapshot_date",
                function=is_not_null,
                name="snapshot_date_nao_nulo",
            ),
        },
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="_ano_arquivo",
                function=is_not_null,
                name="ano_arquivo_nao_nulo",
            ),
        },
    ]


def checks_bronze_hospitais_leitos() -> list[dict]:
    """Checks para bronze_hospitais_leitos (Hospitais e Leitos / CNES)."""
    return [
        # campos de chave
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="COMP",
                function=is_not_null_and_not_empty,
                name="comp_nao_nulo",
            ),
        },
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="CNES",
                function=is_not_null_and_not_empty,
                name="cnes_nao_nulo",
            ),
        },
        # UF deve estar presente (usada no lookup município)
        {
            "criticality": "warn",
            "check": DQColRule(
                col_name="UF",
                function=is_not_null_and_not_empty,
                name="uf_nao_nulo",
            ),
        },
        # metadados de ingestão obrigatórios
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="_snapshot_date",
                function=is_not_null,
                name="snapshot_date_nao_nulo",
            ),
        },
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="_ano_arquivo",
                function=is_not_null,
                name="ano_arquivo_nao_nulo",
            ),
        },
    ]


def checks_silver_srag() -> list[dict]:
    """Checks para silver_srag_municipio_semana."""
    return [
        # PK: municipio_id × semana_epidemiologica
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="municipio_id",
                function=is_not_null_and_not_empty,
                name="municipio_id_nao_nulo",
            ),
        },
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="semana_epidemiologica",
                function=is_not_null_and_not_empty,
                name="semana_epidemiologica_nao_nula",
            ),
        },
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="competencia",
                function=is_not_null_and_not_empty,
                name="competencia_nao_nula",
            ),
        },
        # consolidation_flag deve ter valor esperado
        {
            "criticality": "warn",
            "check": DQColRule(
                col_name="srag_consolidation_flag",
                function=lambda c: value_is_in_list(c, ["consolidado", "estabilizando", "recente"]),
                name="consolidation_flag_valido",
            ),
        },
    ]


def checks_silver_capacity() -> list[dict]:
    """Checks para silver_capacity_municipio_mes."""
    return [
        # PK: municipio_id × competencia
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="municipio_id",
                function=is_not_null_and_not_empty,
                name="municipio_id_nao_nulo",
            ),
        },
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="competencia",
                function=is_not_null_and_not_empty,
                name="competencia_nao_nula",
            ),
        },
        # leitos_totais não pode ser negativo
        {
            "criticality": "error",
            "check": DQRowRule(
                function=lambda row: row["leitos_totais"] >= 0,
                name="leitos_totais_nao_negativo",
            ),
        },
        # capacidade_is_forward_fill deve ser booleano (não nulo)
        {
            "criticality": "warn",
            "check": DQColRule(
                col_name="capacity_is_forward_fill",
                function=is_not_null,
                name="capacity_is_forward_fill_nao_nulo",
            ),
        },
    ]


def checks_gold_features() -> list[dict]:
    """Checks para gold_pressure_features."""
    return [
        # PK: municipio_id × competencia
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="municipio_id",
                function=is_not_null_and_not_empty,
                name="municipio_id_nao_nulo",
            ),
        },
        {
            "criticality": "error",
            "check": DQColRule(
                col_name="competencia",
                function=is_not_null_and_not_empty,
                name="competencia_nao_nula",
            ),
        },
        # feature principal não pode ser negativa
        {
            "criticality": "error",
            "check": DQRowRule(
                function=lambda row: row["casos_por_leito"] >= 0,
                name="casos_por_leito_nao_negativo",
            ),
        },
        # data_quality_score deve estar no range esperado [0, 1]
        {
            "criticality": "warn",
            "check": DQRowRule(
                function=lambda row: 0.0 <= row["data_quality_score"] <= 1.0,
                name="data_quality_score_range_valido",
            ),
        },
        # leitos_totais >= 10 (filtro aplicado upstream — confirma invariante)
        {
            "criticality": "warn",
            "check": DQRowRule(
                function=lambda row: row["leitos_totais"] >= 10,
                name="leitos_totais_filtro_minimo",
            ),
        },
    ]
