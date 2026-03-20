# src/quality/checks.py
# Suítes de checks DQX por camada da Medallion.
#
# Cada função retorna uma lista de DQRowRule prontos para serem passados a
# DQEngineCore.apply_checks_and_split().
#
# criticality="error"  → linha vai para tabela de quarentena (não gravada)
# criticality="warn"   → linha é gravada mas recebe flag de alerta
#
# Uso:
#   from quality.checks import checks_bronze_srag
#   checks = checks_bronze_srag()

from databricks.labs.dqx.check_funcs import (
    is_in_list,
    is_not_null,
    is_not_null_and_not_empty,
    is_valid_date,
)
from databricks.labs.dqx.rule import DQRowRule


def checks_bronze_srag() -> list[DQRowRule]:
    """Checks para bronze_srag (SRAG/SIVEP-Gripe)."""
    return [
        # campos obrigatórios de chave
        DQRowRule(
            column="CO_MUN_RES",
            check_func=is_not_null_and_not_empty,
            name="co_mun_res_nao_nulo",
            criticality="error",
        ),
        DQRowRule(
            column="DT_NOTIFIC",
            check_func=is_not_null_and_not_empty,
            name="dt_notific_nao_nulo",
            criticality="error",
        ),
        # data de primeira sintoma válida (pode ser nula, mas se presente deve ser data)
        DQRowRule(
            column="DT_SIN_PRI",
            check_func=is_valid_date,
            name="dt_sin_pri_formato_valido",
            criticality="warn",
        ),
        # metadados de ingestão obrigatórios
        DQRowRule(
            column="_snapshot_date",
            check_func=is_not_null,
            name="snapshot_date_nao_nulo",
            criticality="error",
        ),
        DQRowRule(
            column="_ano_arquivo",
            check_func=is_not_null,
            name="ano_arquivo_nao_nulo",
            criticality="error",
        ),
    ]


def checks_bronze_hospitais_leitos() -> list[DQRowRule]:
    """Checks para bronze_hospitais_leitos (Hospitais e Leitos / CNES)."""
    return [
        # campos de chave
        DQRowRule(
            column="COMP",
            check_func=is_not_null_and_not_empty,
            name="comp_nao_nulo",
            criticality="error",
        ),
        DQRowRule(
            column="CNES",
            check_func=is_not_null_and_not_empty,
            name="cnes_nao_nulo",
            criticality="error",
        ),
        # UF deve estar presente (usada no lookup município)
        DQRowRule(
            column="UF",
            check_func=is_not_null_and_not_empty,
            name="uf_nao_nulo",
            criticality="warn",
        ),
        # metadados de ingestão obrigatórios
        DQRowRule(
            column="_snapshot_date",
            check_func=is_not_null,
            name="snapshot_date_nao_nulo",
            criticality="error",
        ),
        DQRowRule(
            column="_ano_arquivo",
            check_func=is_not_null,
            name="ano_arquivo_nao_nulo",
            criticality="error",
        ),
    ]


def checks_silver_srag() -> list[DQRowRule]:
    """Checks para silver_srag_municipio_semana."""
    return [
        # PK: municipio_id × semana_epidemiologica
        DQRowRule(
            column="municipio_id",
            check_func=is_not_null_and_not_empty,
            name="municipio_id_nao_nulo",
            criticality="error",
        ),
        DQRowRule(
            column="semana_epidemiologica",
            check_func=is_not_null_and_not_empty,
            name="semana_epidemiologica_nao_nula",
            criticality="error",
        ),
        DQRowRule(
            column="competencia",
            check_func=is_not_null_and_not_empty,
            name="competencia_nao_nula",
            criticality="error",
        ),
        # consolidation_flag deve ter valor esperado
        DQRowRule(
            column="srag_consolidation_flag",
            check_func=is_in_list,
            check_func_args=[["consolidado", "estabilizando", "recente"]],
            name="consolidation_flag_valido",
            criticality="warn",
        ),
    ]


def checks_silver_capacity() -> list[DQRowRule]:
    """Checks para silver_capacity_municipio_mes."""
    return [
        # PK: municipio_id × competencia
        DQRowRule(
            column="municipio_id",
            check_func=is_not_null_and_not_empty,
            name="municipio_id_nao_nulo",
            criticality="error",
        ),
        DQRowRule(
            column="competencia",
            check_func=is_not_null_and_not_empty,
            name="competencia_nao_nula",
            criticality="error",
        ),
        # leitos_totais não pode ser negativo
        DQRowRule(
            check_func=lambda row: row["leitos_totais"] >= 0,
            name="leitos_totais_nao_negativo",
            criticality="error",
        ),
        # capacidade_is_forward_fill deve ser booleano (não nulo)
        DQRowRule(
            column="capacity_is_forward_fill",
            check_func=is_not_null,
            name="capacity_is_forward_fill_nao_nulo",
            criticality="warn",
        ),
    ]


def checks_gold_features() -> list[DQRowRule]:
    """Checks para gold_pressure_features."""
    return [
        # PK: municipio_id × competencia
        DQRowRule(
            column="municipio_id",
            check_func=is_not_null_and_not_empty,
            name="municipio_id_nao_nulo",
            criticality="error",
        ),
        DQRowRule(
            column="competencia",
            check_func=is_not_null_and_not_empty,
            name="competencia_nao_nula",
            criticality="error",
        ),
        # feature principal não pode ser negativa
        DQRowRule(
            check_func=lambda row: row["casos_por_leito"] >= 0,
            name="casos_por_leito_nao_negativo",
            criticality="error",
        ),
        # data_quality_score deve estar no range esperado [0, 1]
        DQRowRule(
            check_func=lambda row: 0.0 <= row["data_quality_score"] <= 1.0,
            name="data_quality_score_range_valido",
            criticality="warn",
        ),
        # leitos_totais >= 10 (filtro aplicado upstream — confirma invariante)
        DQRowRule(
            check_func=lambda row: row["leitos_totais"] >= 10,
            name="leitos_totais_filtro_minimo",
            criticality="warn",
        ),
    ]
