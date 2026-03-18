# src/transforms/gold_pressure_features.py
# Transform Silver → Gold — Feature store de pressão assistencial (v2)
#
# Fontes:
#   silver_srag_municipio_semana    → demanda (casos SRAG hospitalizados)
#   silver_capacity_municipio_mes   → oferta (leitos e estabelecimentos)
# Destino:
#   gold_pressure_features
#
# Grain de saída: municipio_id × competencia (AAAAMM) — mensal
#
# Por que grain mensal (e não semanal):
#   A granularidade semanal amplifica o atraso de notificação do SRAG —
#   as últimas 2-4 semanas de cada ano estão cronicamente incompletas.
#   Agregar por mês suaviza esse ruído e alinha naturalmente com a fonte
#   de capacidade (CNES / Hospitais e Leitos), que é publicada mensalmente.
#
# Join: Capacity (left) × SRAG — todos os municípios com hospital aparecem
#   Municípios sem casos SRAG no mês ficam com campos de demanda zerados.
#   Isso preserva os ~3.330 municípios com leitos_totais >= 10 em toda
#   competência disponível na Capacity, inclusive para scoring ao vivo.
#
# Forward fill de capacity:
#   A fonte Hospitais e Leitos tem atraso típico de 1-2 meses.
#   Competências presentes na SRAG mas ausentes na Capacity recebem
#   os dados do último mês disponível (forward fill).
#   capacity_is_forward_fill = True indica meses com capacity estimada.
#
# Target v2 (target_definition_version = "v2"):
#   target_alta_pressao = 1  se  casos_por_leito(t+1) >= percentil_85_nacional(t+1)
#   Métrica alvo: casos_por_leito = casos_srag_mes / (leitos_totais + 1)
#   O percentil é calculado sobre todos os municípios da mesma competência (t+1).
#
#   Raciocínio da métrica relativa:
#     São Paulo com 37.000 leitos e 8.000 casos não é alerta.
#     Um município com 10 leitos e 17 casos é crise.
#     Pressão relativa à capacidade captura isso; volume absoluto não.
#
#   Semanas sem t+1 (último mês observado) → target = null.
#   Essas linhas são MANTIDAS para scoring ao vivo.
#
# Filtro de qualidade:
#   leitos_totais >= 10 — remove municípios sem hospital (= 0) e prováveis
#   erros cadastrais (1-9 leitos), que distorceriam o denominador e o p85.
#
# Consolidação do SRAG (srag_consolidation_flag):
#   O SRAG tem atraso estrutural de notificação. A flag classifica cada
#   competência por quantos dias se passaram desde o fechamento do mês:
#     >= 90 dias: "consolidado"   → ~95%+ dos casos notificados
#     >= 45 dias: "estabilizando" → ~80-90% dos casos notificados
#      < 45 dias: "recente"       → < 80% — excluído do cálculo do target
#
# Qualidade dos dados (data_quality_score — 0 a 1):
#   Combina capacity_is_forward_fill e srag_consolidation_flag:
#     forward fill ativo           → 0.3
#     SRAG recente                 → 0.5
#     SRAG estabilizando           → 0.8
#     capacity real + consolidado  → 1.0
#
# Colunas de governança:
#   target_definition_version    = "v2"
#   target_metric                = "casos_por_leito"
#   target_percentile            = 0.85
#   capacity_is_forward_fill     = True/False
#   srag_consolidation_flag      = "consolidado" | "estabilizando" | "recente"
#   data_quality_score           = 0.3 | 0.5 | 0.8 | 1.0

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# ── configuração ────────────────────────────────────────────────
CATALOG = "ds_dev_db"
SCHEMA = "dev_christian_van_bellen"

TABLE_SRC_SRAG = f"{CATALOG}.{SCHEMA}.silver_srag_municipio_semana"
TABLE_SRC_CAP = f"{CATALOG}.{SCHEMA}.silver_capacity_municipio_mes"
TABLE_DST = f"{CATALOG}.{SCHEMA}.gold_pressure_features"

# Window padrão do pipeline — reusado por _lags, _dinamica e _target
MUNICIPIO_W = Window.partitionBy("municipio_id").orderBy("competencia")
MUNICIPIO_W2 = MUNICIPIO_W.rowsBetween(-2, -1)  # 2 meses anteriores
MUNICIPIO_W3 = MUNICIPIO_W.rowsBetween(-3, -1)  # 3 meses anteriores


# ── funções ─────────────────────────────────────────────────────
def _agregar_srag_mensal(srag):
    """
    Agrega silver_srag_municipio_semana para grain mensal (municipio_id × competencia).
    A silver já contém a coluna competencia derivada da semana epidemiológica.
    """
    # uf fora do groupBy: alguns municípios têm casos notificados em UFs diferentes
    # da residência (ex: Brasília aparece com BA, GO e DF), o que geraria múltiplas
    # linhas por municipio_id × competencia e quebraria a primary key da feature table.
    return srag.groupBy("municipio_id", "competencia").agg(
        F.first("uf", ignorenulls=True).alias("uf"),
        F.sum("casos_srag").alias("casos_srag_mes"),
        F.sum("casos_obito").alias("casos_obito_mes"),
        F.sum("casos_uti").alias("casos_uti_mes"),
        F.sum("casos_idosos").alias("casos_idosos_mes"),
        F.sum("casos_pediatricos").alias("casos_pediatricos_mes"),
        F.count("*").alias("semanas_com_dado"),
    )


def _forward_fill_capacity(cap, srag_mensal):
    """
    Expande a capacity para cobrir competências presentes na SRAG mas ausentes
    na Capacity (atraso típico de 1-2 meses na publicação da fonte).

    Estratégia:
      1. Identifica o max(competencia) disponível na cap.
      2. Encontra competências da SRAG que excedem esse máximo.
      3. Para cada competência futura, duplica os registros do mês mais recente
         da cap, substituindo o campo competencia pelo mês futuro.
      4. Une as duas partes com capacity_is_forward_fill como marcador.
    """
    max_comp_cap = cap.agg(F.max("competencia").alias("max_comp")).collect()[0]["max_comp"]
    max_comp_srag = srag_mensal.agg(F.max("competencia").alias("max_comp")).collect()[0]["max_comp"]

    # competências da SRAG sem cobertura na capacity
    comps_futuras = [
        row["competencia"]
        for row in srag_mensal.select("competencia").distinct().collect()
        if row["competencia"] > max_comp_cap
    ]

    print("\n── Forward fill de capacity ──")
    print(f"  Max competência na capacity : {max_comp_cap}")
    print(f"  Max competência na SRAG     : {max_comp_srag}")

    # marca competências reais como não-forward-fill
    cap_real = cap.withColumn("capacity_is_forward_fill", F.lit(False))

    if not comps_futuras:
        print("  Nenhuma competência futura — forward fill não necessário.")
        return cap_real

    print(
        f"  Competências sem capacity (usarão forward fill de {max_comp_cap}): {sorted(comps_futuras)}"
    )

    # snapshot do último mês disponível — base do forward fill
    cap_ultimo = cap.filter(F.col("competencia") == max_comp_cap)

    # gera uma cópia por competência futura, substituindo o campo competencia
    from functools import reduce

    from pyspark.sql import DataFrame as SparkDF

    partes_ff = [
        cap_ultimo.withColumn("competencia", F.lit(comp)).withColumn(
            "capacity_is_forward_fill", F.lit(True)
        )
        for comp in comps_futuras
    ]

    cap_ff = reduce(SparkDF.unionByName, partes_ff)
    cap_expandida = cap_real.unionByName(cap_ff)

    print(f"  ✓ Forward fill aplicado para {len(comps_futuras)} competência(s).")
    return cap_expandida


def _join_capacity(srag_mensal, cap):
    """
    LEFT JOIN de cap (base) com srag_mensal — todos os municípios com hospital
    aparecem no Gold em toda competência disponível na Capacity.
    Municípios sem casos SRAG no mês ficam com campos de demanda zerados.
    Aplica forward fill na capacity antes do join para cobrir competências
    presentes na SRAG mas ausentes na fonte de capacidade.
    Filtra leitos_totais >= 10 (remove municípios sem hospital e erros cadastrais).
    """
    cap_expandida = _forward_fill_capacity(cap, srag_mensal)

    cap_sel = cap_expandida.select(
        "municipio_id",
        "competencia",
        "municipio_nome",
        "regiao",
        "leitos_totais",
        "leitos_sus",
        "leitos_uti",
        "leitos_uti_adulto",
        "leitos_uti_pediatrico",
        "leitos_uti_neonatal",
        "num_estabelecimentos",
        "num_hospitais",
        "capacity_is_forward_fill",
    )

    # cap LEFT JOIN srag — inclui todos os municípios com hospital
    df = cap_sel.join(srag_mensal, on=["municipio_id", "competencia"], how="left")

    # campos de demanda nulos (município sem casos SRAG no mês) → 0
    campos_demanda = [
        "casos_srag_mes",
        "casos_obito_mes",
        "casos_uti_mes",
        "casos_idosos_mes",
        "casos_pediatricos_mes",
        "semanas_com_dado",
    ]
    for campo in campos_demanda:
        df = df.withColumn(campo, F.coalesce(F.col(campo), F.lit(0)))

    total_antes = df.count()
    df = df.filter(F.col("leitos_totais") >= 10)
    total_apos = df.count()

    print("\n── Join capacity × srag ──")
    print(f"  Após left join: {total_antes:,}")
    print(f"  Removidos (leitos_totais < 10): {total_antes - total_apos:,}")
    print(f"  Mantidos: {total_apos:,}")
    return df


def _features_razao(df):
    """
    Métricas de pressão relativa do mês atual (sem lag — usadas como feature base
    e como base para o target do mês seguinte).
    """
    return (
        df.withColumn(
            "casos_por_leito",
            F.round(F.col("casos_srag_mes") / (F.col("leitos_totais") + 1), 6),
        )
        .withColumn(
            "casos_por_leito_uti",
            F.round(F.col("casos_srag_mes") / (F.col("leitos_uti") + 1), 6),
        )
        .withColumn(
            "obitos_por_leito",
            F.round(F.col("casos_obito_mes") / (F.col("leitos_totais") + 1), 6),
        )
        .withColumn(
            "uti_por_leito_uti",
            F.round(F.col("casos_uti_mes") / (F.col("leitos_uti") + 1), 6),
        )
        .withColumn(
            "share_idosos",
            F.round(F.col("casos_idosos_mes") / (F.col("casos_srag_mes") + 1), 4),
        )
    )


def _lags_e_medias_moveis(df):
    """
    Lags e médias móveis de casos_por_leito e casos_srag_mes.
    Primeiros meses de cada município ficam nulos → coalesce com 0.
    """
    return (
        df
        # lags de casos_por_leito
        .withColumn(
            "casos_por_leito_lag1",
            F.coalesce(F.lag("casos_por_leito", 1).over(MUNICIPIO_W), F.lit(0.0)),
        )
        .withColumn(
            "casos_por_leito_lag2",
            F.coalesce(F.lag("casos_por_leito", 2).over(MUNICIPIO_W), F.lit(0.0)),
        )
        .withColumn(
            "casos_por_leito_lag3",
            F.coalesce(F.lag("casos_por_leito", 3).over(MUNICIPIO_W), F.lit(0.0)),
        )
        # médias móveis
        .withColumn(
            "casos_por_leito_ma2",
            F.coalesce(F.avg("casos_por_leito").over(MUNICIPIO_W2), F.lit(0.0)),
        )
        .withColumn(
            "casos_por_leito_ma3",
            F.coalesce(F.avg("casos_por_leito").over(MUNICIPIO_W3), F.lit(0.0)),
        )
        # lags de casos_srag_mes (volume absoluto — feature complementar)
        .withColumn(
            "casos_srag_lag1", F.coalesce(F.lag("casos_srag_mes", 1).over(MUNICIPIO_W), F.lit(0))
        )
        .withColumn(
            "casos_srag_lag2", F.coalesce(F.lag("casos_srag_mes", 2).over(MUNICIPIO_W), F.lit(0))
        )
    )


def _features_dinamica(df):
    """
    Features de tendência e aceleração da pressão relativa.
    growth_mom / growth_3m: denominador + 0.001 para evitar divisão por zero
    com valores muito pequenos de casos_por_leito.
    """
    df = df.withColumn(
        "growth_mom",
        F.round(
            (F.col("casos_por_leito") - F.col("casos_por_leito_lag1"))
            / (F.col("casos_por_leito_lag1") + F.lit(0.001)),
            4,
        ),
    ).withColumn(
        "growth_3m",
        F.round(
            (F.col("casos_por_leito") - F.col("casos_por_leito_lag3"))
            / (F.col("casos_por_leito_lag3") + F.lit(0.001)),
            4,
        ),
    )
    # acceleration depende de growth_mom — calculado após derivar growth_mom
    df = df.withColumn(
        "acceleration",
        F.round(
            F.col("growth_mom") - F.coalesce(F.lag("growth_mom", 1).over(MUNICIPIO_W), F.lit(0.0)),
            4,
        ),
    ).withColumn(
        "rolling_std_3m",
        F.round(
            F.coalesce(F.stddev("casos_por_leito").over(MUNICIPIO_W3), F.lit(0.0)),
            4,
        ),
    )
    return df


def _features_sazonais(df):
    """
    Features de sazonalidade derivadas de competencia (AAAAMM).
    is_rainy_season: proxy de sazonalidade respiratória (meses chuvosos no Brasil).
    """
    return (
        df.withColumn("ano", F.col("competencia").substr(1, 4).cast("integer"))
        .withColumn("mes", F.col("competencia").substr(5, 2).cast("integer"))
        .withColumn("quarter", F.ceil(F.col("mes") / 3).cast("integer"))
        .withColumn("is_semester1", F.when(F.col("mes") <= 6, F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "is_rainy_season",
            F.when(F.col("mes").isin(11, 12, 1, 2, 3), F.lit(1)).otherwise(F.lit(0)),
        )
    )


def _adicionar_consolidation_flag(df):
    """
    Classifica o nível de consolidação de cada competência
    baseado em quantos dias se passaram desde o fechamento do mês.

    O SRAG tem atraso estrutural de notificação:
      >= 90 dias: "consolidado"   → ~95%+ dos casos notificados
      >= 45 dias: "estabilizando" → ~80-90% dos casos notificados
       < 45 dias: "recente"       → < 80% dos casos notificados

    Usa a data de hoje como referência.
    competencia AAAAMM → data de fechamento = último dia do mês MM/AAAA
    """
    from datetime import date

    hoje = date.today()

    # converte competencia para data do último dia do mês
    df = df.withColumn(
        "_data_fechamento",
        F.last_day(
            F.to_date(
                F.concat(F.col("competencia"), F.lit("01")),
                "yyyyMMdd",
            )
        ),
    )

    # dias desde o fechamento
    df = df.withColumn(
        "_dias_desde_fechamento",
        F.datediff(F.lit(hoje), F.col("_data_fechamento")),
    )

    # flag de consolidação
    df = df.withColumn(
        "srag_consolidation_flag",
        F.when(F.col("_dias_desde_fechamento") >= 90, F.lit("consolidado"))
        .when(F.col("_dias_desde_fechamento") >= 45, F.lit("estabilizando"))
        .otherwise(F.lit("recente")),
    )

    # score de qualidade combinado (0 a 1)
    # penaliza forward fill E consolidação recente
    df = df.withColumn(
        "data_quality_score",
        F.when(
            F.col("capacity_is_forward_fill"),
            F.lit(0.3),
        )
        .when(
            F.col("srag_consolidation_flag") == "recente",
            F.lit(0.5),
        )
        .when(
            F.col("srag_consolidation_flag") == "estabilizando",
            F.lit(0.8),
        )
        .otherwise(F.lit(1.0)),
    )

    df = df.drop("_data_fechamento", "_dias_desde_fechamento")
    return df


def _calcular_target(df):
    """
    Target v2: casos_por_leito do mês seguinte >= percentil 85 nacional daquele mês.
    Último mês de cada município → target null (mantido para scoring ao vivo).

    Competências excluídas do cálculo do p85 (target null):
      - capacity_is_forward_fill=True em t+1 (capacity estimada)
      - srag_consolidation_flag="recente" em t+1 (SRAG subnotificado, < 80%)
    """
    # passo 1: valor da métrica alvo no mês seguinte
    df = df.withColumn(
        "casos_por_leito_next",
        F.lead("casos_por_leito", 1).over(MUNICIPIO_W),
    )

    # passo 2: marca t+1 como forward fill se capacity do próximo mês for estimada
    df = df.withColumn(
        "next_is_forward_fill",
        F.lead("capacity_is_forward_fill", 1).over(MUNICIPIO_W),
    )

    # passo 3: marca t+1 como recente se SRAG do próximo mês for subnotificado
    df = df.withColumn(
        "next_consolidation_flag",
        F.lead("srag_consolidation_flag", 1).over(MUNICIPIO_W),
    )

    # passo 4: anula casos_por_leito_next quando t+1 é forward fill OU recente
    # → target ficará null nesses meses (scoring only), p85 não será distorcido
    df = df.withColumn(
        "casos_por_leito_next",
        F.when(
            F.col("next_is_forward_fill") | (F.col("next_consolidation_flag") == "recente"),
            F.lit(None).cast("double"),
        ).otherwise(F.col("casos_por_leito_next")),
    )

    # remove colunas auxiliares antes do p85
    df = df.drop("next_is_forward_fill", "next_consolidation_flag")

    # passo 4: percentil 85 nacional de casos_por_leito_next por competencia
    # (linhas com casos_por_leito_next null são ignoradas pelo percentile_approx)
    p85 = df.groupBy("competencia").agg(
        F.percentile_approx("casos_por_leito_next", 0.85).alias("_p85_nacional"),
    )

    # passo 5: join do p85 de volta ao df principal
    df = df.join(p85, on="competencia", how="left")

    # passo 6: target binário (null onde não há mês seguinte ou t+1 é forward fill)
    df = (
        df.withColumn(
            "target_alta_pressao",
            F.when(
                F.col("casos_por_leito_next").isNotNull(),
                F.when(F.col("casos_por_leito_next") >= F.col("_p85_nacional"), F.lit(1)).otherwise(
                    F.lit(0)
                ),
            ),
        )
        # passo 7: colunas de governança
        .withColumn("target_definition_version", F.lit("v2"))
        .withColumn("target_metric", F.lit("casos_por_leito"))
        .withColumn("target_percentile", F.lit(0.85))
        .drop("casos_por_leito_next", "_p85_nacional")
    )
    return df


def _validar_e_filtrar(df):
    """
    Loga anomalias e aplica filtros de qualidade mínima.
    NÃO remove linhas com target null — são usadas para scoring ao vivo.
    """
    total = df.count()
    print("\n── Validação de qualidade ──")
    print(f"  Total antes da filtragem: {total:,}")

    # informativo: último mês por município (sem target)
    sem_target = df.filter(F.col("target_alta_pressao").isNull()).count()
    print(f"  Linhas sem target (scoring only, mantidas): {sem_target:,}")

    # regra 1: municipio_id não nulo
    sem_municipio = df.filter(F.col("municipio_id").isNull()).count()
    if sem_municipio:
        print(f"  ⚠ Descartados por municipio_id nulo: {sem_municipio:,}")
    df = df.filter(F.col("municipio_id").isNotNull())

    # regra 2: competencia no formato AAAAMM (6 dígitos numéricos)
    comp_invalida = df.filter(~F.col("competencia").rlike(r"^\d{6}$")).count()
    if comp_invalida:
        print(f"  ⚠ Descartados por competencia fora do formato AAAAMM: {comp_invalida:,}")
    df = df.filter(F.col("competencia").rlike(r"^\d{6}$"))

    # regra 3: casos_srag_mes >= 0
    srag_negativo = df.filter(F.col("casos_srag_mes") < 0).count()
    if srag_negativo:
        print(f"  ⚠ Descartados por casos_srag_mes < 0: {srag_negativo:,}")
    df = df.filter(F.col("casos_srag_mes") >= 0)

    # regra 4: target_alta_pressao só em {0, 1, null}
    target_invalido = df.filter(
        F.col("target_alta_pressao").isNotNull() & ~F.col("target_alta_pressao").isin(0, 1)
    ).count()
    if target_invalido:
        print(f"  ⚠ Descartados por target_alta_pressao fora de {{0, 1}}: {target_invalido:,}")
    df = df.filter(F.col("target_alta_pressao").isNull() | F.col("target_alta_pressao").isin(0, 1))

    total_apos = df.count()
    print(f"  Total após filtragem: {total_apos:,}  (descartados: {total - total_apos:,})")
    return df


def _adicionar_metadados(df):
    """Adiciona colunas de rastreabilidade do processamento."""
    return (
        df.withColumn("_processed_at", F.current_timestamp())
        .withColumn("_source_srag", F.lit(TABLE_SRC_SRAG))
        .withColumn("_source_cap", F.lit(TABLE_SRC_CAP))
    )


def transformar(spark: SparkSession):
    """
    Executa o pipeline completo Silver → Gold de features de pressão assistencial v2.
    Grava via Databricks Feature Engineering Client (drop + recria para garantir schema limpo).
    """
    fe = FeatureEngineeringClient()

    print("Lendo fontes silver ...")
    srag = spark.table(TABLE_SRC_SRAG)
    cap = spark.table(TABLE_SRC_CAP)
    print(f"  srag: {srag.count():,} linhas | cap: {cap.count():,} linhas")

    df = _agregar_srag_mensal(srag)
    df = _join_capacity(df, cap)
    df = _features_razao(df)
    df = _lags_e_medias_moveis(df)
    df = _features_dinamica(df)
    df = _features_sazonais(df)
    df = _adicionar_consolidation_flag(df)
    df = _calcular_target(df)
    df = _validar_e_filtrar(df)
    df = _adicionar_metadados(df)

    # Feature Store — drop e recria para garantir schema e primary keys limpos
    try:
        fe.get_table(TABLE_DST)
        table_exists = True
    except Exception:
        table_exists = False

    if table_exists:
        spark.sql(f"DROP TABLE IF EXISTS {TABLE_DST}")
        print("  Tabela anterior removida")

    print(f"\nCriando feature table {TABLE_DST} ...")
    fe.create_table(
        name=TABLE_DST,
        primary_keys=["municipio_id", "competencia"],
        df=df,
        description=(
            "Feature store de pressão assistencial respiratória — v2. "
            "Grain: municipio_id x competencia (AAAAMM). "
            "Target: casos_por_leito do mês seguinte >= percentil 85 nacional. "
            "target_definition_version=v2."
        ),
    )
    print(f"✓ Feature table {TABLE_DST} criada.")


def show_summary(spark: SparkSession):
    """
    Imprime estatísticas básicas da gold para validação pós-execução:
    - totais gerais
    - distribuição do target por ano
    - estatísticas de casos_por_leito
    - correlação das features com o target
    """
    df = spark.table(TABLE_DST)

    print(f"Total de linhas: {df.count():,}")
    print(f"Municípios distintos:   {df.select('municipio_id').distinct().count():,}")
    print(f"Competências distintas: {df.select('competencia').distinct().count():,}")

    print("\nDistribuição do target por ano:")
    (
        df.filter(F.col("target_alta_pressao").isNotNull())
        .withColumn("ano", F.col("competencia").substr(1, 4))
        .groupBy("ano")
        .agg(
            F.count("*").alias("total"),
            F.sum("target_alta_pressao").alias("positivos"),
            F.round(F.avg("target_alta_pressao") * 100, 2).alias("pct_positivos"),
        )
        .orderBy("ano")
        .show()
    )

    print("\nEstatísticas de casos_por_leito:")
    df.select(
        F.round(F.min("casos_por_leito"), 6).alias("min"),
        F.round(F.max("casos_por_leito"), 6).alias("max"),
        F.round(F.avg("casos_por_leito"), 6).alias("mean"),
        F.round(F.stddev("casos_por_leito"), 6).alias("stddev"),
    ).show()

    print("\nCorrelação das features com o target:")
    features = [
        "casos_por_leito",
        "casos_por_leito_lag1",
        "casos_por_leito_ma2",
        "casos_por_leito_ma3",
        "casos_srag_lag1",
        "growth_mom",
        "leitos_totais",
        "leitos_uti",
        "obitos_por_leito",
        "uti_por_leito_uti",
        "share_idosos",
        "rolling_std_3m",
    ]
    df_not_null = df.filter(F.col("target_alta_pressao").isNotNull())
    for feat in features:
        corr = df_not_null.stat.corr(feat, "target_alta_pressao")
        barra = "█" * int(abs(corr) * 40)
        sinal = "+" if corr >= 0 else "-"
        print(f"  {feat:<28} {sinal}{abs(corr):.4f}  {barra}")


# ── entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    transformar(spark)
