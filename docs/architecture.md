# Arquitetura — Radar de Pressão Assistencial

## Visão geral

Pipeline MLOps de ponta a ponta para prever risco de sobrecarga hospitalar respiratória
por município, rodando semanalmente no Databricks.

**Pergunta de negócio:** quais municípios têm maior risco de enfrentar pressão
assistencial elevada na próxima competência, dado o comportamento recente da demanda
e a capacidade instalada disponível?

**Tipo de problema:** classificação binária supervisionada, com avaliação operacional
via Precision@K (K = 15% dos municípios — faixa que um gestor monitoraria num ciclo mensal).

---

## Stack tecnológico

| Camada | Tecnologia |
|---|---|
| Processamento | Apache Spark (Databricks Runtime 16.4 ML) |
| Orquestração | Databricks Asset Bundles (DAB) + Jobs |
| Feature Store | Databricks Feature Engineering Client |
| Experiment tracking | MLflow (nativo Databricks) |
| Model registry | Models in Unity Catalog |
| Linguagem | Python 3.11 |
| Dependências | UV 0.9.0 + pyproject.toml + uv.lock |
| CI/CD | GitHub Actions |
| Linting | Ruff |

---

## Arquitetura de dados — Medallion

### Bronze (raw, append-only)

Catálogo/schema: `ds_dev_db.dev_christian_van_bellen`

| Tabela | Fonte | Script |
|---|---|---|
| `bronze_srag` | SRAG / SIVEP-Gripe | `src/ingestion/srag_ingest.py` |
| `bronze_hospitais_leitos` | Hospitais e Leitos | `src/ingestion/hospitais_leitos_ingest.py` |
| `bronze_cnes_estabelecimentos` | CNES (Fase 2) | `src/ingestion/cnes_ingest.py` |
| `bronze_cnes_leitos` | CNES (Fase 2) | `src/ingestion/cnes_ingest.py` |

Regras:
- Schema bruto preservado; tudo como string (`inferSchema=false`)
- Metadados obrigatórios: `_snapshot_date`, `_source_url`, `_ingestion_ts`, `_is_live`, `_ano_arquivo`
- Reprocessamento idempotente: `DELETE WHERE _ano_arquivo = {ano}` + append
- Anos 2023/2024 congelados; 2025/2026 reingeridos toda semana

### Silver (limpo, padronizado)

| Tabela | Grain | Script |
|---|---|---|
| `silver_capacity_municipio_mes` | `municipio_id × competencia (AAAAMM)` | `src/transforms/silver_capacity_municipio_mes.py` |
| `silver_srag_municipio_semana` | `municipio_id × semana_epidemiologica` | `src/transforms/silver_srag_municipio_semana.py` |

Transformações aplicadas:
- Normalização de chaves de município para IBGE 6 dígitos
- Lookup `UF + nome normalizado → municipio_id` para Hospitais/Leitos 2023/2024 (ausência de `CO_IBGE`)
- Agregação de leitos por município-competência
- Derivação de `competencia` (AAAAMM) a partir de `SEM_PRI`
- Tipagem correta (string → int/double/date)
- Deduplicação de casos SRAG entre arquivos anuais
- `uf` via `first(ignorenulls=True)` fora do `groupBy` — municípios com notificações cross-UF (ex: Brasília) gerariam múltiplas linhas por `municipio_id × competencia` e quebrariam a PK da Feature Store

### Gold / Feature Store

| Tabela | Grain | Script |
|---|---|---|
| `gold_pressure_features` | `municipio_id × competencia` | `src/transforms/gold_pressure_features.py` |
| `gold_pressure_scoring` | `municipio_id × competencia × run_date` | `src/scoring/batch_score.py` |
| `monitoring_performance` | `competencia × monitor_date` | `src/monitoring/performance_monitor.py` |

`gold_pressure_features` é registrada como Feature Table com PK `[municipio_id, competencia]`.

---

## Grain analítico: mensal, não semanal

O grain de saída é `municipio_id × competencia (AAAAMM)` — **mensal**, não semanal.

Justificativa:
- A granularidade semanal amplifica o atraso de notificação do SRAG — as últimas 2–4
  semanas de cada ano estão cronicamente incompletas
- Agregar por mês suaviza esse ruído e alinha naturalmente com a fonte de capacidade
  (Hospitais e Leitos), publicada mensalmente
- Reduz ruído cadastral de estabelecimento individual e facilita explicação operacional

Cobertura: ~3.330–3.354 municípios com `leitos_totais >= 10` (filtro remove municípios
sem hospital e prováveis erros cadastrais que distorceriam o denominador e o P85).

---

## Ciclo de vida ML

### 1. Feature Engineering

Join: `silver_capacity_municipio_mes` (left) × `silver_srag_municipio_semana` (agregado para mensal).
Todos os municípios com hospital aparecem — municípios sem casos SRAG no mês ficam com demanda zerada.

**22 features usadas no modelo** (idênticas em treino, validação, teste e scoring):

| Grupo | Features |
|---|---|
| Pressão atual | `casos_por_leito` |
| Lags de pressão | `casos_por_leito_lag1`, `lag2`, `lag3` |
| Médias móveis de pressão | `casos_por_leito_ma2`, `ma3` |
| Demanda bruta | `casos_srag_lag1`, `casos_srag_lag2` |
| Gravidade | `obitos_por_leito`, `uti_por_leito_uti`, `share_idosos` |
| Dinâmica / tendência | `growth_mom`, `growth_3m`, `acceleration`, `rolling_std_3m` |
| Capacidade | `leitos_totais`, `leitos_uti`, `num_hospitais` |
| Sazonalidade | `mes`, `quarter`, `is_semester1`, `is_rainy_season` |

### 2. Target (v2 — `target_definition_version = "v2"`)

```
target_alta_pressao(m, t) = 1  se  casos_por_leito(m, t+1) >= P85_nacional(t+1)

casos_por_leito(m, t) = casos_srag_mes(m, t) / (leitos_totais(m, t) + 1)

P85_nacional(t+1) = percentil 85 de casos_por_leito
                    sobre todos os municípios da competência t+1
```

Raciocínio da métrica relativa: São Paulo com 37.000 leitos e 8.000 casos não é alerta;
um município com 10 leitos e 17 casos é crise. Pressão relativa à capacidade captura
isso; volume absoluto não.

Resultado: ~15% de positivos por ano (2023–2025).

**Exclusões do cálculo do target:**
- `capacity_is_forward_fill = True` em t+1 (capacidade estimada, não publicada)
- `srag_consolidation_flag = "recente"` em t+1 (< 45 dias — < 80% dos casos notificados)

Linhas sem t+1 disponível têm `target = null` e são **mantidas** para scoring ao vivo.

### 3. Split temporal (nunca aleatório)

| Conjunto | Período |
|---|---|
| Treino | até `202412` |
| Validação | `202501` – `202506` |
| Teste | `202507` em diante |
| Scoring live | `2026` |

### 4. Modelos

| Modelo | Script | Registro |
|---|---|---|
| Logistic Regression (baseline) | `src/training/train_baseline_lr.py` | MLflow + Unity Catalog |
| LightGBM (principal) | `src/training/train_gbt.py` | MLflow + Unity Catalog |

**Experimento MLflow:** `/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr`

**Registry:** `ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier`

**Critério de avaliação:** Precision@K(15%) — K representa os municípios que um
gestor monitoraria num ciclo mensal; AUC-ROC mede discriminação geral.

### 5. Human Gate — promoção de modelos

```
Treinamento (LR + LightGBM)
  ↓
evaluate.py: compara os dois modelos + @champion atual (se existir)
  ↓
Se não existe @champion:  promove o melhor automaticamente (first_deploy)
Se existe @champion:
  Novo > champion → registra como @challenger → aguarda aprovação humana no MLflow UI
  Novo ≤ champion → no_change, champion mantido
```

Aliases ativos: `@champion`, `@challenger`

### 6. Batch Scoring

Frequência: semanal (job toda segunda-feira)

**Política de seleção de competência** — a competência é elegível se:
- `target = null` (sem t+1 disponível — linha de scoring ao vivo)
- `capacity_is_forward_fill = False`
- `srag_consolidation_flag != "recente"`
- `data_quality_score >= 0.5`

**Classificação de risco (v2 — percentis relativos por competência):**

| Classe | Critério | Prevalência |
|---|---|---|
| `alto` | score >= P85 desta competência | ~15% |
| `moderado` | score >= P70 desta competência | ~15% |
| `baixo` | score < P70 desta competência | ~70% |

**Modo A/B canary:**
- 20% dos municípios → `@challenger`; 80% → `@champion`
- Roteamento determinístico: `MD5(municipio_id) % 100 < 20`
- Mesmo município sempre no mesmo modelo durante o período A/B
- Ativado via `--ab` no CLI ou `ab_test=True`

### 7. Data Quality Score

Combina `capacity_is_forward_fill` e `srag_consolidation_flag`:

| Condição | `data_quality_score` |
|---|---|
| Capacity real + SRAG consolidado (≥ 90 dias) | 1.0 |
| Capacity real + SRAG estabilizando (≥ 45 dias) | 0.8 |
| Capacity real + SRAG recente (< 45 dias) | 0.5 |
| Capacity forward fill (qualquer flag) | 0.3 |

### 8. Camada de Qualidade de Dados (DQX)

**Databricks Labs DQX** (`databricks-labs-dqx>=0.5.0`) é integrado antes de cada write em todas as camadas da Medallion.

**Padrão:**

```python
from quality.runner import run_checks
from quality.checks import checks_bronze_srag

df = run_checks(spark, df,
                checks=checks_bronze_srag(),
                table_name=TABLE_BRONZE_SRAG,
                quarantine_table=f"{CATALOG}.{SCHEMA}.quarantine_bronze_srag")
df.write...saveAsTable(TABLE_BRONZE_SRAG)
```

`DQEngine.apply_checks_by_metadata_and_split(df, checks)` separa o DataFrame em dois:
- `valid_df` — segue para o write
- `quarantine_df` — gravado em append na tabela de quarentena com timestamps e nome da tabela de destino

**Suítes de checks (`src/quality/checks.py`):**

| Tabela | Checks error (quarentena) | Checks warn (flag) |
|---|---|---|
| `bronze_srag` | `CO_MUN_RES`, `DT_NOTIFIC`, `_snapshot_date`, `_ano_arquivo` não nulos | `DT_SIN_PRI` formato data válido |
| `bronze_hospitais_leitos` | `COMP`, `CNES`, `_snapshot_date`, `_ano_arquivo` não nulos | `UF` não nulo |
| `silver_srag_municipio_semana` | `municipio_id`, `semana_epidemiologica`, `competencia` não nulos | `srag_consolidation_flag` em lista válida |
| `silver_capacity_municipio_mes` | `municipio_id`, `competencia` não nulos; `leitos_totais >= 0` | `capacity_is_forward_fill` não nulo |
| `gold_pressure_features` | `municipio_id`, `competencia` não nulos; `casos_por_leito >= 0` | `data_quality_score` em [0,1]; `leitos_totais >= 10` |

**Tabelas de quarentena** (criadas automaticamente em append):

| Tabela quarentena | Origem |
|---|---|
| `quarantine_bronze_srag` | `srag_ingest.py` |
| `quarantine_bronze_hospitais_leitos` | `hospitais_leitos_ingest.py` |
| `quarantine_silver_srag` | `silver_srag_municipio_semana.py` |
| `quarantine_silver_capacity` | `silver_capacity_municipio_mes.py` |
| `quarantine_gold_features` | `gold_pressure_features.py` |

### 9. Monitoramento

**Performance monitor** (`src/monitoring/performance_monitor.py`):
- Calcula Precision@K realizada comparando `score(T)` com `target_realizado(T+1)`
- Backtesting histórico simulando `@champion` sobre competências passadas
- Threshold de alerta: `Precision@K < 0.55`
- Grava resultados em `monitoring_performance` e loga métricas no MLflow

**Retrain trigger** (`src/monitoring/retrain_trigger.py`):
- Lê `monitoring_performance`
- **Regra principal:** `Precision@K < 0.55` por 2+ competências consecutivas
- **Regra secundária:** queda > 15pp vs. média das competências anteriores (queda abrupta)
- Dispara `job_health_pressure_retrain` via Jobs REST API

---

## Diagrama do pipeline semanal

```
Bronze: srag_ingest + hospitais_leitos_ingest (--live)
  ↓
Silver: capacity_municipio_mes + srag_municipio_semana (paralelo)
  ↓
Gold: gold_pressure_features
  (join Capacity LEFT SRAG, forward fill, lags, target v2, quality flags)
  ↓
Batch Score: @champion, política de confiança, A/B canary
  ↓
Performance Monitor: Precision@K, AUC-PR, backtesting
  ↓
Retrain Trigger: regra 2 consecutivas + queda abrupta
  ↓ (se trigger disparar)
job_health_pressure_retrain: LR → LightGBM → evaluate → @challenger
  ↓ (aprovação humana no MLflow UI)
@champion atualizado
```

---

## Decisões de design

| Decisão | Justificativa |
|---|---|
| Grain mensal em vez de semanal | Suaviza atraso de notificação do SRAG; alinha com fonte de capacidade |
| Capacity LEFT JOIN SRAG | Garante cobertura de todos os ~3.330 municípios com hospital, mesmo sem casos SRAG |
| Métrica relativa no target (casos/leito) | Pressão relativa captura crise em município pequeno; volume absoluto não |
| Percentil relativo no risk_class | Mantém ~15% alto risco por competência, independente do nível absoluto |
| Forward fill de capacity | Leitos mudam < 1% mês a mês — proxy metodologicamente defensável |
| Roteamento A/B por MD5(municipio_id) | Determinístico: mesmo município sempre no mesmo modelo |
| Precision@K como critério de promoção | Mede utilidade operacional real; AUC-ROC mede discriminação mas não ação |
| Human gate para promoção a @champion | Modelo de saúde pública exige revisão humana antes de impactar decisões |

---

## Roadmap de fases

### Fase 1 — MVP técnico (concluída)
- Ingestão SRAG + Hospitais e Leitos
- Medallion Bronze → Silver → Gold
- Target v2 (casos_por_leito, percentil 85)
- Baseline Logistic Regression + LightGBM
- Batch scoring semanal com A/B canary
- Performance monitor + retrain trigger
- CI/CD via GitHub Actions

### Fase 2 — Robustez e latência
- Integração InfoGripe (reduz atraso para 1–2 semanas)
- Registro de Ocupação Hospitalar (capacidade operacional vs. instalada)
- CNES Estabelecimentos (enriquecimento de features)
- SHAP para explicabilidade dos drivers de risco
- Rolling validation (backtesting completo)

### Fase 3 — Maturidade operacional
- Dashboard de visualização por município/UF
- Alertas automáticos por e-mail/Slack
- Benchmark com fontes adicionais (InfoGripe, SIM)
- Auditoria de qualidade de dados automatizada
