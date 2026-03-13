# Arquitetura do projeto — health-pressure-risk

## Visão geral

Pipeline semanal de MLOps para classificação de risco de pressão assistencial
respiratória por município × semana epidemiológica no Brasil.

**Pergunta de negócio:** quais municípios têm maior risco de enfrentar pressão
assistencial elevada na próxima semana, dado o comportamento recente da demanda
e a estrutura disponível?

---

## Grain analítico

`municipio_id (IBGE 6 dígitos) × semana_epidemiologica`

Escolha justificada por:
- Reduz ruído cadastral de hospital individual
- Facilita join com séries epidemiológicas
- Simplifica explicação e visualização
- Preserva valor operacional

---

## Arquitetura de dados — camadas

### Bronze
Dados brutos, particionados por data de ingestão. Append-only.

| Tabela | Fonte |
|---|---|
| `bronze_srag` | SRAG / SIVEP-Gripe (OpenDataSUS) |
| `bronze_cnes_estabelecimentos` | CNES (OpenDataSUS) |
| `bronze_hospitais_leitos` | Hospitais e Leitos (OpenDataSUS) |

Regras:
- Manter schema bruto + metadados de ingestão (`_snapshot_date`, `_source_url`, `_ingestion_ts`)
- Nunca sobrescrever — reprocessamento deleta partição do período e faz append
- Tudo como string (inferSchema=false)

### Silver
Dados limpos, padronizados e reconciliados.

| Tabela | Conteúdo |
|---|---|
| `silver_capacity_municipio_semana` | Capacidade hospitalar agregada por município-semana |
| `silver_srag_municipio_semana` | Demanda SRAG agregada por município-semana |

Transformações aplicadas:
- Padronização de datas e semana epidemiológica
- Normalização de chaves de município (IBGE 6 dígitos)
- Deduplicação
- Harmonização de leitos totais / complementares / UTI
- Tipagem correta (string → int/double/date)

### Gold
Camada pronta para modelagem, serving e analytics.

| Tabela | Conteúdo |
|---|---|
| `gold_pressure_features` | Features + target para treino e scoring |
| `gold_pressure_scoring` | Saída do modelo (risk_score, risk_class, drivers) |
| `gold_pressure_monitoring` | Métricas de drift e qualidade |

---

## Pipeline de jobs

| Job | Frequência | Tarefas |
|---|---|---|
| `job_health_pressure_weekly` | Toda segunda-feira | ingestão → silver → gold → scoring → monitoring |
| `job_health_pressure_retrain` | Mensal | treino → avaliação → registro do champion |
| `job_health_pressure_backfill` | Sob demanda | reprocessa últimas 2–4 semanas (atraso SRAG) |

---

## Arquitetura MLOps

```
Experimentos    → MLflow Tracking (nativo Databricks)
Model Registry  → Unity Catalog (Models in Unity Catalog)
Aliases         → champion / staging / shadow
Serving         → batch scoring semanal (tabela gold_pressure_scoring)
Serving opcional→ endpoint REST (Mosaic AI Model Serving)
Monitoring      → data quality + feature drift + score drift
Retraining      → mensal ou por trigger de drift
```

---

## Features do modelo

### Capacidade (estrutural, estável)
- `leitos_totais`, `leitos_uti`, `num_hospitais`
- `leitos_totais_por_10k`, `leitos_uti_por_10k`

### Demanda recente (dinâmica, semanal)
- `srag_cases_lag1`, `srag_cases_lag2`
- `srag_cases_ma2`, `srag_cases_ma4`
- `srag_severe_cases_lag1`, `srag_deaths_lag1`

### Pressão relativa
- `srag_per_bed_lag1`, `srag_per_icu_bed_lag1`
- `severe_per_icu_bed_lag1`

### Dinâmica / tendência
- `growth_wow`, `growth_2w`, `acceleration`
- `rolling_std_4w`, `pct_change_vs_ma4`

### Sazonalidade
- `epi_week`, `month`, `quarter`

### Regional
- `uf_srag_growth`, `regional_pressure_percentile`

### Histórico de risco
- `prior_pressure_score_lag1`, `prior_high_risk_flag_lag1`
- `weeks_in_high_risk_last_8w`

---

## Saída do modelo

Tabela `gold_pressure_scoring`:

| Campo | Descrição |
|---|---|
| `scoring_date` | Data de execução do scoring |
| `semana_ref` | Semana epidemiológica de referência |
| `municipio_id` | Código IBGE do município |
| `municipio_nome` | Nome do município |
| `uf` | Unidade federativa |
| `risk_score` | Score entre 0 e 1 |
| `risk_class` | baixo / moderado / alto |
| `top_driver_1/2/3` | Principais fatores explicativos |
| `model_version` | Versão do modelo utilizado |

---

## Roadmap de fases

### Fase 1 — MVP técnico (atual)
- Ingestão CNES + SRAG
- Agregação município-semana
- Target v1
- Baseline logística
- Score batch semanal
- Dashboard simples

### Fase 2 — Robustez analítica
- Gradient boosted trees
- Rolling validation
- Monitor de drift
- Explainability (SHAP)
- Tuning de percentis do target

### Fase 3 — Maturidade operacional
- Champion/challenger
- Alertas automáticos
- Retraining programado
- Auditoria de qualidade
- Benchmark com fonte adicional
