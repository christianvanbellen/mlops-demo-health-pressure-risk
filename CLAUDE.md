# Instruções e contexto para o Claude Code — health-pressure-risk

## O que é este projeto
Pipeline de MLOps para prever risco de pressão assistencial respiratória por
município × semana epidemiológica no Brasil. O modelo classifica, para cada município,
a probabilidade de entrar em faixa de alta pressão assistencial na semana seguinte (t+1).
Tipo de problema: classificação binária supervisionada.

---

## Ambiente

| Item | Valor |
|---|---|
| Plataforma | Databricks (AWS) |
| Catálogo Unity Catalog | ds_dev_db |
| Schema de desenvolvimento | dev_christian_van_bellen |
| Prefixo de tabelas bronze | bronze_ |
| Prefixo de tabelas silver | silver_ |
| Prefixo de tabelas gold | gold_ |
| Linguagem principal | Python + PySpark |
| Formato de armazenamento | Delta Lake |
| Experiment tracking | MLflow (nativo Databricks) |
| Model registry | Unity Catalog (Models in Unity Catalog) |

Exemplo de tabela completa: `ds_dev_db.dev_christian_van_bellen.bronze_srag`

---

## Estrutura do repositório

```
health-pressure-risk/
conf/
  dev.yml           # configurações de desenvolvimento
  prod.yml          # configurações de produção
src/
  ingestion/        # leitura de fontes brutas → bronze
  transforms/       # bronze → silver → gold
  training/         # treino, avaliação, registro de modelos
  scoring/          # batch scoring semanal
  monitoring/       # data quality, drift, performance
pipelines/          # definições de pipelines Lakeflow
notebooks/          # notebooks exploratórios e de workshop
jobs/               # definições de jobs Databricks
tests/              # testes unitários
docs/               # documentação de arquitetura e decisões
```

---

## Convenções de código

- Sempre usar `sep=";"` e `encoding="latin1"` para CSVs do OpenDataSUS
- Tabelas bronze são **append-only** — nunca sobrescrever, sempre versionar por `_snapshot_date`
- Toda tabela deve ter colunas de metadados: `_snapshot_date`, `_source_url`, `_ingestion_ts`
- Semana epidemiológica: usar coluna `SEM_NOT` do SRAG; derivar via `F.weekofyear` quando necessário
- Chave principal do grain: `municipio_id` (código IBGE 6 dígitos) × `semana_epidemiologica`
- Pesos do modelo começam heurísticos: `α=0.5`, `β=2.0`, `γ=0.3` — versionados em `target_definition_version`
- Nomes de variáveis e colunas sempre em **snake_case português**
- Comentários de código em **português**

---

## Fontes de dados

| Fonte | Função | Formato | Encoding |
|---|---|---|---|
| SRAG / SIVEP-Gripe | Demanda grave (sinal principal) | CSV | latin1, sep=; |
| CNES Estabelecimentos | Capacidade estrutural | CSV | latin1, sep=; |
| Hospitais e Leitos | Capacidade simplificada | CSV | latin1, sep=; |
| InfoGripe / Fiocruz | Tendência regional (opcional) | API/JSON | UTF-8 |

URLs base: `https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/`
Documentação completa das fontes: `docs/data_sources.md`

---

## Tabelas principais

- `bronze_srag`
- `bronze_cnes_estabelecimentos`
- `bronze_hospitais_leitos`
- `silver_capacity_municipio_semana`
- `silver_srag_municipio_semana`
- `gold_pressure_features`
- `gold_pressure_scoring`
- `gold_pressure_monitoring`

---

## Target do modelo

```
PressureScore(m,t) = Demand(m,t) / (Capacity(m,t) + 1) + γ * growth_wow + δ * regional_trend

Demand(m,t)   = MA2(srag_cases) + 0.5 * MA2(srag_deaths)
Capacity(m,t) = leitos_totais  + 2.0 * leitos_uti

target(m,t) = 1  se  PressureScore(m, t+1) >= percentil_85_nacional(t+1)
```

Versão atual da regra: `target_definition_version = 'v1'`

---

## Modelos

| Etapa | Modelo | Arquivo |
|---|---|---|
| Baseline | Logistic Regression | `src/training/train_baseline_lr.py` |
| Principal | LightGBM ou XGBoost | `src/training/train_gbt.py` |

**Split temporal — nunca aleatório:**
- Treino: 2023–2024
- Validação: 2025 H1
- Teste: 2025 H2
- Scoring live: 2026

---

## Cuidados metodológicos importantes

- **Atraso de notificação SRAG**: semanas recentes são incompletas — sempre reprocessar
  as últimas 2–4 semanas e excluir a semana corrente do treino
- **CNES = capacidade instalada**, não ocupação real — deixar isso explícito em comentários
- **Snapshots temporais de capacidade**: CNES muda ao longo do tempo — versionar por
  `capacity_snapshot_date`
- **Desbalanceamento**: target é top percentil (~15% positivos) — usar `class_weight`

---

## Referências

- Roteiro conceitual do projeto: `docs/architecture.md`
- Decisões técnicas: `docs/decisions.md`
- Fontes de dados detalhadas: `docs/data_sources.md`
