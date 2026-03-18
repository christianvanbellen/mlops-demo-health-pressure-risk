# Radar de Pressão Assistencial

![Python](https://img.shields.io/badge/python-3.11-blue)
![UV](https://img.shields.io/badge/uv-0.9.0-purple)
![Databricks](https://img.shields.io/badge/Databricks-DBR%2016.4%20ML-red)
![MLflow](https://img.shields.io/badge/MLflow-tracked-blue)
![CI](https://github.com/christianvanbellen/mlops-demo-health-pressure-risk/actions/workflows/ci.yml/badge.svg)

Pipeline MLOps para prever risco de sobrecarga hospitalar respiratória por município,
rodando semanalmente no Databricks.

## O que faz

Toda segunda-feira, o pipeline:

1. Ingere dados públicos de SRAG e Hospitais/Leitos
2. Constrói features por município × mês
3. Aplica o modelo `@champion` e classifica municípios em alto / moderado / baixo risco
4. Monitora a performance e dispara retraining se necessário

**Resultado:** ranking semanal de ~3.300 municípios com `risk_score` entre 0 e 1
e principais drivers do risco.

## Horizonte operacional

O pipeline opera com **4–8 semanas de defasagem** em relação ao momento atual,
por conta do atraso de publicação das fontes. Adequado para planejamento preventivo,
não para resposta em tempo real. Ver `docs/pipeline_limitations.md`.

## Arquitetura

```
Bronze → Silver → Gold (Feature Store) → Scoring → Monitor → Trigger
```

Ver `docs/architecture.md` para detalhes completos.

## Quickstart

```bash
# Setup (Linux/macOS):
uv venv --python 3.11 && source .venv/bin/activate
uv sync --all-extras
uv run pytest tests/ -v
```

Ver `CONTRIBUTING.md` para setup completo em Windows, Linux e macOS,
e instruções de contribuição.

## Documentação

| Documento | Conteúdo |
|---|---|
| `docs/architecture.md` | Arquitetura técnica e ML lifecycle |
| `docs/data_sources.md` | Fontes de dados, ingestão e caveats |
| `docs/decisions.md` | Decisões de design (ADRs) |
| `docs/pipeline_limitations.md` | Limitações e horizonte operacional |
| `CONTRIBUTING.md` | Setup de desenvolvedor e fluxo de trabalho |
| `resources/README.md` | Deploy com Databricks Asset Bundles |

## Modelo atual

| Métrica | Valor |
|---|---|
| Modelo | LightGBM `@champion` v3 |
| Precision@K(15%) validação | 0.613 |
| Precision@K(15%) backtesting | ~0.57 médio |
| Municípios scored | 3.304 (jan/2026) |
| Competências de treino | 2023–2024 |

## CI/CD

| Evento | Jobs executados |
|---|---|
| push `feature/**` | lint + testes + bundle validate |
| pull_request → `main` | lint + testes + validate + deploy dev |
| merge em `main` | lint + testes + validate + deploy prod *(approval gate)* |

## Fontes de dados

- **Hospitais e Leitos (CNES)** — capacidade estrutural
- **SRAG / SIVEP-Gripe** — demanda hospitalar grave

Ver `docs/data_sources.md` para detalhes, URLs e caveats.
