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
| Volume de landing | `/Volumes/ds_dev_db/dev_christian_van_bellen/landing/` |

> DBFS público desabilitado no workspace — usar sempre Volumes para arquivos temporários.

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

---

## Fluxo obrigatório de Git

Todo trabalho de código deve seguir este ritual, sem exceção.

### 1. Partir da main atualizada

```bash
git checkout main
git pull origin main
```

### 2. Criar branch descritiva

Padrão: `<tipo>/<topico-curto>`

| Tipo | Quando usar |
|---|---|
| `feat` | nova funcionalidade |
| `fix` | correção de bug |
| `chore` | tarefas de manutenção sem impacto funcional |
| `docs` | documentação |
| `refactor` | refatoração sem mudança de comportamento |
| `test` | testes |

Exemplos: `feat/wheel-packaging`, `fix/personal-compute-temp`, `docs/contributing`

### 3. Fazer as alterações necessárias

### 4. Commitar com Conventional Commits

```
<tipo>(<escopo-opcional>): <descrição curta no imperativo>
```

Exemplos:
- `feat(ci): add wheel build and upload to Unity Catalog volume`
- `fix(jobs): replace job cluster with existing_cluster_id`
- `docs(contributing): add gh CLI setup instructions`

### 5. Abrir PR via gh CLI

```bash
gh pr create \
  --title "<tipo>(<escopo>): <descrição>" \
  --body-file .github/pull_request_template.md \
  --reviewer christianvanbellen
```

**Comportamento quando `gh` não está disponível no ambiente:**

O Claude Code roda em um shell Linux (`/usr/bin/bash`) e pode não ter acesso ao
`gh` CLI instalado no Windows do desenvolvedor. Se o comando falhar com
`command not found` ou exit code 127:

1. **NÃO reportar como erro.**
2. Imprimir o bloco abaixo, com título, body-file e reviewer já preenchidos,
   pronto para copiar e rodar no PowerShell local:

```
─────────────────────────────────────────────
PR pronto para abrir. Rode no PowerShell:

gh pr create `
  --title "<título>" `
  --body-file .github/pull_request_template.md `
  --reviewer christianvanbellen

Ou abra manualmente:
https://github.com/christianvanbellen/mlops-demo-health-pressure-risk/compare/<branch>?expand=1
─────────────────────────────────────────────
```

### Regras adicionais

- **Nunca** commitar diretamente na `main`
- **Nunca** abrir PR sem `--reviewer christianvanbellen`
- **Um PR por tarefa** — não acumular alterações não relacionadas
