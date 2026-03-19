# Guia do Desenvolvedor — Radar de Pressão Assistencial

<!--
  NOTA: Este projeto usa Claude Code (IA assistida) para desenvolvimento.
  A gh CLI é usada pelo Claude Code para abrir pull requests automaticamente,
  atribuir revisor (christianvanbellen) e seguir o template de PR padrão.
  Instale e autentique a gh CLI antes de iniciar (ver seção abaixo).
-->

## Pré-requisitos

- Git
- Python 3.11+
- UV 0.9.0+ (gerenciador de dependências)
- Databricks CLI standalone (não via pip)
- **gh CLI** (GitHub CLI) — necessário para abrir PRs automaticamente
- Acesso ao workspace Databricks (solicitar ao tech lead)

## Instalação do gh CLI

A `gh` CLI é usada pelo fluxo de desenvolvimento assistido por IA (Claude Code)
para abrir pull requests automaticamente com o template padrão e atribuir revisor.

### macOS

```bash
brew install gh
```

### Linux (Debian/Ubuntu)

```bash
sudo apt install gh
# ou, via repositório oficial:
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh
```

### Windows

```powershell
winget install GitHub.cli
```

### Autenticação

```bash
gh auth login
# Escolher: GitHub.com → HTTPS → Login with a web browser
```

Verificar: `gh auth status`

### Configurar revisor padrão (uma vez por repositório)

O revisor `christianvanbellen` é adicionado automaticamente pelo Claude Code
via `--reviewer` no `gh pr create`. Nenhuma configuração adicional necessária —
basta estar autenticado.

---

## Instalação do UV

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verificar: `uv --version` (esperado: 0.9.0+)

## Instalação do Databricks CLI

### Todos os sistemas operacionais

```bash
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks --version  # esperado: 0.286.0+
```

> **Nota:** NÃO instalar via pip/uv — o Databricks CLI standalone é diferente do
> `databricks-cli` PyPI (v0.18, descontinuado).

## Setup do projeto

### Windows (PowerShell)

```powershell
git clone <repo-url>
cd mlops-demo-health-pressure-risk
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv sync --all-extras
```

### Linux / macOS

```bash
git clone <repo-url>
cd mlops-demo-health-pressure-risk
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-extras
```

### Validação

```bash
uv run python -c "import lightgbm, mlflow, sklearn; print('OK')"
uv run pytest tests/ -v
databricks bundle validate --target dev
```

## Configuração do Databricks

Solicitar ao tech lead:
- `DATABRICKS_HOST`
- `DATABRICKS_CLIENT_ID`
- `DATABRICKS_CLIENT_SECRET`

Configurar como variáveis de ambiente ou em `~/.databrickscfg`.

---

## Configuração da aplicação (ConfigLoader)

O `src/config.py` resolve cada variável em cascata, nesta ordem de prioridade:

1. **Spark conf** — `spark.conf.get("health.pressure.<key>")`
   Configure no cluster Databricks → Configuração → Spark Config
2. **Env var** — `os.environ.get("<KEY>")`
   Configure no cluster Databricks → Configuração → Environment Variables
3. **Default hardcodado** — valores de desenvolvimento (definidos em `src/config.py`)
   Ativo automaticamente quando nenhuma das camadas anteriores estiver presente

Se uma variável não for encontrada em nenhuma camada e não tiver default,
o `ConfigLoader` levanta `ValueError` explícito indicando qual variável faltou.

### Rodando scripts localmente (fora do cluster)

Os defaults cobrem o ambiente de desenvolvimento. Se precisar sobrescrever
algum valor, use os scripts abaixo **antes** de executar qualquer script `src/`:

#### bash / zsh

```bash
source scripts/setup-dev.sh
```

Ou exportando variáveis individuais:

```bash
export CATALOG="ds_dev_db"
export SCHEMA="dev_christian_van_bellen"
export TABLE_BRONZE_SRAG="ds_dev_db.dev_christian_van_bellen.bronze_srag"
export TABLE_BRONZE_HOSPITAIS_LEITOS="ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos"
export TABLE_BRONZE_CNES="ds_dev_db.dev_christian_van_bellen.bronze_cnes_estabelecimentos"
export TABLE_SILVER_SRAG="ds_dev_db.dev_christian_van_bellen.silver_srag_municipio_semana"
export TABLE_SILVER_CAPACITY="ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_mes"
export TABLE_GOLD_FEATURES="ds_dev_db.dev_christian_van_bellen.gold_pressure_features"
export TABLE_GOLD_SCORING="ds_dev_db.dev_christian_van_bellen.gold_pressure_scoring"
export TABLE_GOLD_MONITOR="ds_dev_db.dev_christian_van_bellen.monitoring_performance"
export LANDING_PATH="/Volumes/ds_dev_db/dev_christian_van_bellen/landing"
export MLFLOW_EXPERIMENT="/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr"
export MODEL_NAME="ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier"
export RETRAIN_JOB_NAME="job_health_pressure_retrain"
export TRAIN_END="202412"
export VAL_END="202506"
export TEST_START="202507"
export TARGET_PERCENTILE="0.85"
export PRECISION_K_THRESHOLD="0.55"
export MIN_CONSECUTIVE_BELOW="2"
export SCORING_MIN_QUALITY="0.5"
export AB_CHALLENGER_PCT="0.20"
export DRIFT_SEASONAL_FEATURES="mes,quarter,is_semester1,is_rainy_season"
export LGBM_PARAMS_JSON='{"objective":"binary","metric":"binary_logloss","boosting_type":"gbdt","num_leaves":63,"learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"min_child_samples":20,"reg_alpha":0.1,"reg_lambda":0.1,"verbose":-1}'
export NUM_BOOST_ROUND="500"
export EARLY_STOPPING="50"
```

#### PowerShell

```powershell
. .\scripts\setup-dev.ps1
```

Ou definindo variáveis individuais:

```powershell
$env:CATALOG = "ds_dev_db"
$env:SCHEMA  = "dev_christian_van_bellen"
$env:TABLE_BRONZE_SRAG             = "ds_dev_db.dev_christian_van_bellen.bronze_srag"
$env:TABLE_BRONZE_HOSPITAIS_LEITOS = "ds_dev_db.dev_christian_van_bellen.bronze_hospitais_leitos"
$env:TABLE_BRONZE_CNES             = "ds_dev_db.dev_christian_van_bellen.bronze_cnes_estabelecimentos"
$env:TABLE_SILVER_SRAG     = "ds_dev_db.dev_christian_van_bellen.silver_srag_municipio_semana"
$env:TABLE_SILVER_CAPACITY = "ds_dev_db.dev_christian_van_bellen.silver_capacity_municipio_mes"
$env:TABLE_GOLD_FEATURES = "ds_dev_db.dev_christian_van_bellen.gold_pressure_features"
$env:TABLE_GOLD_SCORING  = "ds_dev_db.dev_christian_van_bellen.gold_pressure_scoring"
$env:TABLE_GOLD_MONITOR  = "ds_dev_db.dev_christian_van_bellen.monitoring_performance"
$env:LANDING_PATH      = "/Volumes/ds_dev_db/dev_christian_van_bellen/landing"
$env:MLFLOW_EXPERIMENT = "/Users/christian.bellen@indicium.tech/pressure-risk-baseline-lr"
$env:MODEL_NAME        = "ds_dev_db.dev_christian_van_bellen.pressure_risk_classifier"
$env:RETRAIN_JOB_NAME  = "job_health_pressure_retrain"
$env:TRAIN_END  = "202412"
$env:VAL_END    = "202506"
$env:TEST_START = "202507"
$env:TARGET_PERCENTILE      = "0.85"
$env:PRECISION_K_THRESHOLD  = "0.55"
$env:MIN_CONSECUTIVE_BELOW  = "2"
$env:SCORING_MIN_QUALITY    = "0.5"
$env:AB_CHALLENGER_PCT      = "0.20"
$env:DRIFT_SEASONAL_FEATURES = "mes,quarter,is_semester1,is_rainy_season"
$env:LGBM_PARAMS_JSON = '{"objective":"binary","metric":"binary_logloss","boosting_type":"gbdt","num_leaves":63,"learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"min_child_samples":20,"reg_alpha":0.1,"reg_lambda":0.1,"verbose":-1}'
$env:NUM_BOOST_ROUND  = "500"
$env:EARLY_STOPPING   = "50"
```

### Referência completa das variáveis

Consultar `conf/dev.yml` e `conf/prod.yml` — cada variável está anotada com
o nome da Spark conf key e o nome da env var equivalente.

## Fluxo de trabalho

### Branches

- `main`: branch protegida, requer PR aprovada
- `feature/nome-da-feature`: desenvolvimento
- `fix/nome-do-bug`: correções

### Commits

Usar conventional commits:

```
feat:     nova funcionalidade
fix:      correção de bug
docs:     documentação
ci:       pipeline CI/CD
deps:     dependências
refactor: refatoração sem mudança de comportamento
```

### Abrindo uma PR

1. Criar branch: `git checkout -b feature/minha-feature`
2. Desenvolver e testar localmente
3. Rodar checks: `uv run ruff check src/ && uv run pytest tests/`
4. Push: `git push -u origin feature/minha-feature`
5. Abrir PR via gh CLI (usa o template `.github/pull_request_template.md`):

```bash
gh pr create \
  --title "feat: descrição da mudança" \
  --body-file .github/pull_request_template.md \
  --reviewer christianvanbellen
```

6. Aguardar CI passar (lint + testes + bundle validate)
7. Review e merge

> **Claude Code:** ao abrir PRs automaticamente, usa `gh pr create` com
> `--reviewer christianvanbellen` e o body gerado a partir do template.

### Deploy

- **dev**: automático em PRs (`dab-deploy-dev`)
- **prod**: automático em merge para `main` com approval gate

## Comandos úteis

| Comando | Descrição |
|---|---|
| `uv sync --all-extras` | Instala/atualiza todas as dependências |
| `uv run ruff check src/` | Lint |
| `uv run ruff format src/` | Formata o código |
| `uv run pytest tests/ -v` | Roda os testes |
| `databricks bundle validate` | Valida o bundle DAB |
| `databricks bundle deploy --target dev` | Deploy em dev |
| `uv add <pacote>` | Adiciona dependência de produção |
| `uv add --dev <pacote>` | Adiciona dependência de dev |
| `uv lock` | Regera o uv.lock |
| `gh pr create --reviewer christianvanbellen` | Abre PR com revisor padrão |
| `gh pr list` | Lista PRs abertas |
| `gh pr status` | Status da PR da branch atual |
| `gh auth status` | Verifica autenticação do gh CLI |

## Adicionando dependências

Sempre commitar `pyproject.toml` E `uv.lock` juntos:

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add <pacote>"
```

## Estrutura do projeto

```
src/
  ingestion/   → scripts de ingestão Bronze
  transforms/  → Bronze → Silver → Gold
  training/    → treino e avaliação de modelos
  scoring/     → batch scoring semanal
  monitoring/  → monitor de performance + retrain trigger

resources/
  jobs.yml     → Databricks Asset Bundles (2 Jobs)

docs/
  architecture.md         → arquitetura técnica e ML lifecycle
  data_sources.md         → fontes de dados e ingestão
  decisions.md            → decisões de design (ADRs)
  pipeline_limitations.md → limitações e horizonte operacional

tests/
  test_batch_score.py     → testes do scoring
  test_gold_features.py   → testes do trigger e métricas
```
