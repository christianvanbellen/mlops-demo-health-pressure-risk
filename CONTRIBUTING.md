# Guia do Desenvolvedor — Radar de Pressão Assistencial

## Pré-requisitos

- Git
- Python 3.11+
- UV 0.9.0+ (gerenciador de dependências)
- Databricks CLI standalone (não via pip)
- Acesso ao workspace Databricks (solicitar ao tech lead)

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
4. Push: `git push origin feature/minha-feature`
5. Abrir PR no GitHub usando o template
6. Aguardar CI passar (lint + testes + bundle validate)
7. Solicitar review

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
