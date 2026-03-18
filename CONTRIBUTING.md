# Contribuindo com o projeto

## Setup do ambiente local

### Pré-requisitos
- Python 3.11+
- [UV](https://docs.astral.sh/uv/) instalado:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Instalação

```bash
# Clona o repositório
git clone <repo-url>
cd mlops-demo-health-pressure-risk

# Cria o virtualenv e instala todas as dependências
uv sync --all-extras

# Ativa o virtualenv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Valida a instalação
uv run python -c "import lightgbm; import mlflow; print('OK')"
```

## Comandos úteis

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Testes
uv run pytest tests/ -v

# Deploy do bundle Databricks (dev)
uv run databricks bundle validate
uv run databricks bundle deploy --target dev

# Gerar lock file após alterar pyproject.toml
uv lock
```

## Estrutura do projeto

```
src/
  ingestion/     → scripts de ingestão Bronze
  transforms/    → Bronze → Silver → Gold
  training/      → treino e avaliação de modelos
  scoring/       → batch scoring semanal
  monitoring/    → monitor de performance + trigger

resources/
  jobs.yml       → Databricks Asset Bundles (DAB)

docs/
  pipeline_limitations.md
```

## Adicionando dependências

```bash
# Dependência de produção
uv add <pacote>

# Dependência de desenvolvimento apenas
uv add --dev <pacote>

# Após adicionar, commitar pyproject.toml E uv.lock
git add pyproject.toml uv.lock
git commit -m "deps: add <pacote>"
```
