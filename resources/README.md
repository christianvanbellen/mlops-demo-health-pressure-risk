# Resources — Databricks Asset Bundles

Esta pasta contém as definições de Jobs do projeto
em formato Databricks Asset Bundles (DAB).

## Arquivos

- `jobs.yml` — definição dos dois Jobs do pipeline

## Jobs

### job_health_pressure_weekly
Pipeline semanal completo:
ingestão → silver → gold → scoring → monitoramento → trigger

Schedule: toda segunda-feira às 06h (America/Sao_Paulo)
Status inicial: PAUSED (ativar em produção)

### job_health_pressure_retrain
Pipeline de retraining:
train_lr → train_gbt → evaluate (registra @challenger)

Sem schedule — disparado pelo retrain_trigger ou manualmente.
Após execução, revisar o MLflow UI e promover @challenger
para @champion se as métricas forem satisfatórias.

## Deploy

```bash
# validar sem fazer deploy
databricks bundle validate

# deploy no target dev
databricks bundle deploy --target dev

# deploy no target prod
databricks bundle deploy --target prod
```

## Variáveis

| Variável    | Default                    | Descrição                        |
|-------------|----------------------------|----------------------------------|
| catalog     | ds_dev_db                  | Unity Catalog catalog            |
| schema      | dev_christian_van_bellen   | Schema do projeto                |
| cluster_id  | (obrigatório)              | ID do cluster DBR ML             |

Configurar cluster_id no databricks.yml ou via:
```bash
databricks bundle deploy --var cluster_id=<ID>
```
