# Decisões de Design — Radar de Pressão Assistencial

Registro de decisões técnicas usando formato ADR (Architecture Decision Record) simplificado.
Cada ADR documenta contexto, decisão, justificativa e trade-offs.

---

## ADR-001: Grain mensal em vez de semanal

**Status:** Aceito

**Contexto:**
O SRAG é publicado com granularidade de semana epidemiológica e a silver
`silver_srag_municipio_semana` preserva esse grain. A questão era em qual grain
construir a feature table e o modelo.

**Decisão:**
O grain analítico da Gold é `municipio_id × competencia (AAAAMM)` — **mensal**,
não semanal. A silver do SRAG é agregada para mensal antes do join com capacity.

**Justificativa:**
- A granularidade semanal amplifica o atraso de notificação do SRAG — as últimas
  2–4 semanas de cada ano estão cronicamente incompletas (< 60% dos casos notificados)
- Agregação mensal suaviza esse ruído e alinha naturalmente com a fonte de capacidade
  (Hospitais e Leitos), que é publicada mensalmente por competência
- Reduz ruído cadastral de estabelecimento individual
- Facilita explicação e visualização para gestores municipais

**Trade-offs:**
- Perde resolução temporal para detectar surtos intra-mensais
- Aumenta o atraso estrutural do pipeline (já documentado em `pipeline_limitations.md`)

---

## ADR-002: Capacity LEFT JOIN SRAG

**Status:** Aceito

**Contexto:**
Ao construir a feature table, precisava-se definir qual tabela seria a base do join
entre capacidade e demanda. A alternativa era usar SRAG como base (inner join ou
SRAG left) ou Capacity como base.

**Decisão:**
`silver_capacity_municipio_mes` é a tabela base (LEFT), com `silver_srag` como
tabela direita. Municípios com leitos mas sem casos SRAG no mês ficam com demanda
zerada — não são excluídos.

**Justificativa:**
- Garante cobertura de todos os ~3.330 municípios com `leitos_totais >= 10` em toda
  competência disponível, inclusive para scoring ao vivo
- Municípios sem casos SRAG recentes são relevantes para o modelo — ausência de sinal
  é informação (baixa pressão)
- Inner join excluiria municípios com leitos mas sem internações SRAG no período,
  criando viés de seleção no treino

**Trade-offs:**
- Municípios sem casos SRAG têm features de demanda zeradas — o modelo precisa
  aprender que zero é baixa pressão, não dado ausente
- Exige tratamento de nulls nas features de lag/MA para meses sem casos

---

## ADR-003: Target por percentil relativo (P85)

**Status:** Aceito (v2 ativa)

**Contexto:**
Precisava-se definir o que constitui "alta pressão assistencial". As alternativas eram:
threshold absoluto de casos, taxa de ocupação real de leitos, ou métrica relativa percentilizada.

**Decisão:**
```
target_alta_pressao(m, t) = 1  se  casos_por_leito(m, t+1) >= P85_nacional(t+1)

casos_por_leito = casos_srag_mes / (leitos_totais + 1)
P85 = percentil 85 sobre todos os municípios da mesma competência t+1
```
`target_definition_version = "v2"`, `target_percentile = 0.85`

**Justificativa:**
- **Métrica relativa captura crise local:** São Paulo com 37.000 leitos e 8.000 casos
  não é alerta; um município com 10 leitos e 17 casos é crise. Volume absoluto não captura isso.
- **Percentil relativo é governável:** o corte em P85 mantém ~15% de positivos por
  competência, independente do nível absoluto da temporada. Threshold absoluto variaria
  muito entre verão e inverno epidemiológico.
- **Ocupação real não disponível:** dados de ocupação hospitalar em tempo real não
  estão disponíveis de forma pública, estável e ampla para todos os municípios do Brasil.
- **Versão v1** usava a fórmula `Demand/Capacity + γ*growth + δ*regional_trend` —
  foi simplificada para v2 após validação de que `casos_por_leito` é mais estável e
  interpretável sem perder poder preditivo.

**Trade-offs:**
- O P85 é calculado sobre os municípios do conjunto de treino — há risco de instabilidade
  em competências com poucas observações (ex: início de 2026)
- "Alta pressão" é relativa à distribuição nacional do período, não a um padrão clínico absoluto

---

## ADR-004: Forward fill de capacity

**Status:** Aceito

**Contexto:**
A fonte Hospitais e Leitos tem atraso típico de 1–2 meses na publicação. Competências
recentes da SRAG não têm dados de capacidade correspondentes — precisava-se decidir
como tratar esses meses.

**Decisão:**
Propagar os dados de capacidade do último mês publicado para as competências futuras
ainda não publicadas. Linhas com capacity estimada recebem `capacity_is_forward_fill = True`
e `data_quality_score = 0.3`. Essas linhas **não são usadas para calcular o target**
mas são mantidas para scoring ao vivo.

**Justificativa:**
- Leitos cadastrados mudam muito pouco mês a mês (< 1% de variação típica) — o proxy
  é metodologicamente defensável para scoring de curto prazo
- Excluir completamente os meses sem capacity publicada eliminaria os meses mais
  recentes, justamente onde o scoring ao vivo é necessário
- A flag `capacity_is_forward_fill` permite que consumidores downstream filtrem
  ou penalizem esses registros conforme necessário

**Trade-offs:**
- Capacity estimada pode mascarar abertura ou fechamento de leitos no período de atraso
- Scores produzidos com forward fill devem ser comunicados com ressalva ao usuário final

---

## ADR-005: Clusters efêmeros em vez de cluster fixo

**Status:** Aceito

**Contexto:**
A configuração inicial dos Jobs usava `existing_cluster_id` — um cluster fixo que
precisava estar ativo antes da execução. Isso gerava custo quando o cluster ficava
idle e dependência de configuração manual do ID.

**Decisão:**
Migrar para `job_cluster_key` com `new_cluster` em cada Job. Cada execução cria e
destrói seu próprio cluster efêmero com especificação explícita:
- `spark_version: "16.4.x-cpu-ml-scala2.13"`
- `node_type_id: "i3.xlarge"`
- `num_workers: 1`

**Justificativa:**
- Clusters efêmeros só existem durante a execução — custo zero quando o job não roda
- Configuração reproduzível e versionada no `resources/jobs.yml` — não depende de
  estado externo (ID de cluster existente)
- Elimina a variável `cluster_id` do `databricks.yml` e de todos os targets (dev/prod)
- Ambiente idêntico em todas as execuções — sem risco de drift de configuração

**Trade-offs:**
- Cold start de 2–5 minutos por execução (tempo de provisioning do cluster)
- Não aproveita cache de disco entre execuções semanais

---

## ADR-006: UV como gerenciador de dependências

**Status:** Aceito

**Contexto:**
O projeto precisava de um gerenciador de dependências Python que funcionasse bem em
desenvolvimento local e em CI/CD, com lock file determinístico.

**Decisão:**
Usar UV (`astral-sh/uv`) em vez de pip, conda ou poetry. Lock file: `uv.lock`.
Instalação no CI via `astral-sh/setup-uv@v6`. Sincronização: `uv sync --all-extras --frozen`.

**Justificativa:**
- `uv` é ordens de magnitude mais rápido que pip para instalação
- `uv.lock` é determinístico — CI usa exatamente o mesmo ambiente que o desenvolvedor
- `--frozen` no CI garante que o lock file não diverge do `pyproject.toml`
- Gerencia também a versão do Python (`uv python install 3.11`)
- Substituição do `databricks-cli` pip pelo CLI standalone separa as responsabilidades:
  UV gerencia dependências Python; Databricks CLI é instalado via script oficial

**Trade-offs:**
- UV é relativamente novo — API pode mudar entre versões (fixar `UV_VERSION = "0.9.0"` no CI)
- Databricks Runtime no cluster usa pip nativo — `uv` só é relevante para ambiente local e CI

---

## ADR-007: DAB em vez de JSON manual para Jobs

**Status:** Aceito

**Contexto:**
Jobs Databricks podem ser definidos via UI, JSON manual via API ou Databricks Asset
Bundles (DAB) em YAML. O projeto precisava de Jobs versionados e deployáveis via CI.

**Decisão:**
Usar Databricks Asset Bundles (DAB) com definição em `resources/jobs.yml` e
`databricks.yml` como bundle root. Deploy via `databricks bundle deploy --target dev/prod`.

**Justificativa:**
- YAML versionado no Git — histórico de mudanças nos Jobs junto ao código
- `databricks bundle validate` no CI detecta erros de configuração antes do deploy
- Targets `dev` e `prod` com variáveis distintas (`catalog`, `schema`) sem duplicação
- Suporte nativo a job clusters efêmeros, parâmetros, schedules e email notifications
- Alternativa JSON manual exigiria manutenção de dois estados (código + UI/API)

**Trade-offs:**
- Paths de arquivos Python são relativos à localização do `jobs.yml` (em `resources/`),
  não à raiz do bundle — exige prefixo `../` em todos os `python_file` paths
- DAB é Databricks-specific — não portável para outros orquestradores

---

## ADR-008: Precision@K como métrica primária de monitoramento

**Status:** Aceito

**Contexto:**
Precisava-se escolher qual métrica monitorar continuamente para detectar degradação
do modelo e disparar retraining. As candidatas eram AUC-ROC, AUC-PR, Precision@K e F1.

**Decisão:**
Usar `Precision@K` com `K = 15%` (top ~500 municípios de ~3.330) como métrica
primária de monitoramento. Threshold de alerta: `Precision@K < 0.55`.

**Justificativa:**
- **Mede utilidade operacional real:** K=15% representa o número de municípios que
  um gestor de saúde estadual ou federal monitoraria ativamente num ciclo mensal.
  Precisão nesses K municípios reflete diretamente o valor prático do modelo.
- **AUC-ROC é insensível a degradação local:** um modelo pode manter AUC-ROC alto
  enquanto erra sistematicamente nos municípios de maior risco
- **Calculável via backtesting:** o target realizado de competência T+1 fica disponível
  com 1–2 meses de atraso, permitindo backtesting contínuo

**Trade-offs:**
- K fixo em 15% pode não ser o limiar certo para todos os contextos operacionais
- Precision@K ignora recall — um modelo que acerta os K top municípios mas perde
  municípios críticos fora do top K não seria penalizado
- Threshold de 0.55 é heurístico — precisaria de calibração com dados operacionais reais

---

## ADR-009: Regra de 2 competências consecutivas para trigger

**Status:** Aceito

**Contexto:**
Precisava-se definir quando o monitoramento de performance deve disparar um
retraining automático. Retraining muito frequente é caro; retraining tardio deixa
um modelo degradado em produção.

**Decisão:**
Dois critérios de trigger, com OR lógico:
1. **Regra principal:** `Precision@K < 0.55` por **2+ competências consecutivas**
2. **Regra secundária:** queda > 15pp vs. média das competências anteriores (queda abrupta em 1 competência)

Quando disparado, o trigger chama a Jobs REST API para iniciar `job_health_pressure_retrain`.

**Justificativa:**
- Uma competência abaixo do threshold pode ser variação normal ou anomalia pontual
  de dados — 2 consecutivas indica tendência real de degradação
- A regra de queda abrupta captura mudanças súbitas (ex: nova variante, mudança na
  codificação do SRAG) que a regra de consecutivas demoraria 2 meses para detectar
- O trigger passa `trigger_reason`, `trigger_date` e `retrain_id` como parâmetros
  para rastreabilidade no MLflow

**Trade-offs:**
- Com frequência mensal de monitoramento, 2 competências consecutivas = 2 meses de
  modelo degradado antes do trigger — aceitável para planejamento operacional, não
  para resposta em tempo real
- Falsos triggers podem ser custosos — o human gate (ADR-010) protege contra isso

---

## ADR-010: Human gate para promoção de champion

**Status:** Aceito

**Contexto:**
Precisava-se decidir se a promoção de um novo modelo de `@challenger` para `@champion`
seria automática (baseada em métricas) ou exigiria aprovação humana.

**Decisão:**
Promoção **nunca automática** após o primeiro deploy. O fluxo é:
1. `evaluate.py` compara o novo modelo com `@champion` atual
2. Se novo modelo > champion: registra como `@challenger` no Unity Catalog
3. Revisor humano avalia no MLflow UI: histórico de métricas, curvas P@K, sumário do modelo
4. Promoção manual via MLflow UI: atribuição do alias `@champion` ao challenger

Único caso automático: `first_deploy` — quando não existe `@champion`, o melhor
modelo é promovido diretamente (nenhum usuário afetado ainda).

**Justificativa:**
- Modelo de saúde pública com impacto direto em alocação de recursos — erro em produção
  pode custar vidas ou desperdiçar recursos escassos
- Métricas de holdout podem não capturar mudanças de distribuição em competências recentes
- O revisor humano pode contextualizar anomalias epidemiológicas que o modelo não conhece
  (ex: nova variante, subnotificação anômala, mudança de protocolo de notificação)
- GitHub Environments `production` impõe o mesmo princípio no CI: deploy em prod
  requer aprovação antes de executar

**Trade-offs:**
- Introduz latência de 1–5 dias entre treino e deploy em produção
- Depende de disponibilidade do revisor — bottleneck humano no ciclo de retraining

---

## Decisões de implementação

Registro de achados técnicos durante o desenvolvimento que resultaram em mudanças
de abordagem.

### IMPL-001 — DBFS público desabilitado

O DBFS público estava desabilitado no workspace Databricks. Arquivos temporários
de ingestão devem usar o Volume:
`/Volumes/ds_dev_db/dev_christian_van_bellen/landing/`
em vez de `/tmp/`.

### IMPL-002 — Cast para string no Bronze

Parquet de 2025/2026 do SRAG tem tipos mistos (timestamp, decimal). A tentativa de
gravar com `mergeSchema=true` sobre uma tabela com colunas string gerava conflito
de schema. Padrão consolidado: ler qualquer formato → castear tudo para string →
gravar no Bronze.

### IMPL-003 — DELETE em tabela vazia

`DELETE WHERE _ano_arquivo = {ano}` falha em tabela vazia no Spark Connect com erro
de predicado sem resultado. Envolvido em `try/except` com log de aviso — comportamento
esperado na primeira carga.

### IMPL-004 — Hospitais e Leitos: separadores por período

Descoberto em execução: 2023/2024 usa CSV direto com separador vírgula (`,`);
2025/2026 usa ZIP contendo CSV com separador ponto-e-vírgula (`;`). O código
configura `sep` por ano via dicionário `ANOS`.

### IMPL-005 — SRAG 2025/2026 em Parquet

Descoberto em execução: a partir de 2025, o OpenDataSUS disponibiliza o SRAG em
Parquet (não CSV). O script `srag_ingest.py` detecta o formato pelo nome do arquivo
e usa `spark.read.parquet()` para os anos live.

### IMPL-006 — CO_IBGE ausente em Hospitais e Leitos 2023/2024

Os arquivos históricos (2023/2024) não contêm a coluna `CO_IBGE`. O join com
`municipio_id` nesses anos é feito na Silver via lookup `UF + nome normalizado →
municipio_id` usando a tabela `silver_dim_municipio`.

### IMPL-007 — UF fora do groupBy no Silver SRAG

Alguns municípios têm casos de SRAG notificados em UFs diferentes da residência
(ex: Brasília aparece com BA, GO e DF). Incluir `uf` no `groupBy` gerava múltiplas
linhas por `municipio_id × competencia`, quebrando a PK da Feature Store. Solução:
`F.first("uf", ignorenulls=True)` fora do `groupBy`.

### IMPL-008 — Paths relativos no DAB

O DAB resolve `python_file` paths relativos à localização do arquivo `jobs.yml`
(em `resources/`), não à raiz do bundle. Todos os paths foram prefixados com `../`
(ex: `../src/ingestion/srag_ingest.py`).
