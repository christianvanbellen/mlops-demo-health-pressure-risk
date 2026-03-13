# Log de decisões técnicas — health-pressure-risk

Registro cronológico de decisões de arquitetura, modelagem e engenharia de dados.
Formato: data | decisão | alternativas consideradas | justificativa.

---

## 2026-03

### DEC-001 — Grain analítico: município × semana epidemiológica

**Decisão:** usar `municipio_id × semana_epidemiologica` como unidade analítica principal.

**Alternativas consideradas:**
- Estabelecimento × semana (mais granular)
- Região de saúde × semana (menos granular)

**Justificativa:**
- Reduz ruído cadastral de hospital individual
- Facilita join com séries epidemiológicas públicas
- Simplifica explicação para gestores
- Preserva valor operacional
- Permite expansão futura para grain mais fino (fase 2)

---

### DEC-002 — Target: proxy via PressureScore percentilizado

**Decisão:** definir target como entrada no percentil 85 nacional do PressureScore
futuro (semana t+1), não como ocupação real hospitalar.

**Alternativas consideradas:**
- Ocupação real (taxa de ocupação de leitos)
- Threshold absoluto de casos por leito

**Justificativa:**
- Ocupação real não está disponível de forma pública, estável e ampla
- Percentil relativo é mais honesto e mais fácil de governar
- Framing correto: "risco de pressão assistencial relativa", não "colapso hospitalar"
- Permite ajuste futuro do percentil de corte sem mudar a arquitetura

**Versão atual:** `target_definition_version = 'v1'`
**Parâmetros iniciais:** α=0.5, β=2.0, γ=0.3, percentil=85

---

### DEC-003 — Fontes de dados: SRAG + Hospitais e Leitos como MVP

**Decisão:** usar SRAG/SIVEP-Gripe e Hospitais e Leitos como fontes principais do MVP.
CNES completo e InfoGripe como fontes opcionais/fase 2.

**Justificativa:**
- Hospitais e Leitos é mais simples de processar que CNES completo para o MVP
- SRAG é o sinal de demanda mais direto e confiável disponível publicamente
- Reduz complexidade de joins na primeira iteração
- InfoGripe útil para tendência regional mas não essencial para o modelo base

---

### DEC-004 — Stack: Databricks (AWS) + Unity Catalog + MLflow

**Decisão:** usar Unity Catalog para governança de dados e modelos, MLflow nativo
para tracking, batch scoring semanal como modo de serving principal.

**Alternativas consideradas:**
- Endpoint REST em tempo real (Mosaic AI Model Serving)

**Justificativa:**
- Batch scoring semanal é suficiente para o caso de uso (priorização operacional)
- Unity Catalog centraliza lineage de dados e modelos
- MLflow nativo evita dependência de ferramentas externas
- Endpoint REST fica como opcional para fase 3

---

### DEC-005 — Split temporal, nunca aleatório

**Decisão:** usar time-based split para treino/validação/teste.

**Split definido:**
- Treino: 2023–2024
- Validação: 2025 H1
- Teste: 2025 H2
- Scoring live: 2026

**Cross-validation:** rolling windows ou expanding windows.

**Justificativa:**
- Dados temporais têm dependência serial — split aleatório causa data leakage
- Reproduz cenário real: modelo treinado no passado, avaliado no futuro

---

### DEC-006 — Modelos: LR como baseline, GBT como principal

**Decisão:** Logistic Regression como baseline interpretável, LightGBM/XGBoost como
modelo principal.

**Justificativa:**
- LR é simples, rápido e serve como referência de comparação
- GBT lida bem com não-linearidade e interações em dados tabulares temporais
- Calibração posterior (Platt ou Isotonic) se o score precisar ser interpretado
  como probabilidade operacional

**Critério de promoção do champion:**
- Modelo novo só vira champion se melhorar `precision@top_k` sem piorar calibração significativamente

---

### DEC-007 — Repositório: GitHub pessoal + integração Databricks

**Decisão:** usar GitHub pessoal (`christianvanbellen/mlops-demo-health-pressure-risk`)
como repositório principal, com duas chaves SSH configuradas (corporativa e pessoal).

**Configuração SSH:**
- Chave pessoal: `~/.ssh/id_ed25519_github`
- Alias no config: `github-pessoal`
- Clone via: `git@github-pessoal:christianvanbellen/mlops-demo-health-pressure-risk.git`

**Integração Databricks:**
- Repo espelhado via Databricks Repos (Git integration)
- Token: GitHub Personal Access Token com escopo `repo`

---

### DEC-008 — Contexto do projeto preservado via CLAUDE.md + docs/

**Decisão:** documentar todo o contexto do projeto em arquivos versionados no repo
para preservar continuidade entre sessões do Claude Code.

**Arquivos:**
- `CLAUDE.md` — instruções operacionais e convenções para o Claude Code
- `docs/architecture.md` — decisões de arquitetura e modelo de dados
- `docs/data_sources.md` — fontes, URLs, quirks e normalização
- `docs/decisions.md` — este arquivo, log de decisões técnicas

**Justificativa:**
- Claude Code lê `CLAUDE.md` automaticamente ao iniciar
- Contexto fica no repo, independente de ferramenta ou sessão
- Facilita onboarding de novos colaboradores
