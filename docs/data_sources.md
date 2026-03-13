# Fontes de dados — health-pressure-risk

## Fonte 1 — SRAG / SIVEP-Gripe

**Função no projeto:** sinal principal de demanda grave assistencial

**URL base:**
```
https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/{ANO}/INFLUD{AA}-{DATA}.csv
```

**Importante:** o nome do arquivo inclui a data da última atualização.
Verificar sempre a URL atual no portal antes de rodar a ingestão:
https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024

**Formato:** CSV, separador `;`, encoding `latin1`

**Granularidade:** caso individual hospitalizado

**Atualização:** semanal (banco "vivo" do ano corrente); anos anteriores congelados

**Colunas essenciais utilizadas:**

| Coluna | Descrição |
|---|---|
| `DT_NOTIFIC` | Data de notificação |
| `DT_SIN_PRI` | Data dos primeiros sintomas |
| `CO_MUN_RES` | Município de residência (IBGE) |
| `CO_MUN_NOT` | Município de notificação |
| `SG_UF_NOT` | UF de notificação |
| `NU_IDADE_N` | Idade |
| `CS_SEXO` | Sexo |
| `HOSPITAL` | Internado (1=sim) |
| `UTI` | Foi para UTI (1=sim) |
| `EVOLUCAO` | Evolução (1=cura, 2=óbito) |
| `CLASSI_FIN` | Classificação final |
| `SEM_NOT` | Semana epidemiológica de notificação |
| `SEM_PRI` | Semana epidemiológica dos primeiros sintomas |

**Quirks e cuidados:**
- Semanas recentes estão **sujeitas a revisão e atraso de notificação**
- Sempre reprocessar as últimas 2–4 semanas
- Excluir semana corrente do treino (dados incompletos)
- Completude de campos varia bastante — validar nulls críticos
- `CO_MUN_RES` pode ter 6 ou 7 dígitos dependendo do ano — normalizar para 6

---

## Fonte 2 — CNES Estabelecimentos

**Função no projeto:** capacidade estrutural (tipologia, natureza, localização)

**URL:** https://opendatasus.saude.gov.br/dataset/cnes

**Formato:** CSV, separador `;`, encoding `latin1`

**Granularidade:** estabelecimento de saúde

**Atualização:** mensal (por competência)

**Colunas essenciais utilizadas:**

| Coluna | Descrição |
|---|---|
| `CNES` | Código CNES do estabelecimento |
| `CODUFMUN` | Código IBGE do município |
| `NOMEMUN` | Nome do município |
| `COMP` | Competência (AAAAMM) |
| `TP_UNIDADE` | Tipo de unidade |
| `NAT_JUR` | Natureza jurídica |
| `ESFERA_ADM` | Esfera administrativa |

**Quirks e cuidados:**
- CNES reflete **capacidade instalada/cadastrada**, não ocupação real
- Deixar isso explícito em comentários e documentação do modelo
- Estabelecimentos podem mudar de tipo ao longo do tempo — usar snapshot por competência
- Join com município via `CODUFMUN` (6 dígitos)

---

## Fonte 3 — Hospitais e Leitos

**Função no projeto:** capacidade hospitalar simplificada (leitos gerais e complementares)

**URL:** https://opendatasus.saude.gov.br/dataset/hospitais-e-leitos

**Formato:** CSV, separador `;`, encoding `latin1`

**Granularidade:** estabelecimento hospitalar

**Atualização:** mensal

**Colunas essenciais utilizadas:**

| Coluna | Descrição |
|---|---|
| `CNES` | Código CNES |
| `CODUFMUN` | Código IBGE do município |
| `LEITOS_TOTAL` | Total de leitos |
| `LEITOS_COMPL` | Leitos complementares |
| `UTI_TOTAL_E` | Total de leitos UTI existentes |
| `UTI_TOTAL_SUS` | Leitos UTI SUS |

**Quirks e cuidados:**
- Usar como fonte principal de leitos no MVP (mais simples que CNES completo)
- Harmonizar com CNES quando necessário na fase 2
- Leitos UTI têm alta variabilidade entre municípios pequenos — tratar zeros com cuidado
  (divisor +1 na fórmula do PressureScore evita divisão por zero)

---

## Fonte 4 — InfoGripe / Fiocruz (opcional)

**Função no projeto:** tendência epidemiológica regional

**URL:** http://info.gripe.fiocruz.br

**Formato:** API ou download de boletins

**Atualização:** semanal

**Uso no projeto:**
- Enriquecer feature `regional_alert_signal`
- Validação qualitativa de tendência
- Tratada como fonte opcional — não é dependência do MVP

---

## Normalização de municípios

Todos os joins entre fontes devem usar o código IBGE de 6 dígitos como chave.

Regras de normalização:
```python
# Remove dígito verificador se tiver 7 dígitos
municipio_id = col("CO_MUN_RES").cast("string")
municipio_id = when(length(municipio_id) == 7,
                    municipio_id.substr(1, 6)
               ).otherwise(municipio_id)

# Garante zero-padding para 6 dígitos
municipio_id = lpad(municipio_id, 6, "0")
```

Tabela de referência de municípios: `silver_dim_municipio`
- Fonte: IBGE
- Campos: `municipio_id`, `municipio_nome`, `uf`, `regiao`

---

## Semana epidemiológica

O Brasil usa o calendário epidemiológico do Ministério da Saúde.
A semana começa no domingo e termina no sábado.

Derivação no PySpark:
```python
from pyspark.sql import functions as F

# A partir de uma data
df = df.withColumn("semana_epidemiologica",
        F.concat(
            F.year("DT_NOTIFIC").cast("string"),
            F.lpad(F.weekofyear("DT_NOTIFIC").cast("string"), 2, "0")
        ))
# Resultado: "202301", "202302", ...
```

Quando a coluna `SEM_NOT` já estiver disponível no SRAG, usá-la diretamente
pois já está no calendário epidemiológico correto.

---

## Estrutura dos arquivos SRAG

- Cada arquivo corresponde a um **ano epidemiológico**, não ano calendário
- Datas transbordam entre arquivos por atraso de notificação (esperado)
- 2023 e 2024 estão **congelados** — ingeridos uma vez
- 2025 e 2026 são **bancos vivos** — reprocessados semanalmente
- Sobreposição entre arquivos é esperada — **deduplicação obrigatória no silver**
- Volumes aproximados: 2023=279k, 2024=268k, 2025=336k, 2026=25k (mar/2026)
