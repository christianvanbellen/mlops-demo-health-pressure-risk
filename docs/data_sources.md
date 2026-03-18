# Fontes de Dados e Ingestão

## Visão geral

O pipeline consome três fontes públicas do Ministério da Saúde, todas disponíveis
no Portal de Dados Abertos (dadosabertos.saude.gov.br) e no S3 público do DATASUS.
As fontes cobrem dois eixos do modelo: **demanda** (SRAG — casos graves hospitalizados)
e **capacidade** (Hospitais e Leitos e CNES — leitos disponíveis por município).
A ingestão é feita em Python puro via `requests`, sem dependência de conectores
externos, e grava diretamente nas tabelas Bronze do Unity Catalog via Delta Lake.

---

## Fonte 1 — Hospitais e Leitos

| Campo | Valor |
|---|---|
| Página oficial | https://dadosabertos.saude.gov.br/dataset/hospitais-e-leitos |
| Script | `src/ingestion/hospitais_leitos_ingest.py` |
| Tabela Bronze | `bronze_hospitais_leitos` |
| Grain | estabelecimento × competência (AAAAMM) |
| Frequência da fonte | Mensal — tipicamente com atraso de ~60 dias |
| Linhas aprox. (2023–2026) | 263k |

### Formato por período

| Período | Arquivo | Separador | Encoding |
|---|---|---|---|
| 2023–2024 | `Leitos_{ano}.csv` (CSV direto) | vírgula (`,`) | latin1 |
| 2025–2026 | `Leitos_csv_{ano}.zip` (ZIP contendo CSV interno) | ponto-e-vírgula (`;`) | latin1 |

URLs base no S3:
- CSV direto: `https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/Leitos_SUS/Leitos_{ano}.csv`
- ZIP: `https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/Leitos_SUS/Leitos_csv_{ano}.zip`

As URLs são estáveis e não mudam entre atualizações — o script as monta diretamente,
sem necessidade de scraping da página.

### Colunas selecionadas

`COMP`, `REGIAO`, `UF`, `MUNICIPIO`, `CNES`, `NOME_ESTABELECIMENTO`, `TP_GESTAO`,
`CO_TIPO_UNIDADE`, `DS_TIPO_UNIDADE`, `NATUREZA_JURIDICA`, `DESC_NATUREZA_JURIDICA`,
`LEITOS_EXISTENTES`, `LEITOS_SUS`, `UTI_TOTAL_EXIST`, `UTI_TOTAL_SUS`,
`UTI_ADULTO_EXIST`, `UTI_ADULTO_SUS`, `UTI_PEDIATRICO_EXIST`, `UTI_PEDIATRICO_SUS`,
`UTI_NEONATAL_EXIST`, `UTI_NEONATAL_SUS`, `CO_IBGE` (presente apenas em 2025/2026)

### Caveats

- **Ausência de código IBGE em 2023/2024**: os arquivos históricos contêm apenas o
  nome do município (`MUNICIPIO`), sem `CO_IBGE`. O join com `municipio_id` (IBGE 6
  dígitos) é feito na camada Silver via lookup `UF + nome normalizado → municipio_id`
  usando a tabela de referência `silver_dim_municipio`.
  Em 2025/2026 o `CO_IBGE` está presente — join direto.
- **Capacidade instalada ≠ capacidade operacional**: o CNES registra leitos
  cadastrados, não leitos em operação ou ocupados. Isso deve ser explicitado em
  comentários e na documentação voltada ao usuário final.
- **Versionamento por competência**: o perfil de leitos de um município muda ao
  longo do tempo. A tabela Bronze preserva snapshots por `_snapshot_date`.
- **Leitos UTI em municípios pequenos**: alta variabilidade — o divisor `+1` na
  fórmula do PressureScore evita divisão por zero.

---

## Fonte 2 — SRAG / SIVEP-Gripe

| Campo | Valor |
|---|---|
| Página oficial | https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026 |
| Script | `src/ingestion/srag_ingest.py` |
| Tabela Bronze | `bronze_srag` |
| Grain | caso individual hospitalizado |
| Frequência da fonte | Contínua — notificações chegam ao longo do tempo |
| Linhas aprox. (2023–2026) | 908k (2023≈279k, 2024≈268k, 2025≈336k, 2026≈25k em mar/2026) |

### Formato por período

| Período | Formato | Separador | Encoding |
|---|---|---|---|
| 2023–2024 | CSV | ponto-e-vírgula (`;`) | latin1 |
| 2025–2026 | Parquet | — | — |

### Descoberta de URLs

O script faz scraping da página do OpenDataSUS para descobrir as URLs atuais de cada
ano — o nome do arquivo muda a cada atualização semanal
(ex: `INFLUD25-24-02-2025.parquet`). Em caso de falha no scraping, usa URLs de
fallback hardcoded atualizadas em 2026-03.

### Colunas selecionadas

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

### Caveats críticos

**Atraso de notificação estrutural** — as competências recentes têm casos
artificialmente baixos porque as notificações chegam continuamente após o evento:

| Defasagem | Completude aprox. |
|---|---|
| t − 30 dias | ~60% dos casos notificados |
| t − 60 dias | ~80% dos casos notificados |
| t − 90 dias | ~95% dos casos notificados |
| t − 120+ dias | praticamente consolidado |

As últimas 2–4 semanas são **sempre incompletas** e não devem ser usadas no treino.
O pipeline reprocessa essas semanas semanalmente para capturar notificações tardias.

**Semana de primeiros sintomas vs. notificação**: os casos são classificados por
semana epidemiológica de primeiros sintomas (`SEM_PRI`), não pela data de notificação.
Isso é metodologicamente correto para análise epidemiológica, mas amplifica o atraso
aparente entre o evento e a visibilidade no sistema.

**`CO_MUN_RES` pode ter 6 ou 7 dígitos** dependendo do ano — normalizar para 6 na
Silver (ver seção Normalização de municípios).

**Sobreposição entre arquivos anuais**: datas transbordam entre arquivos por atraso
de notificação — deduplicação obrigatória no Silver.

---

## Fonte 3 — CNES Estabelecimentos (Fase 2 — pausado no MVP)

| Campo | Valor |
|---|---|
| Fonte primária | FTP DATASUS — `ftp://ftp.datasus.gov.br/cnes/` |
| Fallback | https://cnes.datasus.gov.br/pages/downloads/arquivosBaseDados.jsp |
| Script | `src/ingestion/cnes_ingest.py` |
| Tabelas Bronze | `bronze_cnes_estabelecimentos`, `bronze_cnes_leitos` |
| Arquivo | `BASE_DE_DADOS_CNES_{AAAAMM}.ZIP` (~709 MB por competência) |
| Frequência da fonte | Mensal |

**Status**: pausado — `bronze_hospitais_leitos` já cobre toda a capacidade necessária
para o MVP (leitos totais, UTI, município). O CNES de estabelecimentos entra na Fase 2
para enriquecer features com número de estabelecimentos, hospitais e tipologia por município.

### Conteúdo do ZIP relevante

| Arquivo interno | Conteúdo | Tabela destino |
|---|---|---|
| `tbEstabelecimento{AAAAMM}.csv` | Cadastro de estabelecimentos (~286 MB, sep=`;`, latin1) | `bronze_cnes_estabelecimentos` |
| `tbLeito{AAAAMM}.csv` | Leitos por estabelecimento (sep=`;`, latin1) | `bronze_cnes_leitos` |

O ZIP contém ~100 CSVs; apenas os dois acima são extraídos e gravados. O ZIP e os
CSVs temporários são deletados do Volume de landing imediatamente após a gravação.

### Pendências antes de retomar (registradas em 2026-03)

- Validar nomes reais das colunas de `tbLeito` — download anterior corrompeu o ZIP
- Confirmar se `CO_MUNICIPIO_GESTOR` existe em `tbLeito` ou se é necessário join com `tbEstabelecimento`

---

## Fontes opcionais — Fase 2

| Fonte | Função | Impacto esperado |
|---|---|---|
| InfoGripe / Fiocruz | Boletins semanais com nowcasting de tendência regional | Reduziria atraso do sinal de demanda para ~1–2 semanas |
| Registro de Ocupação Hospitalar | Dados quase em tempo real de ocupação de leitos | Substituiria o proxy de capacidade instalada do CNES |

Com ambas integradas, o atraso estrutural do pipeline cairia de 4–8 semanas para
1–2 semanas. Ver detalhes em `docs/pipeline_limitations.md`.

---

## Lógica de ingestão

### Estratégia de carga

Todos os scripts seguem o mesmo padrão:

1. Descobrir URLs ou competências disponíveis (scraping ou FTP, com fallback hardcoded)
2. Baixar arquivo para o Volume de landing (`/Volumes/ds_dev_db/dev_christian_van_bellen/landing/`)
3. Ler, selecionar colunas essenciais e adicionar metadados
4. Gravar via `DELETE WHERE _ano_arquivo = {ano}` + `append` (idempotente por ano)
5. Limpar arquivos temporários do landing quando aplicável

### Frequência e seleção de anos

O job semanal (toda segunda-feira) roda os scripts com `--live`, que reprocessa apenas
os anos marcados como `is_live`:

| Ano | is_live | Comportamento |
|---|---|---|
| 2023 | False | Congelado — pulado na execução semanal |
| 2024 | False | Congelado — pulado na execução semanal |
| 2025 | True | Reingerido toda semana |
| 2026 | True | Reingerido toda semana |

Para carga inicial ou reprocessamento completo, rodar sem `--live`.

### Forward fill de capacity

Quando Hospitais e Leitos ainda não publicou o mês mais recente, o pipeline propaga
os dados do último mês disponível para as competências futuras na camada Silver.
Essas linhas são marcadas com `capacity_is_forward_fill = True` na feature table e
**não são usadas para calcular o target do modelo**. Scores produzidos com forward
fill devem ser interpretados com cautela.

### Flags de consolidação do SRAG

A coluna `srag_consolidation_flag` classifica cada competência com base na defasagem
em relação à data de processamento:

| Flag | Critério | Completude |
|---|---|---|
| `consolidado` | ≥ 90 dias de defasagem | ~95% dos casos notificados |
| `estabilizando` | ≥ 45 dias de defasagem | ~80–90% dos casos notificados |
| `recente` | < 45 dias de defasagem | Excluído do cálculo do target |

---

## Tabelas Bronze

| Tabela | Fonte | Grain | Linhas aprox. |
|---|---|---|---|
| `bronze_hospitais_leitos` | Hospitais e Leitos | estabelecimento × competência | 263k |
| `bronze_srag` | SRAG/SIVEP-Gripe | caso individual hospitalizado | 908k |
| `bronze_cnes_estabelecimentos` | CNES Estabelecimentos (Fase 2) | estabelecimento × competência | — |
| `bronze_cnes_leitos` | CNES Estabelecimentos (Fase 2) | leito × competência | — |

---

## Política de qualidade na ingestão

- **Validação mínima de linhas**: `MIN_LINHAS_VALIDAS = 1.000` — falha explícita se
  o arquivo retornar menos que isso (indica URL inválida ou arquivo corrompido)
- **Bronze é sempre string**: todas as colunas são castadas para `string` na ingestão;
  os tipos corretos são aplicados nas transformações Silver
- **Metadados obrigatórios** em todas as tabelas Bronze:

| Coluna | Conteúdo |
|---|---|
| `_snapshot_date` | Data da execução (YYYY-MM-DD) |
| `_source_url` | URL de origem do arquivo |
| `_ingestion_ts` | Timestamp exato da ingestão |
| `_is_live` | `True` para anos reprocessados semanalmente |
| `_ano_arquivo` | Ano do arquivo (inteiro) |
| `_competencia` | Competência AAAAMM (apenas CNES) |

---

## Normalização de municípios

Todos os joins entre fontes usam o código IBGE de 6 dígitos como chave (`municipio_id`).

```python
# Remove dígito verificador se tiver 7 dígitos
municipio_id = F.col("CO_MUN_RES").cast("string")
municipio_id = F.when(F.length(municipio_id) == 7,
                      municipio_id.substr(1, 6)
               ).otherwise(municipio_id)

# Garante zero-padding para 6 dígitos
municipio_id = F.lpad(municipio_id, 6, "0")
```

Tabela de referência de municípios: `silver_dim_municipio`

| Campo | Descrição |
|---|---|
| `municipio_id` | Código IBGE 6 dígitos |
| `municipio_nome` | Nome normalizado |
| `uf` | Sigla da UF |
| `regiao` | Região geográfica |

---

## Semana epidemiológica

O Brasil usa o calendário epidemiológico do Ministério da Saúde.
A semana começa no domingo e termina no sábado.

Quando a coluna `SEM_NOT` já estiver disponível no SRAG, usá-la diretamente —
já está no calendário epidemiológico correto. Para derivar a partir de uma data:

```python
# Derivação no PySpark
df = df.withColumn(
    "semana_epidemiologica",
    F.concat(
        F.year("DT_NOTIFIC").cast("string"),
        F.lpad(F.weekofyear("DT_NOTIFIC").cast("string"), 2, "0"),
    ),
)
# Resultado: "202301", "202302", ...
```
