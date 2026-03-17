# Limitações e Horizonte Operacional do Pipeline

## Atraso estrutural

O pipeline opera com defasagem de **4-8 semanas** em relação ao momento atual,
decorrente do empilhamento de três fontes de atraso:

| Fonte | Atraso típico | Causa |
|---|---|---|
| Capacity (Hospitais e Leitos) | ~60 dias | Publicação mensal com defasagem |
| SRAG (subnotificação) | ~45-90 dias | Notificações tardias chegando continuamente |
| Pipeline de processamento | ~7 dias | Frequência semanal de atualização |

**Exemplo concreto (março/2026):**
- Score disponível: competência 202601 (janeiro/2026)
- O score prevê: risco de fevereiro/2026
- Fevereiro já ocorreu: o score tem valor retrospectivo parcial

## Caso de uso adequado

✅ **Planejamento operacional antecipado**
  Alocar equipes, insumos e atenção para municípios historicamente vulneráveis
  com 4-8 semanas de antecedência relativa ao ciclo de publicação dos dados.

✅ **Vigilância epidemiológica preventiva**
  Identificar municípios com padrão estrutural de pressão recorrente,
  independentemente de eventos agudos.

✅ **Priorização de recursos escassos**
  Ranking mensal de municípios para direcionar inspeções, transferências
  de pacientes e mobilização de equipes de apoio.

❌ **Resposta a crise em tempo real**
  O pipeline não é adequado para detectar e responder a colapsos
  que estão acontecendo no momento atual.

## Roadmap para reduzir o atraso (Fase 2)

### Fonte 1 — InfoGripe / Fiocruz
  - Boletins semanais com nowcasting de tendência regional
  - Reduziria o atraso do sinal de demanda para ~1-2 semanas
  - Já previsto no documento de escopo (seção 5.4)

### Fonte 2 — Registro de Ocupação Hospitalar
  - Dados quase em tempo real de ocupação de leitos
  - Substituiria o CNES mensal como proxy de capacidade operacional
  - Já previsto no documento de escopo (seção 5.5)

### Impacto esperado com Fase 2
  Com as duas fontes integradas, o atraso estrutural cairia para
  **1-2 semanas** — tornando o pipeline adequado para resposta
  semi-antecipada a surtos respiratórios.

## Notas sobre consolidação dos dados

### Subnotificação do SRAG
  As competências mais recentes têm casos artificialmente baixos:
  - t-1 mês: ~80% dos casos notificados
  - t-2 meses: ~90% dos casos notificados
  - t-3 meses: ~95% dos casos notificados
  - t-4+ meses: praticamente consolidado

  **Implicação para o treino:** as últimas 2-3 competências disponíveis
  têm targets potencialmente incorretos. O pipeline reprocessa essas
  competências semanalmente para capturar notificações tardias.

### Forward fill de capacity
  Quando a fonte de Hospitais e Leitos ainda não publicou o mês mais recente,
  o pipeline usa os dados do último mês disponível como proxy.
  Essas competências são marcadas com `capacity_is_forward_fill = True`
  na feature table e **não são usadas para calcular o target**.
  Scores produzidos com forward fill devem ser interpretados com cautela.

## Versões e governança

| Campo | Valor |
|---|---|
| target_definition_version | v2 |
| target_metric | casos_por_leito |
| target_percentile | 0.85 |
| risk_threshold_version | v2 (percentil 85/70 do score) |
| pressure_formula_version | v1 |
