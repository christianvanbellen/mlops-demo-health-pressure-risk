<!--
  Template de PR — Radar de Pressão Assistencial
  Revisor padrão: christianvanbellen
  Claude Code: usar `gh pr create --reviewer christianvanbellen --body "$(cat .github/pull_request_template.md)"`
-->

## Descrição
<!-- Descreva as mudanças desta PR e a motivação -->

## Tipo de mudança
- [ ] Bug fix
- [ ] Nova feature
- [ ] Refatoração
- [ ] Documentação
- [ ] Dependências
- [ ] CI/CD
- [ ] Temporário / workaround

## Checklist
- [ ] `uv run ruff check src/` sem erros
- [ ] `uv run ruff format src/ --check` sem erros
- [ ] `uv run pytest tests/ -v` passando
- [ ] `databricks bundle validate --target dev` OK
- [ ] `uv.lock` atualizado se dependências mudaram
- [ ] Documentação atualizada se necessário

## Impacto no pipeline
<!-- Alguma das camadas abaixo será afetada? -->
- [ ] Bronze (ingestão)
- [ ] Silver (transforms)
- [ ] Gold / Feature Store
- [ ] Modelo / MLflow / Registry
- [ ] Scoring
- [ ] Monitoramento / Drift
- [ ] CI/CD / DAB
- [ ] Nenhuma (apenas docs ou config local)

## Notas para o revisor
<!-- Contexto adicional, decisões de design, o que testar manualmente -->
