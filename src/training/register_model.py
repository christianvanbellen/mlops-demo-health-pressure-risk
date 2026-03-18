# src/training/register_model.py
# Utilitário CLI para operações manuais no Model Registry.
#
# Uso:
#   python src/training/register_model.py status
#   python src/training/register_model.py listar
#   python src/training/register_model.py promover 4 champion
#   python src/training/register_model.py arquivar 2
#   python src/training/register_model.py comparar 3 4
#
# Complementa o human gate do evaluate.py para operações
# pontuais fora do ciclo automático de retraining.

import argparse
import os
import sys
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MLFLOW_EXPERIMENT, MODEL_NAME

# ── métricas exibidas nas comparações ────────────────────────────
_METRICAS_COMPARACAO = [
    ("val_precision_at_k", "Prec@K val"),
    ("val_auc_pr", "AUC-PR val"),
    ("val_auc_roc", "AUC-ROC val"),
    ("test_precision_at_k", "Prec@K test"),
    ("test_auc_pr", "AUC-PR test"),
]

# aliases reconhecidos como "ativos"
_ALIASES_ATIVOS = ["champion", "challenger", "staging"]


# ── helpers ───────────────────────────────────────────────────────
def _client() -> MlflowClient:
    return MlflowClient()


def _alias_de_versao(client: MlflowClient, versao: str) -> list[str]:
    """Retorna todos os aliases atribuídos a uma versão."""
    mv = client.get_model_version(MODEL_NAME, versao)
    return list(mv.aliases) if mv.aliases else []


def _versao_do_alias(client: MlflowClient, alias: str) -> str | None:
    """Retorna o número de versão associado ao alias, ou None."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
        return mv.version
    except Exception:
        return None


def _metricas_do_run(client: MlflowClient, run_id: str) -> dict:
    """Busca métricas do run MLflow associado à versão."""
    try:
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception:
        return {}


def _formatar_ts(ts_ms: int | None) -> str:
    """Converte timestamp em milissegundos para string legível."""
    if ts_ms is None:
        return "—"
    return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M")


def _sep(largura: int = 72) -> str:
    return "─" * largura


# ── funções principais ────────────────────────────────────────────
def listar_versoes() -> None:
    """
    Lista todas as versões do modelo no registry com seus aliases,
    métricas principais e data de criação.
    """
    client = _client()

    try:
        versoes = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        print(f"  ⚠ Erro ao listar versões: {e}")
        return

    if not versoes:
        print(f"  Nenhuma versão registrada para {MODEL_NAME}.")
        return

    versoes = sorted(versoes, key=lambda v: int(v.version), reverse=True)

    print(f"\n{_sep()}")
    print(f"  Modelo: {MODEL_NAME}")
    print(f"  Total de versões: {len(versoes)}")
    print(_sep())

    cabecalho = f"  {'Versão':<8} {'Aliases':<22} {'Tipo':<12} {'Prec@K val':>10} {'AUC-ROC val':>12} {'Criado em':<18}"
    print(cabecalho)
    print(_sep())

    for mv in versoes:
        aliases = ", ".join(f"@{a}" for a in mv.aliases) if mv.aliases else "—"
        metricas = _metricas_do_run(client, mv.run_id)

        prec_k = metricas.get("val_precision_at_k", None)
        auc_roc = metricas.get("val_auc_roc", None)

        prec_k_str = f"{prec_k:.4f}" if prec_k is not None else "—"
        auc_roc_str = f"{auc_roc:.4f}" if auc_roc is not None else "—"

        # infere tipo do modelo a partir dos parâmetros do run
        try:
            run = client.get_run(mv.run_id)
            model_type = run.data.params.get("model_type", "—")
        except Exception:
            model_type = "—"

        criado = _formatar_ts(mv.creation_timestamp)

        print(
            f"  v{mv.version:<7} {aliases:<22} {model_type:<12} "
            f"{prec_k_str:>10} {auc_roc_str:>12} {criado:<18}"
        )

    print(_sep())


def promover(versao: str, alias: str) -> None:
    """
    Atribui um alias a uma versão do modelo.
    Registra a promoção como run MLflow para rastreabilidade.
    """
    client = _client()

    # valida que a versão existe
    try:
        mv = client.get_model_version(MODEL_NAME, versao)
    except Exception as e:
        print(f"  ⚠ Versão {versao} não encontrada: {e}")
        return

    # versão anterior do alias (para log)
    versao_anterior = _versao_do_alias(client, alias)

    # atribui o novo alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=versao,
    )

    # loga a promoção no MLflow para rastreabilidade
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"promote_{alias}_v{versao}"):
        mlflow.set_tags(
            {
                "operacao": "promoção_manual",
                "alias": alias,
                "versao_nova": versao,
                "versao_anterior": versao_anterior or "nenhuma",
                "model_name": MODEL_NAME,
                "operador": os.environ.get("USER", os.environ.get("USERNAME", "desconhecido")),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    print(f"\n{_sep()}")
    print("  ✓ Promoção concluída")
    print(f"  Alias  : @{alias}")
    print(f"  Versão : v{versao}  ({mv.run_id[:8]}...)")
    if versao_anterior:
        print(f"  Anterior: v{versao_anterior} → removido de @{alias}")
    else:
        print(f"  Anterior: nenhuma versão tinha @{alias}")
    print(_sep())


def arquivar(versao: str) -> None:
    """
    Remove todos os aliases de uma versão e adiciona tag status=archived.
    Não deleta a versão do registry.
    """
    client = _client()

    # valida que a versão existe
    try:
        mv = client.get_model_version(MODEL_NAME, versao)
    except Exception as e:
        print(f"  ⚠ Versão {versao} não encontrada: {e}")
        return

    aliases_removidos = list(mv.aliases) if mv.aliases else []

    # remove todos os aliases
    for alias in aliases_removidos:
        try:
            client.delete_registered_model_alias(MODEL_NAME, alias)
            print(f"  Alias @{alias} removido de v{versao}")
        except Exception as e:
            print(f"  ⚠ Falha ao remover @{alias}: {e}")

    # adiciona tag de arquivamento
    client.set_model_version_tag(MODEL_NAME, versao, "status", "archived")
    client.set_model_version_tag(
        MODEL_NAME, versao, "archived_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    print(f"\n{_sep()}")
    print(f"  ✓ v{versao} arquivada")
    if aliases_removidos:
        print(f"  Aliases removidos: {', '.join(f'@{a}' for a in aliases_removidos)}")
    else:
        print("  Nenhum alias ativo para remover")
    print("  Tag status=archived adicionada")
    print(_sep())


def comparar(versao_a: str, versao_b: str) -> None:
    """
    Compara métricas de duas versões lado a lado com delta.
    """
    client = _client()

    # carrega as duas versões
    try:
        mv_a = client.get_model_version(MODEL_NAME, versao_a)
        mv_b = client.get_model_version(MODEL_NAME, versao_b)
    except Exception as e:
        print(f"  ⚠ Erro ao carregar versões: {e}")
        return

    metricas_a = _metricas_do_run(client, mv_a.run_id)
    metricas_b = _metricas_do_run(client, mv_b.run_id)

    aliases_a = ", ".join(f"@{a}" for a in mv_a.aliases) if mv_a.aliases else "—"
    aliases_b = ", ".join(f"@{a}" for a in mv_b.aliases) if mv_b.aliases else "—"

    print(f"\n{_sep()}")
    print(f"  Comparação de versões — {MODEL_NAME}")
    print(_sep())
    print(f"  {'Métrica':<22} {'v' + versao_a:>12} {'v' + versao_b:>12} {'Delta':>10}  Melhor")
    print(_sep())

    for chave_metrica, label in _METRICAS_COMPARACAO:
        val_a = metricas_a.get(chave_metrica)
        val_b = metricas_b.get(chave_metrica)

        if val_a is None and val_b is None:
            print(f"  {label:<22} {'—':>12} {'—':>12} {'—':>10}")
            continue

        str_a = f"{val_a:.4f}" if val_a is not None else "—"
        str_b = f"{val_b:.4f}" if val_b is not None else "—"

        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            delta_str = f"{delta:+.4f}"
            melhor = f"v{versao_b}" if delta > 0 else (f"v{versao_a}" if delta < 0 else "igual")
        else:
            delta_str = "—"
            melhor = "—"

        print(f"  {label:<22} {str_a:>12} {str_b:>12} {delta_str:>10}  {melhor}")

    print(_sep())
    print(f"  v{versao_a}: {aliases_a}  |  criado em {_formatar_ts(mv_a.creation_timestamp)}")
    print(f"  v{versao_b}: {aliases_b}  |  criado em {_formatar_ts(mv_b.creation_timestamp)}")
    print(_sep())


def status() -> None:
    """
    Mostra o estado atual do registry: aliases ativos e últimas versões.
    """
    client = _client()

    print(f"\n{_sep()}")
    print(f"  Model Registry — {MODEL_NAME}")
    print(f"  Consultado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(_sep())

    # aliases ativos
    print("  Aliases ativos:")
    algum_alias = False
    for alias in _ALIASES_ATIVOS:
        versao = _versao_do_alias(client, alias)
        if versao:
            algum_alias = True
            mv = client.get_model_version(MODEL_NAME, versao)
            metricas = _metricas_do_run(client, mv.run_id)
            prec_k = metricas.get("val_precision_at_k")
            prec_str = f"Prec@K={prec_k:.4f}" if prec_k is not None else ""
            criado = _formatar_ts(mv.creation_timestamp)
            print(f"    @{alias:<12} → v{versao:<5} {prec_str:<18} criado {criado}")

    if not algum_alias:
        print("    Nenhum alias ativo (champion / challenger / staging).")

    # últimas 3 versões
    print("\n  Últimas 3 versões registradas:")
    try:
        versoes = client.search_model_versions(f"name='{MODEL_NAME}'")
        versoes = sorted(versoes, key=lambda v: int(v.version), reverse=True)[:3]
        for mv in versoes:
            aliases = ", ".join(f"@{a}" for a in mv.aliases) if mv.aliases else "sem alias"
            tags = mv.tags or {}
            tag_status = tags.get("status", "")
            sufixo = f"  [{tag_status}]" if tag_status else ""
            criado = _formatar_ts(mv.creation_timestamp)
            print(f"    v{mv.version:<5} {aliases:<25} criado {criado}{sufixo}")
    except Exception as e:
        print(f"    ⚠ Erro ao listar versões: {e}")

    print(_sep())


# ── entrypoint CLI ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utilitário de Model Registry — Radar de Pressão Assistencial"
    )
    subparsers = parser.add_subparsers(dest="comando")

    subparsers.add_parser("listar", help="Lista versões do modelo")
    subparsers.add_parser("status", help="Estado atual do registry")

    p = subparsers.add_parser("promover", help="Atribui alias a uma versão")
    p.add_argument("versao", help="Número da versão (ex: 3)")
    p.add_argument("alias", help="Alias (champion, challenger, staging)")

    p = subparsers.add_parser("arquivar", help="Arquiva uma versão")
    p.add_argument("versao", help="Número da versão")

    p = subparsers.add_parser("comparar", help="Compara duas versões")
    p.add_argument("versao_a")
    p.add_argument("versao_b")

    args = parser.parse_args()

    if args.comando == "listar":
        listar_versoes()
    elif args.comando == "status":
        status()
    elif args.comando == "promover":
        promover(args.versao, args.alias)
    elif args.comando == "arquivar":
        arquivar(args.versao)
    elif args.comando == "comparar":
        comparar(args.versao_a, args.versao_b)
    else:
        parser.print_help()
