#!/usr/bin/env bash
# build-wheel.sh — constrói o wheel do projeto e faz upload para o Unity Catalog Volume
#
# Uso:
#   ./scripts/build-wheel.sh               # build + upload
#   ./scripts/build-wheel.sh --build-only  # apenas build local
#
# Pré-requisitos:
#   - uv instalado e no PATH
#   - databricks CLI configurado (~/.databrickscfg ou variáveis DATABRICKS_HOST etc.)

set -euo pipefail

VOLUME_PATH="dbfs:/Volumes/ds_dev_db/dev_christian_van_bellen/wheels"
BUILD_ONLY=false

for arg in "$@"; do
  case $arg in
    --build-only) BUILD_ONLY=true ;;
    *) echo "Argumento desconhecido: $arg"; exit 1 ;;
  esac
done

echo "==> Limpando dist/ anterior..."
rm -rf dist/

echo "==> Construindo wheel..."
uv build --wheel

WHL=$(ls dist/health_pressure_risk-*.whl | head -1)
WHL_NAME=$(basename "$WHL")
echo "    Wheel gerado: $WHL_NAME"

if [ "$BUILD_ONLY" = true ]; then
  echo "==> Modo --build-only: upload ignorado."
  echo "    Arquivo: dist/$WHL_NAME"
  exit 0
fi

echo "==> Fazendo upload para Volume: ${VOLUME_PATH}/${WHL_NAME}"
databricks fs cp "$WHL" "${VOLUME_PATH}/${WHL_NAME}" --overwrite

echo ""
echo "Concluído. Caminho no cluster:"
echo "  ${VOLUME_PATH}/${WHL_NAME}"
