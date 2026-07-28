#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load .env if present (for POSTGRES_* etc.), but keep a dedicated UI port for HMR.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

DEV_UI_PORT="${DEV_UI_PORT:-5173}"
API_PORT="${API_PORT:-8000}"

echo "[dev] starting stack (hot reload)..."
DEV_UI_PORT="$DEV_UI_PORT" API_PORT="$API_PORT" \
  docker compose -f docker-compose.dev.yml up --build -d

echo
echo "[dev] ready"
echo "  UI:  http://localhost:${DEV_UI_PORT}   (Vite HMR — 必须用这个端口打开)"
echo "  API: http://localhost:${API_PORT}/docs"
echo
docker compose -f docker-compose.dev.yml ps
