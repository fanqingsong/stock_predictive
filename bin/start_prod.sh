#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[prod] starting stack..."
docker compose -f docker-compose.yml up --build -d

echo
echo "[prod] ready"
echo "  UI:  http://localhost:${APP_PORT:-8080}"
echo "  API: via nginx proxy (/api, /health)"
echo
docker compose -f docker-compose.yml ps
