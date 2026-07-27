#!/bin/sh
set -e

echo "Waiting for PostgreSQL at ${POSTGRES_HOST:-db}:${POSTGRES_PORT:-5432}..."
until python - <<'PY'
import os, sys, time
import psycopg2

host = os.environ.get("POSTGRES_HOST", "db")
port = os.environ.get("POSTGRES_PORT", "5432")
user = os.environ.get("POSTGRES_USER", "stock")
password = os.environ.get("POSTGRES_PASSWORD", "stockpass")
dbname = os.environ.get("POSTGRES_DB", "stock_predictive")

for i in range(60):
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname=dbname
        )
        conn.close()
        print("PostgreSQL is ready.")
        sys.exit(0)
    except Exception as exc:
        print(f"Waiting for DB... ({i+1}/60) {exc}")
        time.sleep(1)

print("PostgreSQL is unavailable.")
sys.exit(1)
PY
do
  sleep 1
done

echo "Running migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting application: $*"
exec "$@"
