#!/bin/sh
set -eu

python /app/model_bootstrap.py || true
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-3000}"
