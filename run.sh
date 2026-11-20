#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/backend"
PORT="${PORT:-8000}"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
[ -f .venv/.deps_ok ] || { pip install -q -r requirements.txt && touch .venv/.deps_ok; }
echo "Open http://localhost:$PORT"
exec python -m uvicorn app.main:app --port "$PORT"
