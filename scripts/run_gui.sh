#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if command -v uv >/dev/null 2>&1 && [ -f "$ROOT/uv.lock" ]; then
  exec uv run python -m src.main "$@"
fi

if [ -n "${GTDC_PYTHON:-}" ]; then
  PYTHON="$GTDC_PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" -m src.main "$@"
