#!/usr/bin/env bash
# Collect a one-off snapshot for ops reports (run while TrafficLab is up).
set -euo pipefail
echo "=== TrafficLab runtime snapshot $(date -Iseconds) ==="
echo "--- Host ---"
uname -a
echo "CPUs: $(nproc 2>/dev/null || echo n/a)"
free -h 2>/dev/null || true
df -h / 2>/dev/null || true
echo "--- HTTP checks ---"
curl -sS --connect-timeout 3 http://127.0.0.1:8000/health && echo || echo "backend /health: FAIL"
curl -sS -o /dev/null -w "frontend :3000 HTTP %{http_code}\n" --connect-timeout 3 http://127.0.0.1:3000/ || echo "frontend: FAIL"
echo "--- Docker (if permitted) ---"
if docker info >/dev/null 2>&1; then
  docker compose -f "$(dirname "$0")/../docker-compose.yml" ps 2>/dev/null || true
  docker stats --no-stream 2>/dev/null || true
else
  echo "docker: not available (permission or not running)"
fi
echo "--- Key processes (rss) ---"
ps aux 2>/dev/null | grep -E '[u]vicorn main|[s]umo -c|[n]ode.*vite|postgres: user trafficlab' || true
