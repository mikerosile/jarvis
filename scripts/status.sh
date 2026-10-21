#!/usr/bin/env bash
#
# jarvis-arc stack status (user scope)
#
#   ./scripts/status.sh
#
set -uo pipefail

echo "=== user-scope units ==="
for unit in jarvis-arc jarvis-theo jarvis-phil; do
    printf '%-30s %s\n' "$unit" \
        "$(systemctl --user is-active "${unit}.service" 2>/dev/null || echo unknown)"
done

echo
echo "=== podman ==="
podman ps --filter name=jarvis-arc --format  '  arc    {{.Names}}  {{.Status}}' 2>/dev/null || true
podman ps --filter name=jarvis-theo --format '  theo   {{.Names}}  {{.Status}}' 2>/dev/null || true
podman pod ps --format '  pod    {{.Name}}  {{.Status}}' 2>/dev/null || true

echo
echo "=== health endpoints ==="
for pair in "arc:9931" "theo:8190" "phil:8191"; do
    name="${pair%%:*}"
    port="${pair##*:}"
    status="$(curl -s -m 3 "http://127.0.0.1:${port}/health" || echo offline)"
    printf '%-18s /health: %s\n' "$name" "$status"
done

echo
echo "Recent jarvis-phil log:"
journalctl --user -u jarvis-phil.service -n 10 --no-pager 2>/dev/null || true
