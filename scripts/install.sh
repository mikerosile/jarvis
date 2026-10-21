#!/usr/bin/env bash
#
# Install the jarvis-arc stack (user scope — NO sudo).
#
#   ./scripts/install.sh
#
# Runs as the regular user:
#   * copies the three user-scope units to ~/.config/systemd/user
#   * enables jarvis-phil and jarvis-arc at user login
#     (jarvis-theo is deliberately NOT enabled — on demand only)
#   * creates the jarvis-phil Python venv
#     (phil/.venv) and installs its requirements
#   * builds the FLUX container image (jarvis-theo:latest)
#
# It does NOT start any service. After install, run:
#   systemctl --user start jarvis-phil
#   systemctl --user start jarvis-arc
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_UNIT_DIR="${HOME}/.config/systemd/user"

if [[ ${EUID} -eq 0 ]]; then
    echo "This script must be run as the regular user, not root." >&2
    exit 1
fi

echo "==> Copying user-scope units to ${USER_UNIT_DIR}"
mkdir -p "${USER_UNIT_DIR}"
install -m 0644 "${REPO}/systemd/user/jarvis-phil.service" "${USER_UNIT_DIR}/"
install -m 0644 "${REPO}/systemd/user/jarvis-arc.service" "${USER_UNIT_DIR}/"
install -m 0644 "${REPO}/systemd/user/jarvis-theo.service" "${USER_UNIT_DIR}/"

echo "==> Setting up the jarvis-phil Python venv"
VENV="${REPO}/phil/.venv"
python3 -m venv "${VENV}"
"${VENV}/bin/pip" install -r "${REPO}/phil/requirements.txt"

echo "==> Building the THEO container image (jarvis-theo:latest)"
podman build -t jarvis-theo "${REPO}/theo"

echo "==> Sanity checks (warnings only)"
for path in \
    "${REPO}/arc/compose.yaml" \
    "${REPO}/theo/compose.yaml" \
    "${REPO}/phil/.venv/bin/uvicorn" \
    "${REPO}/models/models-preset.ini" \
    "${REPO}/models/img/FLUX.2-klein-4B"
do
    if [[ -e "${path}" ]]; then
        echo "  ok:      ${path}"
    else
        echo "  MISSING: ${path}   <- place your models under ./models (see ./models/README.md)"
    fi
done

echo "==> Reloading user systemd and enabling services"
systemctl --user daemon-reload
systemctl --user enable jarvis-phil.service
# jarvis-arc is enabled: the LLM is the default GPU owner and comes up at
# boot; the supervisor (PHIL) swaps it down/up around image generations.
systemctl --user enable jarvis-arc.service
# jarvis-theo.service is intentionally NOT enabled: it only runs on demand,
# and must never start at boot while the LLM wants the GPU.

cat <<EOF

Install complete. No services were started.

Bring the stack up when ready:
  systemctl --user start jarvis-phil
  systemctl --user start jarvis-arc

Check state:
  ./scripts/status.sh

For the services to come up at boot without a login session (one-time,
and the only sudo involved anywhere in this project):
  sudo loginctl enable-linger

Then finish the Open-WebUI side (see loki/README.md, "Getting Started", step 4):
  1. Add an OpenAI-compatible connection to the LLM container
     (http://host.containers.internal:9931/v1 from the Open-WebUI container).
  2. Register jarvis-phil as an OpenAPI tool server
     (Settings → Integrations → External Tool Servers → + ;
     URL http://host.containers.internal:8191, spec at /api/openapi.json —
     served by the running supervisor).
  3. Attach the generate_image tool to a chat or the Qwen model,
     with tool calling enabled.
EOF
