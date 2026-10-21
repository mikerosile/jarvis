# T.H.E.O. - Thought Holography and Extraction Operator container (jarvis-theo)
#
# Built by scripts/install.sh:
#   podman build -t jarvis-theo theo/

FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++-14-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with ROCm 7.1 support (AMD GPU). Installed first and from the
# rocm7.1 index only, so the pinned local-version wheel is what gets used.
RUN pip install --no-cache-dir torch==2.11.0+rocm7.1 \
    --index-url https://download.pytorch.org/whl/rocm7.1

ARG CACHEBUST=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py edit_image.py new_image.py .

# Radeon RX 6950 XT (gfx1030) — also set in compose.yaml
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

EXPOSE 8190

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8190"]
