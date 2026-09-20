# JARVIS — Open WebUI Stack

A self-hosted LLM stack built with [podman-compose](https://github.com/containers/podman-compose). It combines a local model server, a chat frontend, and a set of supporting services, all wired together by `podman-compose.yaml`.

## Decisions

The specific choices in this setup were driven by the hardware the stack runs on:

- **AMD RX 6950 XT with 16 GB VRAM (RDNA 2).** The host's GPU is an AMD card, so the llama.cpp service uses the Vulkan build (`ghcr.io/ggml-org/llama.cpp:server-vulkan`) and passes `/dev/kfd` and `/dev/dri` into the container for direct GPU access. The server runs with `--device Vulkan0`, and `GGML_VK_VISIBLE_DEVICES` is set so llama.cpp targets the discrete GPU (the integrated GPU is index 0 on this host).
- **Models on a bind mount.** The model weights live in a host directory (`~/models`) that is bind-mounted into the container at `/models`, so they persist across restarts and are managed on the host. llama.cpp runs in router mode with `--models-max 1`, so only a single model is ever resident in VRAM at once, and the server loads and unloads models on demand.

## Prerequisites

If running containers as rootless ...

* The user running the containers will need to be a member of the "video" and "render" user groups.
```bash
sudo usermod -aG video,render $USER
```
* Verify the discrete GPU is visible to containers by listing the available Vulkan devices:
```bash
podman run -ti --rm --device=/dev/kfd --device=/dev/dri ghcr.io/ggml-org/llama.cpp:server-vulkan --list-devices
```

Example output:
```
Available devices:
  Vulkan0: AMD Ryzen 9 7950X3D 16-Core Processor (RADV RAPHAEL_MENDOCINO) (72540 MiB, 71657 MiB free)
  Vulkan1: AMD Radeon RX 6950 XT (RADV NAVI21) (16368 MiB, 1798 MiB free)
```
* Allow the user to continue running processes after logout
```bash
loginctl enable-linger $USER
```
* Use podman-compose to create a systemd service to run the stack in the background.  See `podman-compose systemd --help` for instruction.

## Services

| Service | Image | Port | Purpose |
| --- | --- | --- | --- |
| `llama-dgpu` | `ghcr.io/ggml-org/llama.cpp:server-vulkan` | 9931 | Local LLM inference server (llama.cpp, Vulkan backend, AMD GPU via `/dev/kfd` and `/dev/dri`). Runs in router mode over `~/models` (bind-mounted at `/models`) with `--models-max 1`, so a single model is resident at a time. |
| `open-webui` | `ghcr.io/open-webui/open-webui:v0.11.3` | `OPEN_WEBUI_APP_PORT` (default 3000) | Web UI for chatting with models. Web search is enabled and delegated to SearXNG. |
| `open-terminal` | `ghcr.io/open-webui/open-terminal` | 8000 | Container terminal tool exposed to Open WebUI; API access protected by `OPEN_TERMINAL_API_KEY`. |
| `docling-serve` | `quay.io/docling-project/docling-serve:latest` | 5001 | Document conversion server (PDF/OCR/tables) used for document ingestion, with a remote-LLM picture description path enabled. |
| `searxng-core` | `docker.io/searxng/searxng` | `SEARXNG_PORT` (default 8080) | Privacy-focused metasearch engine that backs Open WebUI's web search. Config mounted from `searxng-core-config/`. |
| `searxng-valkey` | `docker.io/valkey/valkey:9-alpine` | — | Valkey (Redis-compatible) instance used as the SearXNG result cache, with persistence enabled (`--save 30 1`). |

Persistent state for all services is kept in named volumes (`open-webui`, `open-terminal`, `searxng-core-data`, `searxng-valkey-data`). The llama.cpp model weights are not in a named volume; they live in a host bind mount at `~/models`.

## Environment

All runtime configuration lives in the `.env` file that is consumed by podman-compose for handling compose-level substitution. It holds the llama.cpp GPU settings and the stack-wide settings (ports, keys, SearXNG). The file contains shell-generated secrets and must not be committed.

### Variable reference

| Variable | Used by | Purpose |
| --- | --- | --- |
| `GGML_VK_VISIBLE_DEVICES` | llama.cpp | Selects which Vulkan GPU (by index) the container can see and use. Can contain a single index or multiple separated by commas with values like `0`, `0,1`, `1,0`. |
| `LLAMA_PORT` | llama.cpp | Port the llama.cpp server listens on; also used for the host port mapping. |
| `LLAMA_DEVICE` | llama.cpp | Compute device passed to the server via `--device` (e.g. `Vulkan0`). |
| `LLAMA_MODELS_DIR` | llama.cpp | Directory passed to `--models-dir` that the router scans to discover models. |
| `LLAMA_MODELS_PRESET` | llama.cpp | Path passed to `--models-preset` of the INI file holding per-model runtime settings. |
| `LLAMA_MODELS_MAX` | llama.cpp | Value passed to `--models-max`: maximum number of models resident in memory at once. |
| `LLAMA_ARG_SLEEP_IDLE_SECONDS` | llama.cpp | Value passed to `--sleep-idle-seconds`: seconds of inactivity before the loaded model unloads itself from memory. |
| `LLAMA_ARG_ENDPOINT_SLOTS` | llama.cpp | Exposes the slots monitoring endpoint (`--slots`; enabled by default, `0` = off). |
| `LLAMA_ARG_ENDPOINT_METRICS` | llama.cpp | Enables the Prometheus-compatible metrics endpoint (`--metrics`; disabled by default, `1` = on). |
| `OPEN_WEBUI_APP_PORT` | Open-WebUI | Host port the Open WebUI web interface is published on. |
| `WEBUI_SECRET_KEY` | Open-WebUI | Secret key Open WebUI uses for signing sessions/tokens. **Sensitive — generate a strong random value and never commit it.** |
| `OPEN_TERMINAL_API_KEY` | Open Terminal | API key required to use the `open-terminal` container's API. **Sensitive — generate a strong random value and never commit it.** |
| `DOCLING_PORT` | Docling | Host port the `docling-serve` API is published on. |
| `DOCLING_API_KEY` | Docling | API key required to use the `docling-serve` API. **Sensitive — generate a strong random value and never commit it.** |
| `SEARXNG_VERSION` | SearXNG | Version tag of the SearXNG image to run. |
| `SEARXNG_HOST` | SearXNG | Address SearXNG listens on inside its port mapping (`[::]` = all interfaces, IPv6/IPv4). |
| `SEARXNG_PORT` | SearXNG | Host and container port the SearXNG instance is published on. |
| `SEARXNG_QUERY_URL` | Open-WebUI | Base URL that Open WebUI uses to query SearXNG for web search results (`<query>` is replaced with the search term). |

## Suggested models

The following models were specifically chosen based on their size, quantization, and purpose. The "Suggested Use" column describes the type of work each model is best suited to.

These models were chosen because they fit comfortably in 16 GB of VRAM. Keeping 100% of a model's weights in VRAM matters for performance: when the weights fit entirely on the GPU, every token is generated by the GPU alone. The moment a model is too large to fit, llama.cpp offloads the overflow to system RAM, and each token then requires moving weight data back and forth across the CPU–GPU bus (PCIe), which slows generation dramatically — often to a fraction of the all-VRAM speed. So a model that just barely fits is not as fast as it looks; only a fully resident model runs at full GPU speed.

At the same time, the weights are only part of the memory picture. At runtime the model also needs a KV cache to hold its context (the prompt plus generated tokens): for each token, every attention layer stores key/value tensors, and that memory grows with context length, batch size, and layer count. A 64K-token conversation can require several gigabytes of VRAM even on a modest model. By staying small enough to fit with headroom to spare, these models leave additional VRAM available for that context, so long prompts and long conversations can be served entirely on the GPU rather than being truncated or spilled to slower memory.

| Model | Suggested Use |
| --- | --- |
| `gpt-oss-20b` | General-purpose reasoning and conversation. Open-weight MoE model (20B total / ~5B active) that balances quality with low memory and fast inference. |
| `llava-7b` | Multimodal image understanding. Accepts images alongside text for visual questions, OCR-style extraction, and description of figures/screenshots. |
| `qwen2.5-coder-14b` | Code generation, completion, and debugging. Instruct-tuned coding model (14B) for software-engineering tasks. |
| `qwen3.8-27b` | Long-context general chat and reasoning. 27B model configured with a large context window for summarizing, analyzing, and discussing large documents. |

Per-model server settings (context size, KV cache type, flash attention, reasoning) are not set via environment variables; they live in `~/models/models-preset.ini`, which the server loads in router mode. Each model is a subfolder of `~/models` holding a `model.gguf`, with a matching section in that file referencing it.

## llama.cpp server settings

The `llama-dgpu` service starts the llama.cpp server with the following command-line parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--port` | `9931` (`LLAMA_PORT`) | TCP port the HTTP server listens on. |
| `--device` | required, no default (`LLAMA_DEVICE`) | Compute device used for inference, e.g. `Vulkan0`. |
| `--models-dir` | `/models` (`LLAMA_MODELS_DIR`) | Directory the router scans to discover available models. |
| `--models-preset` | `/models/models-preset.ini` (`LLAMA_MODELS_PRESET`) | INI file holding the per-model runtime settings used in router mode. |
| `--models-max` | `1` (`LLAMA_MODELS_MAX`) | Maximum number of models that may be resident in memory at once. |
| `--sleep-idle-seconds` | `300` (`LLAMA_ARG_SLEEP_IDLE_SECONDS`) | Seconds of inactivity after which the loaded model unloads itself from memory. |
| `--no-mmproj-offload` | *(flag, no value)* | Keeps the multimodal projection (mmproj) weights on the CPU instead of offloading them to the GPU. mmproj models are responsible for image recognition. **This is important to keep larger models (e.g. Qwen3.8-27B) fully loaded in VRAM to ensure fast text responses.** |

## SearXNG configuration

The `searxng-core` container mounts `./searxng-core-config/` read-only into `/etc/searxng`, so the two files in that directory fully control SearXNG's behavior for this stack:

### `searxng-core-config/settings.yml`

General SearXNG settings. It uses SearXNG's default configuration (`use_default_settings: true`) with a few overrides:

- `search.formats` enables the `json` format alongside `html`, which is what Open WebUI needs to consume search results as JSON.
- `server.image_proxy` enables SearXNG's image proxy so result images are relayed through SearXNG (works around referrer/hotlink blocking).
- A few engines (`wikidata`, `ahmia`, `torch`) are disabled.
- **`server.secret_key` must be changed.** The value shipped in the repository is a placeholder and is committed to git. SearXNG uses this key to sign data (e.g. the limiter's link tokens), so anyone who knows it can forge tokens. Replace it with a long, random value of your own (e.g. `openssl rand -hex 32`) before exposing SearXNG to a network.

### `searxng-core-config/limiter.toml`

SearXNG's rate-limiting / anti-bot configuration:

- `[botdetection]` — network prefix sizes used to group client IPs, plus the `trusted_proxies` list of reverse-proxy IPs whose `X-Forwarded-For` / `X-Real-IP` headers should be honored to recover the real client address.
- `[botdetection.ip_limit]` — rate limiting controls, e.g. whether link-local (private LAN) addresses are exempt and whether the `link_token` method is active.
- `[botdetection.ip_lists]` — explicit `block_ip` / `pass_ip` allow/block lists. IPs in `pass_ip` get unrestricted access and skip all other checks.

**If you intend to use SearXNG from anything other than the host it runs on** (other machines on your network, other containers, etc.), add your private LAN subnet to the `pass_ip` list. Otherwise clients outside the host will hit the rate limiter and get their queries blocked or throttled. For example, on a `192.168.1.0/24` LAN:

```toml
pass_ip = [
  '192.168.1.0/24',
]
```

Adjust the CIDR to match your actual subnet. The reverse — restricting access to just the host — is achieved by leaving LAN subnets out of `pass_ip` and not publishing the port beyond localhost.

## References

* podman-compose [https://github.com/containers/podman-compose]
* llama.cpp [https://github.com/ggml-org/llama.cpp]
* Open WebUI [https://openwebui.com]
* Open Terminal [https://github.com/open-webui/open-terminal]
* Docling [https://docling.ai]
* SearXNG [https://searxng.org]
* Valkey [https://valkey.io]
