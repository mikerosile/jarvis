# JARVIS — Open WebUI Stack

A self-hosted LLM stack built with [podman-compose](https://github.com/containers/podman-compose). It combines a local model server, a chat frontend, and a set of supporting services, all wired together by `podman-compose.yaml`.

## Decisions

The specific choices in this setup were driven by the hardware the stack runs on and the goal of fast model switching:

- **AMD RX 6950 XT with 16 GB VRAM (RDNA 2).** The host's GPU is an AMD card, so the Ollama service uses the ROCm image (`ollama/ollama:rocm`) and passes `/dev/kfd` and `/dev/dri` into the container for direct GPU access. `HSA_OVERRIDE_GFX_VERSION` is set so ROCm targets the correct kernel build for the RDNA 2 architecture.
- **128 GB system RAM, 64 GB tmpfs for models.** The host has 128 GB of RAM. Half of that (64 GB) is allocated to a RAM-backed `tmpfs` mount over `/root/.ollama/models`. This lets models load and unload from RAM rather than disk, which is fast. Ollama is configured with `OLLAMA_MAX_LOADED_MODELS=1` so only a single model is ever resident, making rapid switching between models possible — the working set comfortably fits in the 64 GB tmpfs, and the large model files are streamed from RAM each time a model is selected rather than paged in from slower storage.

## Services

| Service | Image | Port | Purpose |
| --- | --- | --- | --- |
| `ollama` | `ollama/ollama:rocm` | 11434 | Local LLM inference server (ROCm build, AMD GPU via `/dev/kfd` and `/dev/dri`). Models are kept in a 64 GB tmpfs mount, so they are not persisted to disk between restarts. |
| `open-webui` | `ghcr.io/open-webui/open-webui:v0.11.3` | `OPEN_WEBUI_APP_PORT` (default 3000) | Web UI for chatting with models. Web search is enabled and delegated to SearXNG. |
| `open-terminal` | `ghcr.io/open-webui/open-terminal` | 8000 | Container terminal tool exposed to Open WebUI; API access protected by `OPEN_TERMINAL_API_KEY`. |
| `docling-serve` | `quay.io/docling-project/docling-serve:latest` | 5001 | Document conversion server (PDF/OCR/tables) used for document ingestion, with a remote-LLM picture description path enabled. |
| `searxng-core` | `docker.io/searxng/searxng` | `SEARXNG_PORT` (default 8080) | Privacy-focused metasearch engine that backs Open WebUI's web search. Config mounted from `searxng-core-config/`. |
| `searxng-valkey` | `docker.io/valkey/valkey:9-alpine` | — | Valkey (Redis-compatible) instance used as the SearXNG result cache, with persistence enabled (`--save 30 1`). |

Persistent state for all services is kept in named volumes (`ollama`, `open-webui`, `open-terminal`, `searxng-core-data`, `searxng-valkey-data`).

## Environment

All runtime configuration lives in the `.env` file that is consumed by podman-compose for handling compose-level substitution. `/.env.rocm` is the reference/template copy of that file for this ROCm setup: it contains both the Ollama GPU settings and the stack-wide settings (ports, keys, SearXNG). Copy it to `.env` (or source it, as it contains shell-generated secrets) before first use.

### Variable reference

| Variable | Used by | Purpose |
| --- | --- | --- |
| `ROCR_VISIBLE_DEVICES` | Ollama | Restricts which ROCm GPU (by index) the Ollama container can see and use. |
| `HSA_OVERRIDE_GFX_VERSION` | Ollama | Overrides the detected GPU GFX version so ROCm picks the correct kernel targets (e.g. for unsupported or newer AMD GPUs). |
| `OLLAMA_CONTEXT_LENGTH` | Ollama | Maximum context window (in tokens) Ollama is allowed to use for models. |
| `OLLAMA_MAX_LOADED_MODELS` | Ollama | Maximum number of models that may be held in memory at once. |
| `OLLAMA_KV_CACHE_TYPE` | Ollama | Quantization type used for the KV cache to reduce GPU memory usage during inference. |
| `OLLAMA_FLASH_ATTENTION` | Ollama | Enables flash attention in Ollama (1 = on) for faster, lower-memory attention computation. |
| `OPEN_WEBUI_APP_PORT` | Open-WebUI | Host port the Open WebUI web interface is published on. |
| `WEBUI_SECRET_KEY` | Open-WebUI | Secret key Open WebUI uses for signing sessions/tokens. **Generate a strong random value.** |
| `OPEN_TERMINAL_API_KEY` | Open Terminal | API key required to use the `open-terminal` container's API. **Generate a strong random value.** |
| `DOCLING_API_KEY` | Docling | API key required to use the `docling-serve` API. **Generate a strong random value.** |
| `SEARXNG_VERSION` | SearXNG | Version tag of the SearXNG image to run. |
| `SEARXNG_HOST` | SearXNG | Address SearXNG listens on inside its port mapping (`[::]` = all interfaces, IPv6/IPv4). |
| `SEARXNG_PORT` | SearXNG |Host and container port the SearXNG instance is published on. |
| `SEARXNG_QUERY_URL` | Open-WebUI | Base URL that Open WebUI uses to query SearXNG for web search results (`<query>` is replaced with the search term). |

## Suggested models

The following models were specifically chosen based on their size, quantization, and purpose. The "Suggested Use" column describes the type of work each model is best suited to.

These models were chosen because they fit comfortably in 16 GB of VRAM. Keeping 100% of a model's weights in VRAM matters for performance: when the weights fit entirely on the GPU, every token is generated by the GPU alone. The moment a model is too large to fit, Ollama offloads the overflow to system RAM, and each token then requires moving weight data back and forth across the CPU–GPU bus (PCIe), which slows generation dramatically — often to a fraction of the all-VRAM speed. So a model that just barely fits is not as fast as it looks; only a fully resident model runs at full GPU speed.

At the same time, the weights are only part of the memory picture. At runtime the model also needs a KV cache to hold its context (the prompt plus generated tokens): for each token, every attention layer stores key/value tensors, and that memory grows with context length, batch size, and layer count. A 128K-token conversation can require several gigabytes of VRAM even on a modest model. By staying small enough to fit with headroom to spare, these models leave additional VRAM available for that context, so long prompts and long conversations can be served entirely on the GPU rather than being truncated or spilled to slower memory.

| Model | Suggested Use |
| --- | --- |
| `gpt-oss:20b` | General-purpose reasoning and conversation. Open-weight MoE model (20B total / ~5B active) that balances quality with low memory and fast inference. |
| `llava:7b` | Multimodal image understanding. Accepts images alongside text for visual questions, OCR-style extraction, and description of figures/screenshots. |
| `qwen2.5-coder:14b-instruct-q5_K_M` | Code generation, completion, and debugging. Instruct-tuned coding model (14B, q5_K_M quantized) for software-engineering tasks. |
| `Qwen3.8-27B-UD-IQ3_XXS-128k:latest` | Long-context general chat and reasoning. 27B model with a 128K context window for summarizing, analyzing, and discussing large documents. |

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
* Ollama [https://ollama.com]
* Open WebUI [https://openwebui.com]
* Open Terminal [https://github.com/open-webui/open-terminal]
* Docling [https://docling.ai]
* SearXNG [https://searxng.org]
* Valkey [https://valkey.io]
