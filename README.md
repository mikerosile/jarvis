# J.A.R.V.I.S. — Just A Rather Very Intelligent System

A self-hosted general use A.I. stack built around the [Open-WebUI](https://openwebui.com) platform designed to leverage an AMD RDNA2 GPU with only 16GBs of VRAM. It contains all the services you would expect from a web based A.I chat portal including web searching, coding, text-to-speech, image generation (and editing) and document parsing.  The services are containized for portability.

## Inspiration

- I was inspired by the "[jarvis](https://github.com/InterGenJLU/jarvis)" project by InterGenJLU and notably their implementation of swapping models with a single GPU to support responsive LLM and quick image generation.
- Disney+'s [VisionQuest](https://www.disneyplus.com/browse/entity-8744ec55-8d14-4631-99b4-c889a2d6013f), for which Vision has multiple A.I.'s compromising his system. In this project, J.A.R.V.I.S. is the entire system and each named service is a different component.
- I don't want to pay for subscriptions or pay fees to Cloud/A.I. organizations to use A.I. or freely give my personal information to them.
- I don't want to worry about token limits.
- I want to maximize investment in existing hardware

## Decisions

The specific choices in this setup were driven by the hardware the stack runs on:

- llama.cpp with vulkan support for the inference engine
- Open-WebUI for the web portal user interface and service integration
- Kokoro for its fast text-to-speech capabilities that can run in-browse
- **AMD RX 6950 XT with 16 GB VRAM (RDNA 2)** the host's secodnary GPU dedicated to A.I. workloads
- 128 GBs of System RAM
- Samsung SSD 990 PRO 2TB for hosting/loading the model to ensure fast loading of models during GPU swap operations

## Prerequisites

The container environment is expected to run rootlees.

* The user running the containers will need to be a member of the "video" and "render" user groups.
```bash
sudo usermod -aG video,render $USER
```
* Allow the user to continue running processes after logout
```bash
loginctl enable-linger $USER
```
* Use podman-compose to create a systemd service to run the stack in the background.  See `podman-compose systemd --help` for instruction.
* Install the HuggingFace CLI (`hf`) tool, which is used by the `models/` Makefile to download community model weights from [Hugging Face](https://huggingface.co). Documentation: <https://huggingface.co/docs/huggingface_hub/guides/cli>

## Services provided by J.A.R.V.I.S.

| Service Name | Application | Conatiner Image | TCP Port | Purpose |
| --- | --- | --- | --- | --- |
| [LOKI](./loki/README.md) | Open-WebUI | ghcr.io/open-webui/open-webui:v0.12.3 | 8080 | Web UI for chatting with models. |
| [ARC](./arc/README.md) | llama.cpp | ghcr.io/ggml-org/llama.cpp:server-vulkan | 9931 | Local LLM inference server (llama.cpp, Vulkan backend) |
| [THEO](./theo/README.md) | FLUX.2 | localhost/jarvis-theo:latest | 8190 | Image generation (txt2img) and editing (img2img) |
| [PHIL](./phil/README.md) | GPU orchestrator | phil/app.py | 8191 | Manages the active GPU workload, swapping between LLM and image generation |
| [ROXIE](./roxie/README.md) | Nginx | docker.io/library/nginx:alpine | 8443 | TLS Reverse Proxy for secure access |
| [SHELLY](./shelly/README.md) | Open-Terminal | ghcr.io/open-webui/open-terminal:0.14.1 | 8001 | Container terminal tool exposed to Open WebUI | 
| [ARCHIE](./archie/README.md) | SearXNG | docker.io/searxng/searxng:latest | 8089 | Privacy-focused metasearch engine that backs Open WebUI's web search. |
| [REMY](./remy/README.md) | Docling | quay.io/docling-project/docling-serve-rocm71:v1.35.0_rocm71 | 5001 | Document conversion server (PDF/OCR/tables) used for document ingestion |
| [SONIA](./sonia/README.md) | Whisper | docker.io/hwdsl2/whisper-server:latest | 9000 | Speech-To-Text for capturing audio from end user |
| [ELOISE](./eloise/README.md) | Kokoro | ghcr.io/remsky/kokoro-fastapi-gpu:latest | 8880 | Text-To-Speech to give the A.I. assistant a voice |

## Project Structure

The repository root is organized around one directory per service, using the same names as the acronyms in the table above:

```text
jarvis/
├── compose.yaml          # main stack definition for podman-compose (LOKI, ROXIE, ARCHIE, REMY, SONIA, ELOISE, SHELLY)
├── .env                  # runtime configuration consumed by podman-compose; contains shell-generated secrets — must not be committed
├── .miopen/              # ROCm/mioPen GPU kernel cache, generated per GPU (e.g. gfx1030 for the RX 6950 XT); safe to delete and rebuild
├── arc/                  # ARC (llama.cpp LLM server): its own compose.yaml plus .env/.env.example; mounts ./models read-only as /models
├── archie/               # ARCHIE (SearXNG): settings.yml and limiter.toml mounted into the container's config directory
├── eloise/               # ELOISE (Kokoro TTS): text-to-speech giving J.A.R.V.I.S. its voice; docs here, service defined in root compose.yaml
├── etc/ssl/certs/        # TLS CA root/intermediary certificates used by Open-WebUI for cert verification
├── loki/                 # LOKI (Open-WebUI): chat web UI for J.A.R.V.I.S.; placeholder directory, service defined in the root compose.yaml
├── models/               # shared model library for ARC (LLMs) and THEO (image generation), see below
├── phil/                 # PHIL: Python GPU orchestrator that swaps the GPU between LLM inference and image generation
├── remy/                 # REMY (Docling): document conversion server (PDF/OCR/tables) used by LOKI for document ingestion; docs here, service defined in root compose.yaml
├── roxie/nginx.conf      # ROXIE: TLS reverse-proxy configuration for Nginx
├── scripts/              # host helpers: install.sh, status.sh, llama_monitor.sh
├── shelly/               # SHELLY (Open-Terminal): sandboxed container terminal exposed to Open WebUI; docs here, service defined in root compose.yaml
├── sonia/data            # SONIA (Whisper) runtime data and downloaded model cache
├── systemd/user          # user-scope systemd units that start ARC/PHIL/THEO at boot and drive the GPU swap flow
├── theo/                 # THEO: FLUX.2 image generation service — Dockerfile, compose.yaml, app code, generated_images/
└── 3rdparty/             # configs/workflows for external tools (ComfyUI JSON workflows, Continue.dev API configs)
```

### `./models/`

This directory is the shared model library: ARC mounts it at `/models`, and THEO uses its image models directly. The important file here is **`models/models-preset.ini`** (referenced as `/models/models-preset.ini` from inside containers): the INI file llama.cpp loads in router mode via `--models-preset`. It holds a global default section (`[*]`) and one `[<model-dir>]` section per model defining its alias, the path to its weights/projection files, and per-model runtime settings such as context size, KV cache types, flash attention, and reasoning effort.

The contents of `models/`:

- **`llm/`** — LLM models for ARC in router mode. One subdirectory per model (e.g. `qwen3.8-27b-gsq-rco-iq3_s-mtp/`), named after the specific model and quantization, containing:
  - `model.gguf` — the GGUF model weights.
  - `mmproj-<dtype>.gguf` (e.g. `mmproj-bf16.gguf`) — optional multimodal projection weights for image understanding; present only on vision-capable models and referenced by the matching section in `models-preset.ini`.
  - a small `model.url` recording where the weights were fetched from, used by the Makefile below.

- **`img/`** — image generation models for THEO. One subdirectory per model (e.g. `FLUX.2-klein-4B/`), each being a whole Hugging Face repository download (weights, text encoder, tokenizer, VAE, scheduler, license and README).

- **`Makefile`** — downloads LLM weights into place: for every subdirectory of `llm/` that contains a `model.url`, it runs `hf download <url>` and renames the result to `model.gguf`. This is how new models are added.

## Environment

All runtime configuration lives in the `.env` file that is consumed by podman-compose for handling compose-level substitution. It holds the llama.cpp GPU settings and the stack-wide settings (ports, keys, SearXNG). The file contains shell-generated secrets and must not be committed.

### Variable reference

| Variable | Used by | Purpose |
| --- | --- | --- |
| `DOCLING_API_KEY` | Docling | API key required to use the `docling-serve` API. **Sensitive — generate a strong random value and never commit it.** |
| `DOCLING_PORT` | Docling | Host port the `docling-serve` API is published on. |
| `GGML_VK_VISIBLE_DEVICES` | llama.cpp | Selects which Vulkan GPU (by index) the container can see and use. Can contain a single index or multiple separated by commas with values like `0`, `0,1`, `1,0`. |
| `LLAMA_ARG_ENDPOINT_METRICS` | llama.cpp | Enables the Prometheus-compatible metrics endpoint (`--metrics`; disabled by default, `1` = on). |
| `LLAMA_ARG_ENDPOINT_SLOTS` | llama.cpp | Exposes the slots monitoring endpoint (`--slots`; enabled by default, `0` = off). |
| `LLAMA_ARG_SLEEP_IDLE_SECONDS` | llama.cpp | Value passed to `--sleep-idle-seconds`: seconds of inactivity before the loaded model unloads itself from memory. |
| `LLAMA_DEVICE` | llama.cpp | Compute device passed to the server via `--device` (e.g. `Vulkan0`). |
| `LLAMA_MODELS_DIR` | llama.cpp | Directory passed to `--models-dir` that the router scans to discover models. |
| `LLAMA_MODELS_MAX` | llama.cpp | Value passed to `--models-max`: maximum number of models resident in memory at once. |
| `LLAMA_MODELS_PRESET` | llama.cpp | Path passed to `--models-preset` of the INI file holding per-model runtime settings. |
| `LLAMA_PORT` | llama.cpp | Port the llama.cpp server listens on; also used for the host port mapping. |
| `OPEN_TERMINAL_API_KEY` | Open Terminal | API key required to use the `open-terminal` container's API. **Sensitive — generate a strong random value and never commit it.** |
| `OPEN_WEBUI_APP_PORT` | Open-WebUI | Host port the Open WebUI web interface is published on. |
| `SEARXNG_QUERY_URL` | Open-WebUI | Base URL that Open WebUI uses to query SearXNG for web search results (`<query>` is replaced with the search term). |
| `WEBUI_SECRET_KEY` | Open-WebUI | Secret key Open WebUI uses for signing sessions/tokens. **Sensitive — generate a strong random value and never commit it.** |
| `SEARXNG_HOST` | SearXNG | Address SearXNG listens on inside its port mapping (`[::]` = all interfaces, IPv6/IPv4). |
| `SEARXNG_PORT` | SearXNG | Host and container port the SearXNG instance is published on. |
| `SEARXNG_VERSION` | SearXNG | Version tag of the SearXNG image to run. |

## References

* podman-compose [https://github.com/containers/podman-compose]
* llama.cpp [https://github.com/ggml-org/llama.cpp]
* Open WebUI [https://openwebui.com]
* Open Terminal [https://github.com/open-webui/open-terminal]
* Docling [https://docling.ai]
* SearXNG [https://searxng.org]
* Valkey [https://valkey.io]
