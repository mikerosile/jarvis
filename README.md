# J.A.R.V.I.S. — Just A Rather Very Intelligent System

A self-hosted general use A.I. stack built around the [Open-WebUI](https://openwebui.com) platform designed to leverage an AMD RDNA2 GPU with only 16GBs of VRAM targeted for single user use. It contains all the services you would expect from a web based A.I chat portal including web searching, coding, text-to-speech, image generation (and editing) and document parsing.  The services are containized for portability.

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

## Operating Environment
| Description | Specification |
| --- | --- |
| Operation System | Fedora 44 (x86_64) |
| CPU | AMD Ryzen 7950X3D |
| Sytem RAM | 128 GBs DDR5 (3600 MT/s) |
| Integrated GPU (iGPU) | AMD Raphael / Radeon 610M<br/>* Provides primary display |
| Dedicated GPU (dGPU) | AMD Radeon RX 6950 XT<br/>* Dedicated to inference workloads |
| Storage | Samsung SSD 990 PRO |

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

- **`Makefile`** — downloads LLM weights into place: execute with `make all`, or download inidividual weights by executing `make <model>`.

## Run-time Configuration

All runtime configuration lives in the `.env` file that is consumed by podman-compose for handling compose-level substitution. It holds the  stack-wide settings (ports, keys, performance settings). The file contains secrets, therefore an example has been provided in [.env.example](.env.example).

## Getting Started

- Follow the [Prerequisites](#prerequisites) section
- Review environemnt variables for configuration directives:
  - [./arc/.env](./arc/.env) - Note `GGML_VK_VISIBLE_DEVICES` and `LLAMA_DEVICE`
  - [.env.example](.env.example) - Copy .env.example to .env, then review and update hostname, API keys, etc.
- Review host CPU architecture from command output `lscpu --extended`, and update or remove `cpuset` directives
  - [./arc/compose.yaml](./arc/compose.yaml) -- `services.arc`
  - [compose.yaml](compose.yaml) -- `services.remy`
- Replace `your.host.name.com` in [.env.example](.env.example), [roxie/nginx.conf](roxie/nginx.conf) to a valid hostname which will host J.A.R.V.I.S.
- Obtain or create an SSL/TLS certificate for ROXIE ( See [roxie/README.md](roxie/README.md#certificates) for details. )
- **Assuming SELinux is enabled**, copy a root certificate or cert bundle (e.g. /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem) to /etc/ssl/certs/ca-certificates.crt to ensure Open WebUI (LOKI) can validate certificates.  Copying the file and bind mounting the file as a 'volume' is easier than dealing with modifying SELinux contexts.
- Execute `make all` from the [./models](./models) subdirectory to download the community A.I. models (**~100 GBs**)
- Execute [./scripts/install.sh](./scripts/install.sh) to install the required Systemd User services for PHIL, ARC and THEO.
  - Verify with [./scripts/status.sh](./scripts/status.sh)
- Start the rest of the stack using `podman-compose up -d` or `docker compose up -d` from the base (this) directory.

## Extending

### Open-WebUI Computer

Consider extending the environment with [Open-WebUI's Computer](https://openwebui.com/computer) service which adds:

- a self-hosted AI workstation accessible from any browser or mobile device
- agentic browser automation & scraping
- deep Open WebUI integration
- messaging & bot connectivity with services like Telegram or WhatsApp for mobile-first remote control

And zero AI required for core tools


### Conduit: Open WebUI & AI

Conduit is an open-source, native mobile client for Open WebUI, direct model endpoints, and self-hosted agents. Your chats live on your device first. Nothing routes through a backend the maintainer operates.

**Great app that supports voice or continued conversation with J.A.R.V.I.S.!**

More info at https://conduit.mobile/

## References

* InterGenJLU's jarvis project [https://github.com/InterGenJLU/jarvis]
* VisionQuest [https://en.wikipedia.org/wiki/VisionQuest]
* Fedora Linux [https://fedoraproject.org/]
* podman-compose [https://github.com/containers/podman-compose]
* Systemd User Services [https://wiki.archlinux.org/title/Systemd/User]<br/>I realize the documentation is for ArchLinux, but its a solid resource and the concepts also apply to Fedora.
* llama.cpp [https://github.com/ggml-org/llama.cpp]
* NGINX Reverse Proxy [https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/]
* Open WebUI [https://openwebui.com]
* Open Terminal [https://github.com/open-webui/open-terminal]
* Docling [https://docling.ai]
* SearXNG [https://searxng.org]
* Valkey [https://valkey.io]
* Kokoro TTS [https://kokorottsai.com]
* Faster Whisper [https://github.com/SYSTRAN/faster-whisper]
* Black Forest Labs - FLUX.2 [https://bfl.ai/models/flux-2]