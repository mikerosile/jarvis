# A.R.C. - Adaptive Reasoning Core

ARC is a self-contained llama.cpp language-model service that powers reasoning
workloads in the Jarvis stack. It runs the official
[`llama.cpp` Vulkan server image](https://github.com/ggml-org/llama.cpp) in
**router mode**: instead of loading a single model, `--models-dir` and
`--models-preset` are passed so that one llama-server instance dynamically hosts,
loads, and unloads multiple GGUF models on demand.

The service is defined by [`compose.yaml`](./compose.yaml) (project name
`jarvis-arc`) and configured through the adjacent [`.env`](./.env) file described below.

## Layout

```
arc/
├── compose.yaml   # Docker Compose definition of the ARC service
└── .env           # Environment variable overrides (port, device, GPU selection, ...)
```

The model library lives one directory up in [`../models`](../models/) — see that
repository's README for how models are downloaded and organized. It is bind-mounted
into the container at `/models`, which matches the paths used inside the preset file.

## Prerequisites

* Verify the discrete GPU is visible to containers by listing the available Vulkan devices:
```bash
podman run -ti --rm --device=/dev/kfd --device=/dev/dri ghcr.io/ggml-org/llama.cpp:server-vulkan --list-devices
```

Example output:
```
Available devices:
  Vulkan0: AMD Radeon RX 6950 XT (RADV NAVI21) (16368 MiB, 1798 MiB free)
```

## Running

From this directory:

```bash
docker compose up -d
```

The server exposes its OpenAI-compatible REST API on port `9931` (configurable via
`LLAMA_PORT`). With the default configuration, metrics are enabled at
`http://localhost:9931/metrics` and one model slot is held.

## The `.env` file

Docker Compose automatically loads a `.env` file from the directory containing the
compose file, so `arc/.env` supplies all `${VAR}` references in `compose.yaml`
without needing to pass them on the command line or export them into your shell.

Every reference in `compose.yaml` uses the `${VAR:-default}` form, meaning **every
variable is optional** — if it is not defined in `.env`, the shown default applies.
Edit `.env` to adapt the service without touching `compose.yaml`.

The current contents:

```dotenv
# System has iGPU at device index 0, force dGPU at index 1
GGML_VK_VISIBLE_DEVICES=1
LLAMA_PORT=9931
LLAMA_DEVICE="Vulkan0"
# Single-model mode (no models-preset.ini)
LLAMA_ARG_ENDPOINT_SLOTS=1
LLAMA_ARG_ENDPOINT_METRICS=1
LLAMA_ARG_SLEEP_IDLE_SECONDS=-1
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `GGML_VK_VISIBLE_DEVICES` | Vulkan GPU index the server may use. The host's iGPU occupies device 0, so this is set to `1` to force inference onto the dGPU; it also makes that GPU appear as `Vulkan0` inside the container (see `LLAMA_DEVICE`). | `1` |
| `LLAMA_PORT` | Host port for the API (`${LLAMA_PORT:-9931}:${LLAMA_PORT:-9931}` keeps host and container ports in sync). | `9931` |
| `LLAMA_DEVICE` | The Vulkan device name passed to `--device`. Must match the GPU made visible by `GGML_VK_VISIBLE_DEVICES` (renumbered to `Vulkan0`). | *(no default — set it)* |
| `REPO_ROOT` | Host root of the models repository, bind-mounted at `/models`. | `/home/stark/jarvis` |
| `LLAMA_MODELS_DIR` | In-container model directory for `--models-dir`. | `/models` |
| `LLAMA_MODELS_PRESET` | Preset file passed to `--models-preset`. | `/models/models-preset.ini` |
| `LLAMA_THREADS` | CPU thread count (`--threads`). | `8` |
| `LLAMA_PARALLEL` | Number of parallel slots. | `1` |
| `LLAMA_MODELS_MAX` | Maximum number of models resident at once. | `1` |
| `LLAMA_ARG_SLEEP_IDLE_SECONDS` | Sleep the model when idle (`-1` disables sleeping). | `-1` |
| `LLAMA_ARG_ENDPOINT_SLOTS` | Expose `/slots` endpoint. | `1` |
| `LLAMA_ARG_ENDPOINT_METRICS` | Expose `/metrics` endpoint. | `1` |

## Models and service configuration

Service configuration, the list of available models, and per-model parameters are all
defined in [`../models/models-preset.ini`](../models/models-preset.ini). The file is
read by llama-server via `--models-preset`:

- A global `[ * ]` section holds settings that apply to every model. Notably:
  - **`n-gpu-layers = -1`** — offloads **all** model layers to the GPU, leaving only
    the CPU for bookkeeping and non-offloadable work.
  - **`threads = 8`** — matches the eight CPU cores pinned for this container by the
    `cpuset` directive in [`compose.yaml`](./compose.yaml).
- One section per model maps a GGUF file to a friendly alias and may override any
  parameter (e.g. context size, KV-cache type, multimodal projection file, reasoning
  effort).

The models currently defined:

| Alias | Model file | Notes |
| --- | --- | --- |
| `gpt-oss-20b` | `llm/gpt-oss-20b-Q8_0.gguf` | Loaded on startup; medium reasoning effort, 64k context, multimodal (mmproj) support. |
| `Qwen2.5-Coder-3B` | `llm/qwen2.5-Coder-3B-Q5_K_M.gguf` | Small code model. |
| `Qwen3-14B` | `llm/Qwen3-14B-Q4_K_M.gguf` | Mid-size general model. |
| `Qwen3.5-35B-A3B` | `llm/Qwen3.5-35B-A3B-Q4_K_M.gguf` | MoE (A3B); multimodal (mmproj) support. |
| `Qwen3.8-27B-UD-IQ4_XS` | `llm/Qwen3.8-27B-UD-IQ4_XS.gguf` | Medium reasoning effort, 64k context; multimodal support. |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp` | `llm/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | Uses MTP draft-model speculative decoding; multimodal support. |

Models are fetched into the repository with its Makefile (see
[`../models/README.md`](../models/README.md)); they are referenced as `/models/llm/*.gguf`
from inside the container.

By default, llama.cpp is configured to keep only **a single LLM model loaded at a
time**: `LLAMA_MODELS_MAX` (default `1`) caps how many models may be resident in memory
simultaneously, so requesting another preset means loading it on demand and unloading
the one that was active. Since each load/unload cycle reads and evicts multi-gigabyte
GGUF files from disk, **fast storage such as an SSD or NVMe drive is important** — it
directly determines how quickly a model can be brought into memory when you switch.

## CPU pinning (`cpuset`)

The host CPU is an **AMD Ryzen 7950X3D**, which is built from **two core complex dies
(CCDs)**. CCD1 offers a higher clock frequency than CCD0, and inference engines favour
that sustained frequency over raw core count. The `cpuset` directive in
[`compose.yaml`](./compose.yaml) —

```yaml
cpuset: "8-15,24-31" # AMD Ryzen 9 7950X3D, CCD1 has higher clock freq
```

— pins the container to a fixed set of logical CPUs on the host so that llama.cpp's
threads land on (and stay on) the faster complex. Together with `--threads 8`, this
lets every thread run at full frequency instead of being migrated by the scheduler onto
CCD0, where it would be down-clocked.

## Monitoring

For live performance monitoring, use [`llamagputop`](https://github.com/mikerosile/llamagputop) — my fork of
the original project (see its own documentation for usage). Its
[`model_argument` branch](https://github.com/mikerosile/llamagputop/tree/model_argument) supports monitoring the performance
of a specific model when llama.cpp is running in router mode. ARC exposes its `/metrics`
endpoint (`LLAMA_ARG_ENDPOINT_METRICS=1`, see above), which llamagputop reads for its data.

## References

- [llama.cpp website](https://llama.cpp)
- [llama.cpp documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs) — includes the [llama-server / OpenAI-compatible API docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) used to configure this service
- [llama.cpp primary GitHub repository](https://github.com/ggml-org/llama.cpp)
