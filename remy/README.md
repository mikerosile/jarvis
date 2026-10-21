# R.E.M.Y. - Reference Evaluation, Manipulation and Yielding

R.E.M.Y. is the document processing service of the Jarvis stack. It ingests documents (PDFs, images, Office files, etc.), converts them into structured data, and serves that data over HTTP so that **LOKI** — the Open WebUI chat assistant (`jarvis_loki`) — can process uploaded documents for RAG and other workflows. LOKI is wired up to this service via its `DOCLING_PARAMS` and `DOCLING_API_KEY` configuration in [../compose.yaml](../compose.yaml).

The service itself is provided by the [Docling](https://docling.ai/) application, run as a containerized API server (docling-serve) with an optional web UI for interactive testing.

> **Note:** This service does not offload processing to a GPU — all document processing occurs on the CPU. The `cpuset` directive in [../compose.yaml](../compose.yaml) (`cpuset: "8-15,24-31"`) pins the container to those logical CPUs because they are the higher-frequency cores of CCD1 on an AMD Ryzen 7950X3D processor, so document conversion workloads run on the faster half of the chip.

## Service Definition

R.E.M.Y. is defined in [../compose.yaml](../compose.yaml) under the `remy` service:

```yaml
  remy:
    image: quay.io/docling-project/docling-serve-rocm71:v1.35.0_rocm71
    container_name: jarvis_remy

    restart: unless-stopped

    ports:
      - ${DOCLING_PORT:-5001}:5001

    cpuset: "8-15,24-31"
```

Key points:

- **Image**: `quay.io/docling-project/docling-serve-rocm71:v1.35.0_rocm71` — a docling-serve build compiled against ROCm 7.1 (see [Building the Image](#building-the-image)).
- **Port**: The container port 5001 is published on host port `${DOCLING_PORT:-5001}`. LOKI talks to REMY at that endpoint using the shared `DOCLING_API_KEY`.
- **CPU pinning**: `cpuset: "8-15,24-31"` reserves a set of cores for document conversion workloads (OCR, layout analysis, table extraction).

The service exposes an API key (`DOCLING_API_KEY`) and enables optional features such as the web UI and remote LLM-powered picture descriptions — see the `environment` block in [../compose.yaml](../compose.yaml) for the full list of runtime settings.

## Configuration (Environment Variables)

The variables that configure R.E.M.Y. are defined in [../.env](../.env) under a `# Docling Settings` section:

| Variable | Example Value | Purpose |
|---|---|---|
| `DOCLING_API_KEY` | *(secret)* | API key required to authenticate requests, used by LOKI when submitting documents for processing. |
| `DOCLING_NUM_THREADS` | `8` | Number of CPU threads used for Docling's model inference (CPU portion of layout/OCR work). |
| `DOCLING_PORT` | `5001` | Host port the service is published on, mapped to container port 5001. |

A new API key can be generated with:

```sh
openssl rand -hex 32
```

Additional runtime settings specific to the docling-serve application (UI toggle, sync request timeout, worker counts, ROCm/GPU visibility such as `ROCR_VISIBLE_DEVICES`, etc.) are set inline in the `remy` environment block of [../compose.yaml](../compose.yaml).

## Building the Image

The [docling-serve](https://github.com/docling-project/docling-serve) project has been forked to
[github.com/mikerosile/docling-serve](https://github.com/mikerosile/docling-serve) by
[mikerosile](https://github.com/mikerosile).

Tag **`v1.35.0_rocm71`** of that fork adds an extra Makefile target,
`docling-serve-rocm71-image`, which builds the container image using **ROCm 7.1** — the same version of ROCm running on the host operating system (Fedora 44). This ensures the prebuilt models and kernels in the image match the driver/stack available at runtime:

```sh
git clone https://github.com/mikerosile/docling-serve.git
cd docling-serve
git checkout v1.35.0_rocm71
make docling-serve-rocm71-image
```

**Note:** If using Podman instead of Docker, build the image with:

```sh
CONTAINER_RUNTIME=podman make docling-serve-rocm71-image
```

The resulting image is what [../compose.yaml](../compose.yaml) consumes as
`quay.io/docling-project/docling-serve-rocm71:v1.35.0_rocm71`. AMD's guidance on GPU isolation
(`ROCR_VISIBLE_DEVICES`) referenced in the compose file is available in the [ROCm documentation](https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html#rocr-visible-devices).

## TODO

- Consider GPU offload as a future feature enhancement. PHIL may be able to swap in the GPU workload, as it does for THEO.

## References

- [Docling website](https://docling.ai/)
- [Docling GitHub repository](https://github.com/docling-project/docling)
- [Docling documentation](https://docling-project.github.io/docling/)
- [docling-serve upstream repository](https://github.com/docling-project/docling-serve)
- [mikerosile's docling-serve fork (`v1.35.0_rocm71`)](https://github.com/mikerosile/docling-serve/tree/v1.35.0_rocm71)
- [ROCm GPU isolation documentation](https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html#rocr-visible-devices)
