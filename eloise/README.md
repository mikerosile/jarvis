# E.L.O.I.S.E. - Expressive Language Output and Intonational Speech Engine

ELOISE is the text-to-speech component of the J.A.R.V.I.S. stack. It gives J.A.R.V.I.S. a "voice" by providing TTS services for LOKI (Open-WebUI): conversation replies are sent to ELOISE, which synthesizes them into audio that can be played back inside the chat UI. All speech is generated locally on this host — no text leaves the system for a third-party service.

The synthesis engine behind ELOISE is [Kokoro](https://github.com/hexgrad/kokoro), an open-weight TTS model with 82 million parameters that delivers near-large-model quality while remaining fast and cheap enough to run without a GPU.

## Role in the stack

| Property | Value |
| --- | --- |
| Application | Kokoro-FastAPI (`ghcr.io/remsky/kokoro-fastapi-gpu`) |
| Container name | `jarvis_eloise` |
| Published port | `8880:8880` |
| API style | OpenAI-compatible TTS API (`/v1/audio/speech`, model `kokoro`) |
| Engine | Kokoro-82M (Apache-licensed), CPU inference |

ELOISE declares a dependency on LOKI in the compose file, so the web UI that consumes it is up before ELOISE starts.

## Configuration

The service is defined in [`../compose.yaml`](../compose.yaml) as part of the `jarvis` stack:

```yaml
eloise:
  image: ghcr.io/remsky/kokoro-fastapi-gpu:latest
  container_name: jarvis_eloise

  restart: always

  depends_on:
    - loki

  ports:
      - 8880:8880
```

Notes on the configuration:

- **`image`** — this is the GPU build of Kokoro-FastAPI, but no `deploy.resources.reservations.devices` (NVIDIA) section or `--gpus` option is configured. The container therefore performs inference on the host CPU, which handles Kokoro-82M's lightweight model adequately for interactive use. If GPU acceleration is ever desired, an NVIDIA device reservation can be added without any other changes.
- **`ports`** — exposes the OpenAI-compatible TTS API on host port 8880:

  ```bash
  curl http://localhost:8880/v1/audio/speech \
       -H "Content-Type: application/json" \
       -d '{"model": "kokoro", "input": "Hello, world.", "voice": "af_heart"}' > speech.wav
  ```

- **`depends_on` / `restart`** — LOKI starts before ELOISE, and ELOISE is restarted automatically if it ever stops.

No environment variables or `.env` entries are needed: the deployment uses all of Kokoro-FastAPI's defaults except for what is pinned in the compose file above. There are also no bind mounts; model weights are cached inside the container on first run.

## Voices and Open-WebUI integration

Kokoro offers a small variety of built-in voices (English American `af_*` / `am_*`, English British `bf_*` / `bm_*`) rather than custom voice models. ELOISE does not require any local configuration of its own — the TTS endpoint URL and voice selection are configured through Open-WebUI's text-to-speech settings, which point at the API published on port 8880 and let each user pick a Kokoro voice per conversation. The integration procedure is described in the [Open-WebUI documentation](https://docs.openwebui.com/features/chat-conversations/audio/text-to-speech/kokoro-web-integration/).

J.A.R.V.I.S. itself is configured to speak using a blend of the `bm_fable` and `bm_george` voices (`bm` = British Male).

## References

- Kokoro software (project repository): https://github.com/hexgrad/kokoro
- Kokoro-82M model page and documentation: https://huggingface.co/hexgrad/Kokoro-82M
- Kokoro library on PyPI: https://pypi.org/project/kokoro/
- Kokoro web demo: https://voice-generator.pages.dev/
- Open-WebUI TTS — Kokoro integration: https://docs.openwebui.com/features/chat-conversations/audio/text-to-speech/kokoro-web-integration/
- Open-WebUI TTS — Kokoro-FastAPI (this image's) integration: https://docs.openwebui.com/features/chat-conversations/audio/text-to-speech/Kokoro-FastAPI-integration/
