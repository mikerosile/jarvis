# S.O.N.I.A. - Speech Observation and Neural Interpretation Assistant

SONIA is the speech-to-text component of the J.A.R.V.I.S. stack. It runs the [Whisper server](https://hub.docker.com/r/hwdsl2/whisper-server) software in a container to provide transcription (and translation) services for LOKI, so audio captured from an end user can be turned into text before it reaches the assistant. All processing happens locally on the host — audio is never sent to a third-party service.

## Role in the stack

| Property | Value |
| --- | --- |
| Application | Whisper server (`hwdsl2/whisper-server`) |
| Container name | `jarvis_sonia` |
| Published port | `9000:9000` |
| API style | OpenAI-compatible (Whisper) transcription/translation API |
| Active model | `base` (faster-whisper, ~145 MB), pre-downloaded and cached in this directory |

LOKI declares a dependency on SONIA in the compose file so the transcription service is up before the web UI that uses it.

## Configuration

The service is defined in [`../compose.yaml`](../compose.yaml) as part of the `jarvis` stack:

```yaml
sonia:
  image: hwdsl2/whisper-server:latest
  container_name: jarvis_sonia

  volumes:
    - ./sonia/data:/var/lib/whisper:Z

  ports:
    - 9000:9000

  depends_on:
    - loki

  restart: always
```

Notes on the configuration:

- **`image`** — the CPU build of `hwdsl2/whisper-server`; no GPU device or environment variables are required.
- **`volumes`** — `./sonia/data` is bind-mounted into the container at `/var/lib/whisper`, which is where whisper-server keeps its persistent state: selected model, cached weights, usage metadata, and an auto-generated API key. The `:Z` suffix relabels the directory for SELinux. This mount is what makes restarts fast — models are not re-downloaded after the first run.
- **`ports`** — exposes the transcription API on host port 9000.
- **`depends_on` / `restart`** — SONIA starts before LOKI and is restarted automatically if it ever stops.

No environment variables or `.env` entries are needed: this deployment uses all of whisper-server's defaults except for what is pinned in the compose file above.

## Runtime state (`./data`)

The mounted data directory currently holds:

- **`.model`** — records that the active model is `base` (weights live in `models--Systran--faster-whisper-base/`).
- **`.api_key` / `.auth_enabled`** — authentication is enabled; on a fresh install whisper-server generated an API key (`whisper-…`) automatically. Requests must present it as a bearer token:

  ```bash
  curl http://localhost:9000/v1/audio/transcriptions \
       -H "Authorization: Bearer $(cat data/.api_key)" \
       -F file=@audio.mp3 \
       -F model=whisper-1 \
       -F response_format=json
  ```

- **`.port` / `.server_addr`** — the published port and address recorded at startup.
- Everything else (`blobs/`, `xet/`, `.whisper-usage/`) is Hugging Face cache internals used to store the downloaded model files.

The API also supports OpenAI's standard options: `response_format=json|srt|vtt`, `language` hints, per-request beam size, and SSE streaming (`stream=true`), as well as a translation endpoint at `/v1/audio/translations`.

## References

- Whisper (OpenAI): https://github.com/openai/whisper
- whisper-server Docker image: https://hub.docker.com/r/hwdsl2/whisper-server
- docker-whisper project page: https://github.com/hwdsl2/docker-whisper
- faster-whisper engine: https://github.com/Systran/faster-whisper
- OpenAI audio transcription API reference (the interface this server mirrors): https://platform.openai.com/docs/api-reference/audio/createTranscription
