# P.H.I.L. - Protocol Handler for Intelligent Logistics

PHIL is a host-side FastAPI service (see [app.py](app.py)) that acts as the **GPU workload orchestrator** for JARVIS. The host GPU has limited VRAM (16 GB) and can hold only one model at a time, so PHIL arbitrates access to it, ensuring that **only one service is using the GPU at any given moment**.

PHIL controls access for two containerized workloads:

| Service | Role | Container | systemd unit |
| --- | --- | --- | --- |
| **ARC** | LLM | `jarvis-arc` | `jarvis-arc.service` — default GPU owner, enabled at boot |
| **THEO** | Image generation (FLUX) | `jarvis-theo` | `jarvis-theo.service` — started on demand only |

ARC owns the GPU by default. When an image request arrives, PHIL temporarily hands the GPU to THEO for the duration of that single request, then returns it to ARC afterwards.

## Inspiration

PHIL was created from the inspiration provided by the [InterGenJLU "jarvis" project](https://github.com/InterGenJLU/jarvis#local-image-generation-flux2), and specifically from the GPU swap mechanism it implements: temporarily stopping one workload so another can claim the VRAM, running its job, then handing the GPU back.

## Running as a systemd service

PHIL runs as a **user-scope systemd service** defined in [../systemd/user/jarvis-phil.service](../systemd/user/jarvis-phil.service). Everything is user-scoped — `systemctl --user` and rootless podman, no sudo anywhere.

```
$ systemctl --user start jarvis-phil
$ systemctl --user status jarvis-phil
$ journalctl --user -u jarvis-phil -f        # logs (SyslogIdentifier=jarvis-phil)
```

The unit runs the app from the project venv (`phil/.venv`, created by `scripts/install.sh`):

```
ExecStart=%h/jarvis/phil/.venv/bin/python -m uvicorn phil.app:app --host 0.0.0.0 --port 8191
```

PHIL binds to **all interfaces on port 8191** (unlike the GPU containers, which stay loopback-only), because the containerized Open-WebUI reaches it via `host.containers.internal` or the LAN IP. PHIL starts ordered after `jarvis-arc.service`, but self-heals if ARC is down — a failed supervisor start never blocks boot.

## How an image request flows

PHIL sits in front of THEO's FLUX service as an OpenAI-compatible images API proxy (the built-in "openai" engine of Open-WebUI v0.12.x POSTs to PHIL's `/v1/images/*` endpoints). Each request runs the following swap sequence:

1. **Stop ARC** — `systemctl --user stop jarvis-arc.service`, with a `podman pod stop pod_jarvis-arc` fallback, then poll ARC's health endpoint until its port is down (VRAM released).
2. **Start THEO** — `systemctl --user start jarvis-theo.service`, then poll `http://127.0.0.1:8190/health` until it reports `"ready"` (model loaded into VRAM).
3. **Generate** — forward the request unchanged to FLUX's `/v1/images/*`.
4. **Stop THEO** and **restart ARC** — so the GPU goes back to its default owner.

Guarantees:

- **One generation at a time.** A lock serializes swaps; a concurrent request gets HTTP `409` instead of double-claiming the GPU.
- **ARC is always restored.** Steps 4–5 run in a `finally` block, so even if the client (Open-WebUI) times out or the browser goes away mid-generation, the LLM comes back and the GPU is never left half-claimed.
- **Self-healing.** If PHIL finds both workloads down (e.g. after a crash), it proceeds anyway — the `finally` block ends with ARC running.
- **Health ports are the source of truth.** The unit process (`podman-compose up`) can die or sit in a `failed` state while the container keeps holding the GPU; that's why PHIL falls back to direct `podman pod stop/start` when `systemctl` alone doesn't release or restore the port.

Bad payloads (unparseable JSON, empty prompts, undecodable images) are rejected with HTTP `422` *before* any GPU swap starts, so invalid requests don't burn ~95 s of swapping for nothing.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | PHIL's own state: `{status, llm, flux, generating, base_url}` (for scripts / status checks; always unauthenticated) |
| `POST` | `/v1/images/generations` (also served at `/images/generations`) | Text-to-image — full swap + generate flow, JSON body forwarded to FLUX and response returned as-is (minus the cosmetic `url` field, keeping `b64_json`) |
| `POST` | `/v1/images/edits` (also served at `/images/edits`) | Image edit — multipart body forwarded verbatim (boundary intact) to FLUX |

The auto-generated OpenAPI schema lives under the usual API paths: interactive docs at [`/api/docs`](http://localhost:8191/api/docs), schema at `/api/openapi.json`.

Error responses mirror the failure mode: `403` bad token, `409` another generation in flight, `422` rejected payload (either from PHIL's pre-swap validation or surfaced from FLUX's own 4xx detail), `502` FLUX request failed upstream, `503` the GPU swap itself could not happen (e.g. ARC couldn't be stopped).

## Configuration (environment variables)

All configuration is via environment variables, set in the unit file:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://127.0.0.1:9931` | ARC's API base URL (port comes from ARC's compose) |
| `FLUX_BASE_URL` | `http://127.0.0.1:8190` | THEO's FLUX base URL |
| `SWAP_PORT` | `8191` | Port PHIL binds to |
| `IMAGE_OUTPUT_DIR` | `/home/stark/jarvis/theo/generated_images` | Where generated images land (must match the mount in THEO) |
| `IMAGE_BASE_URL` | auto-detected LAN IP (`http://<lan-ip>:8191`) | Informational: reported at startup and in `/health`; override if PHIL sits behind a proxy or on a different interface. Use it when configuring Open-WebUI's backend card |
| `SWAP_API_TOKEN` | *(unset = no auth)* | If set, the `/v1/images/*` endpoints require the token as an `X-Swap-Token: <token>` header **or** `Authorization: Bearer <token>` (the form Open-WebUI's openai engine sends). `/health` stays open. |

## Troubleshooting

If PHIL logs a `CRITICAL:` message — it could not stop FLUX before restoring ARC, or could not bring ARC back up — the GPU may be left in an inconsistent state and needs manual intervention: stop THEO's workload so the VRAM is released, then start ARC again (verify via the `/health` endpoint that `flux` is false and `llm` is true). After that PHIL self-heals on the next image request.
