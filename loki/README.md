# L.O.K.I. - Logical Operations and Knowledge Interface

LOKI is the primary user interface to the J.A.R.V.I.S. system, built on the [Open-WebUI](https://openwebui.com) platform. Every conversation with the stack — chatting with models, ingesting documents and knowledge, web searches, invoking tools — happens here; all of the other services (ARC, ROXIE, SHELLY, ARCHIE, REMY, SONIA, ELOISE…) are headless back ends that exist to serve LOKI. It is named for Loki of Norse myth — the trickster who brings chaos into an ordered world — because humans play exactly that role in a computer environment: we are the unknown element, the interface through which human unpredictability enters J.A.R.V.I.S.'s A.I. system, as is expanded on below.

The service is defined under `services.loki` in [`../compose.yaml`](../compose.yaml), with its runtime values supplied through the variables documented in [`../.env`](../.env) (see also the root [README](../README.md)).

## Why "L.O.K.I."?

In Norse mythology, Loki was the trickster — cunning, unpredictable, and a deliberate disruptor of order. He wandered among the gods of Asgard bringing mischief and mayhem into their rigid world, yet nothing in those myths happened *around* him: every story he broke was also reshaped by his interference. He was chaos given a name.

Humans occupy exactly that role in any computer environment. To an engineered, deterministic system we are the unknown element — the agent who improvises, breaks patterns, asks questions no training run anticipated, and injects genuine unpredictability into otherwise orderly machinery. Humans are, in effect, the chaos bringer to every A.I. system.

LOKI's name is a deliberate mirror of that role: it is *the* human interface, the gateway through which human chaos enters J.A.R.V.I.S.'s ordered stack. The inference server, proxy, search engine, and voice services all behave predictably; LOKI is where intent — creative, inconsistent, chaotic human intent — arrives from outside the system. As Loki did in the myths, it disrupts the status quo of every model loaded into ARC and reshapes what comes back out.

## Container Service (from `../compose.yaml`)

| Property | Value |
|---|---|
| Image | `ghcr.io/open-webui/open-webui:v0.12.3` |
| Container name | `jarvis_loki` |
| Published ports | **none** — intentionally not exposed (see [Network Posture](#network-posture-no-exposed-port)) |
| Internal port | `8080` (container network only) |
| Restart policy | `unless-stopped` |

### Volumes

- `loki-data:/app/backend/data` — named volume backing Open-WebUI's state: its database, user accounts, conversation history, uploaded knowledge files, and settings. Persists across restarts and container rebuilds.
- `./etc/ssl/certs/ca-certificates.crt → /etc/ssl/certs/ca-certificates.crt:ro,Z` — the host's CA trust bundle, mounted read-only with SELinux relabeling, so Open-WebUI verifies outbound TLS against the same store as the rest of the system (including private CAs). Consumed via `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE`.

## Environment Configuration (from `../.env`)

The `loki` service block in [`../compose.yaml`](../compose.yaml) reads every runtime value from [`../.env`](../.env), with safe fallback defaults written inline in the compose file. The `.env` file holds generated secrets and must never be committed — none of those values are repeated here.

| Variable | Default (compose) / Value (`.env`) | Purpose |
|---|---|---|
| `WEBUI_URL` | default `http://localhost:8080`; `.env`: `https://your.host.name.com:8443` | The public URL Open-WebUI advertises as its own endpoint (links, CORS context). Must match ROXIE's HTTPS entry point, since that is how clients actually reach LOKI. |
| `WEBUI_SECRET_KEY` | default `change-me`; generated value in `.env`. **Sensitive** | Signing key for sessions and tokens. Generate a strong random value; never commit or print it. |
| `WEBUI_NAME` | fixed: `Open WebUI` | Instance name displayed by the UI (header/about). |
| `CORS_ALLOW_ORIGIN` | default `*`; not set in `.env` | Origins permitted to make cross-origin API calls against this instance's HTTP API. Browser traffic arriving through ROXIE is additionally constrained there. |
| `AIOHTTP_CLIENT_SESSION_SSL` | fixed: `"false"` | Disables TLS certificate verification for Open-WebUI's outbound `aiohttp` requests, so it can call internal or self-signed endpoints that are not in the public trust store (e.g. other J.A.R.V.I.S. services). |
| `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` | fixed: `/etc/ssl/certs/ca-certificates.crt` | Point Python's HTTP clients at the mounted host CA bundle, so TLS verification uses the full system trust store rather than the image's baked-in defaults. |
| `ENABLE_CONTEXT_COMPACTION` | fixed: `"true"` | Long conversations are compacted (summarized/trimmed) to fit within a model's context window instead of hard-failing at its limit. |
| `CONTEXT_COMPACTION_TOKEN_THRESHOLD` | default `65000`; `.env`: `120000` | Context length, in tokens, beyond which compaction is triggered. Sized for the models ARC serves locally. |
| `ENABLE_OPENAI_API` | fixed: `"false"` | Disables Open-WebUI's own OpenAI-compatible API endpoint — LOKI is exposed to humans through its web UI, not as an API server for other systems. |
| `ENABLE_OLLAMA_API` | fixed: `"false"` | Same, for the Ollama-compatible endpoint. J.A.R.V.I.S. talks to ARC directly rather than an external model host. |
| `ENABLE_RAG_LOCAL_WEB_FETCH` | default `true`; also set in `.env` | Lets users supply URLs that Open-WebUI fetches itself and folds into knowledge/RAG context. |
| `ENABLE_WEB_SEARCH` | fixed: `"true"` | Enables in-chat web search, backed by ARCHIE's SearXNG instance. |
| `WEB_SEARCH_ENGINE` | fixed: `searxng` | Search backend used for in-chat queries — points at [ARCHIE](../archie/README.md). |
| `WEB_SEARCH_RESULT_COUNT` | fixed: `3` | Number of search results returned per query. |
| `WEB_SEARCH_CONCURRENT_REQUESTS` | fixed: `10` | Maximum concurrent requests issued while searching. |
| `SEARXNG_QUERY_URL` | default `http://host.docker.internal:8089/search?q=<query>`; `.env`: the public SearXNG URL on port 8089 | Endpoint LOKI queries for web-search results; `<query>` is substituted with the search term. Reaches ARCHIE, which listens on port 8089. |
| `DOCLING_PARAMS` | fixed JSON preset | Pipeline parameters handed to [REMY](../remy/README.md) (Docling) when LOKI ingests documents: OCR enabled (`do_ocr`, tesseract/eng), PDF backend `pypdfium2`, table mode `accurate`. |

## Network Posture — No Exposed Port

LOKI does **not** require its port to be exposed, and the `ports` mapping in [`../compose.yaml`](../compose.yaml) is intentionally left commented out:

```yaml
    #ports:
    #  - "${OPEN_WEBUI_APP_PORT:-3000}:8080"
```

It is intended to sit behind a reverse proxy, which is exactly what the [ROXIE](../roxie/README.md) service provides. ROXIE (`nginx:alpine`) terminates TLS at the edge on port `8443` and proxies all traffic — API calls, WebSockets, static assets — upstream to LOKI's internal-only address `loki:8080`. Clients therefore never touch Open-WebUI directly; they reach J.A.R.V.I.S. over HTTPS at `https://your.host.name.com:8443` (the same URL configured as `WEBUI_URL`).

Keeping LOKI unexposed means the single public entry point gets ROXIE's hardening for free: TLS-only with modern protocols/ciphers, HSTS and clickjacking/security headers, a restricted CORS origin map, streaming-optimized no-buffering proxy rules for long LLM completions, and 24-hour WebSocket timeouts. See the [ROXIE README](../roxie/README.md) for how that is achieved.

If you ever *do* want direct access (skipping TLS termination and ROXIE's security layer — not recommended), uncomment the `ports` block: with `.env`'s `OPEN_WEBUI_APP_PORT=8086` that publishes host port 8086 to the container's 8080. Point `WEBUI_URL` at whatever direct address you expose, then restart the service.

## Role in the J.A.R.V.I.S. Stack

- **Front door:** LOKI is where humans enter the system and where human "chaos" (see above) meets the A.I. back ends.
- **Model access:** ARC's llama.cpp server is added as an OpenAI-compatible model connection from inside the UI at `http://host.containers.internal:9931/v1`.
- **Knowledge & search:** document uploads are parsed by REMY (Docling, per `DOCLING_PARAMS`) into the RAG knowledge base; in-chat web searches go to ARCHIE via `SEARXNG_QUERY_URL`; user-supplied URLs can be fetched locally (`ENABLE_RAG_LOCAL_WEB_FETCH`).
- **Tools & shell:** SHELLY (Open-Terminal) is connectable from Open-WebUI's settings using its `OPEN_TERMINAL_API_KEY` — its CORS is restricted to LOKI's origin, and multi-user mode is enabled. PHIL, the GPU supervisor, can be registered as an external OpenAPI tool server (`http://host.containers.internal:8191`) so image generation (THEO) is callable from chat.
- **Voice in/out:** SONIA (Whisper speech-to-text) and ELOISE (Kokoro text-to-speech) both declare `depends_on: loki` — they give J.A.R.V.I.S. its ears and voice, but LOKI is the conversation itself.
- **Persistence:** everything a user does in the UI — accounts, chats, knowledge base, settings — lives on the `loki-data` volume.

## Getting Started

1. Ensure the prerequisites from the [root README](../README.md) are met (podman environment, GPU groups, `loginctl enable-linger`).
2. Bring up the main stack from the repository root: `podman-compose -f compose.yaml up` — LOKI starts together with ROXIE and the other services in [`compose.yaml`](../compose.yaml).
3. Open `https://your.host.name.com:8443` (the address behind which LOKI sits) and create an account on first run; the first registered user becomes admin.
4. Wire up J.A.R.V.I.S.: add ARC as a model connection (`http://host.containers.internal:9931/v1`), connect SHELLY, register PHIL's tool server, then start talking.

## References

- [Open-WebUI website](https://openwebui.com)
- [Open-WebUI documentation](https://docs.openwebui.com) — including the [reverse proxy (Nginx) guide ROXIE follows](https://docs.openwebui.com/reference/https/nginx) and the [custom CA store reference used for the mounted trust bundle](https://docs.openwebui.com/troubleshooting/custom-ca)
- [Open-WebUI GitHub repository](https://github.com/open-webui/open-webui)
