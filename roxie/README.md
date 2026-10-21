# R.O.X.I.E. - Routing and Origin eXposure Interception Engine

ROXIE is the `nginx:alpine` reverse proxy that sits in front of **LOKI** (Open-WebUI). It terminates TLS at the edge so Open-WebUI can run unexposed inside the Docker network, adding a hardened security layer between clients and the LLM UI.

## Container Service (from `../compose.yaml`)

| Property | Value |
|---|---|
| Image | `docker.io/library/nginx:alpine` |
| Container name | `jarvis_roxie` |
| Published port | `8443:8443` (HTTPS only) |
| Restart policy | `unless-stopped` |

Mounted volumes (all read-only with SELinux relabeling):

- `./roxie/nginx.conf` → `/etc/nginx/conf.d/default.conf:ro,Z` — proxy configuration.
- `/etc/pki/tls/private/roxie.key` → `/etc/nginx/certs/privkey.pem:ro,Z` — TLS private key.
- `/etc/pki/tls/certs/roxie.crt` → `/etc/nginx/certs/fullchain.pem:ro,Z` — full certificate chain for `your.host.name.com`.

The upstream **LOKI** service (`ghcr.io/open-webui/open-webui:v0.12.3`, container `jarvis_loki`) has its host port mapping commented out, so it is only reachable on the Docker network — clients must go through ROXIE over HTTPS at `https://your.host.name.com:8443` (which matches its `WEBUI_URL`).

## Nginx Configuration (`./nginx.conf`)

> ## ⚠️ IMPORTANT — replace every reference to `your.host.name.com` with a valid hostname before use!
>
> The placeholder hostname **`your.host.name.com`** appears in three places that must all be changed to your real, reachable FQDN (matching the certificate you provision in [Certificates](#certificates)):
>
> 1. **`./nginx.conf`** — the `server_name your.host.name.com;` directive and both entries of the trusted-origin (`$cors_allow_origin`) map at the top.
> 2. **LOKI's environment** — the `WEBUI_URL` variable (set in `.env`, templated by `${WEBUI_URL}` on the `loki` service in `../compose.yaml`; see `.env.example`).
> 3. **The TLS certificate / key** — a cert issued for another name will not validate, so make sure it covers your chosen hostname (including any SANs).
>
> Change all of these consistently; if they disagree, browsers will show certificate errors or cross-origin requests will be blocked by the CORS map.

**Listening port.** The server block accepts HTTPS on port **8443**, set by the `listen 8443 ssl;` directive (which implicitly enables SSL for that listener). To change it, edit that directive to a new port and keep things in sync with two related places: the published port mapping `"8443:8443"` on the `roxie` service in `../compose.yaml`, and any references to `https://your.host.name.com:8443` (the trusted-origin map at the top of `nginx.conf` and LOKI's `WEBUI_URL`). Then restart ROXIE for the change to take effect.

**Upstream & TLS termination.** An `upstream openwebui` block points to `loki:8080` with an aggressive keepalive pool (128 connections, 30-minute timeout) so nginx keeps warm HTTP/1.1 connections to Open-WebUI. The single virtual server listens on port **8443** with TLS for `your.host.name.com`, enabling **HTTP/2** and only allowing **TLSv1.2/1.3** (no TLSv1/v1.1) with a modern ECDHE/DHE AES-GCM cipher suite, server-cipher preference off, and session tickets disabled with a 1-day shared session cache for fast resumption.

**Security headers.** Every response gets hardening headers: strict HSTS (`max-age=63072000; includeSubDomains; preload`), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and a restrictive referrer policy. The `Access-Control-Allow-Origin` header is driven by an `ngx_map` of trusted origins (currently only the host's own origin, `https://your.host.name.com:8443`) — unknown origins are not echoed back, so browsers block cross-origin abuse from random sites.

**Streaming-optimized `/api/`.** The API location is tuned for long-lived LLM completions and SSE streaming: gzip, response/request buffering, and caching are all disabled (`proxy_buffering off`, `tcp_nodelay on`, `X-Accel-Buffering: no`), proxy timeouts are extended to 1800 s (30 min) per phase, WebSocket `Upgrade`/`Connection` headers are forwarded via the `$connection_upgrade` map for chat connections, and CORS preflight (`OPTIONS`) requests are answered directly with a 204.

**WebSockets.** Paths under `/ws/` and `/socket.io/` get the same no-buffering treatment with 24-hour timeouts to keep persistent socket connections alive through idle periods.

**Static assets & default.** `/static/` is allowed to buffer, with `Cache-Control: public, max-age=604800, immutable` for efficient browser caching of the frontend bundle. The catch-all `location /` proxies everything else to Open-WebUI with standard `Host`, `X-Real-IP`, and `X-Forwarded-For/Proto` headers.

## Bypassing ROXIE

ROXIE is not strictly required. Open-WebUI (LOKI) can be accessed directly by simply exposing its port: uncomment the `ports` block in the `loki` service in `../compose.yaml` and remap its internal port 8080 to a host port of your choosing, then restart the service.

```yaml
    ports:
      - "${OPEN_WEBUI_APP_PORT:-3000}:8080"
```

Keep in mind that doing so exposes Open-WebUI without TLS termination or the security hardening ROXIE provides (TLS enforcement, HSTS, restricted CORS origins), and `WEBUI_URL` would need to point at whatever direct host/port is exposed instead.

## Certificates

To use ROXIE you must provide a valid SSL/TLS certificate and its matching private key for your hostname (`your.host.name.com` in this deployment). The files are mounted into the container as:

- Private key → `/etc/nginx/certs/privkey.pem`
- Certificate (full chain) → `/etc/nginx/certs/fullchain.pem`

If you don't have a certificate issued by a CA, a self-signed one can be generated with OpenSSL. This example creates both the private key and a 365-day self-signed certificate for `your.host.name.com`, including an extra Subject Alternative Name:

```bash
# Generate a 2048-bit RSA private key
openssl genrsa -out /etc/pki/tls/private/roxie.key 2048

# Create a self-signed certificate with CN + SAN, valid for 365 days
openssl req -x509 -newkey rsa:2048 -nodes \
    -key /etc/pki/tls/private/roxie.key \
    -out /etc/pki/tls/certs/roxie.crt \
    -days 365 \
    -subj "/CN=your.host.name.com" \
    -addext "subjectAltName=DNS:your.host.name.com,DNS:localhost,IP:127.0.0.1"
```

Browsers will warn about an untrusted self-signed certificate; accept the exception or install it into your trust store. For production use a CA-issued chain (e.g., Let's Encrypt) — place its full-chain file where `fullchain.pem` is expected and restart ROXIE.

## References

- [Open-WebUI documentation: configuring Nginx as a reverse proxy for Open-WebUI](https://docs.openwebui.com/reference/https/nginx) — this configuration follows that guide's requirements (buffering disabled for streaming, WebSocket upgrade support, and proper forwarded-protocol headers).
