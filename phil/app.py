"""PHIL - Protocol Handler for Intelligent Logistics
GPU workload supervisor for JARVIS

Host-side FastAPI service, run by the user-scope unit
jarvis-phil.service under a standard user. No sudo anywhere.

Open-WebUI has no way to run commands on the host. The "signal" it sends
is an HTTP call to this service (bound on 0.0.0.0:8191). In Open-WebUI
v0.12.x the **built-in image backends** (Settings → Interfaces → Image
Generation / Edit Image cards, engine "openai") POST to the /v1/images/*
endpoints below.

The 16 GB GPU can hold only one model at a time. This service arbitrates
between the two containerized GPU workloads:

    LLM   podman pod `pod_jarvis-arc` — unit `jarvis-arc` (project-managed,
          default GPU owner, enabled at boot)
    FLUX  podman container `jarvis-theo` — unit `jarvis-theo` (on demand)

Swap sequence for one image request:

    1. stop  the LLM   (systemctl --user, plus `podman pod stop` fallback)
    2. start the FLUX  (systemctl --user start jarvis-theo)
    3. poll  http://127.0.0.1:8190/health until "ready"
    4. POST  to FLUX /v1/images/* (OpenAI-compatible proxy; the body is
        forwarded unchanged to the same FLUX path)
    5. stop  the FLUX  (systemctl --user stop jarvis-theo)
    6. start the LLM   (systemctl --user, plus `podman pod start` fallback)

Guarantees:
  * Only one generation at a time (threading.Lock; concurrent requests get
    HTTP 409).
  * Steps 5-6 always run in a finally block, so even if the HTTP client
    (Open-WebUI) times out or the browser goes away, the LLM is restored
    and the GPU is never left half-claimed.
  * Self-healing: if it finds both workloads down (e.g. after a crash),
    it simply proceeds — the finally block ends with the LLM running.

Why both `systemctl --user` and direct `podman` calls? The unit process
(`podman-compose up`) can die or sit in a `failed` state while the
container keeps running; in that case `systemctl stop` is a no-op and
the pod keeps holding the GPU. The health endpoints are the source of
truth, and direct pod commands are the fallback.

Endpoints:
    GET  /health              -> service state (for scripts/status.sh)
    POST /v1/images/generations -> OpenAI Images API text-to-image proxy
                                   (also served at /images/generations; full
                                   swap + gen flow, forwards to FLUX)
    POST /v1/images/edits         -> OpenAI Images API image-editing proxy
                                   (also served at /images/edits; full swap +
                                   gen flow, forwards the multipart body to FLUX)
"""

from contextlib import contextmanager
import io
import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from starlette.datastructures import Headers

logger = logging.getLogger("jarvis_phil")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_UNIT = "jarvis-arc.service"  # project-managed (systemd/user/jarvis-arc.service)
LLM_POD = "pod_jarvis-arc"
FLUX_UNIT = "jarvis-theo.service"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:9931")
LLM_HEALTH_URL = LLM_BASE_URL.rstrip("/") + "/health"
FLUX_BASE_URL = os.environ.get("FLUX_BASE_URL", "http://127.0.0.1:8190")
FLUX_HEALTH_URL = FLUX_BASE_URL.rstrip("/") + "/health"
FLUX_GEN_V1_URL = FLUX_BASE_URL.rstrip("/") + "/v1/images/generations"
FLUX_EDIT_V1_URL = FLUX_BASE_URL.rstrip("/") + "/v1/images/edits"

STOP_TIMEOUT = 60            # s to wait for a workload's port to go down
FLUX_STARTUP_TIMEOUT = 120  # s for the FLUX container to load the model
LLM_STARTUP_TIMEOUT = 120   # s for the LLM container to load the model
GENERATION_TIMEOUT = 600    # s for the diffusion run itself

OUTPUT_DIR = Path(
    os.environ.get("IMAGE_OUTPUT_DIR", "/home/stark/jarvis/theo/generated_images")
)
SWAP_PORT = int(os.environ.get("SWAP_PORT", "8191"))
SWAP_API_TOKEN = os.environ.get("SWAP_API_TOKEN", "")  # "" = no auth (trusted LAN)


def _default_base_url() -> str:
    """Base URL for image links: env override, else auto-detected LAN IP."""
    override = os.environ.get("IMAGE_BASE_URL")
    if override:
        return override.rstrip("/")
    host = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host = s.getsockname()[0]
        s.close()
    except OSError:
        pass
    return f"http://{host}:{SWAP_PORT}"


IMAGE_BASE_URL = _default_base_url()

# ---------------------------------------------------------------------------
# App + state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FLUX Image Tools",
    version="2.0.0",
    # Keep FastAPI's auto-generated schema under /api/ (docs at
    # /api/docs) instead of the default /openapi.json.
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url=None,
)
_LOCK = threading.Lock()  # only one GPU swap/generation at a time


class _MultipartBodyCache:
    """Read multipart request bodies once, before the endpoint handlers.

    FastAPI resolves `Form`/`File` parameters (consuming the request
    stream) before the handler body runs, so a later
    ``await request.body()`` would raise ``RuntimeError: Stream consumed``.
    This middleware drains the body first, stashes it in the ASGI scope,
    and replays it through a wrapped ``receive`` — form parsing keeps
    working exactly as before, and handlers that need to forward the
    verbatim multipart bytes can read ``scope["_multipart_body"]``.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        content_type = Headers(scope=scope).get("content-type", "").lower()
        if not content_type.startswith("multipart/form-data"):
            await self.app(scope, receive, send)
            return

        # Drain the request body now and remember it for downstream use.
        # Stop when the final body chunk arrives (more_body False); on a
        # keep-alive connection a later receive() would block until the
        # client disconnects.
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break
        body = b"".join(chunks)
        scope["_multipart_body"] = body

        sent = False

        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


app.add_middleware(_MultipartBodyCache)


# ---------------------------------------------------------------------------
# Workload plumbing (no sudo — user-scope systemd + rootless podman)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 90) -> tuple[int, str]:
    """Run a command, returning (returncode, stderr_or_message)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, (result.stderr or result.stdout).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timeout: {' '.join(cmd)}"


def _systemctl_user(action: str, unit: str) -> tuple[bool, str]:
    """Run `systemctl --user <action> <unit>` (own user bus, no sudo)."""
    logger.info("systemctl --user %s %s", action, unit)
    rc, err = _run(["systemctl", "--user", action, unit])
    return rc == 0, err


def _podman_pod(action: str, pod: str) -> tuple[bool, str]:
    """Run `podman pod <action> <pod>` (rootless, own pods, no sudo)."""
    logger.info("podman pod %s %s", action, pod)
    rc, err = _run(["podman", "pod", action, pod], timeout=120)
    return rc == 0, err


def _llm_running() -> bool:
    """True while the LLM API answers (source of truth: the health port)."""
    try:
        return requests.get(LLM_HEALTH_URL, timeout=3).status_code == 200
    except requests.RequestException:
        return False


def _flux_running() -> bool:
    try:
        return requests.get(FLUX_HEALTH_URL, timeout=3).status_code == 200
    except requests.RequestException:
        return False


def _wait_llm_down(timeout: int = STOP_TIMEOUT) -> bool:
    """Poll until the LLM health port stops answering (VRAM released)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _llm_running():
            return True
        time.sleep(1)
    logger.error("Timed out waiting for the LLM (%s) to go down", LLM_HEALTH_URL)
    return False


def _wait_llm_up(timeout: int = LLM_STARTUP_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(LLM_HEALTH_URL, timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2)
    logger.error("Timed out waiting for the LLM (%s) to come up", LLM_HEALTH_URL)
    return False


def _wait_flux_ready(timeout: int = FLUX_STARTUP_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(FLUX_HEALTH_URL, timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ready":
                return True
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2)
    logger.error("Timed out waiting for FLUX (%s) to become ready", FLUX_HEALTH_URL)
    return False


def _wait_flux_down(timeout: int = STOP_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _flux_running():
            return True
        time.sleep(1)
    logger.error("Timed out waiting for FLUX (%s) to go down", FLUX_HEALTH_URL)
    return False


# ---------------------------------------------------------------------------
# GPU swap lifecycle
# ---------------------------------------------------------------------------

def _stop_llm() -> bool:
    """Free the GPU: stop the LLM workload.

    `systemctl --user stop` is a no-op when the unit is already inactive
    (e.g. `failed`) even though the pod keeps running, so fall back to
    `podman pod stop` when the port is still answering.
    """
    ok, err = _systemctl_user("stop", LLM_UNIT)
    if not ok:
        logger.warning("systemctl stop %s returned non-zero (unit may already be inactive): %s",
                       LLM_UNIT, err)
    if not _llm_running():
        return True
    logger.info("LLM still up after unit stop — stopping the pod directly")
    ok, err = _podman_pod("stop", LLM_POD)
    if not ok:
        logger.error("podman pod stop %s failed: %s", LLM_POD, err)
        return False
    return _wait_llm_down()


def _start_llm() -> bool:
    """Restore the LLM workload. The unit's ExecStart runs
    `podman-compose up`; if the unit misbehaves, start the pod directly.
    Health is the proof."""
    ok, err = _systemctl_user("start", LLM_UNIT)
    if not ok:
        logger.warning("systemctl start %s returned non-zero: %s (verifying via health)",
                       LLM_UNIT, err)
    if _wait_llm_up():
        return True
    logger.warning("LLM not up via unit — starting the pod directly")
    ok, err = _podman_pod("start", LLM_POD)
    if not ok:
        logger.error("podman pod start %s failed: %s", LLM_POD, err)
        return False
    return _wait_llm_up(timeout=60)


def _start_flux() -> bool:
    ok, err = _systemctl_user("start", FLUX_UNIT)
    if not ok:
        logger.error("Could not start %s: %s", FLUX_UNIT, err)
        return False
    return _wait_flux_ready()


def _stop_flux() -> bool:
    ok, err = _systemctl_user("stop", FLUX_UNIT)
    if not ok:
        logger.error("Could not stop %s: %s", FLUX_UNIT, err)
        return False
    return _wait_flux_down()


def _restore_llm() -> None:
    """Best-effort restore: stop FLUX (if up), bring the LLM back.

    Idempotent — safe to call multiple times. On failure it logs loudly;
    the manual recovery procedure is in README.md.
    """
    if _flux_running():
        if not _stop_flux():
            logger.error(
                "CRITICAL: FLUX is still running; not starting the LLM to avoid a "
                "VRAM conflict — manual intervention needed (see README.md)"
            )
            return
    if not _llm_running():
        if _start_llm():
            logger.info("LLM restored, GPU back on the LLM container")
        else:
            logger.error(
                "CRITICAL: could not restore the LLM — manual intervention needed "
                "(see README.md)"
            )


@contextmanager
def _flux_session(timings: dict, stop_llm_failure: str):
    """Hold the GPU for FLUX for the duration of one request.

    Entry: stop the LLM and start FLUX (both skipped if FLUX is already
    up). A failed swap raises 503; a failed FLUX start also restores the
    LLM first so the GPU goes back to its owner. Exit: restore the LLM
    (idempotent, best-effort). Populates `timings` with llm_stop_s,
    flux_start_s, restore_start and llm_restore_s.
    """
    if not _flux_running():
        t0 = time.time()
        if not _stop_llm():
            logger.error("Could not stop the LLM container before starting FLUX")
            raise HTTPException(503, detail=stop_llm_failure)
        timings["llm_stop_s"] = time.time() - t0
        t0 = time.time()
        if not _start_flux():
            _restore_llm()
            raise HTTPException(503, detail="Could not start the FLUX container.")
        timings["flux_start_s"] = time.time() - t0
    else:
        timings["llm_stop_s"] = 0.0
        timings["flux_start_s"] = 0.0
    try:
        yield
    finally:
        timings["restore_start"] = time.time()
        _restore_llm()
        timings["llm_restore_s"] = time.time() - timings["restore_start"]


# ---------------------------------------------------------------------------
# Auth (optional)
# ---------------------------------------------------------------------------

def _check_token(request: Request) -> None:
    """Accept the token as X-Swap-Token <token> or Authorization: Bearer <token>.

    The Bearer form is what Open-WebUI's built-in "openai" engine sends
    (Authorization: Bearer <API key>), so the backend card can use the
    same secret as its API key.
    """
    if not SWAP_API_TOKEN:
        return
    if request.headers.get("X-Swap-Token") == SWAP_API_TOKEN:
        return
    bearer = request.headers.get("Authorization", "")
    if bearer.startswith("Bearer ") and bearer[len("Bearer "):] == SWAP_API_TOKEN:
        return
    raise HTTPException(
        403, "Missing or invalid token (send X-Swap-Token or Authorization: Bearer)"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
def log_config():
    logger.info(
        "jarvis-phil up: llama=%s flux=%s base_url=%s token=%s",
        _llm_running(), _flux_running(),
        IMAGE_BASE_URL, "set" if SWAP_API_TOKEN else "off",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm": _llm_running(),
        "flux": _flux_running(),
        "generating": _LOCK.locked(),
        "base_url": IMAGE_BASE_URL,
    }


def _response_detail(resp: requests.Response) -> str:
    """Best-effort detail string from an error response body.

    Prefers the JSON `detail` field when the body is JSON; otherwise
    returns the raw text.
    """
    if resp.headers.get("content-type", "").startswith("application/json"):
        try:
            detail = resp.json().get("detail")
            if detail is not None:
                return str(detail)
        except ValueError:
            pass
    return resp.text


def _validate_image_bytes(image_bytes: bytes) -> None:
    """Confirm the raw image bytes are a real, decodable image.

    Runs before the GPU swap so invalid payloads fail fast with 422
    instead of burning ~95s swapping the GPU for nothing.
    """
    if not image_bytes:
        raise HTTPException(422, "Uploaded image is empty")
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception as e:
        raise HTTPException(422, f"image is not a decodable image (PNG/JPG/WebP expected): {e}")


# ---------------------------------------------------------------------------
# OpenAI Images API proxies (Open WebUI v0.12.x built-in "openai" engine)
#
# These forward to FLUX's own /v1/images/* endpoints, running the same
# GPU-swap lifecycle around each request. Open WebUI's built-in image
# generation (openai engine) uses these endpoints. Responses are returned
# as-is apart from stripping the cosmetic `url` field (Open WebUI prefers
# `url` over `b64_json` when both exist).
# ---------------------------------------------------------------------------

def _strip_urls(body: dict) -> dict:
    """Drop the `url` field from each generated-image item in place.

    FLUX's `url` is a cosmetic `/images/...` path that nothing serves.
    Open WebUI's openai engine prefers `url` over `b64_json` when both are
    present, so only the b64_json payload is kept.
    """
    for item in body.get("data", []):
        if isinstance(item, dict):
            item.pop("url", None)
    return body


def _forward_edit(raw_body: bytes, content_type: str) -> requests.Response:
    """Forward the raw multipart edit body to FLUX verbatim (boundary intact)."""
    return requests.post(
        FLUX_EDIT_V1_URL,
        data=raw_body,
        headers={"content-type": content_type},
        timeout=GENERATION_TIMEOUT,
    )


@app.post("/v1/images/generations")
@app.post("/images/generations")
async def openai_generate(request: Request):
    """OpenAI Images API proxy: text-to-image (forwards the JSON body to FLUX)."""
    _check_token(request)

    if not _LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another image is already being generated; try again shortly.")

    try:
        # Pre-swap validation: parse the JSON body so bad payloads fail fast
        # instead of burning a ~95s GPU swap for nothing.
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            raise HTTPException(422, "Request body must be a JSON object")

        logger.info("OpenAI generations request: prompt=%.80s",
                    body.get("prompt") or "")

        t0 = time.time()
        timings: dict = {}
        with _flux_session(
            timings,
            "Could not stop the LLM container; aborting to avoid a VRAM conflict.",
        ):
            # 1+2. Stop LLM and start FLUX (see _flux_session), then generate.
            t_gen = time.time()
            try:
                resp = await run_in_threadpool(
                    requests.post,
                    FLUX_GEN_V1_URL,
                    json=body,
                    timeout=GENERATION_TIMEOUT,
                )
            except requests.RequestException as e:
                raise HTTPException(502, f"FLUX image request failed: {e}")

            if resp.status_code != 200:
                detail = _response_detail(resp)
                if 400 <= resp.status_code < 500:
                    # FLUX rejected the payload (e.g. bad size/prompt): surface
                    # the real 4xx detail instead of masking it as an upstream failure.
                    raise HTTPException(422, f"FLUX rejected the request: {detail}")
                raise HTTPException(502, f"FLUX image request failed: {detail}")

            result = resp.json()
            _strip_urls(result)
            # 3. GPU is returned to the LLM by _flux_session on exit.

        total_s = time.time() - t0
        logger.info(
            "openai_generate: llm_stop=%.1fs flux_start=%.1fs generate=%.1fs "
            "llm_restore=%.1fs total=%.1fs",
            timings["llm_stop_s"], timings["flux_start_s"],
            timings["restore_start"] - t_gen, timings["llm_restore_s"], total_s,
        )
        return result
    finally:
        _LOCK.release()


@app.post("/v1/images/edits")
@app.post("/images/edits")
async def openai_edit(
    request: Request,
    image: UploadFile | None = File(default=None),
    image_list: list[UploadFile] | None = File(default=None, alias="image[]"),
    prompt: str = Form(...),
):
    """OpenAI Images API proxy: image edit (forwards the multipart body to FLUX).

    The multipart body is forwarded verbatim (original content-type and
    boundary intact) so FLUX sees exactly the file parts Open WebUI sent.
    The `image`/`image[]`/`prompt` params above are only read for
    pre-swap validation; `n`/`seed` and other stock OpenAI fields pass
    through untouched in the raw body.
    """
    _check_token(request)

    if not _LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another image is already being generated; try again shortly.")

    try:
        # Pre-swap validation mirroring FLUX's edits endpoint: bad payloads
        # fail fast instead of burning a ~95s GPU swap for nothing.
        if not prompt.strip():
            raise HTTPException(422, "prompt must not be empty")

        files = []
        if image is not None:
            files.append(image)
        if image_list:
            files.extend(image_list)
        if not files:
            raise HTTPException(
                422, "Missing image file (send an 'image' part in the form body)"
            )
        image_bytes = await files[0].read()
        _validate_image_bytes(image_bytes)

        # FastAPI's Form parsing consumed the stream before this handler
        # ran, so request.body() would raise "Stream consumed". The
        # _MultipartBodyCache middleware stashed the original multipart
        # bytes (boundary included) in the scope instead.
        raw_body = request.scope.get("_multipart_body")
        if raw_body is None:
            raise HTTPException(500, "multipart body was not cached")
        content_type = request.headers.get("content-type", "")

        logger.info("OpenAI edit request: %d bytes prompt=%.80s",
                    len(image_bytes), prompt)

        t0 = time.time()
        timings: dict = {}
        with _flux_session(timings, "Could not stop the LLM container; aborting."):
            # 1+2. Stop LLM and start FLUX (see _flux_session), then edit.
            t_gen = time.time()
            try:
                resp = await run_in_threadpool(_forward_edit, raw_body, content_type)
            except requests.RequestException as e:
                raise HTTPException(502, f"FLUX image request failed: {e}")

            if resp.status_code != 200:
                detail = _response_detail(resp)
                if 400 <= resp.status_code < 500:
                    # FLUX rejected the payload: surface the real 4xx detail
                    # instead of masking it as an upstream failure.
                    raise HTTPException(422, f"FLUX rejected the request: {detail}")
                raise HTTPException(502, f"FLUX image request failed: {detail}")

            result = resp.json()
            _strip_urls(result)
            # 3. GPU is returned to the LLM by _flux_session on exit.

        total_s = time.time() - t0
        logger.info(
            "openai_edit: llm_stop=%.1fs flux_start=%.1fs generate=%.1fs "
            "llm_restore=%.1fs total=%.1fs",
            timings["llm_stop_s"], timings["flux_start_s"],
            timings["restore_start"] - t_gen, timings["llm_restore_s"], total_s,
        )
        return result
    finally:
        _LOCK.release()
