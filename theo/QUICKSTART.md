# T.H.E.O. - Thought Holography and Extraction Operator - Quick Start Guide

## 🚀 30-Second Setup

```bash
# 1. Build and start the container (from the repo root)
podman build -t jarvis-theo flux/
podman compose -f theo/compose.yaml up -d

# 2. Wait for model load (~30-60 seconds)
while ! curl -s http://localhost:8190/health | grep -q '"status":"ready"'; do sleep 5; done

# 3. Generate your first image!
curl -X POST http://localhost:8190/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat with sunglasses", "seed": 42}'
```

The model loads at container startup. The server listens on loopback only (`127.0.0.1:8190`) — the host-side `jarvis-phil` supervisor is its normal client.

---

## 📝 Text-to-Image

### cURL Example

```bash
curl -X POST http://localhost:8190/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Futuristic cityscape at sunset",
    "width": 1024,
    "height": 768,
    "steps": 4,
    "seed": 42
  }'
```

### Python Example

```python
import requests

resp = requests.post(
    "http://localhost:8190/generate",
    json={"prompt": "A serene lake at dawn", "width": 1024, "height": 1024, "seed": 42},
    timeout=600,
)
print(resp.json())
```

The response is JSON; the PNG itself is written to `generated_images/` on the host (container path `/images`):

```json
{"path": "/images/flux_1760600000_42.png", "seed": 42, "elapsed_seconds": 12.3}
```

---

## 🖼️ Image-to-Image

The input image is sent as **raw base64 in a JSON body** (no `data:` URI prefix). Output dimensions are derived from the input: scaled to ~1 MP preserving aspect ratio, then floored to a multiple of 16.

### cURL Example

```bash
# Encode the input image as raw base64
B64=$(base64 -w0 input.jpg)

curl -X POST http://localhost:8190/img2img \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Make it more dramatic with golden hour lighting\", \"image_base64\": \"$B64\", \"steps\": 10, \"seed\": 42}"
```

### Python Example

```python
import base64
import requests

with open("input.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:8190/img2img",
    json={
        "prompt": "Convert to an oil painting",
        "image_base64": b64,
        "cfg_scale": 1.0,
        "steps": 10,
        "seed": 42,
    },
    timeout=600,
)
print(resp.json())
```

```json
{
  "path": "/images/flux_img2img_1792170343_42.png",
  "seed": 42,
  "elapsed_seconds": 52.4,
  "input_dimensions": {"width": 960, "height": 1280},
  "output_dimensions": {"width": 880, "height": 1168}
}
```

### Parameters

| Field | Required | Default | Range / Notes |
|-------|----------|---------|---------------|
| `prompt` | yes | — | 1–500 characters |
| `image_base64` | yes | — | Raw base64 of a PNG/JPEG (no `data:` prefix) |
| `cfg_scale` | no | 1.0 | 0–20. Above 1.0 adds an empty-prompt pass per step (roughly doubles denoise time) |
| `steps` | no | 4 | 1–100 |
| `seed` | no | -1 | -1 = random; a fixed seed reproduces the result |

> **Note on `strength`:** the old `strength` parameter is no longer part of the API — edit intensity is controlled by the prompt and step count instead. Old clients that still send `strength` keep working: unknown fields are silently ignored.

---

## 🤖 OpenAI-Compatible Endpoints

These two endpoints speak the stock **OpenAI Images API** shape, so the built-in image backends of Open WebUI v0.12+ work unmodified (in production they are reached through the `jarvis-phil` proxy on port `8191`, which performs the GPU swap first).

### `POST /v1/images/generations` (JSON)

```bash
curl -X POST http://localhost:8190/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Futuristic cityscape at sunset",
    "n": 1,
    "size": "1024x768",
    "steps": 4,
    "seed": 42
  }'
```

```python
import requests

resp = requests.post(
    "http://localhost:8190/v1/images/generations",
    json={"prompt": "A serene lake at dawn", "size": "1024x1024", "seed": 42},
    timeout=600,
)
print(resp.json())
```

### `POST /v1/images/edits` (multipart/form-data)

```bash
curl -X POST http://localhost:8190/v1/images/edits \
  -F "image=@input.jpg" \
  -F "prompt=Convert to an oil painting" \
  -F "seed=42"
```

```python
import requests

with open("input.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8190/v1/images/edits",
        files={"image": ("input.jpg", f, "image/jpeg")},
        data={"prompt": "Convert to an oil painting", "seed": 42},
        timeout=600,
    )
print(resp.json())
```

### Response (both endpoints)

OpenAI Images API shape:

```json
{
  "created": 1760600000,
  "data": [
    {
      "b64_json": "<base64-encoded PNG>",
      "url": "/images/flux_1760600000_42.png",
      "revised_prompt": "Futuristic cityscape at sunset",
      "seed": 42
    }
  ]
}
```

Notes:

- `model`, `background`, `quality`, `style`, `response_format`, … are accepted and ignored — the stock Open WebUI client works unmodified.
- `seed: -1` = random (ms timestamp mod 2³²); `n: 1–4` generates `n` images with seeds `seed`, `seed+1`, … (mod 2³²).
- Generations: `size` is `WxH` with each side 256–2048 (default `1024x1024`); `steps` 1–50 (default 4); `cfg_scale` 0–20 (default 1.0).
- Edits: `image` (or `image[]`) — the first image wins; `size`, `steps`, `cfg_scale`, … are accepted and ignored (edits run at 4 steps, `cfg_scale` 1.0); output ≈ 1 MP, floored to a multiple of 16.
- `url` is cosmetic — the PNG is always written to `generated_images/` on the host (container path `/images`). The supervisor proxy strips `url` from responses so Open WebUI consumes `b64_json`.

---

## 🔍 Health & Diagnostics

```bash
# Check if model is ready
curl http://localhost:8190/health

# View system diagnostics (VRAM, device)
curl http://localhost:8190/info | jq .
```

`/info` reports: `model_ready`, `cuda_available`, `device`, `total_gb`, `free_gb`, `used_gb`.

---

## 📚 Full Documentation

| Document | Purpose |
|----------|---------|
| [`README.md`](./README.md) | Overview, architecture, memory strategy, benchmarks, troubleshooting |
| `test_txt2img.py` | End-to-end text-to-image test (`/generate` + `/v1/images/generations`, positives + 422 checks) |
| `test_img2img.py` | End-to-end image-edit test (`/img2img` + `/v1/images/edits`, positives + 422 checks) |

---

## ⚡ Performance Benchmarks

Measured on an RX 6950 XT (16 GB, ROCm, sequential CPU offload):

| Operation | Steps | Input → Output | Time |
|-----------|-------|----------------|------|
| Image-to-Image | 4 | 960×1280 → 880×1168 | ~52 s |
| Text-to-Image | 4 | — → 1024×1024 | ~25–31 s |

Denoise time scales roughly linearly with step count.

---

## 🐛 Quick Troubleshooting

### Model not ready?

The model loads at container startup (~30–60 s). Poll until it's ready:

```bash
while ! curl -s http://localhost:8190/health | grep -q '"status":"ready"'; do sleep 5; done
echo "Model is ready!"
```

### 500 CUDA out-of-memory?

The GPU is shared with the LLM container — one model resident at a time. Let `jarvis-phil` handle switching, or stop the LLM container before using THEO.

### 400 "input image too small"?

Only tiny inputs or extreme aspect ratios fail the ~1 MP scaling. Use a larger input image.

### Slow inference times?

Check VRAM state with `/info` (watch `free_gb`); restart the container if memory looks fragmented. The MIOpen kernel cache (`.miopen/`) persists across restarts, so restarts get faster, not slower.

---

## 💡 Pro Tips

1. **Use fixed seeds for reproducibility**: `"seed": 42` produces the same result every time
2. **Inputs are auto-scaled to ~1 MP** (aspect preserving); the output grid is the scaled size floored to a multiple of 16 — no need to set dimensions yourself
3. **`cfg_scale` above 1.0** (e.g. 3.5) pushes a stronger edit, but each step runs an extra empty-prompt pass — roughly double the denoise time
4. **Check `/info` before heavy usage** to monitor VRAM availability
5. **Run `theo/test_txt2img.py` / `theo/test_img2img.py`** for quick end-to-end requests against the running container (positives + 422 checks)

---

## 📞 Support

- Endpoints, setup & troubleshooting: [`README.md`](./README.md)
- Smoke tests: `python3 theo/test_txt2img.py` / `python3 theo/test_img2img.py`
- Container logs: `podman logs --tail 50 jarvis-theo`

---

**Version:** 1.1.0  
**Last Updated:** 2026-10-17
