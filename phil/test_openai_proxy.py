#!/usr/bin/env python3
"""jarvis_phil OpenAI Images API proxy tests (P1-P5).

Hits the supervisor on :8191 (the port Open WebUI talks to), so these
exercise the proxy layer itself: auth (off in the user unit), the
GPU-swap lock (409), pre-swap validation (422), verbatim multipart
forwarding, and response url-stripping.

Run:  python3 test_openai_proxy.py

Positive cases (P1, P4) each burn a ~95s GPU swap; the 409
probe and every 422 fail in milliseconds. After each swap the
supervisor restores the LLM on its own (_restore_llm), so no manual
GPU restore is needed. Works whether FLUX is currently up or down.
"""
import base64
import io
import math
import sys
import threading
import time
from pathlib import Path

import requests
from PIL import Image

BASE = 'http://localhost:8191'
GENERATIONS_URL = f'{BASE}/v1/images/generations'
EDITS_URL = f'{BASE}/v1/images/edits'

# Same prompts/assets as the FLUX-direct tests so the supervisor sees
# exactly what the other chunks exercised.
image_prompt = 'The image features a can of Tropical Fruit Sparkling Water standing upright on a light gray, marbled kitchen countertop. The can has a vibrant magenta-pink upper half and a pale pink lower half, separated by a wavy boundary. The brand name “Paul\'s Lemonade” appears prominently in large, stylized cream-colored lettering, with “SPARKLING WATER” printed beneath it. An illustration of colorful citrus fruits, including a sliced pink-red grapefruit and orange-yellow fruits with a green leaf, decorates the center of the can. Near the bottom, bold dark-pink text reads “TROPICAL FRUIT,” followed by “NATURALLY FLAVORED WITH OTHER NATURAL FLAVORS.” White lettering around the top advertises zero sugar and zero calories. The can has a reflective silver rim and glossy surface, while the softly blurred background contains kitchen fixtures, a black backsplash, and partially visible household items.'
edit_prompt = 'Change the top label, "Waterloo", to read “Paul’s Lemonade”. Use the same style font as “Waterloo”, and give the words a slight downward arc'

img_path = Path('/home/michael/Downloads/Waterloo soda can.jpg')
with open(img_path, 'rb') as f:
    input_bytes = f.read()

input_img = Image.open(io.BytesIO(input_bytes))
input_w, input_h = input_img.size
print(f'Input image: {input_w}x{input_h} {input_img.format}')

# Mirror _run_img2img's sizing: ~1 MP target, aspect preserving, each
# side floored to a multiple of 16 (VAE scale 8 * 2).
TARGET_PIXELS = 1024 * 1024
GRID = 16
_scale = math.sqrt(TARGET_PIXELS / (input_w * input_h))
grid_w = (int(input_w * _scale) // GRID) * GRID
grid_h = (int(input_h * _scale) // GRID) * GRID
print(f'Expected edit output grid: {grid_w}x{grid_h}')

failures = []


def check(label, cond, extra=''):
    tag = 'PASS' if cond else 'FAIL'
    print(f'  [{tag}] {label}' + (f'  ({extra})' if extra else ''))
    if not cond:
        failures.append(label)


def gen_body(**overrides):
    """OpenAI-shaped body, as sent by Open WebUI v0.12.3 (model ignored)."""
    body = {
        'model': 'flux',
        'prompt': image_prompt,
        'n': 1,
        'response_format': 'b64_json',
        'size': '1024x1024',
        'steps': 4,
        'cfg_scale': 1.0,
        'seed': 42,
    }
    body.update(overrides)
    return body


def check_gen_response(resp, label):
    """v0.12.3 generations shape via the proxy: `url` keys must be stripped."""
    if resp.status_code != 200:
        check(f'{label} -> 200', False, f'got {resp.status_code}: {resp.text[:200]}')
        return
    body = resp.json()
    check(f'{label} "created" is int', isinstance(body.get('created'), int),
          str(body.get('created')))
    data = body.get('data', [])
    check(f'{label} 1 item', len(data) == 1, f'got {len(data)}')
    for i, item in enumerate(data):
        check(f'{label} item {i} has no url key (stripped by proxy)',
              'url' not in item, str(item.get('url')))
        check(f'{label} item {i} seed=42', item.get('seed') == 42,
              str(item.get('seed')))
        check(f'{label} item {i} revised_prompt echoed',
              item.get('revised_prompt') == image_prompt)
        try:
            img = Image.open(io.BytesIO(base64.b64decode(item['b64_json'])))
            ok = img.format == 'PNG' and img.size == (1024, 1024)
            check(f'{label} item {i} decodes to 1024x1024 PNG', ok,
                  f'got {img.format} {img.size}')
        except Exception as e:
            check(f'{label} item {i} decodes to 1024x1024 PNG', False, repr(e))


# ---------------------------------------------------------------------------
# P1+P2: generations through the proxy; the second request must hit the
# 409 while P1 holds the GPU-swap lock (acquired pre-validation on arrival)
# ---------------------------------------------------------------------------
print('Testing /v1/images/generations via supervisor (P1)...')
p1 = {}


def _run_p1():
    p1['resp'] = requests.post(GENERATIONS_URL, json=gen_body(), timeout=600)


t0 = time.time()
th = threading.Thread(target=_run_p1)
th.start()

time.sleep(10)  # P1 is mid-swap by now; the lock is definitely held
probe = requests.post(GENERATIONS_URL, json=gen_body(), timeout=60)
print(f'P2 probe status: {probe.status_code} '
      f'({time.time() - t0:.1f}s since P1 start)')
check('P2 concurrent generations -> 409', probe.status_code == 409,
      f'got {probe.status_code}: {probe.text[:120]}')
if probe.status_code == 409:
    detail = str(probe.json().get('detail', ''))
    check('P2 409 detail mentions "already being generated"',
          'already being generated' in detail, detail[:120])

th.join()
print(f'P1 status: {p1["resp"].status_code} ({time.time() - t0:.1f}s total)')
check_gen_response(p1['resp'], 'P1')

# ---------------------------------------------------------------------------
# P3: non-object JSON body -> instant 422 before the GPU swap
# ---------------------------------------------------------------------------
print('Testing /v1/images/generations with a JSON array body (P3)...')
resp = requests.post(GENERATIONS_URL, json=[1, 2, 3], timeout=60)
print(f'Status: {resp.status_code}')
check('P3 array body -> 422', resp.status_code == 422,
      f'got {resp.status_code}: {resp.text[:200]}')
if resp.status_code == 422:
    detail = str(resp.json().get('detail', ''))
    check('P3 detail mentions "Request body must be a JSON object"',
          'request body must be a json object' in detail.lower(),
          detail[:120])

# ---------------------------------------------------------------------------
# P4: edits multipart through the proxy (raw body forwarded verbatim)
# ---------------------------------------------------------------------------
def check_edit_response(resp, label):
    if resp.status_code != 200:
        check(f'{label} -> 200', False, f'got {resp.status_code}: {resp.text[:200]}')
        return
    body = resp.json()
    check(f'{label} "created" is int', isinstance(body.get('created'), int),
          str(body.get('created')))
    data = body.get('data', [])
    check(f'{label} 1 item', len(data) == 1, f'got {len(data)}')
    for i, item in enumerate(data):
        check(f'{label} item {i} has no url key (stripped by proxy)',
              'url' not in item, str(item.get('url')))
        check(f'{label} item {i} seed=42', item.get('seed') == 42,
              str(item.get('seed')))
        check(f'{label} item {i} revised_prompt echoed',
              item.get('revised_prompt') == edit_prompt)
        try:
            img = Image.open(io.BytesIO(base64.b64decode(item['b64_json'])))
            ok = img.format == 'PNG' and img.size == (grid_w, grid_h)
            check(f'{label} item {i} decodes to {grid_w}x{grid_h} PNG (~1 MP grid)',
                  ok, f'got {img.format} {img.size}')
        except Exception as e:
            check(f'{label} item {i} decodes to {grid_w}x{grid_h} PNG (~1 MP grid)',
                  False, repr(e))


print('Testing /v1/images/edits via supervisor (P4)...')
t0 = time.time()
resp = requests.post(
    EDITS_URL,
    files={'image': ('Waterloo soda can.jpg', input_bytes, 'image/jpeg')},
    # seed/n are undeclared on the supervisor side; they pass through
    # untouched in the forwarded raw multipart body.
    data={'prompt': edit_prompt, 'seed': '42', 'n': '1'},
    timeout=600,
)
print(f'Status: {resp.status_code} ({time.time() - t0:.1f}s)')
check_edit_response(resp, 'P4')

# ---------------------------------------------------------------------------
# P5: pre-swap validation 422s (all instant, no GPU swap)
# ---------------------------------------------------------------------------
def check_422_edits(label, files=None, prompt=edit_prompt, timeout=60,
                    expect_in_detail=None, **fields):
    data = {'prompt': prompt}
    data.update(fields)
    print(f'Testing /v1/images/edits ({label})...')
    resp = requests.post(EDITS_URL, files=files, data=data, timeout=timeout)
    print(f'Status: {resp.status_code}')
    if resp.status_code != 422:
        check(f'{label} -> 422', False, f'got {resp.status_code}: {resp.text[:200]}')
        return
    detail = ''
    try:
        detail = str(resp.json().get('detail', ''))
    except Exception:
        pass
    check(f'{label} -> 422', True)
    if not detail:
        check(f'{label} 422 has detail', False)
    elif expect_in_detail:
        check(f'{label} detail mentions {expect_in_detail!r}',
              expect_in_detail.lower() in detail.lower(), detail[:120])


# P5a: no image part at all (plain form body, no multipart file part)
check_422_edits('P5a no image part', files=None,
                expect_in_detail="send an 'image' part")
# P5b: empty prompt -> the required `prompt` Form validation fires first
# (an empty multipart field arrives as missing), so FastAPI's own
# "Field required" message is what we get
check_422_edits('P5b empty prompt',
                files={'image': ('Waterloo soda can.jpg', input_bytes,
                                 'image/jpeg')},
                prompt='', expect_in_detail='Field required')
# P5c: whitespace-only prompt -> the handler's own check
check_422_edits('P5c whitespace prompt',
                files={'image': ('Waterloo soda can.jpg', input_bytes,
                                 'image/jpeg')},
                prompt='   ', expect_in_detail='prompt must not be empty')
# P5d: zero-byte image
check_422_edits('P5d zero-byte image',
                files={'image': ('empty.jpg', b'', 'image/jpeg')},
                expect_in_detail='Uploaded image is empty')
# P5e: undecodable bytes (supervisor wording, stricter than FLUX's)
check_422_edits('P5e undecodable bytes',
                files={'image': ('junk.bin', b'not an image at all',
                                 'application/octet-stream')},
                expect_in_detail='not a decodable image')


print()
if failures:
    print(f'{len(failures)} check(s) FAILED:')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
print('All proxy checks passed.')
