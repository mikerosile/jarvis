#!/usr/bin/env python3
import requests, base64
from pathlib import Path

image_prompt = 'Change the top label, "Waterloo", to read “Paul’s Lemonade”. Use the same style font as “Waterloo”, and give the words a slight downward arc'

img_path = Path('/home/michael/Downloads/Waterloo soda can.jpg')
with open(img_path, 'rb') as f:
    base64_str = base64.b64encode(f.read()).decode('utf-8')

data = {
    'prompt': image_prompt,
    'image_base64': base64_str,
    'steps': 4,
    'seed': 42
}

print(f'Testing with {len(base64_str)} char base64 string...')
resp = requests.post('http://localhost:8190/img2img', json=data, timeout=600)
print(f'Status: {resp.status_code}')
print(resp.json())

# ---------------------------------------------------------------------------
# OpenAI Images API compat: POST /v1/images/edits  (E1-E7)
# ---------------------------------------------------------------------------
import io
import math
import sys
import time
from PIL import Image

failures = []


def check(label, cond, extra=''):
    tag = 'PASS' if cond else 'FAIL'
    print(f'  [{tag}] {label}' + (f'  ({extra})' if extra else ''))
    if not cond:
        failures.append(label)


with open(img_path, 'rb') as f:
    input_bytes = f.read()
input_img = Image.open(io.BytesIO(input_bytes))
input_w, input_h = input_img.size
print(f'Input image: {input_w}x{input_h} {input_img.format}')

# _run_img2img scales the input to ~1 MP (1024x1024 target, aspect
# preserving) and floors each side to a multiple of 16 (VAE scale 8 * 2);
# mirror that formula here so we can assert the expected output size.
TARGET_PIXELS = 1024 * 1024
GRID = 16
_scale = math.sqrt(TARGET_PIXELS / (input_w * input_h))
grid_w = (int(input_w * _scale) // GRID) * GRID
grid_h = (int(input_h * _scale) // GRID) * GRID
print(f'Expected output grid: {grid_w}x{grid_h}')


def post_edits(files=None, prompt=image_prompt, label='', timeout=600, **fields):
    data = {'prompt': prompt}
    data.update(fields)
    print(f'Testing /v1/images/edits ({label})...')
    t0 = time.time()
    resp = requests.post(
        'http://localhost:8190/v1/images/edits',
        files=files, data=data, timeout=timeout,
    )
    print(f'Status: {resp.status_code} ({time.time() - t0:.1f}s)')
    return resp


def check_edit_response(resp, label, n=1, seed0=42):
    if resp.status_code != 200:
        check(f'{label} -> 200', False, f'got {resp.status_code}: {resp.text[:200]}')
        return
    body = resp.json()
    check(f'{label} "created" is int', isinstance(body.get('created'), int),
          str(body.get('created')))
    data = body.get('data', [])
    check(f'{label} {n} item(s)', len(data) == n, f'got {len(data)}')
    for i, item in enumerate(data):
        expect_seed = (seed0 + i) % (2 ** 32)
        check(f'{label} item {i} seed={expect_seed}', item.get('seed') == expect_seed,
              str(item.get('seed')))
        check(f'{label} item {i} revised_prompt echoed',
              item.get('revised_prompt') == image_prompt)
        try:
            img = Image.open(io.BytesIO(base64.b64decode(item['b64_json'])))
            ok = img.format == 'PNG' and img.size == (grid_w, grid_h)
            check(f'{label} item {i} decodes to {grid_w}x{grid_h} PNG (~1 MP grid)',
                  ok, f'got {img.format} {img.size}')
        except Exception as e:
            check(f'{label} item {i} decodes to {grid_w}x{grid_h} PNG (~1 MP grid)',
                  False, repr(e))


def can_files(name='Waterloo soda can.jpg', ctype='image/jpeg'):
    return {'image': (name, input_bytes, ctype)}


# E1: image + prompt, seed 42 (steps/cfg_scale run at FLUX defaults)
check_edit_response(post_edits(can_files(), label='E1 basic edit',
                               seed='42'), 'E1')

# E2: n=2 -> two items, seeds 42 and 43
check_edit_response(post_edits(can_files(), label='E2 n=2', n='2',
                               seed='42'), 'E2', n=2)

# E3: stock `size` field ignored -> output stays at the ~1 MP grid size
check_edit_response(post_edits(can_files(), label='E3 size ignored',
                               size='512x512', seed='42'), 'E3')


def check_422(label, files=None, prompt=image_prompt, timeout=60,
              expect_in_detail=None, **fields):
    resp = post_edits(files, prompt=prompt, label=label, timeout=timeout,
                      **fields)
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


# E4: no image part
check_422('E4 no image part', files=None,
          expect_in_detail="send an 'image' part")
# E5: empty prompt -> the required `prompt` Form validation fires first
# (an empty multipart field arrives as missing), so FastAPI's own
# "Field required" message is what we get
check_422('E5 empty prompt', files=can_files(), prompt='',
          expect_in_detail='Field required')
# E5b: whitespace-only prompt -> the handler's own check
check_422('E5b whitespace prompt', files=can_files(), prompt='   ',
          expect_in_detail='prompt must not be empty')
# E6: zero-byte image
check_422('E6 zero-byte image',
          files={'image': ('empty.jpg', b'', 'image/jpeg')},
          expect_in_detail='Uploaded image is empty')
# E7: undecodable bytes
check_422('E7 undecodable bytes',
          files={'image': ('junk.bin', b'not an image at all',
                           'application/octet-stream')},
          expect_in_detail='Could not decode the uploaded image')

print()
if failures:
    print(f'{len(failures)} check(s) FAILED:')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
print('All edits checks passed.')
