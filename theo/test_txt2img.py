#!/usr/bin/env python3
import json
import requests

image_prompt = 'The image features a can of Tropical Fruit Sparkling Water standing upright on a light gray, marbled kitchen countertop. The can has a vibrant magenta-pink upper half and a pale pink lower half, separated by a wavy boundary. The brand name “Paul\'s Lemonade” appears prominently in large, stylized cream-colored lettering, with “SPARKLING WATER” printed beneath it. An illustration of colorful citrus fruits, including a sliced pink-red grapefruit and orange-yellow fruits with a green leaf, decorates the center of the can. Near the bottom, bold dark-pink text reads “TROPICAL FRUIT,” followed by “NATURALLY FLAVORED WITH OTHER NATURAL FLAVORS.” White lettering around the top advertises zero sugar and zero calories. The can has a reflective silver rim and glossy surface, while the softly blurred background contains kitchen fixtures, a black backsplash, and partially visible household items.'

data = {
    'prompt': image_prompt,
    'width': 1024,
    'height': 1024,
    'cfg_scale': 1.0,
    'steps': 4,
    'seed': 42
}

print('Testing /generate...')
resp = requests.post('http://localhost:8190/generate', json=data, timeout=600)
print(f'Status: {resp.status_code}')
print(resp.json())

# ---------------------------------------------------------------------------
# OpenAI Images API compat: POST /v1/images/generations  (G1-G7)
# ---------------------------------------------------------------------------
import base64
import io
import sys
import time
from PIL import Image

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


def post_generations(body, label, timeout=600):
    print(f'Testing /v1/images/generations ({label})...')
    t0 = time.time()
    resp = requests.post(
        'http://localhost:8190/v1/images/generations', json=body, timeout=timeout
    )
    print(f'Status: {resp.status_code} ({time.time() - t0:.1f}s)')
    return resp


def check_gen_response(resp, label, n=1, seed0=42, width=1024, height=1024):
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
            ok = img.format == 'PNG' and img.size == (width, height)
            check(f'{label} item {i} decodes to {width}x{height} PNG', ok,
                  f'got {img.format} {img.size}')
        except Exception as e:
            check(f'{label} item {i} decodes to {width}x{height} PNG', False, repr(e))


# G1: stock v0.12.3 body (model + response_format accepted and ignored)
check_gen_response(post_generations(gen_body(), 'G1 stock body'), 'G1')

# G2: n=2 -> two items, seeds 42 and 43
check_gen_response(post_generations(gen_body(n=2), 'G2 n=2'), 'G2', n=2)

# G3: size 512x768 honored
check_gen_response(post_generations(gen_body(size='512x768'), 'G3 size 512x768'),
                   'G3', width=512, height=768)


def check_422(label, body, expect_in_detail=None):
    resp = post_generations(body, label, timeout=60)
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


# G4: empty prompt
check_422('G4 empty prompt', gen_body(prompt=''))
# G5: size out of range (handler-raised 422)
check_422('G5 size 99999x1', gen_body(size='99999x1'), expect_in_detail='out of range')
# G6: steps below minimum
check_422('G6 steps=0', gen_body(steps=0))
# G7: n above maximum
check_422('G7 n=5', gen_body(n=5))

print()
if failures:
    print(f'{len(failures)} check(s) FAILED:')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
print('All generations checks passed.')
