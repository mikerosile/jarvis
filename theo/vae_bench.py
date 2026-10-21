#!/usr/bin/env python3
"""VAE dtype benchmark for FLUX.2-klein-4B (AutoencoderKLFlux2).

Loads only the VAE (no transformer) and times the exact encode/decode paths
that edit_image.py runs, for each dtype configuration:

  configs:
    fp32   - whole VAE float32 (also the quality reference)
    bf16   - whole VAE bfloat16 (current production default = baseline)
    fp16   - whole VAE float16
    mixed  - bfloat16 encoder + quant_conv, float32 decoder + post_quant_conv

  encode path (mirrors Flux2KleinPipeline.prepare_image_latents /
  _encode_vae_image exactly):
      vae.encode(img) -> retrieve_latents(..., sample_mode="argmax")
      -> _patchify_latents  (2x2: (B,32,H/8,W/8) -> (B,128,H/16,W/16))
      -> batch-norm normalize, stats cast to the latent dtype
      -> _pack_latents       ((B,C,H,W) -> (B,H*W,C))

  decode path (mirrors the VAEDecode block in edit_image.py):
      _unpack_latents_with_ids  (scatter tokens by their position ids)
      -> batch-norm denormalize, stats cast to the decoder dtype
      -> _unpatchify_latents
      -> vae.decode(z.to(device, decoder_dtype), return_dict=False)[0]

Decode input convention: every configuration decodes the SAME reference
latent - the fp32 reference encode's output rounded to bfloat16 (what the
bf16 transformer emits). Ground truth for PSNR is the fp32 decode of the
unrounded fp32 latent.

Measurement: 1 untimed warmup (timing only; under MIOPEN_FIND_MODE=2 no solver
search occurs) + 3 timed iterations, median reported,
torch.cuda.synchronize() around every timed op. Peak GPU memory per phase via
torch.cuda.reset_peak_memory_stats (process-wide, includes resident VAEs).

Decision gate (printed at the end): a configuration PASSES if its total
(encode + decode) is at least 2x faster than bf16 AND its decode PSNR vs the
fp32 ground truth is at least 35 dB. Only a passing config should be adopted
as the new --vae-dtype default in edit_image.py.

Run (one-shot container invocation, script mounted read-only):
    podman run --rm -it \
      -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -e MIOPEN_FIND_MODE=2 \
      -e MIOPEN_USER_DB_PATH=/miopen/db \
      -e MIOPEN_CUSTOM_CACHE_DIR=/miopen/cache \
      --device /dev/kfd --device /dev/dri \
      -v "$PWD/models:/models:Z" -v "$PWD/.miopen:/miopen:Z" \
      -v "$PWD/theo/vae_bench.py:/vae_bench.py:ro,Z" \
      jarvis-theo:latest python /vae_bench.py

Sanity anchor: under MIOPEN_FIND_MODE=2 no solver search occurs, so bf16 @
1024x1024 should land in seconds per phase, not minutes (the old production
numbers, ~59 s encode / ~167 s decode from the 1 MP default run, included the
one-time search cost and no longer apply).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

import diffusers
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_latents

DEVICE = torch.device("cuda")
# Bench order; bf16 is the baseline (current production default).
CONFIG_ORDER = ("fp32", "bf16", "fp16", "mixed")
# Preference order for the --vae-dtype default if several configs pass.
CONFIG_PREFERENCE = ("fp16", "mixed", "fp32")
CONFIG_LABELS = {
    "fp32": "fp32",
    "bf16": "bf16",
    "fp16": "fp16",
    "mixed": "bf16-enc/fp32-dec",
}
# Images live in [-1, 1], so the largest possible pixel difference is 2.
PSNR_PEAK = 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAE dtype benchmark for the FLUX.2-klein-4B VAE "
        "(encode/decode paths exactly as run by edit_image.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("IMG_MODEL_PATH", "/models/flux/FLUX.2-klein-4B"),
        help="diffusers-format FLUX.2-klein-4B repo (env: IMG_MODEL_PATH)",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[1024, 512],
        help="square resolutions to benchmark (multiples of 16)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup iterations per phase")
    parser.add_argument("--iters", type=int, default=3, help="timed iterations per phase (median reported)")
    parser.add_argument("--seed", type=int, default=1234, help="seed for the synthetic test image")
    parser.add_argument(
        "--image",
        default=None,
        help="optional input image (PIL, resized BILINEAR); default: deterministic synthetic content",
    )
    parser.add_argument("--gate-speedup", type=float, default=2.0, help="gate: minimum total (enc+dec) speedup vs bf16")
    parser.add_argument("--gate-psnr", type=float, default=35.0, help="gate: minimum decode PSNR (dB) vs fp32 ground truth")
    args = parser.parse_args(argv)
    for r in args.resolutions:
        if r < 16 or r % 16 != 0:
            parser.error(f"--resolutions entries must be multiples of 16 >= 16 (got {r})")
    return args


def make_synthetic_image(size: int, seed: int) -> torch.Tensor:
    """Deterministic multi-frequency test content (per-channel sinusoids at
    8 spatial frequencies + Gaussian noise), as (1, 3, size, size) float32
    in [-1, 1] on the GPU."""
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(0.0, 1.0, size, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    img = torch.empty(3, size, size, dtype=torch.float64)
    for c in range(3):
        sig = torch.zeros(size, size, dtype=torch.float64)
        for k in range(1, 9):  # spatial frequencies 1..8 cycles per image
            fx = k / 8.0
            fy = (k + 1) / 8.0 * (-1.0 if c % 2 else 1.0)
            phase = 2.0 * math.pi * torch.rand(1, generator=g).item()
            amp = torch.rand(1, generator=g).item()
            sig += amp * torch.sin(2.0 * math.pi * (fx * xx + fy * yy) + phase)
        gain = 0.2 + 0.8 * torch.rand(1, generator=g).item()
        noise = 0.15 * torch.randn(size, size, dtype=torch.float64, generator=g)
        img[c] = 0.5 + 0.45 * gain * (sig / 8.0) + noise
    img = (img - img.min()) / (img.max() - img.min())
    return (2.0 * img - 1.0).to(dtype=torch.float32).unsqueeze(0).to(DEVICE)


def load_image(path: str, size: int) -> torch.Tensor:
    """Load a PIL image, resize BILINEAR to size x size, return (1, 3, S, S)
    float32 in [-1, 1] on the GPU."""
    pil = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    t = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return (2.0 * t - 1.0).to(DEVICE)


def encode_image(vae, image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Mirror Flux2KleinPipeline.prepare_image_latents -> _encode_vae_image.

    Returns the batch-norm-normalized, packed reference latent
    (1, H/16 * W/16, 128) in ``dtype``.
    """
    img = image.to(DEVICE, dtype)
    latents = retrieve_latents(vae.encode(img), generator=None, sample_mode="argmax")
    latents = Flux2KleinPipeline._patchify_latents(latents)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = (latents - mean) / std
    return Flux2KleinPipeline._pack_latents(latents)


def make_latent_ids(half_h: int, half_w: int) -> torch.Tensor:
    """Mirror Flux2KleinPipeline._prepare_latent_ids on a full (1, C, H, W)
    grid: (1, H*W, 4) int64 rows (T=0, H, W, L=0), row-major."""
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(half_h), torch.arange(half_w), torch.arange(1)
    )
    return ids.unsqueeze(0).to(DEVICE)


def decode_latents(
    vae,
    packed: torch.Tensor,
    ids: torch.Tensor,
    half_h: int,
    half_w: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mirror the VAEDecode block in edit_image.py (lines 515-528).

    ``packed`` is the normalized packed latent (1, H/16*W/16, 128); ``dtype``
    is the decoder dtype (the batch-norm stats are cast to it, mirroring the
    production fix).
    """
    z = Flux2KleinPipeline._unpack_latents_with_ids(packed, ids, half_h, half_w)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(z.device, dtype)
    std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(z.device, dtype)
    z = z * std + mean
    z = Flux2KleinPipeline._unpatchify_latents(z)
    return vae.decode(z.to(DEVICE, dtype), return_dict=False)[0]


def build_vaes(master) -> tuple[dict, dict]:
    """fp32 master + per-config copies.

    Returns ({name: vae on GPU}, {name: (enc_dtype, dec_dtype)}).
    """
    vaes = {"fp32": master}
    dtypes = {"fp32": (torch.float32, torch.float32)}
    for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        vae = copy.deepcopy(master)
        vae.to(dt)
        vaes[name] = vae
        dtypes[name] = (dt, dt)
    mixed = copy.deepcopy(master)
    mixed.encoder.to(torch.bfloat16)
    mixed.quant_conv.to(torch.bfloat16)
    # decoder, post_quant_conv and the fp32 batch-norm stats stay float32
    vaes["mixed"] = mixed
    dtypes["mixed"] = (torch.bfloat16, torch.float32)
    return vaes, dtypes


def time_phase(fn, warmup: int, iters: int):
    """Run ``warmup`` untimed + ``iters`` timed iterations of ``fn``.

    Returns (median_seconds, peak_gib, last_output).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(DEVICE)
    times, peaks, out = [], [], None
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize(DEVICE)
        times.append(time.perf_counter() - t0)
        peaks.append(torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 3)
    return statistics.median(times), max(peaks), out


def psnr_db(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB for tensors in [-1, 1] (peak difference 2.0)."""
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    mse = torch.nn.functional.mse_loss(a.float(), b.float())
    if not torch.isfinite(mse) or mse.item() <= 0.0:
        return float("inf")
    return 10.0 * math.log10(PSNR_PEAK ** 2 / mse.item())


def main() -> int:
    args = parse_args()
    props = torch.cuda.get_device_properties(0)
    print("=" * 78)
    print("VAE dtype benchmark - FLUX.2-klein-4B (AutoencoderKLFlux2)")
    print("=" * 78)
    hip = f", HIP {torch.version.hip}" if torch.version.hip else ""
    print(f"torch {torch.__version__}{hip} | diffusers {diffusers.__version__}")
    print(f"GPU: {props.name} ({props.total_memory / 1024 ** 3:.1f} GiB)")
    print(f"model dir: {args.model_dir}")
    print(
        f"resolutions: {' '.join(str(r) for r in args.resolutions)} | "
        f"warmup {args.warmup} + {args.iters} timed (median) | seed {args.seed}"
    )
    print(f"test image: {args.image or f'synthetic multi-frequency (seed {args.seed})'}")
    print(f"gate: total >= {args.gate_speedup:g}x bf16 AND decode PSNR >= {args.gate_psnr:g} dB")
    print("sanity anchor: bf16 @ 1024x1024 ~ 59 s encode / 167 s decode (production log)")

    t0 = time.perf_counter()
    master = AutoencoderKLFlux2.from_pretrained(
        args.model_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(DEVICE)
    master.requires_grad_(False)
    master.eval()
    n_params = sum(p.numel() for p in master.parameters())
    print(
        f"VAE loaded in {time.perf_counter() - t0:.1f}s | {n_params / 1e6:.1f}M params | "
        f"fp32 master kept on GPU as the quality reference"
    )

    vaes, dtypes = build_vaes(master)
    print(
        "configs: "
        + ", ".join(
            # f"{CONFIG_LABELS[n]} (enc={dtypes[n][0].__name__}, dec={dtypes[n][1].__name__})"
            f"{CONFIG_LABELS[n]} (enc={str(dtypes[n][0]).removeprefix('torch.')}, dec={str(dtypes[n][1]).removeprefix('torch.')})"
            for n in CONFIG_ORDER
        )
    )

    results: dict[int, dict] = {}
    with torch.inference_mode():
        for res in args.resolutions:
            half = res // 16
            image = load_image(args.image, res) if args.image else make_synthetic_image(res, args.seed)
            ids = make_latent_ids(half, half)

            # Reference (untimed): fp32 encode, then fp32 decode of the
            # unrounded latent = ground truth for PSNR.
            ref_z = encode_image(master, image, torch.float32)
            out_ref = decode_latents(master, ref_z, ids, half, half, torch.float32)
            # Decode input for every config: the same latent rounded to bf16
            # (simulating the bf16 transformer output).
            ref_z_bf16 = ref_z.to(torch.bfloat16)

            print()
            print(
                f"--- {res}x{res} ---  raw latent (1,32,{res // 8},{res // 8}) "
                f"-> patchified (1,128,{half},{half}) -> packed {tuple(ref_z.shape)}"
            )

            rows = {}
            for name in CONFIG_ORDER:
                vae = vaes[name]
                enc_dt, dec_dt = dtypes[name]
                t_enc, peak_enc, z_c = time_phase(
                    lambda vae=vae, enc_dt=enc_dt: encode_image(vae, image, enc_dt),
                    args.warmup,
                    args.iters,
                )
                t_dec, peak_dec, out_c = time_phase(
                    lambda vae=vae, dec_dt=dec_dt: decode_latents(vae, ref_z_bf16, ids, half, half, dec_dt),
                    args.warmup,
                    args.iters,
                )
                diff = (z_c.float() - ref_z.float()).abs()
                rows[name] = {
                    "enc": t_enc,
                    "dec": t_dec,
                    "total": t_enc + t_dec,
                    "psnr": psnr_db(out_c, out_ref),
                    "lat_max": diff.max().item(),
                    "lat_mean": diff.mean().item(),
                    "peak_enc": peak_enc,
                    "peak_dec": peak_dec,
                }
                del z_c, out_c

            base_total = rows["bf16"]["total"]
            print(
                f"{'config':<19} {'enc(s)':>8} {'dec(s)':>8} {'total(s)':>9} {'vs-bf16':>8} "
                f"{'PSNR(dB)':>9} {'lat.max':>10} {'lat.mean':>10} {'pkEnc':>7} {'pkDec':>7}"
            )
            for name in CONFIG_ORDER:
                r = rows[name]
                tag = "  (baseline)" if name == "bf16" else ""
                print(
                    f"{CONFIG_LABELS[name]:<19} {r['enc']:8.2f} {r['dec']:8.2f} {r['total']:9.2f} "
                    f"{base_total / r['total']:7.2f}x {r['psnr']:9.2f} {r['lat_max']:10.3e} "
                    f"{r['lat_mean']:10.3e} {r['peak_enc']:7.2f} {r['peak_dec']:7.2f}{tag}"
                )
            results[res] = rows

            del image, ids, ref_z, ref_z_bf16, out_ref
            torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print(
        f"GATE CHECK (pass iff total >= {args.gate_speedup:g}x bf16 "
        f"AND decode PSNR >= {args.gate_psnr:g} dB)"
    )
    print("=" * 78)
    per_cfg_pass = {n: True for n in CONFIG_ORDER if n != "bf16"}
    for res in args.resolutions:
        rows = results[res]
        base_total = rows["bf16"]["total"]
        parts = ["bf16 = baseline"]
        for name in (n for n in CONFIG_ORDER if n != "bf16"):
            r = rows[name]
            ok = r["total"] <= base_total / args.gate_speedup and r["psnr"] >= args.gate_psnr
            per_cfg_pass[name] &= ok
            parts.append(
                f"{name}: {'PASS' if ok else 'FAIL'} "
                f"(total {r['total']:.1f}s vs {base_total:.1f}s, PSNR {r['psnr']:.1f} dB)"
            )
        print(f"  {res}x{res}: " + " | ".join(parts))
    print()
    winners = [n for n in CONFIG_PREFERENCE if per_cfg_pass[n]]
    if winners:
        print(f"OVERALL: {', '.join(winners)} pass the gate -> proceed "
              f"(add --vae-dtype to edit_image.py, default = {winners[0]})")
    else:
        print("OVERALL: no configuration passed the gate -> make no changes to edit_image.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
