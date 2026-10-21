#!/usr/bin/env python3
"""Standalone FLUX.2-klein-4B image editing.

Replicates the ComfyUI workflow in ``comfyui_image_edit_workflow.json`` without the
ComfyUI runtime. The workflow JSON is the source of truth for the pipeline
structure; two CLI defaults intentionally deviate from it to match the
distilled-checkpoint behaviour of ``new_image.py``: ``--steps`` defaults to
4 (the workflow node uses 20) and ``--cfg`` defaults to 1.0 (the workflow
node uses 5). Pass ``--steps 20 --cfg 5`` to reproduce the workflow exactly.
Node-by-node mapping:

  LoadImage (76)
      -> open with PIL, convert to RGB.

  ImageScaleToTotalPixels (75:129, megapixels=1, nearest-exact, steps=1)
      -> scale so the total pixel count is 1,048,576, floor both dimensions,
         resize with PIL NEAREST.

  GetImageSize (75:133)
      -> the scaled (w, h) sizes the denoise grid and the reference encode.

  VAEEncode (75:132) + ReferenceLatent (75:131, 75:134)
      -> the scaled image is VAE-encoded (argmax retrieval) and the resulting
         reference latent is concatenated onto BOTH the positive and negative
         conditioning sequences.

  CLIPTextEncode (75:74 positive, 75:73 negative "" only when --cfg > 1)
      -> the positive prompt is the edit text; the negative prompt is "" and
         is encoded only when CFG is actually used (--cfg > 1). The text
         sequence is fitted to the actual prompt length (capped by
         --max-seq-len) so short prompts don't pay for padding tokens.

  EmptyFlux2LatentImage (75:140) + RandomNoise (75:139, seed=72054227065427)
      -> denoising starts from pure Gaussian noise, not a noised input image.

  Flux2Scheduler (75:142, steps=20) + KSamplerSelect (75:141, euler)
      -> flow-match euler sampling with sigmas linspace(1, 1/steps, steps) and
         the empirical timestep shift mu derived from the latent seq length.
         The workflow node uses steps=20; the CLI default is 4, the
         BFL-recommended count for this distilled checkpoint (new_image.py).

  CFGGuider (75:137, cfg=5) + SamplerCustomAdvanced (75:136)
      -> when CFG is used, both the positive and negative passes run and are
         combined as neg + cfg * (pos - neg). The workflow's guider is
         unconditional (cfg=5), but the checkpoint is distilled, so the CLI
         default is --cfg 1.0 and only the positive pass runs; --cfg > 1
         enables the negative pass (pass --cfg 5 to match the workflow).

  VAEDecode (75:144) + SaveImage (75:145, prefix "Flux2-Klein-4b-base")
      -> unpack latents via their position ids, apply the VAE batch-norm stats,
         decode, and save a PNG.

Grid alignment note: ComfyUI encodes the *exact* scaled size (its VAE handles
arbitrary dimensions), while EmptyFlux2LatentImage truncates the denoise grid
to a multiple of 16. The diffusers pipeline requires the reference and noise
grids to be identical, so after the 1 MP scale the image is additionally
NEAREST-resized to a multiple of 16. This is identical to the ComfyUI result
when the scaled size is already a multiple of 16 and differs by a few pixels
otherwise.

GPU memory note: at the 1 MP grid the transformer sees an ~8k-token joint
sequence (denoise + reference + text). On a 16 GB GPU the bf16 transformer
(~7 GiB) plus SDPA's math fallback, which materializes the full S x S score
matrix and its softmax copy (~12 GiB transient), does not fit. The script
probes for a flash / memory-efficient / cuDNN SDPA backend (none materialize
S x S) and uses the first one available. If none is available on the
platform (e.g. consumer ROCm gfx1030), torch.nn.functional.scaled_dot_
product_attention is swapped for a query-chunked math implementation around
the inference block: queries are processed in blocks sized from the free
VRAM (ComfyUI's attention_sub_quad rule), so the transient score matrix is
(B*H*chunk*S) instead of (B*H*S*S). Each block's softmax still spans the
full key range, so the math is unchanged; the patch is installed and
restored around the block only.

Requires a diffusers version that ships Flux2KleinPipeline.

Usage:
    python edit_image.py --prompt "..." --image /path/to/input.jpg
    python edit_image.py --prompt "..." --image in.jpg --megapixels 0.25
    python edit_image.py --prompt "..." --image in.jpg --steps 20 --cfg 5
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    compute_empirical_mu,
    retrieve_timesteps,
)

# ComfyUI ImageScaleToTotalPixels: megapixels=1 -> 1,048,576 total pixels.
# --megapixels scales this budget (1.0 matches the workflow node).
TARGET_PIXELS = 1024 * 1024
# ComfyUI SaveImage prefix.
OUTPUT_PREFIX = "Flux2-Klein-4b-base"
# Container environment (compose.yaml); fall back to host-friendly defaults.
DEFAULT_MODEL_PATH = os.environ.get("IMG_MODEL_PATH", "/models/flux/FLUX.2-klein-4B")
DEFAULT_OUTPUT_DIR = os.environ.get("FLUX_OUTPUT_DIR", "/generated_images")

logger = logging.getLogger("edit_image")


def scale_to_total_pixels(image: Image.Image, target_pixels: int) -> Image.Image:
    """ComfyUI ImageScaleToTotalPixels (nearest-exact, resolution_steps=1).

    Scales the image so its total pixel count is ~``target_pixels``, flooring
    both dimensions (matches ComfyUI's int-truncation behaviour), and resizes
    with PIL NEAREST (the workflow's "nearest-exact" upscale method).
    """
    w, h = image.size
    scale = math.sqrt(target_pixels / (w * h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    if (new_w, new_h) != (w, h) and new_w > 0 and new_h > 0:
        image = image.resize((new_w, new_h), Image.Resampling.NEAREST)
    return image


def probe_sdpa_backend(device: torch.device) -> tuple[contextlib.AbstractContextManager, str | None]:
    """Pick an SDPA backend that avoids materializing the S x S attention matrix.

    Returns ``(context, name)``: a context manager that forces the chosen
    backend around the inference block, and its name (``None`` if no fast
    backend is available and SDPA's default selection should be left alone).
    A tiny probe forward verifies eligibility on this platform before
    committing to the backend.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return contextlib.nullcontext(), None
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    cudnn = getattr(SDPBackend, "CUDNN_ATTENTION", None)
    if cudnn is not None:
        backends.append(cudnn)
    for backend in backends:
        try:
            probe = torch.randn(1, 24, 16, 128, dtype=torch.bfloat16, device=device)
            with warnings.catch_warnings():
                # The probe is expected to fail on gfx1030 (no compiled
                # flash/mem-efficient/cuDNN kernels); silence the kernel
                # eligibility UserWarnings it emits on every attempt.
                warnings.simplefilter("ignore", UserWarning)
                with sdpa_kernel(backend):
                    torch.nn.functional.scaled_dot_product_attention(probe, probe, probe)
        except Exception:
            continue
        return sdpa_kernel(backend), backend.name
    return contextlib.nullcontext(), None


# Candidate query-chunk sizes (largest first), matching ComfyUI's
# attention_sub_quad list.
_CHUNK_SIZES = (4096, 2048, 1024, 512, 256)
# Below this sequence length the full S x S math fallback is small enough
# (text-encoder attention) that native SDPA is preferred.
_CHUNK_MIN_SEQ = 2048
# The real F.scaled_dot_product_attention; set by chunked_sdpa_patch().
_ORIG_SDPA = None
# One-time log flag for the chunked SDPA path.
_CHUNK_LOGGED = False


def _chunked_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs,
) -> torch.Tensor:
    """Query-chunked stand-in for F.scaled_dot_product_attention.

    Mirrors ComfyUI's default ``attention_sub_quad`` on platforms where
    torch ships no fast SDPA kernel: the math-backend fallback materializes
    the full S x S score matrix (plus its softmax copy), which OOMs on 16 GB
    cards for the ~8k-token FLUX.2 joint sequence. Instead, the queries are
    processed in blocks; each block's softmax still spans the full key range,
    so the math is the same as full math attention (modulo floating-point
    associativity). Masked, causal, dropout, GQA, non-CUDA and small-S calls
    are delegated to the real SDPA.
    """
    if (
        attn_mask is not None
        or is_causal
        or dropout_p > 0.0
        or not query.is_cuda
        or query.dim() != 4
        or query.shape[1] != key.shape[1]  # GQA / MQA
        or max(query.shape[-2], key.shape[-2]) <= _CHUNK_MIN_SEQ
    ):
        return _ORIG_SDPA(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs,
        )
    if scale is None:
        scale = query.shape[-1] ** -0.5
    return _chunked_attention_core(query, key, value, scale)


def _chunked_attention_core(query, key, value, scale):
    """Query-chunked math attention (the actual computation).

    Queries are processed in blocks; each block's softmax still spans the
    full key range, so this is the same math as the full-S math backend
    (modulo floating-point associativity). The transient score tensor is
    (B*H*chunk*S_k) instead of (B*H*S_q*S_k).
    """
    global _CHUNK_LOGGED
    batch, heads, seq_q, dim = query.shape
    seq_k = key.shape[-2]
    chunk = 256
    if query.is_cuda:
        try:
            free_bytes = torch.cuda.mem_get_info(query.device)[0]
            # ComfyUI's rule (attention_sub_quad): the per-chunk score matrix
            # (batch*heads*chunk*seq_k elements) must fit in a quarter of the
            # free VRAM, leaving headroom for the softmax reuse, the output
            # slice, and the MIOPEN workspace.
            for x in _CHUNK_SIZES:
                if free_bytes / (batch * heads * query.element_size() * x * 4.0) >= seq_k:
                    chunk = x
                    break
        except Exception:
            pass

    if not _CHUNK_LOGGED:
        _CHUNK_LOGGED = True
        logger.debug(
            "chunked SDPA: seq %dx%d, batch*heads=%d, chunk=%d",
            seq_q, seq_k, batch * heads, chunk,
        )

    key_t = key.transpose(-2, -1)
    out = torch.empty_like(query)
    for start in range(0, seq_q, chunk):
        scores = torch.matmul(query[..., start:start + chunk, :], key_t)
        scores.mul_(scale)
        # In-place, max-subtracted softmax: same math as torch's softmax
        # without the extra (B*H*chunk*S_k) allocation of an out-of-place call.
        scores.sub_(scores.amax(dim=-1, keepdim=True)).exp_()
        scores.div_(scores.sum(dim=-1, keepdim=True))
        out[..., start:start + chunk, :] = torch.matmul(scores, value)
    return out


def chunked_sdpa_patch() -> contextlib.AbstractContextManager:
    """Swap ``torch.nn.functional.scaled_dot_product_attention`` (the exact
    lookup diffusers' attention dispatch performs) for the query-chunked
    implementation above while active; restore the original on exit.
    """

    @contextlib.contextmanager
    def _patch():
        global _ORIG_SDPA
        _ORIG_SDPA = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = _chunked_scaled_dot_product_attention
        try:
            yield
        finally:
            torch.nn.functional.scaled_dot_product_attention = _ORIG_SDPA
            _ORIG_SDPA = None

    return _patch()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone FLUX.2-klein-4B image editing "
        "(replicates theo/comfyui_image_edit_workflow.json).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", required=True, help="edit prompt (positive CLIPTextEncode text)")
    parser.add_argument("--image", required=True, help="input image to edit (LoadImage)")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="diffusers-format FLUX.2-klein-4B repo (env: IMG_MODEL_PATH)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"output PNG path (default: <FLUX_OUTPUT_DIR|/generated_images>/{OUTPUT_PREFIX}-<timestamp>.png)",
    )
    parser.add_argument("--steps", type=int, default=4, help="denoising steps (Flux2Scheduler); 4 is the BFL-recommended count for this distilled checkpoint (the workflow node uses 20)")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale (CFGGuider); 1.0 disables CFG, as intended for this distilled checkpoint (the workflow node uses 5)")
    parser.add_argument("--megapixels", type=float, default=1.0, help="total pixel budget for the aspect-preserving scale (ImageScaleToTotalPixels; 1.0 = 1,048,576 px, matching the workflow)")
    parser.add_argument("--max-seq-len", type=int, default=512, help="cap on encoded text tokens (encode_prompt max_sequence_length); the sequence is auto-fitted to the actual prompt length up to this cap")
    parser.add_argument("--seed", type=int, default=None, help="noise seed (RandomNoise); random when omitted")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="model dtype",
    )
    parser.add_argument(
        "--vae-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float16",
        help="VAE dtype (cast after load); float16 is ~3.7-4.5x faster than bfloat16 on gfx1030 with ~64 dB decode PSNR (see vae_bench.log)",
    )
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="offload models between CPU and GPU as needed (enable_model_cpu_offload)",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if int(args.megapixels * TARGET_PIXELS) < 16 * 16:
        parser.error("--megapixels too small: the scaled grid must be at least 16x16")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t_start = time.monotonic()
    if args.seed is None:
        args.seed = random.getrandbits(63)
    logger.info("seed=%d", args.seed)
    miopen_env = {k: v for k, v in sorted(os.environ.items())
                  if k.startswith("MIOPEN_")}
    logger.debug(
        "torch %s (hip %s), MIOPEN env: %s",
        torch.__version__, torch.version.hip, miopen_env or "{}",
    )
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    # --- LoadImage ----------------------------------------------------------
    image = Image.open(args.image).convert("RGB")
    src_w, src_h = image.size

    # --- ImageScaleToTotalPixels (nearest-exact, steps=1) -------------------
    # The workflow node uses megapixels=1 (TARGET_PIXELS); --megapixels scales
    # that budget so quick edits can run on smaller grids.
    target_pixels = int(args.megapixels * TARGET_PIXELS)
    image = scale_to_total_pixels(image, target_pixels)
    width, height = image.size  # --- GetImageSize
    logger.info(
        "input %dx%d -> scaled %dx%d (%g MP, NEAREST)",
        src_w, src_h, width, height, args.megapixels,
    )

    # --- GPU memory strategy ---------------------------------------------------
    # At the 1 MP grid the transformer's joint sequence is ~8k tokens; SDPA's
    # math fallback materializes the full S x S score matrix (~12 GiB
    # transient with the softmax copy), which OOMs on 16 GB cards alongside
    # the bf16 transformer. Prefer a flash / memory-efficient / cuDNN SDPA
    # backend; if the platform has none, swap in a query-chunked math
    # implementation (ComfyUI's attention_sub_quad equivalent) around the
    # inference block.
    sdpa_ctx, sdpa_backend = probe_sdpa_backend(device)
    if sdpa_backend is None:
        sdpa_ctx = chunked_sdpa_patch()
        logger.warning(
            "SDPA backend: none available; "
            "using query-chunked math attention (ComfyUI sub-quadratic equivalent)"
        )
    else:
        logger.info("SDPA backend: %s", sdpa_backend)

    # --- Model load ----------------------------------------------------------
    t0 = time.monotonic()
    pipe = Flux2KleinPipeline.from_pretrained(args.model_path, dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    logger.info("model loaded in %.1fs (cpu_offload=%s)", time.monotonic() - t0, args.cpu_offload)

    # --- VAE dtype ---------------------------------------------------------
    # RDNA2 has no native bf16, and the bench (vae_bench.log) shows float16
    # VAE encode/decode ~3.7-4.5x faster than bfloat16 at ~64 dB decode PSNR,
    # so float16 is the default. .to() is in-place, so the CPU offload hooks
    # attached above are preserved.
    vae_dtype = getattr(torch, args.vae_dtype)
    if vae_dtype != pipe.vae.dtype:
        logger.info("casting VAE %s -> %s (--vae-dtype)", pipe.vae.dtype, vae_dtype)
        pipe.vae = pipe.vae.to(vae_dtype)

    # The denoise grid (EmptyFlux2LatentImage) and the reference encode
    # (VAEEncode) must share the same spatial grid: floor to vae_scale_factor*2
    # (= 16) and NEAREST-resize the image so the reference latents align
    # exactly with the denoise latents.
    multiple_of = pipe.vae_scale_factor * 2
    width = (width // multiple_of) * multiple_of
    height = (height // multiple_of) * multiple_of
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.NEAREST)
    logger.info("denoise/reference grid: %dx%d", width, height)

    with sdpa_ctx, torch.inference_mode():
        # The checkpoint is distilled, so CFG is off by default (cfg <= 1.0),
        # matching new_image.py; the negative (empty-prompt) encode and the
        # uncond pass are only paid for when --cfg > 1.0.
        do_cfg = args.cfg > 1.0

        # Fit the text sequence to the actual prompt length (capped), so short
        # prompts don't pad to the full default: padded tokens run through the
        # text encoder and are attended to in the transformer's joint
        # self-attention for no benefit.
        templated = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        n_prompt_tokens = len(pipe.tokenizer(templated)["input_ids"])
        max_seq_len = min(args.max_seq_len, max(n_prompt_tokens, 16))
        logger.info(
            "text seq: %d tokens (prompt=%d, cap=%d)",
            max_seq_len, n_prompt_tokens, args.max_seq_len,
        )

        # --- CLIPTextEncode (positive prompt; negative "" only when do_cfg) --
        t_phase = time.monotonic()
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=args.prompt, device=device, max_sequence_length=max_seq_len
        )
        if do_cfg:
            negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                prompt="", device=device, max_sequence_length=max_seq_len
            )
        logger.info("encode: %.1fs", time.monotonic() - t_phase)

        # --- RandomNoise -------------------------------------------------------
        generator = torch.Generator(device=device).manual_seed(args.seed)

        # --- EmptyFlux2LatentImage + noise latents -----------------------------
        # prepare_latents is called before prepare_image_latents, matching the
        # pipeline's RNG consumption order.
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )

        # --- VAEEncode -> ReferenceLatent ---------------------------------------
        # The image is already exactly (width, height), so the crop preprocess
        # is an identity transform.
        t_phase = time.monotonic()
        condition_image = pipe.image_processor.preprocess(
            image, height=height, width=width, resize_mode="crop"
        )
        image_latents, image_latent_ids = pipe.prepare_image_latents(
            images=[condition_image],
            batch_size=1,
            generator=generator,
            device=device,
            dtype=pipe.vae.dtype,
        )
        logger.info("VAE encode: %.1fs", time.monotonic() - t_phase)

        # --- Flux2Scheduler ------------------------------------------------------
        sigmas = np.linspace(1.0, 1.0 / args.steps, args.steps)
        if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=args.steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            pipe.scheduler, args.steps, device, sigmas=sigmas, mu=mu
        )
        pipe.scheduler.set_begin_index(0)
        # CPU copy of the schedule for progress display only; reading t.item()
        # inside the loop would force a D2H sync on every logged step.
        timesteps_cpu = timesteps.to("cpu").tolist()

        # --- KSamplerSelect (euler) + CFGGuider + SamplerCustomAdvanced ----------
        # Denoise phase timing starts here (noise prep + sampling loop).
        t_phase = time.monotonic()
        latent_seq = latents.size(1)
        # The reference id tensor is constant across steps; concat once.
        latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
        logger.info(
            "latent shapes: denoise %s, reference %s, joint %d tokens",
            tuple(latents.shape), tuple(image_latents.shape),
            latents.shape[1] + image_latents.shape[1],
        )
        logger.info("denoising %d steps (euler, cfg=%g, mu=%.3f)", num_inference_steps, args.cfg, mu)
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            # Reference latent tokens are appended after the denoise tokens.
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(latents.dtype)

            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
            noise_pred = noise_pred[:, :latent_seq]

            if do_cfg:
                with pipe.transformer.cache_context("uncond"):
                    neg_noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, :latent_seq]
                noise_pred = neg_noise_pred + args.cfg * (noise_pred - neg_noise_pred)

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if (i + 1) % 5 == 0 or i == num_inference_steps - 1:
                logger.info("step %d/%d  t=%.4f", i + 1, num_inference_steps, timesteps_cpu[i])
        logger.info("denoise: %.1fs", time.monotonic() - t_phase)

        # --- VAEDecode -------------------------------------------------------------
        t_phase = time.monotonic()
        latent_height = 2 * (int(height) // multiple_of)
        latent_width = 2 * (int(width) // multiple_of)
        latents = pipe._unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)

        # The BN stats and the decode input run in the VAE's own dtype
        # (--vae-dtype), not the transformer latent dtype.
        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, pipe.vae.dtype)
        latents_bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).to(latents.device, pipe.vae.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = pipe._unpatchify_latents(latents)
        torch.cuda.synchronize()  # kernels above are async; stop the clock on GPU completion
        t_prep = time.monotonic() - t_phase

        output = pipe.vae.decode(latents.to(device, pipe.vae.dtype), return_dict=False)[0]
        torch.cuda.synchronize()
        t_vae = time.monotonic() - t_phase

        output = pipe.image_processor.postprocess(output, output_type="pil")[0]
        t_post = time.monotonic() - t_phase
        logger.info(
            "decode: %.1fs (prep %.1fs, vae.decode %.1fs, postprocess %.1fs)",
            t_post, t_prep, t_vae - t_prep, t_post - t_vae,
        )
        peak = torch.cuda.max_memory_allocated() / 2**30
        free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
        logger.info("VRAM: peak %.2f GiB, free %.2f/%.2f GiB", peak, free, total)

    pipe.maybe_free_model_hooks()

    # --- SaveImage (prefix "Flux2-Klein-4b-base") -----------------------------------
    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".png")
    else:
        out_path = Path(DEFAULT_OUTPUT_DIR) / f"{OUTPUT_PREFIX}-{datetime.now():%Y%m%d-%H%M%S}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_phase = time.monotonic()
    output.save(out_path)
    logger.info(
        "saved %s (%dx%d) (save %.1fs, total %.1fs)",
        out_path, output.width, output.height,
        time.monotonic() - t_phase, time.monotonic() - t_start,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
