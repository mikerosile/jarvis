#!/usr/bin/env python3
"""Standalone FLUX.2-klein-4B text-to-image generation.

Generates a new image from a text prompt only (no reference image), using the
same ``Flux2KleinPipeline`` components as ``edit_image.py``. The denoising is
driven manually through the pipeline's component methods so the sampling
matches the ComfyUI-style workflow behaviour:

   CLIPTextEncode (positive prompt; negative "" only when --cfg > 1)
       -> the prompt is the generation text; the negative prompt is "" and
          is encoded only when CFG is actually used (--cfg > 1). The text
          sequence is fitted to the actual prompt length (capped by
          --max-seq-len) so short prompts don't pay for padding tokens.

   EmptyFlux2LatentImage (width/height from the CLI) + RandomNoise (seed)
       -> the denoise grid is the CLI --width/--height floored to a multiple
          of vae_scale_factor * 2 (= 16); sampling starts from pure Gaussian
          noise seeded by --seed (a random 63-bit seed when omitted).

   Flux2Scheduler (steps) + KSamplerSelect (euler)
       -> flow-match euler sampling with sigmas linspace(1, 1/steps, steps) and
          the empirical timestep shift mu derived from the latent seq length.

   CFGGuider (cfg) + SamplerCustomAdvanced
       -> the default cfg is 1.0, matching the checkpoint's distilled
          training, so only the positive pass runs. With --cfg > 1 both the
          positive and negative passes run and are combined as
          neg + cfg * (pos - neg). (The pipeline's own __call__ ignores
          guidance_scale for step-wise distilled checkpoints, which is why
          the components are driven manually here.)

   VAEDecode + SaveImage (prefix "Flux2-Klein-4b-base")
       -> unpack latents via their position ids, apply the VAE batch-norm
          stats, decode, and save a PNG.

GPU memory note: at the default 1024x1024 grid the transformer sees a joint
sequence of ~4k image tokens plus the text tokens. On a 16 GB GPU the bf16
transformer (~7 GiB) plus SDPA's math fallback, which materializes the full
S x S score matrix and its softmax copy, does not fit. The script probes for
a flash / memory-efficient / cuDNN SDPA backend (none materialize S x S) and
uses the first one available. If none is available on the platform (e.g.
consumer ROCm gfx1030), torch.nn.functional.scaled_dot_product_attention is
swapped for a query-chunked math implementation around the inference block:
queries are processed in blocks sized from the free VRAM (ComfyUI's
attention_sub_quad rule), so the transient score matrix is (B*H*chunk*S)
instead of (B*H*S*S). Each block's softmax still spans the full key range, so
the math is unchanged; the patch is installed and restored around the block
only.

Requires a diffusers version that ships Flux2KleinPipeline.

Usage:
    python new_image.py --prompt "a watercolor painting of a red fox"
    python new_image.py --prompt "..." --width 896 --height 1152 --seed 1234
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    compute_empirical_mu,
    retrieve_timesteps,
)

# ComfyUI SaveImage prefix.
OUTPUT_PREFIX = "Flux2-Klein-4b-base"
# Container environment (compose.yaml); fall back to host-friendly defaults.
DEFAULT_MODEL_PATH = os.environ.get("IMG_MODEL_PATH", "/models/flux/FLUX.2-klein-4B")
DEFAULT_OUTPUT_DIR = os.environ.get("FLUX_OUTPUT_DIR", "/generated_images")
# The pipeline's documented best-result sample size.
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

logger = logging.getLogger("new_image")


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
    cards for the FLUX.2 joint sequence. Instead, the queries are processed
    in blocks; each block's softmax still spans the full key range, so the
    math is the same as full math attention (modulo floating-point
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
        description="Standalone FLUX.2-klein-4B text-to-image generation "
        "(prompt only, no reference image).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", required=True, help="text prompt to generate (positive CLIPTextEncode text)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="output width in pixels (floored to a multiple of 16)")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="output height in pixels (floored to a multiple of 16)")
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
    parser.add_argument("--steps", type=int, default=4, help="denoising steps (Flux2Scheduler); 4 is the BFL-recommended count for this distilled checkpoint")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale (CFGGuider); 1.0 disables CFG, as intended for this distilled checkpoint")
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
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="offload models between CPU and GPU as needed (enable_model_cpu_offload)",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.width < 16 or args.height < 16:
        parser.error("--width and --height must be >= 16 (the denoise grid is a multiple of 16)")
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
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    # --- GPU memory strategy ---------------------------------------------------
    # At the 1 MP grid the transformer's joint sequence is ~4k image tokens
    # plus text; SDPA's math fallback materializes the full S x S score
    # matrix (plus its softmax copy), which OOMs on 16 GB cards alongside
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

    # --- EmptyFlux2LatentImage --------------------------------------------------
    # The denoise grid is floored to a multiple of vae_scale_factor * 2 (= 16).
    multiple_of = pipe.vae_scale_factor * 2
    width = (args.width // multiple_of) * multiple_of
    height = (args.height // multiple_of) * multiple_of
    logger.info("denoise grid: %dx%d", width, height)

    with sdpa_ctx, torch.inference_mode():
        # The checkpoint is distilled, so CFG is off by default (cfg <= 1.0);
        # the negative (empty-prompt) encode is only paid for when CFG is used.
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

        # --- Pure-noise latents -------------------------------------------------
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
        logger.info("denoising %d steps (euler, cfg=%g, mu=%.3f)", num_inference_steps, args.cfg, mu)
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents

            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]

            if do_cfg:
                with pipe.transformer.cache_context("uncond"):
                    neg_noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]
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

        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = pipe._unpatchify_latents(latents)

        output = pipe.vae.decode(latents.to(device), return_dict=False)[0]
        output = pipe.image_processor.postprocess(output, output_type="pil")[0]
        logger.info("decode: %.1fs", time.monotonic() - t_phase)

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
