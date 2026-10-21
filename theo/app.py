"""T.H.E.O. - Thought Holography and Extraction Operator

FastAPI app running inside the `jarvis-theo` podman container. The
host-side supervisor (phil/app.py) starts and stops this
container via the user-scope unit `jarvis-theo` so it can share the
single 16 GB GPU with the LLM container: one model resident at a time.

Endpoints:
    GET  /health     -> {"status": "loading" | "ready", "model": ..., "device": ...}
    GET  /info       -> device / VRAM diagnostics
    POST /generate   -> {"prompt": str, "width": int, "height": int, "cfg_scale": float,
                         "steps": int, "seed": int}
                        -> {"path": str, "seed": int, "elapsed_seconds": float}
    POST /v1/images/generations  OpenAI Images API compat (also served at
                         /images/generations); stock OpenAI fields (model, n,
                         size, response_format, ...) are accepted, extras
                         ignored; FLUX adds steps/cfg_scale/seed
                         -> {"created": int,
                             "data": [{"b64_json": str, "url": str,
                                       "revised_prompt": str, "seed": int}]}
    POST /v1/images/edits    OpenAI Images API compat (also served at
                         /images/edits); multipart form with `image`
                         (or `image[]`) + `prompt`; steps/cfg_scale run at
                         the FLUX defaults; same response shape as
                         /v1/images/generations
    POST /img2img    -> {"prompt": str, "image_base64": str, "cfg_scale": float,
                         "steps": int, "seed": int}
                        -> {"path": str, "seed": int, "elapsed_seconds": float,
                            "input_dimensions": dict, "output_dimensions": dict}

Configuration (environment, set in theo/compose.yaml):
    IMG_MODEL_PATH     FLUX model directory inside the container
                       (default: /models/img/FLUX.2-klein-4B)
    FLUX_OUTPUT_DIR    where generated PNGs are written (host-shared)

Model note:
    FLUX.2 klein-4B (~4B params, bfloat16) with Qwen3-4B text encoder is
    ~17 GB total (transformer ~9 GB, text encoder ~8 GB, + VAE), which
    exceeds 16 GB VRAM -- so it runs with sequential CPU offload.
    Distilled to 4 steps for fast generation. The transformer runs with
    guidance=None (the distilled checkpoint ignores guidance_scale), so
    both endpoints expose cfg_scale for explicit CFG instead.

txt2img approach (same component-level logic as new_image.py):
    The requested width/height is floored to a multiple of 16 and denoising
    starts from pure Gaussian noise. The Qwen3 text sequence is fit to the
    prompt length (cap 512); the negative (empty-prompt) pass runs only when
    cfg_scale > 1.0. Components are offloaded sequentially (one on the GPU at
    a time; peak VRAM ~7-8 GB on 16 GB GPUs).

img2img approach (same reference-latent logic as edit_image.py):
    The input image is scaled to ~1 MP (aspect preserving, NEAREST), floored to a
    multiple of 16, and VAE-encoded (argmax). Denoising starts from pure Gaussian
    noise and the reference latents are concatenated onto the denoise sequence, so
    the reference tokens condition the transformer's joint self-attention. The
    Qwen3 text sequence is fit to the prompt length (cap 512); the negative
    (empty-prompt) pass runs only when cfg_scale > 1.0. Components are offloaded
    sequentially (one on the GPU at a time; peak VRAM ~7-8 GB on 16 GB GPUs).
"""

import base64
import gc
import logging
import math
import os
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("theo_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUTPUT_DIR = os.environ.get("FLUX_OUTPUT_DIR", "/images")
DEVICE = "cuda"  # torch.cuda is HIP-mapped on AMD ROCm

# ---------------------------------------------------------------------------
# Model path (FLUX.2 klein-4B) -- overridden by IMG_MODEL_PATH env var in compose.yaml
# ---------------------------------------------------------------------------
PIPELINE_MODEL_PATH: str = os.environ.get(
    "IMG_MODEL_PATH", "/models/img/FLUX.2-klein-4B"
)

# Defaults for FLUX.2 klein (distilled to 4 steps; CFG off by default)
_DEFAULT_STEPS = 4
_DEFAULT_CFG_SCALE = 1.0        # CFG off by default; only >1.0 runs the negative pass
_TARGET_PIXELS = 1024 * 1024    # img2img pixel budget (ComfyUI ImageScaleToTotalPixels)


# ---------------------------------------------------------------------------
# App + state
# ---------------------------------------------------------------------------

app = FastAPI(title="FLUX.2 klein image server", version="1.0.0")
_pipeline = None  # Flux2KleinPipeline (handles txt2img & img2img)
_model_ready = False


class GenerateRequest(BaseModel):
    prompt: str
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)
    cfg_scale: float = Field(
        default=_DEFAULT_CFG_SCALE, ge=0.0, le=20.0,
        description="CFG scale; only >1.0 runs the negative (empty-prompt) pass",
    )
    steps: int = Field(default=_DEFAULT_STEPS, ge=1, le=100)
    seed: int = Field(default=-1)  # -1 = random


class GenerateResponse(BaseModel):
    path: str
    seed: int
    elapsed_seconds: float


class Img2ImgRequest(BaseModel):
    """Image-to-image edit request (reference-latent logic from edit_image.py).

    The denoise grid is derived from the input image: scaled to ~1 MP (aspect
    preserving, NEAREST), then floored to a multiple of 16. Denoising starts from
    pure noise and the VAE-encoded input image is concatenated as reference tokens
    onto the denoise sequence.
    """

    prompt: str = Field(
        ..., min_length=1, max_length=500,
        description="Text prompt describing the desired edit",
    )

    image_base64: str = Field(..., description="Input image as base64-encoded PNG/JPEG")

    cfg_scale: float = Field(
        default=_DEFAULT_CFG_SCALE, ge=0.0, le=20.0,
        description="CFG scale; only >1.0 runs the negative (empty-prompt) pass",
    )
    seed: int = Field(default=-1, ge=-1)
    steps: int = Field(default=_DEFAULT_STEPS, ge=1, le=100)


class Img2ImgResponse(BaseModel):
    """Response model for image-to-image generation."""

    path: str = Field(..., description="Absolute file path to generated image")
    seed: int = Field(..., description="Random seed used for generation")
    elapsed_seconds: float = Field(..., description="Total generation time in seconds")
    input_dimensions: dict = Field(
        ..., description="Original image dimensions {'width': int, 'height': int}"
    )
    output_dimensions: dict = Field(
        ..., description="Output image dimensions {'width': int, 'height': int}"
    )


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_model():
    """Load FLUX.2 klein-4B pipeline with sequential CPU offload. Takes ~30s."""
    global _pipeline, _model_ready

    logger.info("Loading FLUX.2 klein-4B pipeline from %s ...", PIPELINE_MODEL_PATH)
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import after env setup so PyTorch config is applied first
    from diffusers import Flux2KleinPipeline

    torch.cuda.empty_cache()  # Clean slate before loading

    # Load FLUX.2 klein-4B in bfloat16. The transformer (~9 GB) + Qwen3 text encoder
    # (~8 GB) + VAE = ~17 GB total -- exceeds 16 GB VRAM for inference (activations).
    _pipeline = Flux2KleinPipeline.from_pretrained(
        PIPELINE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    # Enable VAE optimizations (slicing/tiling reduce decode VRAM)
    if hasattr(_pipeline.vae, "enable_slicing"):
        _pipeline.vae.enable_slicing()
    if hasattr(_pipeline.vae, "enable_tiling"):
        _pipeline.vae.enable_tiling()

    # Sequential CPU offload: peak VRAM ~8-10 GB (one component on GPU at a time)
    _pipeline.enable_sequential_cpu_offload()

    logger.info(
        "FLUX.2 klein-4B loaded in %.1fs -- VRAM alloc=%.1fGB reserved=%.1fGB",
        time.time() - t0,
        torch.cuda.memory_allocated() / 2**30,
        torch.cuda.memory_reserved() / 2**30,
    )

    _model_ready = True


@app.on_event("shutdown")
def unload_model():
    """Free VRAM on container stop so the LLM can reload."""
    global _pipeline, _model_ready
    _model_ready = False

    if _pipeline is not None:
        del _pipeline
        _pipeline = None

    torch.cuda.empty_cache()
    gc.collect()
    logger.info("FLUX.2 klein-4B pipeline unloaded, VRAM freed")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ready" if _model_ready else "loading",
        "model": Path(PIPELINE_MODEL_PATH).name,
        "device": DEVICE,
    }


def _get_vram():
    """Get actual GPU memory via ROCm HIP (PyTorch's CUDA abstraction doesn't track correctly on AMD)."""
    import ctypes
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        total = ctypes.c_size_t()
        free = ctypes.c_size_t()
        if hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0:
            return {
                "total_gb": round(total.value / 2**30, 1),
                "free_gb": round(free.value / 2**30, 1),
                "used_gb": round((total.value - free.value) / 2**30, 1),
            }
    except OSError:
        pass
    return {"total_gb": None, "free_gb": None, "used_gb": None}


@app.get("/info")
def info():
    """Device + VRAM diagnostics."""
    cuda_ok = torch.cuda.is_available()
    vram = _get_vram() if cuda_ok else {"total_gb": None, "free_gb": None, "used_gb": None}
    return {
        "model_ready": _model_ready,
        "cuda_available": cuda_ok,
        "device": torch.cuda.get_device_name(0) if cuda_ok else "cpu",
        **vram,
    }


def _run_txt2img(prompt: str, width: int, height: int, steps: int,
                 cfg_scale: float, seed: int) -> tuple[str, float]:
    """Txt2img core (shared by /generate and /v1/images/generations).

    The requested width/height is floored to a multiple of 16 and denoising
    starts from pure Gaussian noise. The Qwen3 text sequence is fit to the
    prompt length (cap 512); the negative (empty-prompt) pass runs only when
    cfg_scale > 1.0. The caller must have confirmed the model is ready.

    Returns (out_path, total_elapsed).
    """
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )

    pipeline = _pipeline
    device = torch.device(DEVICE)

    t_total = time.time()
    logger.info(
        "txt2img: %dx%d steps=%d seed=%d cfg=%.1f: %.80s...",
        width, height, steps, seed, cfg_scale, prompt[:80],
    )

    # --- EmptyFlux2LatentImage: floor the grid to the denoise multiple
    multiple_of = pipeline.vae_scale_factor * 2
    gen_w = (width // multiple_of) * multiple_of
    gen_h = (height // multiple_of) * multiple_of
    logger.info("grid: %dx%d -> %dx%d", width, height, gen_w, gen_h)
    width, height = gen_w, gen_h

    with torch.inference_mode():
        # Distilled checkpoint: CFG off by default; the negative (empty-prompt)
        # encode and the uncond pass are only paid for when cfg > 1.0.
        do_cfg = cfg_scale > 1.0

        # Fit the text sequence to the actual prompt length (capped), so short
        # prompts don't pad to the full default.
        templated = pipeline.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        max_seq_len = min(512, max(len(pipeline.tokenizer(templated)["input_ids"]), 16))

        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, device=device, max_sequence_length=max_seq_len
        )
        if do_cfg:
            negative_prompt_embeds, negative_text_ids = pipeline.encode_prompt(
                prompt="", device=device, max_sequence_length=max_seq_len
            )

        # Pure-Gaussian noise latents, seeded.
        generator = torch.Generator(device=device).manual_seed(seed)
        latents, latent_ids = pipeline.prepare_latents(
            batch_size=1,
            num_latents_channels=pipeline.transformer.config.in_channels // 4,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )

        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        if getattr(pipeline.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler, steps, device, sigmas=sigmas, mu=mu
        )
        pipeline.scheduler.set_begin_index(0)
        logger.info(
            "denoise: %d steps, cfg=%.1f, mu=%.3f, %d latent tokens",
            num_inference_steps, cfg_scale, mu, latents.shape[1],
        )
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents
            with pipeline.transformer.cache_context("cond"):
                noise_pred = pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
            if do_cfg:
                with pipeline.transformer.cache_context("uncond"):
                    neg_noise_pred = pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]
                noise_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode: unpack, apply the VAE's batch-norm stats in the VAE dtype,
        # unpatchify, VAE decode -> PIL.
        latent_height = 2 * (height // multiple_of)
        latent_width = 2 * (width // multiple_of)
        latents = pipeline._unpack_latents_with_ids(
            latents, latent_ids, latent_height // 2, latent_width // 2
        )
        latents_bn_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, pipeline.vae.dtype)
        latents_bn_std = torch.sqrt(
            pipeline.vae.bn.running_var.view(1, -1, 1, 1) + pipeline.vae.config.batch_norm_eps
        ).to(latents.device, pipeline.vae.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = pipeline._unpatchify_latents(latents)
        output = pipeline.vae.decode(latents.to(device, pipeline.vae.dtype), return_dict=False)[0]
        image = pipeline.image_processor.postprocess(output, output_type="pil")[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - t_total
    out_path = str(Path(OUTPUT_DIR) / f"flux_{int(time.time())}_{seed}.png")
    image.save(out_path)
    logger.info("txt2img done in %.1fs (%dx%d) -> %s", total_elapsed, width, height, out_path)

    return out_path, total_elapsed


@app.post("/generate", response_model=GenerateResponse)
def txt2img(req: GenerateRequest):
    """Text-to-image generation (same component-level logic as new_image.py).

    The requested width/height is floored to a multiple of 16 and denoising
    starts from pure Gaussian noise. The Qwen3 text sequence is fit to the
    prompt length (cap 512); the negative (empty-prompt) pass runs only when
    cfg_scale > 1.0.
    """
    if not _model_ready or _pipeline is None:
        raise HTTPException(503, "Model not loaded yet")

    seed = req.seed if req.seed >= 0 else int(time.time()) % (2**32)
    out_path, total_elapsed = _run_txt2img(
        req.prompt, req.width, req.height, req.steps, req.cfg_scale, seed
    )

    return GenerateResponse(
        path=out_path,
        seed=seed,
        elapsed_seconds=round(total_elapsed, 2),
    )


def _run_img2img(prompt: str, image_bytes: bytes, steps: int,
                 cfg_scale: float, seed: int) -> tuple[str, float, int, int, int, int]:
    """Img2img core (shared by /img2img and /v1/images/edits).

    The input is scaled to ~1 MP (aspect preserving, NEAREST), floored to a
    multiple of 16, and VAE-encoded (argmax). Denoising starts from pure
    Gaussian noise and the reference latents are concatenated onto the
    denoise sequence, so the reference tokens condition the transformer's
    joint self-attention. The caller must have confirmed the model is
    ready.

    Returns (out_path, total_elapsed, inp_w, inp_h, width, height).
    """
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )

    pipeline = _pipeline
    device = torch.device(DEVICE)

    t_total = time.time()
    logger.info(
        "img2img: steps=%d seed=%d cfg=%.1f: %.80s...",
        steps, seed, cfg_scale, prompt[:80],
    )

    # --- Input: scale to ~1 MP (ComfyUI ImageScaleToTotalPixels), floor to grid
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inp_w, inp_h = image.size
    scale = math.sqrt(_TARGET_PIXELS / (inp_w * inp_h))
    multiple_of = pipeline.vae_scale_factor * 2
    width = (int(inp_w * scale) // multiple_of) * multiple_of
    height = (int(inp_h * scale) // multiple_of) * multiple_of
    if width < multiple_of or height < multiple_of:
        raise HTTPException(
            400,
            f"input image too small: scaled grid is {width}x{height}, "
            f"must be at least {multiple_of}x{multiple_of}",
        )
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.NEAREST)
    logger.info("grid: input %dx%d -> %dx%d", inp_w, inp_h, width, height)

    with torch.inference_mode():
        # Distilled checkpoint: CFG off by default; the negative (empty-prompt)
        # encode and the uncond pass are only paid for when cfg > 1.0.
        do_cfg = cfg_scale > 1.0

        # Fit the text sequence to the actual prompt length (capped), so short
        # prompts don't pad to the full default.
        templated = pipeline.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        max_seq_len = min(512, max(len(pipeline.tokenizer(templated)["input_ids"]), 16))

        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, device=device, max_sequence_length=max_seq_len
        )
        if do_cfg:
            negative_prompt_embeds, negative_text_ids = pipeline.encode_prompt(
                prompt="", device=device, max_sequence_length=max_seq_len
            )

        # Pure-noise denoise latents first, then the reference encode, matching
        # the pipeline's RNG consumption order.
        generator = torch.Generator(device=device).manual_seed(seed)
        latents, latent_ids = pipeline.prepare_latents(
            batch_size=1,
            num_latents_channels=pipeline.transformer.config.in_channels // 4,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )
        # The image is already exactly (width, height), so the crop preprocess
        # is an identity transform.
        condition_image = pipeline.image_processor.preprocess(
            image, height=height, width=width, resize_mode="crop"
        )
        image_latents, image_latent_ids = pipeline.prepare_image_latents(
            images=[condition_image],
            batch_size=1,
            generator=generator,
            device=device,
            dtype=pipeline.vae.dtype,
        )

        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        if getattr(pipeline.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler, steps, device, sigmas=sigmas, mu=mu
        )
        pipeline.scheduler.set_begin_index(0)

        # The reference id tensor is constant across steps; concat once.
        latent_seq = latents.size(1)
        latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
        logger.info(
            "denoise: %d steps, cfg=%.1f, mu=%.3f, joint %d tokens",
            num_inference_steps, cfg_scale, mu, latent_seq + image_latents.shape[1],
        )
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            # Reference latent tokens are appended after the denoise tokens.
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(latents.dtype)
            with pipeline.transformer.cache_context("cond"):
                noise_pred = pipeline.transformer(
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
                with pipeline.transformer.cache_context("uncond"):
                    neg_noise_pred = pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, :latent_seq]
                noise_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode: unpack, apply the VAE's batch-norm stats in the VAE dtype,
        # unpatchify, VAE decode -> PIL.
        latent_height = 2 * (height // multiple_of)
        latent_width = 2 * (width // multiple_of)
        latents = pipeline._unpack_latents_with_ids(
            latents, latent_ids, latent_height // 2, latent_width // 2
        )
        latents_bn_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, pipeline.vae.dtype)
        latents_bn_std = torch.sqrt(
            pipeline.vae.bn.running_var.view(1, -1, 1, 1) + pipeline.vae.config.batch_norm_eps
        ).to(latents.device, pipeline.vae.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = pipeline._unpatchify_latents(latents)
        output = pipeline.vae.decode(latents.to(device, pipeline.vae.dtype), return_dict=False)[0]
        image = pipeline.image_processor.postprocess(output, output_type="pil")[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - t_total
    out_path = str(Path(OUTPUT_DIR) / f"flux_img2img_{int(time.time())}_{seed}.png")
    image.save(out_path)
    logger.info("img2img done in %.1fs -> %s", total_elapsed, out_path)

    return out_path, total_elapsed, inp_w, inp_h, width, height


@app.post("/img2img", response_model=Img2ImgResponse)
def img2img(req: Img2ImgRequest):
    """Image-to-image edit (same reference-latent logic as edit_image.py).

    The input is scaled to ~1 MP (aspect preserving, NEAREST), floored to a
    multiple of 16, and VAE-encoded (argmax). Denoising starts from pure
    Gaussian noise and the reference latents are concatenated onto the
    denoise sequence, so the reference tokens condition the transformer's
    joint self-attention.
    """
    if not _model_ready or _pipeline is None:
        raise HTTPException(503, "Model not loaded yet")

    seed = req.seed if req.seed >= 0 else int(time.time()) % (2**32)
    out_path, total_elapsed, inp_w, inp_h, width, height = _run_img2img(
        req.prompt, base64.b64decode(req.image_base64),
        req.steps, req.cfg_scale, seed,
    )

    return Img2ImgResponse(
        path=out_path,
        seed=seed,
        elapsed_seconds=round(total_elapsed, 2),
        input_dimensions={"width": inp_w, "height": inp_h},
        output_dimensions={"width": width, "height": height},
    )


# ---------------------------------------------------------------------------
# OpenAI Images API compat (serves Open WebUI v0.12.3's built-in
# "Image Generation" and "Edit Image" backends on the openai engine)
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(r"^(\d+)\s*[xX]\s*(\d+)$")
_SIZE_MIN_SIDE = 256
_SIZE_MAX_SIDE = 2048


class OpenAIGenerationsRequest(BaseModel):
    """OpenAI Images API request (POST /v1/images/generations).

    Accepts and ignores the stock OpenAI fields (model, background,
    quality, style, response_format, api-version, ...) so the stock
    Open WebUI client needs no modification.
    """

    model_config = ConfigDict(extra="allow")

    prompt: str = Field(min_length=1)
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = Field(
        default=None,
        description="'<width>x<height>', 256-2048 per side (default 1024x1024)",
    )
    steps: int = Field(default=_DEFAULT_STEPS, ge=1, le=50)
    cfg_scale: float = Field(
        default=_DEFAULT_CFG_SCALE, ge=0.0, le=20.0,
        description="CFG scale; only >1.0 runs the negative (empty-prompt) pass",
    )
    seed: int = Field(default=-1, ge=-1)


def _parse_openai_size(size: str | None) -> tuple[int, int]:
    """Parse an OpenAI-style 'WxH' size; None means the 1024x1024 default."""
    if size is None:
        return 1024, 1024
    m = _SIZE_RE.match(size.strip())
    if m is None:
        raise HTTPException(
            422, f"Invalid size {size!r}; expected '<width>x<height>'"
        )
    w, h = int(m.group(1)), int(m.group(2))
    if not (_SIZE_MIN_SIDE <= w <= _SIZE_MAX_SIDE
            and _SIZE_MIN_SIDE <= h <= _SIZE_MAX_SIDE):
        raise HTTPException(
            422, f"Size {w}x{h} out of range; each side must be 256-2048"
        )
    return w, h


def _openai_image_item(prompt: str, out_path: str, seed: int) -> dict:
    """One entry of the OpenAI `data` list (b64_json payload + metadata)."""
    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {
        "b64_json": b64,
        "url": f"/images/{Path(out_path).name}",  # cosmetic; not served
        "revised_prompt": prompt,
        "seed": seed,
    }


@app.post("/v1/images/generations")
@app.post("/images/generations")
def openai_image_generations(req: OpenAIGenerationsRequest):
    """OpenAI Images API compat: text-to-image (same core as /generate).

    Returns {"created": <unix ts>, "data": [{"b64_json", "url",
    "revised_prompt", "seed"}, ...]} with `n` images (1-4).
    """
    if not _model_ready or _pipeline is None:
        raise HTTPException(503, "Model not loaded yet")

    width, height = _parse_openai_size(req.size)
    data = []
    for i in range(req.n):
        if req.seed >= 0:
            seed = (req.seed + i) % (2**32)
        else:
            seed = (int(time.time() * 1000) + i) % (2**32)
        out_path, _elapsed = _run_txt2img(
            req.prompt, width, height, req.steps, req.cfg_scale, seed
        )
        data.append(_openai_image_item(req.prompt, out_path, seed))

    return {"created": int(time.time()), "data": data}


# --- Image edits (multipart form) ------------------------------------------


@app.post("/v1/images/edits")
@app.post("/images/edits")
async def openai_image_edits(
    image: list[UploadFile] | None = File(default=None),
    image_list: list[UploadFile] | None = File(default=None, alias="image[]"),
    prompt: str = Form(...),
    n: int = Form(default=1, ge=1, le=4),
    seed: int = Form(default=-1, ge=-1),
):
    """OpenAI Images API compat: image edit (same core as /img2img).

    Multipart form: `image` (or `image[]`; first wins) plus `prompt`.
    `n` (1-4) and `seed` are optional; steps/cfg_scale run at the FLUX
    defaults and other stock OpenAI fields are ignored. Returns
    {"created": <unix ts>, "data": [...]} like /v1/images/generations.
    """
    if not _model_ready or _pipeline is None:
        raise HTTPException(503, "Model not loaded yet")

    if not prompt.strip():
        raise HTTPException(422, "prompt must not be empty")

    files = (image or []) + (image_list or [])
    if not files:
        raise HTTPException(
            422, "Missing image file (send an 'image' part in the form body)"
        )

    image_bytes = await files[0].read()
    if not image_bytes:
        raise HTTPException(422, "Uploaded image is empty")

    data = []
    for i in range(n):
        if seed >= 0:
            seed_i = (seed + i) % (2**32)
        else:
            seed_i = (int(time.time() * 1000) + i) % (2**32)
        try:
            out_path, _elapsed, _iw, _ih, _w, _h = await run_in_threadpool(
                _run_img2img, prompt, image_bytes,
                _DEFAULT_STEPS, _DEFAULT_CFG_SCALE, seed_i,
            )
        except OSError:
            raise HTTPException(
                422, "Could not decode the uploaded image (expected PNG/JPEG)"
            )
        data.append(_openai_image_item(prompt, out_path, seed_i))

    return {"created": int(time.time()), "data": data}
