"""Pinokio launcher for Microsoft Lens — all configurable settings exposed in UI."""

from __future__ import annotations

import gc
import os
import random
import sys
import time

import gradio as gr
import torch
from huggingface_hub import (
    list_repo_files,
    hf_hub_download,
    snapshot_download,
    try_to_load_from_cache,
)
from huggingface_hub import utils as hf_utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "app")
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from lens import LensGptOssEncoder, LensPipeline  # noqa: E402
from lens.resolution import SUPPORTED_ASPECT_RATIOS, SUPPORTED_BASE_RESOLUTIONS  # noqa: E402

# ── Model registry ────────────────────────────────────────────────────────────

REPOS = {
    "Lens-Turbo (4 steps, fast)":     os.environ.get("LENS_TURBO_REPO", "microsoft/Lens-Turbo"),
    "Lens (20 steps, RL quality)":    os.environ.get("LENS_REPO",       "microsoft/Lens"),
    "Lens-Base (50 steps, baseline)": os.environ.get("LENS_BASE_REPO",  "microsoft/Lens-Base"),
}
MODEL_CHOICES = list(REPOS.keys())

MODEL_DEFAULTS: dict[str, tuple[int, float]] = {
    "Lens-Turbo (4 steps, fast)":     (4,  1.0),
    "Lens (20 steps, RL quality)":    (20, 5.0),
    "Lens-Base (50 steps, baseline)": (50, 5.0),
}

DTYPE_MAP = {
    "bfloat16 (recommended)": torch.bfloat16,
    "float16":                 torch.float16,
    "float32 (slowest)":       torch.float32,
}

MAX_SEED = 2**31 - 1
_SKIP_PREFIXES = ("assets/", "README", ".git")

# ── Hardware detection ────────────────────────────────────────────────────────

def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return 0.0

def _default_cpu_offload() -> bool:
    v = os.environ.get("LENS_LOW_VRAM", "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    # Text encoder ~13 GB + transformer ~16 GB; need headroom for activations
    return _vram_gb() < 48

def _default_mxfp4() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 9  # Hopper+ (H100/H200) natively supports MXFP4
    except Exception:
        return False

VRAM = _vram_gb()
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected"

# ── Runtime state ─────────────────────────────────────────────────────────────

_text_encoder: LensGptOssEncoder | None = None
_pipes: dict[str, LensPipeline] = {}
_local_cache: dict[str, str] = {}
_loaded_dtype: torch.dtype | None = None
_loaded_mxfp4: bool | None = None

# ── Download ──────────────────────────────────────────────────────────────────

def _ensure_cached(repo: str) -> str:
    if repo in _local_cache:
        return _local_cache[repo]
    hf_utils.disable_progress_bars()
    try:
        all_files = [
            f for f in list_repo_files(repo_id=repo)
            if not f.startswith(_SKIP_PREFIXES)
        ]
        to_download = [
            f for f in all_files
            if try_to_load_from_cache(repo_id=repo, filename=f) is None
        ]
        if to_download:
            print(f"\nDownloading {len(to_download)} file(s) from {repo}:", flush=True)
            for i, filename in enumerate(to_download, 1):
                print(f"  [{i}/{len(to_download)}] {filename}", flush=True)
                t0 = time.time()
                local_file = hf_hub_download(repo_id=repo, filename=filename)
                elapsed = time.time() - t0
                size_mb = os.path.getsize(local_file) / (1024 ** 2)
                print(f"         -> {size_mb:.0f} MB  {size_mb / elapsed:.1f} MB/s", flush=True)
        else:
            print(f"\n{repo} already cached.", flush=True)
        _local_cache[repo] = snapshot_download(repo_id=repo, local_files_only=True)
    finally:
        hf_utils.enable_progress_bars()
    return _local_cache[repo]

# ── Model loading ─────────────────────────────────────────────────────────────

def _unload_all() -> None:
    global _text_encoder, _loaded_dtype, _loaded_mxfp4
    _text_encoder = None
    _loaded_dtype = None
    _loaded_mxfp4 = None
    _pipes.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Model unloaded.", flush=True)


def _get_text_encoder(dtype: torch.dtype, use_mxfp4: bool) -> LensGptOssEncoder:
    global _text_encoder, _loaded_dtype, _loaded_mxfp4
    if _text_encoder is not None:
        return _text_encoder

    kwargs: dict = {"subfolder": "text_encoder", "dtype": dtype}
    try:
        from transformers import Mxfp4Config
        # dequantize=True on pre-Hopper GPUs that lack native MXFP4 kernels
        kwargs["quantization_config"] = Mxfp4Config(dequantize=not use_mxfp4)
    except ImportError:
        pass

    # Both repos share the same text encoder; use Turbo (smaller download)
    repo = REPOS["Lens-Turbo (4 steps, fast)"]
    local_path = _ensure_cached(repo)
    mxfp4_label = "native" if use_mxfp4 else "dequantized"
    print(f"Loading text encoder [{dtype} / mxfp4={mxfp4_label}]...", flush=True)
    _text_encoder = LensGptOssEncoder.from_pretrained(local_path, disable_mmap=True, **kwargs)
    _loaded_dtype = dtype
    _loaded_mxfp4 = use_mxfp4
    return _text_encoder


def _get_pipe(model_name: str, dtype: torch.dtype, use_mxfp4: bool, cpu_offload: bool) -> LensPipeline:
    repo = REPOS[model_name]

    if repo in _pipes:
        return _pipes[repo]

    # In CPU-offload mode keep only one pipeline in RAM at a time
    if cpu_offload and _pipes:
        for key in list(_pipes.keys()):
            del _pipes[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    enc = _get_text_encoder(dtype, use_mxfp4)
    local_path = _ensure_cached(repo)
    print(f"Loading {model_name}...", flush=True)

    pipe = LensPipeline.from_pretrained(
        local_path,
        text_encoder=enc,
        torch_dtype=dtype,
        disable_mmap=True,
    )

    if cpu_offload:
        pipe.enable_model_cpu_offload()
        print("CPU offload enabled.", flush=True)
    else:
        pipe.to("cuda")

    _pipes[repo] = pipe
    return pipe

# ── Inference ─────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    model_name: str,
    base_resolution: int,
    aspect_ratio: str,
    steps: int,
    cfg: float,
    num_images: int,
    seed: int,
    randomize_seed: bool,
    cpu_offload: bool,
    use_mxfp4: bool,
    dtype_label: str,
    enable_reasoner: bool,
    reasoner_url: str,
    reasoner_key: str,
    reasoner_model_id: str,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    dtype = DTYPE_MAP.get(dtype_label, torch.bfloat16)

    # Reload text encoder if dtype or mxfp4 changed
    if _loaded_dtype is not None and (_loaded_dtype != dtype or _loaded_mxfp4 != use_mxfp4):
        _unload_all()

    pipe = _get_pipe(model_name, dtype, use_mxfp4, cpu_offload)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)

    call_kwargs: dict = dict(
        prompt=prompt.strip(),
        base_resolution=int(base_resolution),
        aspect_ratio=aspect_ratio,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        num_images_per_prompt=int(num_images),
        generator=generator,
    )

    if enable_reasoner and reasoner_url.strip():
        call_kwargs["enable_reasoner"] = True
        call_kwargs["api_url"] = reasoner_url.strip()
        if reasoner_key.strip():
            call_kwargs["api_key"] = reasoner_key.strip()
        if reasoner_model_id.strip():
            call_kwargs["api_model"] = reasoner_model_id.strip()

    out = pipe(**call_kwargs)
    return out.images, seed


def do_reload(model_name, dtype_label, use_mxfp4, cpu_offload):
    _unload_all()
    dtype = DTYPE_MAP.get(dtype_label, torch.bfloat16)
    _get_pipe(model_name, dtype, use_mxfp4, cpu_offload)
    return gr.update(value="Model reloaded successfully.", visible=True)

# ── UI ────────────────────────────────────────────────────────────────────────

CSS = "#col-container { max-width: 1140px; margin: 0 auto; }"


def main() -> None:
    if not torch.cuda.is_available():
        sys.stderr.write("\nLens requires an NVIDIA GPU with CUDA.\n\n")
        sys.exit(1)

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    print(f"GPU: {GPU_NAME} ({VRAM:.1f} GB) — CPU offload default: {_default_cpu_offload()}", flush=True)

    with gr.Blocks(theme=gr.themes.Citrus(), css=CSS, title="Microsoft Lens") as demo:
        with gr.Column(elem_id="col-container"):

            gr.Markdown(f"""
# Microsoft Lens
3.8B foundational text-to-image model by Microsoft · **{GPU_NAME}** ({VRAM:.1f} GB VRAM)

[Paper](https://arxiv.org/abs/2605.21573) · [Code](https://github.com/microsoft/Lens) · [Lens](https://huggingface.co/microsoft/Lens) · [Lens-Turbo](https://huggingface.co/microsoft/Lens-Turbo)
""")

            with gr.Row():

                # ── Left: controls ────────────────────────────────────────
                with gr.Column(scale=3):

                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A cinematic mountain lake at sunrise, soft golden light, mist rising off the water",
                        lines=3,
                    )
                    model = gr.Radio(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")
                    run_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Accordion("Generation", open=True):
                        with gr.Row():
                            base_res = gr.Radio(
                                choices=list(SUPPORTED_BASE_RESOLUTIONS),
                                value=1024,
                                label="Base resolution",
                            )
                            aspect = gr.Dropdown(
                                choices=list(SUPPORTED_ASPECT_RATIOS),
                                value="1:1",
                                label="Aspect ratio",
                            )
                        with gr.Row():
                            steps = gr.Slider(1, 60, value=4, step=1, label="Steps")
                            cfg = gr.Slider(0.0, 15.0, value=1.0, step=0.1, label="Guidance scale")
                        with gr.Row():
                            num_images = gr.Slider(1, 4, value=1, step=1, label="Images per prompt")
                        with gr.Row():
                            seed_sl = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                            randomize = gr.Checkbox(value=True, label="Randomize seed")

                    with gr.Accordion("Prompt Reasoner", open=False):
                        enable_reasoner = gr.Checkbox(
                            value=False,
                            label="Enable prompt reasoner",
                            info="Refines your prompt via an LLM before generation.",
                        )
                        reasoner_url = gr.Textbox(
                            label="API URL",
                            placeholder="https://api.openai.com/v1",
                            visible=False,
                        )
                        reasoner_key = gr.Textbox(
                            label="API key",
                            type="password",
                            visible=False,
                        )
                        reasoner_model_id = gr.Textbox(
                            label="Model ID",
                            placeholder="gpt-4o-mini",
                            visible=False,
                        )

                        def _toggle_reasoner(enabled):
                            return [gr.update(visible=enabled)] * 3

                        enable_reasoner.change(
                            _toggle_reasoner,
                            inputs=enable_reasoner,
                            outputs=[reasoner_url, reasoner_key, reasoner_model_id],
                        )

                    with gr.Accordion("Hardware", open=False):
                        gr.Markdown(
                            f"**{GPU_NAME}** · {VRAM:.1f} GB VRAM  \n"
                            "Changes here take effect after clicking **Reload model**."
                        )
                        cpu_offload = gr.Checkbox(
                            value=_default_cpu_offload(),
                            label="CPU offload",
                            info="Moves model layers to RAM between steps. Slower but fits any VRAM.",
                        )
                        use_mxfp4 = gr.Checkbox(
                            value=_default_mxfp4(),
                            label="MXFP4 quantization (text encoder)",
                            info="Native on Hopper+ (H100/H200). Saves ~10 GB. May reduce quality on older GPUs.",
                        )
                        dtype_sel = gr.Dropdown(
                            choices=list(DTYPE_MAP.keys()),
                            value="bfloat16 (recommended)",
                            label="Precision (dtype)",
                        )
                        with gr.Row():
                            reload_btn = gr.Button("Reload model", variant="secondary")
                        reload_status = gr.Textbox(
                            label="",
                            interactive=False,
                            visible=False,
                            show_label=False,
                        )

                # ── Right: output ─────────────────────────────────────────
                with gr.Column(scale=4):
                    gallery = gr.Gallery(
                        label="Output",
                        type="pil",
                        height=660,
                        columns=2,
                        object_fit="contain",
                    )
                    used_seed = gr.Number(label="Seed used", interactive=False)

            gr.Examples(
                examples=[
                    ["A generous portion of classic British fish and chips on white paper, golden crispy beer-battered cod, thick-cut chips, lemon wedge, mushy peas, wooden pub table, overhead shot", MODEL_CHOICES[0]],
                    ["A crystal dragon soaring through an aurora borealis sky, transparent faceted body refracting green and purple light, ice trail from its wings", MODEL_CHOICES[0]],
                    ["Aerial view of Yuanyang rice terraces at sunrise, cascading water-filled paddies reflecting pink sky, morning mist between layers, drone photography", MODEL_CHOICES[1]],
                    ["A green iguana basking on a moss-covered log in a tropical rainforest, every scale sharp, dewdrops on its skin, National Geographic style", MODEL_CHOICES[1]],
                ],
                inputs=[prompt, model],
            )

        # ── Wiring ────────────────────────────────────────────────────────────

        def _sync_defaults(m):
            s, g = MODEL_DEFAULTS[m]
            return gr.update(value=s), gr.update(value=g)

        model.change(_sync_defaults, inputs=model, outputs=[steps, cfg])

        gen_inputs = [
            prompt, model, base_res, aspect, steps, cfg, num_images,
            seed_sl, randomize, cpu_offload, use_mxfp4, dtype_sel,
            enable_reasoner, reasoner_url, reasoner_key, reasoner_model_id,
        ]
        gen_outputs = [gallery, used_seed]

        run_btn.click(generate, inputs=gen_inputs, outputs=gen_outputs)
        prompt.submit(generate, inputs=gen_inputs, outputs=gen_outputs)

        reload_btn.click(
            do_reload,
            inputs=[model, dtype_sel, use_mxfp4, cpu_offload],
            outputs=reload_status,
        )

    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
