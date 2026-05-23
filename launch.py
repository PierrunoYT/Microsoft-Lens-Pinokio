"""Pinokio launcher for Microsoft Lens and Lens-Turbo text-to-image."""

from __future__ import annotations

import gc
import os
import random
import sys
import time

import gradio as gr
import torch
from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download, try_to_load_from_cache
from huggingface_hub import utils as hf_utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "app")
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from lens import LensGptOssEncoder, LensPipeline  # noqa: E402
from lens.resolution import SUPPORTED_ASPECT_RATIOS, SUPPORTED_BASE_RESOLUTIONS  # noqa: E402

DTYPE = torch.bfloat16
TURBO_REPO = os.environ.get("LENS_TURBO_REPO", "microsoft/Lens-Turbo")
LENS_REPO = os.environ.get("LENS_REPO", "microsoft/Lens")
MAX_SEED = 2**31 - 1

MODEL_CHOICES = ["Lens-Turbo (4 steps)", "Lens (20 steps, RL)"]


def model_defaults(model_name: str) -> tuple[int, float]:
    if "Turbo" in model_name:
        return 4, 1.0
    return 20, 5.0


def _low_vram_enabled() -> bool:
    override = os.environ.get("LENS_LOW_VRAM", "").strip().lower()
    if override in ("1", "true", "yes"):
        return True
    if override in ("0", "false", "no"):
        return False
    if not torch.cuda.is_available():
        return True
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return vram_gb < 20
    except Exception:
        return False


LOW_VRAM = _low_vram_enabled()

_text_encoder = None
_pipes: dict[str, LensPipeline] = {}
_local_cache: dict[str, str] = {}

_SKIP_PREFIXES = ("assets/", "README", ".git")


def _ensure_cached(repo: str) -> str:
    if repo not in _local_cache:
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


def _should_disable_mxfp4() -> bool:
    if os.environ.get("LENS_DISABLE_MXFP4", "").strip() in ("1", "true", "yes"):
        return True
    if LOW_VRAM:
        return True
    if not torch.cuda.is_available():
        return True
    try:
        major, _minor = torch.cuda.get_device_capability()
        return major < 9
    except Exception:
        return True


def _get_text_encoder():
    global _text_encoder
    if _text_encoder is not None:
        return _text_encoder

    kwargs: dict = {"subfolder": "text_encoder", "dtype": DTYPE}
    disable_mxfp4 = _should_disable_mxfp4()
    try:
        from transformers import Mxfp4Config
        kwargs["quantization_config"] = Mxfp4Config(dequantize=disable_mxfp4)
    except ImportError:
        pass

    repo = TURBO_REPO if LOW_VRAM else LENS_REPO
    local_path = _ensure_cached(repo)
    print(f"Loading GPT-OSS text encoder from cache (disable_mxfp4={disable_mxfp4})...", flush=True)
    _text_encoder = LensGptOssEncoder.from_pretrained(local_path, disable_mmap=True, **kwargs)
    return _text_encoder


def _unload_pipes(keep_repo: str | None = None) -> None:
    for repo in list(_pipes.keys()):
        if keep_repo is not None and repo == keep_repo:
            continue
        del _pipes[repo]
    if keep_repo is None:
        _pipes.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _get_pipe(model_name: str) -> LensPipeline:
    repo = TURBO_REPO if "Turbo" in model_name else LENS_REPO
    if repo in _pipes:
        return _pipes[repo]

    if LOW_VRAM:
        _unload_pipes()

    local_path = _ensure_cached(repo)
    print(f"Loading {model_name} from cache...", flush=True)
    pipe = LensPipeline.from_pretrained(
        local_path,
        text_encoder=_get_text_encoder(),
        torch_dtype=DTYPE,
        disable_mmap=True,
    )
    if LOW_VRAM or os.environ.get("LENS_OFFLOAD", "").strip() in ("1", "true", "yes"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    _pipes[repo] = pipe
    return pipe


def generate(
    prompt: str,
    model_name: str = MODEL_CHOICES[0],
    base_resolution: int = 1024,
    aspect_ratio: str = "1:1",
    steps: int | None = None,
    cfg: float | None = None,
    seed: int = 0,
    randomize_seed: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    pipe = _get_pipe(model_name)
    default_steps, default_cfg = model_defaults(model_name)
    steps = default_steps if steps is None else int(steps)
    cfg = default_cfg if cfg is None else float(cfg)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)

    out = pipe(
        prompt=prompt.strip(),
        base_resolution=int(base_resolution),
        aspect_ratio=aspect_ratio,
        num_inference_steps=steps,
        guidance_scale=cfg,
        num_images_per_prompt=1,
        generator=generator,
    )
    return out.images[0], seed


CSS = """
#col-container { max-width: 1100px; margin: 0 auto; }
"""


def main() -> None:
    if not torch.cuda.is_available():
        sys.stderr.write(
            "\nLens requires an NVIDIA GPU with CUDA.\n"
            "Install finished, but no CUDA device was detected.\n\n"
        )
        sys.exit(1)

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    mode = "Low VRAM (CPU offload)" if LOW_VRAM else "Standard"
    print(f"Lens Pinokio launcher — {mode}", flush=True)

    with gr.Blocks(theme=gr.themes.Citrus(), css=CSS, title="Microsoft Lens") as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown(
                f"""
                # Microsoft Lens
                3.8B foundational text-to-image model by Microsoft. **{mode}** — switch between
                **Lens-Turbo** (4-step distilled, fast) and **Lens** (20-step RL-tuned, higher quality).

                [Paper](https://arxiv.org/abs/2605.21573) · [Code](https://github.com/microsoft/Lens) · [Lens](https://huggingface.co/microsoft/Lens) · [Lens-Turbo](https://huggingface.co/microsoft/Lens-Turbo)
                """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A cinematic mountain lake at sunrise, soft golden light, mist rising off the water",
                        lines=3,
                    )
                    with gr.Row():
                        model = gr.Radio(
                            choices=MODEL_CHOICES,
                            value=MODEL_CHOICES[0],
                            label="Model",
                        )
                    run_btn = gr.Button("Generate", variant="primary")

                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            base_res = gr.Radio(
                                choices=list(SUPPORTED_BASE_RESOLUTIONS),
                                value=1024,
                                label="Base resolution",
                            )
                            aspect = gr.Dropdown(
                                choices=list(SUPPORTED_ASPECT_RATIOS),
                                value="1:1",
                                label="Aspect ratio (W:H)",
                            )
                        with gr.Row():
                            steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                            cfg = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="Guidance scale")
                        with gr.Row():
                            seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                            randomize = gr.Checkbox(value=True, label="Randomize seed")

                with gr.Column(scale=4):
                    image = gr.Image(label="Output", type="pil", height=640)
                    used_seed = gr.Number(label="Seed used", interactive=False)

            gr.Examples(
                examples=[
                    ["A generous portion of classic British fish and chips on white paper, golden crispy beer-battered cod, thick-cut chips, lemon wedge, mushy peas, wooden pub table, overhead shot", MODEL_CHOICES[0]],
                    ["A crystal dragon soaring through an aurora borealis sky, transparent faceted body refracting green and purple light, ice trail from its wings, high fantasy digital art", MODEL_CHOICES[0]],
                    ["Aerial view of Yuanyang rice terraces at sunrise, cascading water-filled paddies reflecting pink sky, morning mist between layers, drone photography", MODEL_CHOICES[1]],
                    ["A green iguana basking on a moss-covered log in a tropical rainforest, every scale rendered sharply, dewdrops on its skin, National Geographic style", MODEL_CHOICES[1]],
                ],
                inputs=[prompt, model],
                outputs=[image, used_seed],
                fn=generate,
                cache_examples=False,
            )

        def _sync_defaults(model_name):
            s, g = model_defaults(model_name)
            return gr.update(value=s), gr.update(value=g)

        model.change(_sync_defaults, inputs=model, outputs=[steps, cfg])

        inputs = [prompt, model, base_res, aspect, steps, cfg, seed, randomize]
        outputs = [image, used_seed]
        run_btn.click(generate, inputs=inputs, outputs=outputs)
        prompt.submit(generate, inputs=inputs, outputs=outputs)

    demo.queue(max_size=4 if LOW_VRAM else 8).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
