# Microsoft Lens — Pinokio Launcher

A 1-click Pinokio launcher for [Microsoft Lens](https://github.com/microsoft/Lens) — a 3.8B foundational text-to-image model with efficient training and fast high-resolution generation.

## What It Does

Lens generates images from text prompts using a compact MMDiT denoiser, FLUX.2 VAE latents, and GPT-OSS text features. This launcher provides a Gradio web UI with two checkpoints:

| Model | Steps | CFG | Notes |
|-------|-------|-----|-------|
| **Lens-Turbo** | 4 | 1.0 | Distilled, fast |
| **Lens** | 20 | 5.0 | RL-tuned, higher quality |

**Requirements:** NVIDIA GPU with CUDA, Python 3.12 (via Pinokio `ai` bundle), ~30 GB disk for model weights (downloaded on first run from Hugging Face).

> macOS is not supported. Lens needs CUDA on Windows or Linux.

## How to Use

1. Click **Install** to clone [microsoft/Lens](https://github.com/microsoft/Lens) and install dependencies.
2. Click **Start** to launch the Gradio web UI.
3. Click **Open Web UI** when the server is ready.
4. Enter a prompt and click **Generate**. Weights download automatically on the first run.

On GPUs under 20 GB VRAM, `launch.py` automatically enables CPU offload and dequantizes the text encoder on pre-Hopper cards. Set `LENS_LOW_VRAM=0` to force full-GPU mode, or `LENS_LOW_VRAM=1` to force low-VRAM mode.

## Models

Downloaded from Hugging Face on first generation:

- [microsoft/Lens-Turbo](https://huggingface.co/microsoft/Lens-Turbo) — default in the UI
- [microsoft/Lens](https://huggingface.co/microsoft/Lens) — optional, higher quality

The GPT-OSS text encoder and FLUX.2 VAE are pulled as dependencies of those repos. Some components may be gated — set `HF_TOKEN` in your environment if needed.

## CLI (advanced)

After install, you can also run the upstream CLI from the `app` folder:

```bash
python inference.py --repo_id microsoft/Lens-Turbo --prompt "a red fox in snow" --steps 4 --cfg 1.0 --out ./outputs
```

Add `--offload` and `--disable_mxfp4` for low-VRAM setups.

## License

Lens is released under the [MIT License](https://github.com/microsoft/Lens/blob/main/LICENSE). See the [model card](https://huggingface.co/microsoft/Lens) for responsible AI terms (research use).
