# Microsoft Lens — Pinokio Launcher

A 1-click Pinokio launcher for [Microsoft Lens](https://github.com/microsoft/Lens) — a 3.8B foundational text-to-image model with efficient training and fast high-resolution generation.

## What It Does

Lens generates images from text prompts using a compact MMDiT denoiser, FLUX.2 VAE latents, and GPT-OSS text features. This launcher provides a Gradio web UI with three checkpoints:

| Model | Steps | CFG | Notes |
|-------|-------|-----|-------|
| **Lens-Turbo** | 4 | 1.0 | Distilled, fast |
| **Lens** | 20 | 5.0 | RL-tuned, higher quality |
| **Lens-Base** | 50 | 5.0 | Baseline |

**Requirements:** NVIDIA GPU with CUDA, Python 3.12 (via Pinokio `ai` bundle), ~30 GB disk for model weights (downloaded on first run from Hugging Face).

> macOS is not supported. Lens needs CUDA on Windows or Linux.

## How to Use

1. Click **Install** to clone [microsoft/Lens](https://github.com/microsoft/Lens) and install dependencies.
2. Click **Start** to launch the Gradio web UI.
3. Click **Open Web UI** when the server is ready.
4. Enter a prompt and click **Generate**. Weights download automatically on the first run.

On GPUs under 48 GB VRAM, `launch.py` automatically enables CPU offload. Set `LENS_LOW_VRAM=0` to default to full-GPU mode, or `LENS_LOW_VRAM=1` to default to CPU offload. The Hardware controls override these defaults and apply on the next generation. Offload still requires enough RAM for the model and enough VRAM for each active component.

The text encoder is dequantized by default. MXFP4 is an advanced opt-in requiring a compatible GPU and PyTorch/Triton kernels; GPU generation alone does not guarantee support. The launcher does not install that optional kernel stack.

Only one model pipeline is retained at a time. Generation and reload requests share a queue so they cannot change model state concurrently.

After updating from an older launcher, run **Install** once if prompted. The launcher only shows **Start** after dependency and import checks succeed. **Reset** removes the entire `app` folder, including any outputs saved there; copy those outputs elsewhere first. Hugging Face's external cache is not removed.

**Upstream availability:** On 2026-09-10, the configured `microsoft/Lens` GitHub repository returned “Repository not found.” Fresh installs require that repository to be accessible. An existing checkout can still be used, but upstream updates and a complete fresh-install test are blocked until access is restored.

## Models

Downloaded from Hugging Face on first generation:

- [microsoft/Lens-Turbo](https://huggingface.co/microsoft/Lens-Turbo) — default in the UI
- [microsoft/Lens](https://huggingface.co/microsoft/Lens) — optional, higher quality
- [microsoft/Lens-Base](https://huggingface.co/microsoft/Lens-Base) — optional baseline

The GPT-OSS text encoder and FLUX.2 VAE are pulled as dependencies of those repos. Some components may be gated — set `HF_TOKEN` in your environment if needed.

## CLI (advanced)

After install, you can also run the upstream CLI from the `app` folder:

```bash
python inference.py --repo_id microsoft/Lens-Turbo --prompt "a red fox in snow" --steps 4 --cfg 1.0 --out ./outputs
```

Add `--offload` and `--disable_mxfp4` for low-VRAM setups.

## Web API

Use the local URL printed by **Start**; the examples below use port 7860. The named Gradio endpoint is `/generate`. Arguments, in order, are prompt, model, base resolution, aspect ratio, steps, guidance scale, image count, seed, randomize seed, CPU offload, MXFP4, precision, reasoner enabled, reasoner URL, reasoner key, and reasoner model ID. The result contains gallery images and the seed used. API calls require the same model downloads and hardware as the UI.

Python (`pip install gradio_client`):

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
images, seed = client.predict(
    "a red fox in snow", "Lens-Turbo (4 steps, fast)", 1024, "1:1",
    4, 1.0, 1, 42, False, True, False, "bfloat16 (recommended)",
    False, "", "", "", api_name="/generate",
)
```

JavaScript (`npm install @gradio/client`):

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");
const result = await client.predict("/generate", [
  "a red fox in snow", "Lens-Turbo (4 steps, fast)", 1024, "1:1",
  4, 1.0, 1, 42, false, true, false, "bfloat16 (recommended)",
  false, "", "", ""
]);
console.log(result.data);
```

Curl (POSIX shell; on Windows, use `curl.exe` with equivalent quoting):

```bash
curl -X POST http://127.0.0.1:7860/gradio_api/call/generate \
  -H 'Content-Type: application/json' \
  -d '{"data":["a red fox in snow","Lens-Turbo (4 steps, fast)",1024,"1:1",4,1.0,1,42,false,true,false,"bfloat16 (recommended)",false,"","",""]}'
# Replace EVENT_ID with the event_id returned above:
curl -N http://127.0.0.1:7860/gradio_api/call/generate/EVENT_ID
```

## Validation

```bash
python -m unittest discover -s tests -v
node --test tests/launcher.test.js
```

The Python regressions use mocked model dependencies; they do not download weights. With Gradio installed, an additional test builds the real UI and checks its event configuration. Full CUDA inference and installation require access to the upstream repository and models.

## License

Lens is released under the [MIT License](https://github.com/microsoft/Lens/blob/main/LICENSE). See the [model card](https://huggingface.co/microsoft/Lens) for responsible AI terms (research use).
