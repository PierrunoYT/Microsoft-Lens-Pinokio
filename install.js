module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      when: "{{platform === 'darwin'}}",
      method: "notify",
      params: {
        html: "macOS is not supported. Lens requires Windows or Linux with an NVIDIA GPU and CUDA."
      },
      next: null
    },
    {
      when: "{{!gpus.includes('nvidia')}}",
      method: "notify",
      params: {
        html: "Lens requires an NVIDIA GPU with CUDA. AMD and CPU-only setups are not supported."
      },
      next: null
    },
    {
      when: "{{exists('app/.installed')}}",
      method: "fs.rm",
      params: { path: "app/.installed" }
    },
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: "git clone https://github.com/microsoft/Lens app"
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r ../requirements.txt",
          "uv pip check",
          "python -c \"import torch, gradio; from lens import LensGptOssEncoder, LensPipeline\""
        ]
      }
    },
    {
      method: "fs.write",
      params: { path: "app/.installed", json: { ready: true } }
    },
    {
      method: "notify",
      params: { html: "Installation finished! Click <strong>Start</strong> to launch the Lens web UI. Models download from Hugging Face on first generation (~30 GB)." }
    }
  ]
}
