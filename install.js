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
      when: "{{gpu !== 'nvidia'}}",
      method: "notify",
      params: {
        html: "Lens requires an NVIDIA GPU with CUDA. AMD and CPU-only setups are not supported."
      },
      next: null
    },
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: "git clone https://github.com/microsoft/Lens app"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["uv pip install -r ../requirements.txt"]
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
      method: "notify",
      params: {
        html: "Installation finished! Click <strong>Start</strong> to launch the Lens web UI. Models download from Hugging Face on first generation (~30 GB)."
      }
    }
  ]
}
