module.exports = {
  run: [
    // Lens only supports NVIDIA CUDA on Windows and Linux.
    {
      when: "{{!gpus.includes('nvidia') || (platform !== 'win32' && platform !== 'linux')}}",
      method: "notify",
      params: { html: "Lens requires Windows or Linux with an NVIDIA GPU and CUDA." },
      next: null
    },
    // nvidia windows
    {
      when: "{{gpus.includes('nvidia') && platform === 'win32'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio"
        ]
      },
      next: null
    },
    // nvidia linux
    {
      when: "{{gpus.includes('nvidia') && platform === 'linux'}}",
      method: "shell.run",
      params: {
        bluefairy: "off",
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio"
        ]
      },
      next: null
    }
  ]
}
