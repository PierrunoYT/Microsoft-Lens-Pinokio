module.exports = {
  daemon: true,
  run: [
    {
      when: "{{platform === 'darwin'}}",
      method: "notify",
      params: {
        html: "macOS is not supported. Lens requires Windows or Linux with an NVIDIA GPU."
      },
      next: null
    },
    {
      when: "{{!gpus.includes('nvidia')}}",
      method: "notify",
      params: { html: "Lens requires an NVIDIA GPU with CUDA." },
      next: null
    },
    {
      when: "{{!exists('app/.installed') || !exists('app/env')}}",
      method: "notify",
      params: { html: "Run Install successfully before starting Lens." },
      next: null
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          GRADIO_SERVER_PORT: "{{port}}"
        },
        path: "app",
        message: ["python ../launch.py"],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
