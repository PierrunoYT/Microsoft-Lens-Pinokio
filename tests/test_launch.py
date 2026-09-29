"""Runtime regression tests without CUDA, model downloads, or upstream code."""
import importlib.util
from pathlib import Path
import sys
import threading
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def load_launcher():
    gradio = ModuleType("gradio")
    gradio.Progress = Mock(return_value=None)
    gradio.Error = type("Error", (Exception,), {})
    gradio.update = lambda **kwargs: kwargs
    torch = ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
    torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=Mock(),
                                 OutOfMemoryError=type("OutOfMemoryError", (Exception,), {}))
    torch.Generator = Mock(return_value=SimpleNamespace(manual_seed=lambda seed: seed))
    torch.inference_mode = nullcontext
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = Mock(return_value="cached-snapshot")
    lens = ModuleType("lens")
    lens.LensGptOssEncoder = Mock()
    lens.LensPipeline = Mock()
    resolution = ModuleType("lens.resolution")
    resolution.SUPPORTED_ASPECT_RATIOS = ["1:1"]
    resolution.SUPPORTED_BASE_RESOLUTIONS = [1024]
    modules = {"gradio": gradio, "torch": torch, "huggingface_hub": hub,
               "lens": lens, "lens.resolution": resolution}
    spec = importlib.util.spec_from_file_location("launcher_under_test", Path(__file__).parents[1] / "launch.py")
    module = importlib.util.module_from_spec(spec)
    # Importing the launcher should not require creating an actual app checkout.
    with patch.dict(sys.modules, modules), patch("os.chdir"), patch.object(sys, "path", sys.path.copy()):
        spec.loader.exec_module(module)
    return module


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.app = load_launcher()
        self.first, self.second = self.app.MODEL_CHOICES[:2]

    def populate(self, offload=False):
        app = self.app
        pipe = Mock()
        app._pipes[app.REPOS[self.first]] = pipe
        app._text_encoder = Mock()
        app._loaded_dtype = "bf16"
        app._loaded_mxfp4 = False
        app._loaded_cpu_offload = offload
        return pipe

    def generate(self, **changes):
        args = dict(prompt="a fox", model_name=self.first, base_resolution=1024,
                    aspect_ratio="1:1", steps=4, cfg=1.0, num_images=1, seed=42,
                    randomize_seed=False, cpu_offload=False, use_mxfp4=False,
                    dtype_label="bfloat16 (recommended)", enable_reasoner=False,
                    reasoner_url="", reasoner_key="", reasoner_model_id="")
        args.update(changes)
        return self.app.generate(**args)

    def test_same_settings_reuse_pipeline(self):
        old = self.populate()
        self.assertIs(self.app._get_pipe(self.first, "bf16", False, False), old)

    def test_offload_change_rebuilds_pipeline_and_encoder(self):
        old = self.populate()
        old_encoder = self.app._text_encoder
        with patch.object(self.app, "_get_text_encoder", return_value=Mock()) as encoder:
            pipe = self.app._get_pipe(self.first, "bf16", False, True)
        self.assertIsNot(pipe, old)
        self.assertIsNot(self.app._text_encoder, old_encoder)
        encoder.assert_called_once()
        pipe.enable_model_cpu_offload.assert_called_once()
        self.assertTrue(self.app._loaded_cpu_offload)

    def test_switching_model_releases_previous_pipeline(self):
        self.populate()
        with patch.object(self.app, "_get_text_encoder", return_value=Mock()):
            self.app._get_pipe(self.second, "bf16", False, False)
        self.assertEqual(list(self.app._pipes), [self.app.REPOS[self.second]])

    def test_snapshot_download_is_atomic_and_memoized(self):
        self.assertEqual(self.app._ensure_cached("model"), "cached-snapshot")
        self.app._ensure_cached("model")
        self.app.snapshot_download.assert_called_once_with(
            repo_id="model", ignore_patterns=self.app._IGNORE_PATTERNS)

    def test_failed_download_can_be_retried(self):
        self.app.snapshot_download.side_effect = [OSError("offline"), "retry"]
        with self.assertRaises(OSError):
            self.app._ensure_cached("model")
        self.assertNotIn("model", self.app._local_cache)
        self.assertEqual(self.app._ensure_cached("model"), "retry")

    def test_loading_oom_is_reported_and_state_cleared(self):
        self.populate()
        with patch.object(self.app, "_get_pipe", side_effect=self.app.torch.cuda.OutOfMemoryError):
            with self.assertRaisesRegex(self.app.gr.Error, "CUDA out of memory"):
                self.generate()
        self.assertFalse(self.app._pipes)
        self.assertIsNone(self.app._text_encoder)

    def test_inference_error_keeps_loaded_pipeline(self):
        pipe = self.populate()
        pipe.side_effect = RuntimeError("reasoner unavailable")
        pipe._execution_device = "cuda"
        with self.assertRaisesRegex(self.app.gr.Error, "Generation failed"):
            self.generate()
        self.assertIs(self.app._pipes[self.app.REPOS[self.first]], pipe)
        self.assertIsNotNone(self.app._text_encoder)

    def test_reload_failure_clears_partial_encoder(self):
        def fail(*args):
            self.app._text_encoder = Mock()
            raise RuntimeError("load failed")
        with patch.object(self.app, "_get_pipe", side_effect=fail):
            result = self.app.do_reload(self.first, "bfloat16 (recommended)", False, False)
        self.assertIn("Reload failed", result["value"])
        self.assertIsNone(self.app._text_encoder)

    def test_seed_and_inference_mode(self):
        pipe = Mock(return_value=SimpleNamespace(images=["image"]))
        pipe._execution_device = "cuda"
        with patch.object(self.app, "_get_pipe", return_value=pipe), \
             patch.object(self.app.torch, "inference_mode", return_value=nullcontext()) as mode:
            self.assertEqual(self.generate(), (["image"], 42))
        mode.assert_called_once()
        self.assertEqual(pipe.call_args.kwargs["generator"], 42)

    def test_empty_reasoner_url_rejected_before_loading(self):
        with patch.object(self.app, "_get_pipe") as load:
            with self.assertRaisesRegex(self.app.gr.Error, "API URL"):
                self.generate(enable_reasoner=True)
        load.assert_not_called()

    def test_reload_waits_for_generation_lock(self):
        started, finished = threading.Event(), threading.Event()
        def reload():
            started.set()
            self.app.do_reload(self.first, "bfloat16 (recommended)", False, False)
            finished.set()
        with patch.object(self.app, "_get_pipe"):
            with self.app._model_lock:
                worker = threading.Thread(target=reload)
                worker.start()
                self.assertTrue(started.wait(1))
                self.assertFalse(finished.wait(0.05))
            worker.join(2)
        self.assertTrue(finished.is_set())


if __name__ == "__main__":
    unittest.main()
