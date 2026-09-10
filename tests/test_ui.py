"""Optional UI construction check using the real installed Gradio package."""
import importlib.util
import unittest
from unittest.mock import patch

from test_launch import load_launcher


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Install Gradio to check UI construction")
class GradioSmokeTest(unittest.TestCase):
    def test_ui_builds_and_registers_serialized_events(self):
        import gradio
        app = load_launcher()
        app.gr = gradio
        app.torch.cuda.is_available = lambda: True
        with patch.object(gradio.Blocks, "launch", autospec=True) as launch:
            app.main()
        demo = launch.call_args.args[0]
        self.assertEqual(launch.call_args.kwargs["server_name"], "127.0.0.1")
        self.assertIn("theme", launch.call_args.kwargs)
        events = [fn for fn in demo.fns.values() if fn.fn in (app.generate, app.do_reload)]
        self.assertEqual(len(events), 3)
        self.assertTrue(all(fn.concurrency_id == "model" and fn.concurrency_limit == 1 for fn in events))
        self.assertIn("generate", [fn.api_name for fn in events])
        demo.close()


if __name__ == "__main__":
    unittest.main()
