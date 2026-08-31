"""Behavioral contracts for interactive and developer-shell commands."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from app import shell  # noqa: E402


class FakeRuntime:
    def __init__(self):
        self.events = []

    def reset_context(self):
        self.events.append(("reset",))

    def load_checkpoint(self, path):
        self.events.append(("load", path))

    def interactive_predict(self, text, *, top_k, reset_context):
        self.events.append(("predict", text, top_k, reset_context))
        return [("NEXT", 0.75)]

    def generate(self, text, *, max_tokens, top_k, reset_context, verbose):
        self.events.append(
            ("generate", text, max_tokens, top_k, reset_context, verbose)
        )
        return ["A", "B"]

    def decode_tokens(self, tokens):
        return " ".join(tokens)

    def runtime_status_snapshot(self):
        self.events.append(("status",))
        return {
            "training": {
                "training_mode": "single_pass",
                "token_duration": 0.02,
                "token_duration_source": "default",
            },
            "invocations": {
                "present_calls": 3,
                "reset_context_calls": 1,
                "sim_run_count": 4,
                "simulated_seconds": 0.08,
                "sim_run_seconds": 0.5,
            },
            "step_time": 0.001,
            "compile_fingerprint": {},
        }


class ShellCommandTests(unittest.TestCase):
    def handle(self, runtime, command, argument="", *, mode="shell"):
        return shell._handle_slash_command(
            runtime,
            command,
            argument,
            mode=mode,
            checkpoint_path="test.pkl",
            testing_set=[["A", "B"]],
            top_k=3,
            max_tokens=7,
            max_examples=5,
            max_demo_examples=2,
        )

    def test_exit_aliases_stop_the_prompt_loop_without_runtime_work(self):
        for command in ("/exit", "/quit"):
            with self.subTest(command=command):
                runtime = FakeRuntime()
                self.assertFalse(self.handle(runtime, command))
                self.assertEqual(runtime.events, [])

    def test_reset_help_and_unknown_commands_preserve_the_continue_contract(self):
        runtime = FakeRuntime()

        with redirect_stdout(io.StringIO()) as stdout:
            self.assertTrue(self.handle(runtime, "/reset"))
            self.assertTrue(self.handle(runtime, "/help"))
            self.assertTrue(self.handle(runtime, "/unknown"))

        output = stdout.getvalue()
        self.assertEqual(runtime.events, [("reset",)])
        self.assertIn("[context reset]", output)
        self.assertIn("Developer shell commands", output)
        self.assertIn("Unknown command: /unknown", output)

    def test_interactive_mode_rejects_shell_only_commands(self):
        runtime = FakeRuntime()

        with redirect_stdout(io.StringIO()) as stdout:
            self.assertTrue(self.handle(runtime, "/help", mode="interactive"))
            self.assertTrue(self.handle(runtime, "/status", mode="interactive"))

        output = stdout.getvalue()
        self.assertIn("/reset  - clear context memory", output)
        self.assertIn("Unknown command: /status", output)
        self.assertEqual(runtime.events, [])

    def test_predict_and_generate_validate_arguments_and_preserve_context(self):
        runtime = FakeRuntime()

        with redirect_stdout(io.StringIO()) as stdout:
            self.assertTrue(self.handle(runtime, "/predict"))
            self.assertTrue(self.handle(runtime, "/generate"))
            self.assertTrue(self.handle(runtime, "/predict", "hello"))
            self.assertTrue(self.handle(runtime, "/generate", "seed"))

        output = stdout.getvalue()
        self.assertIn("Usage: /predict <text>", output)
        self.assertIn("Usage: /generate <text>", output)
        self.assertIn("NEXT (0.750)", output)
        self.assertIn("generated:\nA B", output)
        self.assertEqual(
            runtime.events,
            [
                ("predict", "hello", 3, False),
                ("generate", "seed", 7, 3, False, False),
            ],
        )

    def test_status_eval_demo_and_load_apply_documented_state_effects(self):
        runtime = FakeRuntime()

        with (
            patch.object(shell, "evaluate_model", return_value={"accuracy": 0.5}),
            patch.object(shell, "run_demo_predictions") as demo,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertTrue(self.handle(runtime, "/status"))
            self.assertTrue(self.handle(runtime, "/eval"))
            self.assertTrue(self.handle(runtime, "/demo"))
            self.assertTrue(self.handle(runtime, "/load"))

        output = stdout.getvalue()
        self.assertIn("checkpoint:              test.pkl", output)
        self.assertIn("present calls:           3", output)
        self.assertIn("[evaluation complete; context reset]", output)
        self.assertIn("[demo complete; context reset]", output)
        self.assertIn("[checkpoint reloaded; context reset]", output)
        demo.assert_called_once_with(
            runtime,
            [["A", "B"]],
            max_examples=2,
            top_k=3,
        )
        self.assertEqual(
            runtime.events,
            [
                ("status",),
                ("reset",),
                ("reset",),
                ("load", "test.pkl"),
                ("reset",),
            ],
        )


if __name__ == "__main__":
    unittest.main()
