"""Independent public CLI and workflow-resolution contract tests."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from app.args import parse_args, resolve_workflow  # noqa: E402


class CliWorkflowContractTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(sys, "argv", ["model/main.py", *arguments]):
            return parse_args()

    def assert_parse_exits_two(self, *arguments):
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stderr(io.StringIO()):
                self.parse(*arguments)
        self.assertEqual(raised.exception.code, 2)

    def test_no_stage_flags_resolves_to_train_only(self):
        self.assertEqual(
            resolve_workflow(self.parse()),
            {
                "train": True,
                "eval": False,
                "demo": False,
                "interactive": False,
                "shell": False,
            },
        )

    def test_full_selects_four_stages(self):
        self.assertEqual(
            resolve_workflow(self.parse("--full")),
            {
                "train": True,
                "eval": True,
                "demo": True,
                "interactive": True,
                "shell": False,
            },
        )

    def test_no_flags_suppress_their_full_workflow_stages(self):
        self.assertEqual(
            resolve_workflow(
                self.parse(
                    "--full",
                    "--no-eval",
                    "--no-demo",
                    "--no-interactive",
                )
            ),
            {
                "train": True,
                "eval": False,
                "demo": False,
                "interactive": False,
                "shell": False,
            },
        )

    def test_each_explicit_individual_stage_does_not_imply_other_stages(self):
        stage_flags = {
            "train": "--train",
            "eval": "--eval",
            "demo": "--demo",
            "interactive": "--interactive",
            "shell": "--shell",
        }
        for selected_stage, flag in stage_flags.items():
            with self.subTest(stage=selected_stage):
                expected = {stage: False for stage in stage_flags}
                expected[selected_stage] = True
                self.assertEqual(resolve_workflow(self.parse(flag)), expected)

    def test_incompatible_shell_and_interactive_combinations_exit_two(self):
        invalid_combinations = (
            ("--shell", "--interactive"),
            ("--full", "--shell"),
        )
        for arguments in invalid_combinations:
            with self.subTest(arguments=arguments):
                self.assert_parse_exits_two(*arguments)

    def test_invalid_checkpoint_comparison_combinations_exit_two(self):
        invalid_combinations = (
            ("--compare-current-architecture",),
            ("--compare-current-architecture", "--inspect-checkpoint"),
        )
        for arguments in invalid_combinations:
            with self.subTest(arguments=arguments):
                self.assert_parse_exits_two(*arguments)

    def test_invalid_tokenizer_budgets_and_benchmark_repeats_exit_two(self):
        invalid_values = (
            ("--tokenizer-vocab-size", "0"),
            ("--tokenizer-max-subword-length", "0"),
            ("--benchmark-repeats", "0"),
        )
        for arguments in invalid_values:
            with self.subTest(arguments=arguments):
                self.assert_parse_exits_two(*arguments)


if __name__ == "__main__":
    unittest.main()
