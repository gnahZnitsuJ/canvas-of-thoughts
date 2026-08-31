import io
import sys
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

from app import workflow  # noqa: E402


class FakeTokenizer:
    def fingerprint(self):
        return "tokenizer-fingerprint"

    def metadata(self):
        return {"name": "word-v1", "fingerprint": self.fingerprint()}


def workflow_args(**overrides):
    values = {
        "full": False,
        "train": False,
        "eval": False,
        "demo": False,
        "interactive": False,
        "shell": False,
        "no_eval": False,
        "no_demo": False,
        "no_interactive": False,
        "use_runtime_profile": False,
        "train_mode": "single-pass",
        "token_duration": None,
        "calibrate_token_duration": False,
        "force_retrain": False,
        "dry_run": False,
        "inspect_decoder_cache": False,
        "inspect_checkpoint": False,
        "build_only": False,
        "checkpoint_path": "test.pkl",
        "compare_current_architecture": False,
        "no_telemetry": False,
        "probe_mode": "minimal",
        "compile_profile": "fast-solver",
        "learned_init_mode": "zero-nosolver",
        "learned_init_seed": None,
        "decoder_cache_mode": "auto",
        "architecture": "root-context-v1",
        "tokenizer": "word-v1",
        "tokenizer_normalization": "NFC",
        "tokenizer_vocab_size": 512,
        "tokenizer_max_subword_length": 12,
        "opencl_platform_index": None,
        "opencl_device_index": None,
        "first_run_warmup": False,
        "profile_compile": False,
        "max_examples": 3,
        "max_demo_examples": 2,
        "top_k": 2,
        "generate": False,
        "max_tokens": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeRuntime:
    def __init__(self, events):
        self.events = events
        self.invocation_index = 0

    def configure_training(self, **_configuration):
        self.events.append("configure")

    def simulator_invocation_telemetry(self):
        self.invocation_index += 1
        self.events.append(f"snapshot-{self.invocation_index}")
        return {
            "present_calls": self.invocation_index,
            "reset_context_calls": 0,
            "sim_run_count": self.invocation_index,
            "sim_run_seconds": 0.0,
            "simulated_seconds": 0.0,
        }

    def load_checkpoint(self, _path):
        self.events.append("load-checkpoint")


class WorkflowOrchestrationTests(unittest.TestCase):
    def test_telemetry_context_preserves_run_payload_contract(self):
        counters_before = {
            "present_calls": 1,
            "reset_context_calls": 1,
            "sim_run_count": 1,
            "sim_run_seconds": 1.0,
            "simulated_seconds": 1.0,
        }
        counters_after_training = {
            key: value + 1 for key, value in counters_before.items()
        }
        counters_after_evaluation = {
            key: value + 1 for key, value in counters_after_training.items()
        }
        runtime = SimpleNamespace(
            sim=object(),
            training_mode="single_pass",
            tokenizer=None,
            training_configuration=lambda: {"training_mode": "single_pass"},
        )
        model_result = SimpleNamespace(
            model=object(),
            sub_lengths=[1, 20],
            probe_mode="minimal",
            created_probe_labels=["prediction"],
            skipped_probe_labels=["debug"],
        )
        compiled = workflow.CompiledRun(
            runtime=runtime,
            model_result=model_result,
            opencl_selection={
                "platform": SimpleNamespace(name="platform"),
                "device": SimpleNamespace(name="device"),
                "platform_index": 2,
                "device_index": 3,
            },
            compile_profile={"name": "fast-solver"},
            compile_fingerprint={"backend": "nengo_ocl"},
        )
        context = workflow.RunTelemetryContext(
            compiled=compiled,
            timings={"Model build": 1.0},
            train_test=SimpleNamespace(
                training_set=[["a", "b"]],
                testing_set=[["a", "b"]],
            ),
            max_examples=1,
            training_invocations_before=counters_before,
            training_invocations_after=counters_after_training,
            evaluation_invocations_after=counters_after_evaluation,
            evaluation_result={"accuracy": 0.5},
            calibration_result={"selected_k": 1},
        )

        with (
            patch.object(workflow, "environment_telemetry", return_value={"python": "test"}),
            patch.object(workflow, "network_telemetry", return_value={"nodes": 1}),
            patch.object(workflow, "operator_telemetry", return_value={"Copy": 1}),
            patch.object(workflow, "training_invocation_estimate", return_value={"runs": 1}),
            patch.object(workflow, "evaluation_invocation_estimate", return_value={"runs": 1}),
            patch.object(workflow, "save_telemetry", return_value=Path("telemetry.json")) as save,
        ):
            with redirect_stdout(io.StringIO()):
                workflow.save_run_telemetry(context)

        payload = save.call_args.args[1]
        self.assertEqual(payload["kind"], "model_run")
        self.assertEqual(payload["environment"]["opencl_platform"], "platform")
        self.assertEqual(payload["environment"]["opencl_device_index"], 3)
        self.assertIs(payload["compile_profile"], compiled.compile_profile)
        self.assertIs(payload["compile_fingerprint"], compiled.compile_fingerprint)
        self.assertEqual(
            payload["actual_simulator_invocations"]["training"]["sim_run_count"],
            1,
        )
        self.assertEqual(payload["evaluation"], {"accuracy": 0.5})
        self.assertEqual(payload["calibration"], {"selected_k": 1})

    def test_decoder_cache_inspection_exits_before_runtime_planning(self):
        args = workflow_args(inspect_decoder_cache=True)

        with (
            patch.object(
                workflow,
                "inspect_decoder_cache",
                return_value={"path": "cache"},
            ) as inspect,
            patch.object(
                workflow,
                "format_decoder_cache_summary",
                return_value="decoder cache summary",
            ) as format_summary,
            patch.object(workflow, "load_requested_runtime_profile") as load_profile,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            workflow.run_application(args)

        self.assertIn("decoder cache summary", stdout.getvalue())
        inspect.assert_called_once_with()
        format_summary.assert_called_once_with({"path": "cache"})
        load_profile.assert_not_called()

    def test_dry_run_reports_resolved_plan_without_loading_data(self):
        args = workflow_args(dry_run=True)
        training_config = {
            "training_mode": "single_pass",
            "token_duration": 0.02,
            "token_duration_source": "default",
            "step_time": 0.02,
        }

        with (
            patch.object(workflow, "load_requested_runtime_profile", return_value=None),
            patch.object(
                workflow,
                "resolve_training_configuration",
                return_value=training_config,
            ),
            patch.object(workflow, "build_train_test") as build_data,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            workflow.run_application(args)

        output = stdout.getvalue()
        self.assertIn("Dry run summary", output)
        self.assertIn("'train': True", output)
        self.assertIn("training mode:           single_pass", output)
        self.assertIn("decoder cache mode:      auto", output)
        build_data.assert_not_called()

    def test_calibration_preconditions_fail_before_loading_data(self):
        cases = (
            (
                workflow_args(
                    train=True,
                    calibrate_token_duration=True,
                    force_retrain=True,
                ),
                {
                    "training_mode": "single_pass",
                    "token_duration": 0.02,
                    "token_duration_source": "default",
                    "step_time": 0.02,
                },
                "requires --train-mode scheduled",
            ),
            (
                workflow_args(
                    train=True,
                    calibrate_token_duration=True,
                    force_retrain=False,
                ),
                {
                    "training_mode": "scheduled",
                    "token_duration": 0.02,
                    "token_duration_source": "explicit",
                    "step_time": 0.001,
                },
                "requires an actual retraining run",
            ),
        )

        for args, training_config, message in cases:
            with self.subTest(message=message):
                with (
                    patch.object(
                        workflow,
                        "load_requested_runtime_profile",
                        return_value=None,
                    ),
                    patch.object(
                        workflow,
                        "resolve_training_configuration",
                        return_value=training_config,
                    ),
                    patch.object(workflow, "build_train_test") as build_data,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        workflow.run_application(args)

                build_data.assert_not_called()

    def test_checkpoint_inspection_prints_metadata_without_loading_data(self):
        args = workflow_args(inspect_checkpoint=True)
        metadata = {
            "timestamp": "2026-08-30T12:00:00Z",
            "architecture": {"training_semantics_version": "v1"},
            "compile_fingerprint": {},
        }

        with (
            patch.object(workflow, "load_requested_runtime_profile", return_value=None),
            patch.object(
                workflow,
                "resolve_training_configuration",
                return_value={
                    "training_mode": "single_pass",
                    "token_duration": 0.02,
                    "token_duration_source": "default",
                    "step_time": 0.02,
                },
            ),
            patch.object(
                workflow,
                "load_checkpoint_metadata",
                return_value=(Path("resolved-test.pkl"), metadata),
            ),
            patch.object(workflow, "build_train_test") as build_data,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            workflow.run_application(args)

        output = stdout.getvalue()
        self.assertIn("Checkpoint inspection", output)
        self.assertIn("resolved-test.pkl", output)
        self.assertIn("training semantics:      v1", output)
        build_data.assert_not_called()

    def test_missing_inspection_checkpoint_is_not_hidden_without_build_only(self):
        args = workflow_args(inspect_checkpoint=True)

        with (
            patch.object(workflow, "load_requested_runtime_profile", return_value=None),
            patch.object(
                workflow,
                "resolve_training_configuration",
                return_value={
                    "training_mode": "single_pass",
                    "token_duration": 0.02,
                    "token_duration_source": "default",
                    "step_time": 0.02,
                },
            ),
            patch.object(
                workflow,
                "load_checkpoint_metadata",
                side_effect=FileNotFoundError("missing checkpoint"),
            ),
            patch.object(workflow, "build_train_test") as build_data,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "missing checkpoint"):
                workflow.run_application(args)

        build_data.assert_not_called()

    def test_disabled_run_telemetry_does_not_persist(self):
        with (
            patch.object(workflow, "save_run_telemetry") as save,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            workflow.maybe_save_run_telemetry(False, object())

        save.assert_not_called()
        self.assertIn("Telemetry recording disabled", stdout.getvalue())

    def test_runtime_stages_follow_application_order_without_compile(self):
        events = []
        args = workflow_args(eval=True, demo=True, interactive=True, shell=True)
        train_test = SimpleNamespace(
            vocab=["TOKEN"],
            training_set=[["a", "b"]],
            testing_set=[["a", "b"]],
            tokenizer=FakeTokenizer(),
        )
        runtime = FakeRuntime(events)
        compiled = workflow.CompiledRun(
            runtime=runtime,
            model_result=object(),
            opencl_selection={
                "platform": SimpleNamespace(name="platform"),
                "device": SimpleNamespace(name="device"),
                "platform_index": 0,
                "device_index": 0,
            },
            compile_profile={"name": "fast-solver"},
            compile_fingerprint={"backend": "nengo_ocl"},
        )
        captured_contexts = []

        def record(name, result=None):
            def call(*_args, **_kwargs):
                events.append(name)
                return result

            return call

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    workflow,
                    "load_requested_runtime_profile",
                    side_effect=record("load-profile", None),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "resolve_training_configuration",
                    side_effect=record(
                        "resolve-training",
                        {
                            "training_mode": "single_pass",
                            "token_duration": 0.02,
                            "token_duration_source": "default",
                            "step_time": 0.02,
                        },
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "load_seed_vocab_model",
                    side_effect=record("load-seed", object()),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "build_train_test",
                    side_effect=record("build-data", train_test),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "build_model_vocab",
                    side_effect=record("build-vocab", object()),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "build_runtime",
                    side_effect=record("build-runtime", compiled),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "evaluate_model",
                    side_effect=record("evaluate", {"accuracy": 0.0}),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "maybe_save_run_telemetry",
                    side_effect=lambda enabled, context: (
                        events.append("telemetry"),
                        captured_contexts.append((enabled, context)),
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "run_demo_predictions",
                    side_effect=record("demo"),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "launch_interactive_prompt",
                    side_effect=record("interactive"),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "launch_runtime_shell",
                    side_effect=record("shell"),
                )
            )

            with redirect_stdout(io.StringIO()):
                workflow.run_application(args)

        self.assertEqual(
            events,
            [
                "load-profile",
                "resolve-training",
                "build-data",
                "load-seed",
                "build-vocab",
                "build-runtime",
                "configure",
                "snapshot-1",
                "load-checkpoint",
                "evaluate",
                "snapshot-2",
                "telemetry",
                "demo",
                "interactive",
                "shell",
            ],
        )
        self.assertEqual(len(captured_contexts), 1)
        telemetry_enabled, context = captured_contexts[0]
        self.assertTrue(telemetry_enabled)
        self.assertIs(context.compiled, compiled)
        self.assertEqual(context.evaluation_result, {"accuracy": 0.0})

    def test_build_only_checkpoint_comparison_never_compiles(self):
        events = []
        args = workflow_args(
            build_only=True,
            inspect_checkpoint=True,
            compare_current_architecture=True,
        )
        train_test = SimpleNamespace(
            vocab=["TOKEN"],
            training_set=[["TOKEN"]],
            tokenizer=FakeTokenizer(),
        )
        model_result = object()

        def record(name, result=None):
            def call(*_args, **_kwargs):
                events.append(name)
                return result

            return call

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(workflow, "load_requested_runtime_profile", return_value=None)
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "resolve_training_configuration",
                    return_value={
                        "training_mode": "single_pass",
                        "token_duration": 0.02,
                        "token_duration_source": "default",
                        "step_time": 0.02,
                    },
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "load_checkpoint_metadata",
                    side_effect=record("inspect", (Path("test.pkl"), {"metadata": True})),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "print_checkpoint_metadata",
                    side_effect=record("print-metadata"),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "load_seed_vocab_model",
                    side_effect=record("load-seed", object()),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "build_train_test",
                    side_effect=record("build-data", train_test),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "build_model_vocab",
                    side_effect=record("build-vocab", object()),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "build_model_result",
                    side_effect=record(
                        "build-model",
                        (
                            model_result,
                            {"name": "fast-solver", "settings": {}},
                        ),
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "compare_architecture_to_checkpoint",
                    side_effect=record("compare", {"matches": True}),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "save_build_only_telemetry",
                    side_effect=record("telemetry"),
                )
            )
            stack.enter_context(
                patch.object(
                    workflow,
                    "print_architecture_comparison",
                    side_effect=record("print-comparison"),
                )
            )
            build_runtime = stack.enter_context(
                patch.object(workflow, "build_runtime")
            )

            with redirect_stdout(io.StringIO()):
                workflow.run_application(args)

        build_runtime.assert_not_called()
        self.assertEqual(
            events,
            [
                "inspect",
                "print-metadata",
                "build-data",
                "load-seed",
                "build-vocab",
                "build-model",
                "compare",
                "telemetry",
                "print-comparison",
            ],
        )


if __name__ == "__main__":
    unittest.main()
