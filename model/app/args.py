"""CLI argument parsing and workflow-mode resolution for model/main.py."""

import argparse

BENCHMARK_MODE_MAP = {
    "compile-current": "current",
    "compile-components": "components",
    "compile-full": "full",
}

DEFAULT_CALIBRATION_CANDIDATES = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]


def _parse_float_list(value):
    """Parse a comma-separated list of floats for calibration candidates."""
    try:
        parsed = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid float list: {value}"
        ) from exc

    if not parsed:
        raise argparse.ArgumentTypeError("At least one calibration candidate is required")

    return parsed


def parse_args():
    """Build and parse the top-level CLI for normal runs and benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run the Nengo language-model workflow."
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARK_MODE_MAP),
        help="Run a compile benchmark and exit.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full workflow: train, eval, demo, and interactive.",
    )
    parser.add_argument("--train", action="store_true", help="Run training/load.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation.")
    parser.add_argument("--demo", action="store_true", help="Print demo predictions.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the interactive prompt.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation in the default workflow.",
    )
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Skip demo predictions in the default workflow.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive mode in the default workflow.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="reuters_checkpoint.pkl",
        help="Checkpoint file name under model/checkpoints.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing checkpoint and retrain.",
    )
    parser.add_argument(
        "--train-mode",
        choices=["single-pass", "scheduled"],
        default="single-pass",
        help="Training driver for corpus training.",
    )
    parser.add_argument(
        "--token-duration",
        type=float,
        help="Token duration for scheduled training or calibration.",
    )
    parser.add_argument(
        "--calibrate-token-duration",
        action="store_true",
        help="Profile scheduled training token durations before retraining.",
    )
    parser.add_argument(
        "--calibration-train-sequences",
        type=int,
        default=2,
        help="Training sequences to use during token-duration calibration.",
    )
    parser.add_argument(
        "--calibration-eval-examples",
        type=int,
        default=50,
        help="Evaluation examples to use during token-duration calibration.",
    )
    parser.add_argument(
        "--calibration-candidates",
        type=_parse_float_list,
        default=list(DEFAULT_CALIBRATION_CANDIDATES),
        help="Comma-separated token durations to test during calibration.",
    )
    parser.add_argument(
        "--use-runtime-profile",
        action="store_true",
        help="Load token-duration defaults from model/config/runtime_profile.json.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Maximum evaluation examples.",
    )
    parser.add_argument(
        "--max-demo-examples",
        type=int,
        default=10,
        help="Maximum demo predictions to print.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k predictions for evaluation/demo/interactive output.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Use autoregressive generation in interactive mode.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=15,
        help="Maximum generated tokens in interactive mode.",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable telemetry recording and results-file output for this run.",
    )
    parser.add_argument(
        "--opencl-platform-index",
        type=int,
        help=(
            "Explicit OpenCL platform index. Defaults to "
            "CANVAS_OPENCL_PLATFORM_INDEX if set, otherwise 0."
        ),
    )
    parser.add_argument(
        "--opencl-device-index",
        type=int,
        help=(
            "Explicit OpenCL device index within the selected platform. "
            "Defaults to CANVAS_OPENCL_DEVICE_INDEX if set, otherwise 0."
        ),
    )
    return parser.parse_args()


def resolve_workflow(args):
    """Translate raw CLI flags into the concrete stages this run should execute."""
    explicit_workflow = any(
        [
            args.full,
            args.train,
            args.eval,
            args.demo,
            args.interactive,
        ]
    )

    if explicit_workflow:
        if args.full:
            workflow = {
                "train": True,
                "eval": True,
                "demo": True,
                "interactive": True,
            }
        else:
            workflow = {
                "train": args.train,
                "eval": args.eval,
                "demo": args.demo,
                "interactive": args.interactive,
            }
    else:
        workflow = {
            "train": True,
            "eval": False,
            "demo": False,
            "interactive": False,
        }

    if explicit_workflow:
        if args.no_eval:
            workflow["eval"] = False
        if args.no_demo:
            workflow["demo"] = False
        if args.no_interactive:
            workflow["interactive"] = False

    return workflow