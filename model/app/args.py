"""CLI argument parsing and workflow-mode resolution for model/main.py."""

import argparse

BENCHMARK_MODE_MAP = {
    "compile-current": "current",
    "compile-components": "components",
    "compile-full": "full",
}

DEFAULT_CALIBRATION_CANDIDATES = None


def _parse_int_list(value):
    """Parse a comma-separated list of integer timestep multipliers."""
    try:
        parsed = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid integer list: {value}"
        ) from exc

    if not parsed:
        raise argparse.ArgumentTypeError("At least one calibration candidate is required")
    if any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("Calibration candidates must be positive integers")

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
        "--shell",
        action="store_true",
        help="Launch the developer runtime shell.",
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
        help="Token duration for scheduled training.",
    )
    parser.add_argument(
        "--calibrate-token-duration",
        action="store_true",
        help="Match scheduled training against the single-pass reference before retraining.",
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
        type=_parse_int_list,
        default=DEFAULT_CALIBRATION_CANDIDATES,
        help=(
            "Comma-separated integer timestep multipliers k to test during "
            "calibration. Defaults to testing every k from 1 through baseline_k."
        ),
    )
    parser.add_argument(
        "--use-runtime-profile",
        action="store_true",
        help="Load token-duration defaults from model/config/runtime_profile.json.",
    )
    parser.add_argument(
        "--first-run-warmup",
        action="store_true",
        help="Run one post-construction warmup step and record its startup cost.",
    )
    parser.add_argument(
        "--profile-compile",
        action="store_true",
        help="Write a cProfile .prof artifact for simulator construction under model/results/.",
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
    args = parser.parse_args()

    if args.shell and args.interactive:
        parser.error("--shell cannot be combined with --interactive.")

    if args.shell and args.full and not args.no_interactive:
        parser.error(
            "--full includes --interactive. Use --full --no-interactive --shell "
            "if you want the full workflow before the developer shell."
        )

    return args


def resolve_workflow(args):
    """Translate raw CLI flags into the concrete stages this run should execute."""
    explicit_workflow = any(
        [
            args.full,
            args.train,
            args.eval,
            args.demo,
            args.interactive,
            args.shell,
        ]
    )

    if explicit_workflow:
        if args.full:
            workflow = {
                "train": True,
                "eval": True,
                "demo": True,
                "interactive": True,
                "shell": args.shell,
            }
        else:
            workflow = {
                "train": args.train,
                "eval": args.eval,
                "demo": args.demo,
                "interactive": args.interactive,
                "shell": args.shell,
            }
    else:
        workflow = {
            "train": True,
            "eval": False,
            "demo": False,
            "interactive": False,
            "shell": False,
        }

    if explicit_workflow:
        if args.no_eval:
            workflow["eval"] = False
        if args.no_demo:
            workflow["demo"] = False
        if args.no_interactive:
            workflow["interactive"] = False

    return workflow
