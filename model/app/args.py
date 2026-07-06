"""CLI argument parsing and workflow-mode resolution for model/main.py."""

import argparse

BENCHMARK_MODE_MAP = {
    "compile-current": "current",
    "compile-components": "components",
    "compile-full": "full",
    "compile-repeat-current": "repeat-current",
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
        "--dry-run",
        action="store_true",
        help="Resolve workflow, runtime, and checkpoint plans without building the Nengo model.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the Python Nengo model and record complexity, but stop before simulator compilation.",
    )
    parser.add_argument(
        "--inspect-checkpoint",
        action="store_true",
        help="Inspect checkpoint metadata without compiling the simulator.",
    )
    parser.add_argument(
        "--compare-current-architecture",
        action="store_true",
        help="With --build-only and --inspect-checkpoint, compare checkpoint metadata against the current build signature.",
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
        "--compile-profile",
        choices=["full", "fast-solver"],
        default="full",
        help=(
            "Compile-time build profile. 'fast-solver' lowers ensemble eval-point "
            "counts during build without changing the architecture layout."
        ),
    )
    parser.add_argument(
        "--learned-init-mode",
        choices=["random-function", "zero-nosolver", "seeded-nosolver"],
        default="random-function",
        help=(
            "Initialization strategy for PES-learned decoded connections. "
            "'random-function' preserves the current behavior."
        ),
    )
    parser.add_argument(
        "--learned-init-seed",
        type=int,
        help="Optional deterministic seed for learned-connection initialization experiments.",
    )
    parser.add_argument(
        "--probe-mode",
        choices=["minimal", "debug"],
        default="debug",
        help=(
            "Instrumentation surface for model probes. "
            "'minimal' keeps only required probes, while 'debug' keeps the "
            "current richer probe set."
        ),
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
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=2,
        help="Repeat count for the compile-repeat-current benchmark mode.",
    )
    parser.add_argument(
        "--include-first-run-warmup",
        action="store_true",
        help="For compile-repeat-current, run one post-compile warmup step per repeat and record it.",
    )
    args = parser.parse_args()

    if args.shell and args.interactive:
        parser.error("--shell cannot be combined with --interactive.")

    if args.shell and args.full and not args.no_interactive:
        parser.error(
            "--full includes --interactive. Use --full --no-interactive --shell "
            "if you want the full workflow before the developer shell."
        )

    if args.compare_current_architecture and not args.inspect_checkpoint:
        parser.error("--compare-current-architecture requires --inspect-checkpoint.")

    if args.compare_current_architecture and not args.build_only:
        parser.error(
            "--compare-current-architecture currently requires --build-only so "
            "the current architecture can be compared without simulator compile."
        )

    if args.learned_init_mode == "seeded-nosolver" and args.learned_init_seed is None:
        parser.error("--learned-init-mode seeded-nosolver requires --learned-init-seed.")

    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be at least 1.")

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
