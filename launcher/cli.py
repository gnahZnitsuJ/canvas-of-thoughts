"""Standard-library application launcher and model command dispatcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from launcher.bootstrap.checks import (
    format_missing_package_message,
    load_requirements,
    missing_required_packages,
    run_doctor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m launcher",
        description="Run Canvas of Thoughts or diagnose its prerequisites.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "doctor",
        help="Check Python packages, data resources, and OpenCL.",
    )
    run_parser = subparsers.add_parser(
        "run",
        add_help=False,
        help="Run the model; remaining arguments are model options.",
    )
    return parser


def _report_late_missing_dependency(exc):
    requirements = load_requirements()
    missing_root = (exc.name or "").split(".", 1)[0]
    matching = tuple(
        requirement
        for requirement in requirements
        if requirement.import_name == missing_root
    )
    if not matching:
        return False
    print(format_missing_package_message(matching), file=sys.stderr)
    return True


def run_model_cli(argv=None, prog=None):
    """Import and run the model stack after bootstrap checks pass."""
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))

    from app.args import BENCHMARK_MODE_MAP, parse_args
    from app.workflow import run_application
    from utils.benchmark_compile import benchmark as run_compile_benchmark

    args = parse_args(argv, prog=prog)
    if args.benchmark:
        run_compile_benchmark(
            BENCHMARK_MODE_MAP[args.benchmark],
            platform_index=args.opencl_platform_index,
            device_index=args.opencl_device_index,
            probe_mode=args.probe_mode,
            compile_profile_name=args.compile_profile,
            learned_init_mode=args.learned_init_mode,
            learned_init_seed=args.learned_init_seed,
            architecture_name=args.architecture,
            repeats=args.benchmark_repeats,
            include_first_run_warmup=args.include_first_run_warmup,
            decoder_cache_mode=args.decoder_cache_mode,
        )
        return 0

    run_application(args)
    return 0


def run_command(argv=None, prog=None):
    """Perform the quick bootstrap gate and dispatch to the model CLI."""
    try:
        missing = missing_required_packages()
    except (OSError, ValueError) as exc:
        print(f"Could not read project requirements: {exc}", file=sys.stderr)
        return 1

    if missing:
        print(format_missing_package_message(missing), file=sys.stderr)
        return 1

    try:
        return run_model_cli(argv, prog=prog)
    except ModuleNotFoundError as exc:
        if not _report_late_missing_dependency(exc):
            raise
        return 1


def main(argv=None):
    """Run the canonical ``doctor`` or ``run`` application command."""
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    if not argv:
        parser.print_help()
        return 0
    if argv[0] == "run":
        return run_command(
            argv[1:],
            prog="python -m launcher run",
        )

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return run_doctor()
    return 0


def legacy_main(argv=None):
    """Preserve the historical ``python model/main.py`` command surface."""
    if argv is None:
        argv = sys.argv[1:]
    if "--check-environment" in argv:
        if argv != ["--check-environment"]:
            print("--check-environment must be used by itself.", file=sys.stderr)
            return 2
        return run_doctor()
    return run_command(argv)
