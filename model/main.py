"""Command-line entrypoint for Canvas model and benchmark workflows."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.environment_check import (
    format_missing_package_message,
    load_requirements,
    missing_required_packages,
    run_environment_check,
)


def run_model_cli(argv=None):
    """Load the model stack only after the lightweight bootstrap checks pass."""
    from app.args import BENCHMARK_MODE_MAP, parse_args
    from app.workflow import run_application
    from utils.benchmark_compile import benchmark as run_compile_benchmark

    args = parse_args(argv)

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


def main(argv=None):
    """Run bootstrap diagnostics before importing heavyweight dependencies."""
    if argv is None:
        argv = sys.argv[1:]

    if "--check-environment" in argv:
        if argv != ["--check-environment"]:
            print("--check-environment must be used by itself.", file=sys.stderr)
            return 2
        return run_environment_check()

    try:
        missing = missing_required_packages()
    except (OSError, ValueError) as exc:
        print(f"Could not read project requirements: {exc}", file=sys.stderr)
        return 1

    if missing:
        print(format_missing_package_message(missing), file=sys.stderr)
        return 1

    try:
        return run_model_cli(argv)
    except ModuleNotFoundError as exc:
        requirements = load_requirements()
        required_imports = {requirement.import_name for requirement in requirements}
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root not in required_imports:
            raise
        matching = tuple(
            requirement
            for requirement in requirements
            if requirement.import_name == missing_root
        )
        print(format_missing_package_message(matching), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
