"""Command-line entrypoint for Canvas model and benchmark workflows."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.args import BENCHMARK_MODE_MAP, parse_args
from app.workflow import run_application
from utils.benchmark_compile import benchmark as run_compile_benchmark


def main():
    args = parse_args()

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
        )
        return

    run_application(args)


if __name__ == "__main__":
    main()
