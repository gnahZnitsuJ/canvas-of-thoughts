import argparse
import gc
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR.parent

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import nengo
import nengo_ocl
import nengo_spa as spa
import numpy as np

import components.net_comp as nc
import components.net_classes as ncls
from config import model_parameters as mp
from utils.build_config import compile_profile_scope, resolve_compile_profile
from utils.input import InputModule
from utils.opencl import print_opencl_selection, select_opencl_device
from utils.telemetry import (
    environment_telemetry,
    network_telemetry,
    operator_telemetry,
    print_compile_benchmark_summary,
    render_compile_benchmark_summary,
    save_telemetry,
    save_text_artifact,
)


RESULTS_DIR = MODEL_DIR / "results"


@contextmanager
def representation_dimension(dimensions):
    previous = mp.rep_vocab_dim
    mp.rep_vocab_dim = dimensions
    try:
        yield
    finally:
        mp.rep_vocab_dim = previous


def make_vocab(dimensions):
    vocab = spa.Vocabulary(dimensions, strict=False, pointer_gen=None)
    vocab.add("POS", np.ones(dimensions) / np.sqrt(dimensions))
    vocab.add(mp.pad_token, np.zeros(dimensions))
    return vocab


def run_first_step_warmup(simulator):
    """Measure one post-compile backend warmup step for repeat benchmarks."""
    start = perf_counter()
    run_steps = getattr(simulator, "run_steps", None)
    if callable(run_steps):
        run_steps(1)
    else:
        simulator.run(float(getattr(simulator, "dt", 0.001)))

    reset = getattr(simulator, "reset", None)
    if callable(reset):
        reset()
    return perf_counter() - start


def compile_case(
    name,
    sub_lengths,
    dimensions,
    simulator_name,
    context,
    probe_mode,
    *,
    compile_profile_name="full",
    learned_init_mode="random-function",
    learned_init_seed=None,
    include_first_run_warmup=False,
    repeat_index=None,
):
    with representation_dimension(dimensions):
        vocab = make_vocab(dimensions)
        compile_profile = resolve_compile_profile(compile_profile_name)

        start = perf_counter()
        with compile_profile_scope(compile_profile):
            model_result = nc.Model(
                sub_lengths,
                vocab,
                strict=False,
                probe_mode=probe_mode,
                learned_init_mode=learned_init_mode,
                learned_init_seed=learned_init_seed,
                compile_profile_name=compile_profile["name"],
                compile_profile_settings=compile_profile["settings"],
            )
        model_build_seconds = perf_counter() - start

        start = perf_counter()
        if simulator_name == "nengo":
            simulator = nengo.Simulator(
                model_result.model,
                progress_bar=False,
            )
        else:
            simulator = nengo_ocl.Simulator(
                model_result.model,
                context=context,
                progress_bar=False,
            )
        simulator_compile_seconds = perf_counter() - start
        first_run_warmup_seconds = (
            run_first_step_warmup(simulator)
            if include_first_run_warmup
            else None
        )

        result = {
            "name": name,
            "simulator": simulator_name,
            "repeat_index": repeat_index,
            "sub_lengths": sub_lengths,
            "context_length": max(sub_lengths),
            "rep_vocab_dim": dimensions,
            "probe_mode": model_result.probe_mode,
            "compile_profile": compile_profile,
            "learned_init_mode": learned_init_mode,
            "learned_init_seed": learned_init_seed,
            "model_build_seconds": model_build_seconds,
            "simulator_compile_seconds": simulator_compile_seconds,
            "first_run_warmup_seconds": first_run_warmup_seconds,
            "network": network_telemetry(model_result.model),
            "operators": operator_telemetry(simulator),
            "probes": {
                "mode": model_result.probe_mode,
                "created_labels": model_result.created_probe_labels,
                "skipped_labels": model_result.skipped_probe_labels,
            },
        }

        simulator.close()
        del simulator
        del model_result
        gc.collect()
        return result


def component_case(
    name,
    builder,
    dimensions,
    simulator_name,
    context,
    *,
    compile_profile_name="full",
):
    with representation_dimension(dimensions):
        vocab = make_vocab(dimensions)
        compile_profile = resolve_compile_profile(compile_profile_name)
        start = perf_counter()
        with compile_profile_scope(compile_profile):
            network = builder(vocab)
        model_build_seconds = perf_counter() - start

        start = perf_counter()
        if simulator_name == "nengo":
            simulator = nengo.Simulator(network, progress_bar=False)
        else:
            simulator = nengo_ocl.Simulator(
                network,
                context=context,
                progress_bar=False,
            )
        simulator_compile_seconds = perf_counter() - start

        result = {
            "name": name,
            "simulator": simulator_name,
            "rep_vocab_dim": dimensions,
            "compile_profile": compile_profile,
            "model_build_seconds": model_build_seconds,
            "simulator_compile_seconds": simulator_compile_seconds,
            "network": network_telemetry(network),
            "operators": operator_telemetry(simulator),
        }
        simulator.close()
        return result


def build_base_component(
    vocab,
    *,
    learned_init_mode="random-function",
    learned_init_seed=None,
):
    with spa.Network(seed=mp.seed) as network:
        context = InputModule(vocab.dimensions)
        target = InputModule(vocab.dimensions)
        ncls.BaseComponent(
            vocab,
            context,
            target,
            learned_init_mode=learned_init_mode,
            learned_init_seed=learned_init_seed,
        )
    return network


def benchmark(
    mode,
    platform_index=None,
    device_index=None,
    probe_mode="debug",
    compile_profile_name="full",
    learned_init_mode="random-function",
    learned_init_seed=None,
    repeats=2,
    include_first_run_warmup=False,
):
    opencl_selection = select_opencl_device(
        platform_index=platform_index,
        device_index=device_index,
    )
    print_opencl_selection(opencl_selection)
    platform = opencl_selection["platform"]
    device = opencl_selection["device"]
    context = opencl_selection["context"]

    scaling_cases = [
        ("baseline", [1, 20], 64),
        ("sub_lengths_1", [20], 64),
        ("sub_lengths_4", [1, 5, 10, 20], 64),
        ("context_length_5", [1, 5], 64),
        ("context_length_100", [1, 100], 64),
        ("dimension_32", [1, 20], 32),
        ("dimension_128", [1, 20], 128),
    ]

    scaling = []
    simulator_comparison = []
    repeat_compile = []
    if mode == "full":
        scaling = [
            compile_case(
                name,
                sub_lengths,
                dimensions,
                "nengo_ocl",
                context,
                probe_mode,
                compile_profile_name=compile_profile_name,
                learned_init_mode=learned_init_mode,
                learned_init_seed=learned_init_seed,
            )
            for name, sub_lengths, dimensions in scaling_cases
        ]

        simulator_comparison = [
            compile_case(
                "comparison",
                [1, 20],
                64,
                simulator,
                context,
                probe_mode,
                compile_profile_name=compile_profile_name,
                learned_init_mode=learned_init_mode,
                learned_init_seed=learned_init_seed,
            )
            for simulator in ("nengo", "nengo_ocl")
        ]
    elif mode == "current":
        scaling = [
            compile_case(
                "current_configuration",
                [1, mp.context_length],
                mp.rep_vocab_dim,
                "nengo_ocl",
                context,
                probe_mode,
                compile_profile_name=compile_profile_name,
                learned_init_mode=learned_init_mode,
                learned_init_seed=learned_init_seed,
            )
        ]
    elif mode == "repeat-current":
        repeat_compile = [
            compile_case(
                "repeat_current_configuration",
                [1, mp.context_length],
                mp.rep_vocab_dim,
                "nengo_ocl",
                context,
                probe_mode,
                compile_profile_name=compile_profile_name,
                learned_init_mode=learned_init_mode,
                learned_init_seed=learned_init_seed,
                include_first_run_warmup=include_first_run_warmup,
                repeat_index=repeat_index,
            )
            for repeat_index in range(repeats)
        ]

    component_builders = [
        ("State", lambda vocab: spa.State(vocab)),
        ("Bind", lambda vocab: spa.Bind(vocab)),
        ("ContextModule", lambda vocab: ncls.ContextModule(vocab)),
        (
            "BaseComponent",
            lambda vocab: build_base_component(
                vocab,
                learned_init_mode=learned_init_mode,
                learned_init_seed=learned_init_seed,
            ),
        ),
    ]
    component_costs = []
    if mode in ("full", "components"):
        component_costs = [
            component_case(
                name,
                builder,
                64,
                simulator,
                context,
                compile_profile_name=compile_profile_name,
            )
            for simulator in ("nengo", "nengo_ocl")
            for name, builder in component_builders
        ]

    payload = {
        "kind": f"compile_benchmark_{mode}",
        "environment": {
            **environment_telemetry(),
            "opencl_platform": platform.name,
            "opencl_device": device.name,
            "opencl_platform_index": opencl_selection["platform_index"],
            "opencl_device_index": opencl_selection["device_index"],
        },
        "probe_mode": probe_mode,
        "compile_profile": resolve_compile_profile(compile_profile_name),
        "learned_init_mode": learned_init_mode,
        "learned_init_seed": learned_init_seed,
        "scaling": scaling,
        "simulator_comparison": simulator_comparison,
        "component_costs": component_costs,
        "repeat_compile": repeat_compile,
    }
    result_path = save_telemetry(RESULTS_DIR, payload)
    summary_text = render_compile_benchmark_summary(
        payload,
        telemetry_path=result_path,
    )
    summary_path = save_text_artifact(
        RESULTS_DIR,
        summary_text,
        prefix="summary",
    )
    print_compile_benchmark_summary(payload)
    print(f"Saved compile benchmark telemetry to: {result_path}")
    print(f"Saved compile benchmark summary to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("full", "components", "current", "repeat-current"),
        default="full",
    )
    parser.add_argument(
        "--platform-index",
        type=int,
        help=(
            "Explicit OpenCL platform index. Defaults to "
            "CANVAS_OPENCL_PLATFORM_INDEX if set, otherwise 0."
        ),
    )
    parser.add_argument(
        "--device-index",
        type=int,
        help=(
            "Explicit OpenCL device index within the selected platform. "
            "Defaults to CANVAS_OPENCL_DEVICE_INDEX if set, otherwise 0."
        ),
    )
    parser.add_argument(
        "--probe-mode",
        choices=("minimal", "debug"),
        default="debug",
        help="Instrumentation surface to use while building benchmark cases.",
    )
    parser.add_argument(
        "--compile-profile",
        choices=("full", "fast-solver"),
        default="full",
        help="Build profile to use during benchmark model construction.",
    )
    parser.add_argument(
        "--learned-init-mode",
        choices=("random-function", "zero-nosolver", "seeded-nosolver"),
        default="random-function",
        help="Initialization strategy for PES-learned connections in benchmark runs.",
    )
    parser.add_argument(
        "--learned-init-seed",
        type=int,
        help="Optional seed for deterministic learned-connection initialization.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Repeat count for repeat-current benchmark mode.",
    )
    parser.add_argument(
        "--include-first-run-warmup",
        action="store_true",
        help="Run one post-compile warmup step per repeat and record it.",
    )
    args = parser.parse_args()
    if args.learned_init_mode == "seeded-nosolver" and args.learned_init_seed is None:
        parser.error("--learned-init-mode seeded-nosolver requires --learned-init-seed.")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    benchmark(
        args.mode,
        platform_index=args.platform_index,
        device_index=args.device_index,
        probe_mode=args.probe_mode,
        compile_profile_name=args.compile_profile,
        learned_init_mode=args.learned_init_mode,
        learned_init_seed=args.learned_init_seed,
        repeats=args.repeats,
        include_first_run_warmup=args.include_first_run_warmup,
    )
