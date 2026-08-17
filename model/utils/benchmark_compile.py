"""Compile benchmarks for full models and representative Nengo components."""

import argparse
import gc
import sys
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
from architecture.variants import DEFAULT_ARCHITECTURE_NAME, available_architectures
from config import cache_defaults, data_defaults, model_defaults
from utils.build_config import (
    DEFAULT_COMPILE_PROFILE_NAME,
    DEFAULT_LEARNED_INIT_MODE,
    DEFAULT_LEARNED_INIT_SEED,
    LEARNED_INIT_MODES,
    available_compile_profiles,
    compile_profile_scope,
    resolve_compile_profile,
    validate_learned_init_configuration,
)
from utils.input import InputModule
from utils.decoder_cache import create_decoder_cache, make_backend_build_model
from utils.opencl import print_opencl_selection, select_opencl_device
from utils.probes import DEFAULT_PROBE_MODE, VALID_PROBE_MODES
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
BENCHMARK_MODES = ("full", "components", "current", "repeat-current")


def make_vocab(dimensions):
    vocab = spa.Vocabulary(dimensions, strict=False, pointer_gen=None)
    vocab.add("POS", np.ones(dimensions) / np.sqrt(dimensions))
    vocab.add(data_defaults.PAD_TOKEN, np.zeros(dimensions))
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
    compile_profile_name=DEFAULT_COMPILE_PROFILE_NAME,
    learned_init_mode=DEFAULT_LEARNED_INIT_MODE,
    learned_init_seed=DEFAULT_LEARNED_INIT_SEED,
    include_first_run_warmup=False,
    repeat_index=None,
    architecture_name=DEFAULT_ARCHITECTURE_NAME,
    decoder_cache_mode=cache_defaults.DEFAULT_DECODER_CACHE_MODE,
):
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
            architecture_name=architecture_name,
        )
    model_build_seconds = perf_counter() - start

    decoder_cache = create_decoder_cache(
        decoder_cache_mode,
        learning_connections=model_result.learning_connections,
        learning_connection_metadata=model_result.learning_connection_metadata,
    )
    build_model = make_backend_build_model(simulator_name, decoder_cache)

    start = perf_counter()
    if simulator_name == "nengo":
        simulator = nengo.Simulator(
            model_result.model,
            model=build_model,
            progress_bar=False,
        )
    else:
        simulator = nengo_ocl.Simulator(
            model_result.model,
            model=build_model,
            context=context,
            progress_bar=False,
        )
    simulator_compile_seconds = perf_counter() - start
    decoder_cache.finalize_after_build()
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
        "architecture_name": architecture_name,
        "architecture_signature": model_result.architecture_topology_signature,
        "model_build_seconds": model_build_seconds,
        "simulator_compile_seconds": simulator_compile_seconds,
        "first_run_warmup_seconds": first_run_warmup_seconds,
        "network": network_telemetry(model_result.model),
        "operators": operator_telemetry(simulator),
        "decoder_cache": decoder_cache.telemetry(),
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
    compile_profile_name=DEFAULT_COMPILE_PROFILE_NAME,
    decoder_cache_mode=cache_defaults.DEFAULT_DECODER_CACHE_MODE,
):
    vocab = make_vocab(dimensions)
    compile_profile = resolve_compile_profile(compile_profile_name)
    start = perf_counter()
    with compile_profile_scope(compile_profile):
        network = builder(vocab)
    model_build_seconds = perf_counter() - start

    decoder_cache = create_decoder_cache(decoder_cache_mode)
    build_model = make_backend_build_model(simulator_name, decoder_cache)

    start = perf_counter()
    if simulator_name == "nengo":
        simulator = nengo.Simulator(
            network,
            model=build_model,
            progress_bar=False,
        )
    else:
        simulator = nengo_ocl.Simulator(
            network,
            model=build_model,
            context=context,
            progress_bar=False,
        )
    simulator_compile_seconds = perf_counter() - start
    decoder_cache.finalize_after_build()

    result = {
        "name": name,
        "simulator": simulator_name,
        "rep_vocab_dim": dimensions,
        "compile_profile": compile_profile,
        "model_build_seconds": model_build_seconds,
        "simulator_compile_seconds": simulator_compile_seconds,
        "network": network_telemetry(network),
        "operators": operator_telemetry(simulator),
        "decoder_cache": decoder_cache.telemetry(),
    }
    simulator.close()
    return result


def build_base_component(
    vocab,
    *,
    learned_init_mode=DEFAULT_LEARNED_INIT_MODE,
    learned_init_seed=DEFAULT_LEARNED_INIT_SEED,
):
    with spa.Network(seed=model_defaults.MODEL_SEED) as network:
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
    probe_mode=DEFAULT_PROBE_MODE,
    compile_profile_name=DEFAULT_COMPILE_PROFILE_NAME,
    learned_init_mode=DEFAULT_LEARNED_INIT_MODE,
    learned_init_seed=DEFAULT_LEARNED_INIT_SEED,
    repeats=2,
    include_first_run_warmup=False,
    architecture_name=DEFAULT_ARCHITECTURE_NAME,
    decoder_cache_mode=cache_defaults.DEFAULT_DECODER_CACHE_MODE,
):
    if mode not in BENCHMARK_MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")

    opencl_selection = select_opencl_device(
        platform_index=platform_index,
        device_index=device_index,
    )
    print_opencl_selection(opencl_selection)
    platform = opencl_selection["platform"]
    device = opencl_selection["device"]
    context = opencl_selection["context"]

    # Refresh is an invocation-level action. Clear once, then allow later cases
    # (especially repeat-current) to demonstrate reuse within this benchmark.
    if decoder_cache_mode == "refresh":
        create_decoder_cache("refresh")
        effective_decoder_cache_mode = "auto"
    else:
        effective_decoder_cache_mode = decoder_cache_mode

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
                architecture_name=architecture_name,
                decoder_cache_mode=effective_decoder_cache_mode,
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
                architecture_name=architecture_name,
                decoder_cache_mode=effective_decoder_cache_mode,
            )
            for simulator in ("nengo", "nengo_ocl")
        ]
    elif mode == "current":
        scaling = [
            compile_case(
                "current_configuration",
                [1, model_defaults.CONTEXT_LENGTH],
                model_defaults.VOCAB_DIMENSIONS,
                "nengo_ocl",
                context,
                probe_mode,
                compile_profile_name=compile_profile_name,
                learned_init_mode=learned_init_mode,
                learned_init_seed=learned_init_seed,
                architecture_name=architecture_name,
                decoder_cache_mode=effective_decoder_cache_mode,
            )
        ]
    elif mode == "repeat-current":
        repeat_compile = [
            compile_case(
                "repeat_current_configuration",
                [1, model_defaults.CONTEXT_LENGTH],
                model_defaults.VOCAB_DIMENSIONS,
                "nengo_ocl",
                context,
                probe_mode,
                compile_profile_name=compile_profile_name,
                learned_init_mode=learned_init_mode,
                learned_init_seed=learned_init_seed,
                include_first_run_warmup=include_first_run_warmup,
                repeat_index=repeat_index,
                architecture_name=architecture_name,
                decoder_cache_mode=effective_decoder_cache_mode,
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
                decoder_cache_mode=effective_decoder_cache_mode,
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
        "architecture_name": architecture_name,
        "decoder_cache_mode": decoder_cache_mode,
        "decoder_cache_refresh_performed": decoder_cache_mode == "refresh",
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
        choices=BENCHMARK_MODES,
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
        "--architecture",
        choices=available_architectures(),
        default=DEFAULT_ARCHITECTURE_NAME,
        help="Named architecture used by full-model benchmark cases.",
    )
    parser.add_argument(
        "--probe-mode",
        choices=VALID_PROBE_MODES,
        default=DEFAULT_PROBE_MODE,
        help="Instrumentation surface to use while building benchmark cases.",
    )
    parser.add_argument(
        "--compile-profile",
        choices=available_compile_profiles(),
        default=DEFAULT_COMPILE_PROFILE_NAME,
        help="Build profile to use during benchmark model construction.",
    )
    parser.add_argument(
        "--learned-init-mode",
        choices=LEARNED_INIT_MODES,
        default=DEFAULT_LEARNED_INIT_MODE,
        help="Initialization strategy for PES-learned connections in benchmark runs.",
    )
    parser.add_argument(
        "--learned-init-seed",
        type=int,
        default=DEFAULT_LEARNED_INIT_SEED,
        help="Seed for deterministic learned-connection initialization.",
    )
    parser.add_argument(
        "--decoder-cache-mode",
        choices=cache_defaults.DECODER_CACHE_MODES,
        default=cache_defaults.DEFAULT_DECODER_CACHE_MODE,
        help="Persistent decoder-cache reuse policy.",
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
    try:
        validate_learned_init_configuration(
            args.learned_init_mode,
            args.learned_init_seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
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
        architecture_name=args.architecture,
        decoder_cache_mode=args.decoder_cache_mode,
    )
