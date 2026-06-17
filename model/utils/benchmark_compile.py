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


def compile_case(name, sub_lengths, dimensions, simulator_name, context):
    with representation_dimension(dimensions):
        vocab = make_vocab(dimensions)

        start = perf_counter()
        model_result = nc.Model(sub_lengths, vocab, strict=False)
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

        result = {
            "name": name,
            "simulator": simulator_name,
            "sub_lengths": sub_lengths,
            "context_length": max(sub_lengths),
            "rep_vocab_dim": dimensions,
            "model_build_seconds": model_build_seconds,
            "simulator_compile_seconds": simulator_compile_seconds,
            "network": network_telemetry(model_result.model),
            "operators": operator_telemetry(simulator),
        }

        simulator.close()
        del simulator
        del model_result
        gc.collect()
        return result


def component_case(name, builder, dimensions, simulator_name, context):
    with representation_dimension(dimensions):
        vocab = make_vocab(dimensions)
        start = perf_counter()
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
            "model_build_seconds": model_build_seconds,
            "simulator_compile_seconds": simulator_compile_seconds,
            "network": network_telemetry(network),
            "operators": operator_telemetry(simulator),
        }
        simulator.close()
        return result


def build_base_component(vocab):
    with spa.Network(seed=mp.seed) as network:
        context = InputModule(vocab.dimensions)
        target = InputModule(vocab.dimensions)
        ncls.BaseComponent(vocab, context, target)
    return network


def benchmark(mode, platform_index=None, device_index=None):
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
    if mode == "full":
        scaling = [
            compile_case(name, sub_lengths, dimensions, "nengo_ocl", context)
            for name, sub_lengths, dimensions in scaling_cases
        ]

        simulator_comparison = [
            compile_case("comparison", [1, 20], 64, simulator, context)
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
            )
        ]

    component_builders = [
        ("State", lambda vocab: spa.State(vocab)),
        ("Bind", lambda vocab: spa.Bind(vocab)),
        ("ContextModule", lambda vocab: ncls.ContextModule(vocab)),
        ("BaseComponent", build_base_component),
    ]
    component_costs = []
    if mode in ("full", "components"):
        component_costs = [
            component_case(name, builder, 64, simulator, context)
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
        "scaling": scaling,
        "simulator_comparison": simulator_comparison,
        "component_costs": component_costs,
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
        choices=("full", "components", "current"),
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
    args = parser.parse_args()
    benchmark(
        args.mode,
        platform_index=args.platform_index,
        device_index=args.device_index,
    )
