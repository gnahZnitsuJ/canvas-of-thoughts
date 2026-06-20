"""High-level workflow helpers used by the main model CLI entrypoint."""

from pathlib import Path
from time import perf_counter

import nengo_ocl
import nengo_spa as spa
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import reuters

import components.net_comp as nc
from components.runtime import ModelRuntime
from config import model_parameters as mp
from utils import seed_vocab
from utils.input import make_unitary
from utils.opencl import print_opencl_selection, select_opencl_device
from utils.processing import WordsToSPAVocab
from utils.telemetry import (
    environment_telemetry,
    evaluation_invocation_estimate,
    network_telemetry,
    operator_telemetry,
    save_telemetry,
    training_invocation_estimate,
)
from utils.train_partition import multiple_data_partition

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
SEED_VOCAB_PATH = BASE_DIR / "utils" / "seed_vocab.model"
DATASETS = [reuters]


def print_timing(label, elapsed):
    """Print one timing line using the same alignment as the rest of the CLI."""
    print(f"{label + ':':<23}{elapsed:.3f} sec")


def invocation_delta(after, before):
    """Compute per-phase deltas from cumulative runtime invocation counters."""
    return {
        key: after[key] - before[key]
        for key in after
    }


def load_seed_vocab_model():
    """Load the cached seed Word2Vec model, generating it once if needed."""
    if SEED_VOCAB_PATH.is_file():
        print("seed_vocab.model exists.")
        return Word2Vec.load(str(SEED_VOCAB_PATH))

    print("seed_vocab.model does NOT exist.")
    print("Generating seed vocabulary model...")
    seed_vocab.generate_seed_vocab(DATASETS)

    if not SEED_VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f"Failed to generate {SEED_VOCAB_PATH}"
        )

    return Word2Vec.load(str(SEED_VOCAB_PATH))


def build_train_test(timings):
    """Partition datasets into train/test sequences and record partition time."""
    start = perf_counter()
    train_test = multiple_data_partition(
        DATASETS,
        training_restriction=mp.training_restriction,
        testing_restriction=mp.testing_restriction,
        strict=mp.strict_vocab,
    )
    timings["Data partition"] = perf_counter() - start
    return train_test


def build_model_vocab(seed_vocab_model, vocab, timings):
    """Construct the SPA vocabulary used by the Nengo model for this run."""
    spa_vocab = WordsToSPAVocab(vocab)

    if not mp.strict_vocab:
        seed_vocab_vectors = {
            token: seed_vocab_model.wv.get_vector(token)
            for token in spa_vocab
            if token != mp.pad_token
        }
    else:
        seed_vocab_vectors = {
            token: seed_vocab_model.wv.get_vector(token)
            for token in spa_vocab
            if token not in (mp.pad_token, mp.unknown_token)
        }

    # Seed the runtime vocabulary from the Word2Vec model, then add the
    # special vectors the architecture expects explicitly.
    start = perf_counter()
    model_vocab = spa.Vocabulary(
        dimensions=mp.rep_vocab_dim,
        strict=mp.strict_vocab,
        pointer_gen=None,
        max_similarity=mp.rep_vocab_max_sim,
    )
    pos_vec = make_unitary(dim=mp.rep_vocab_dim)
    model_vocab.add("POS", pos_vec)

    for key, pointer in seed_vocab_vectors.items():
        model_vocab.add(key=key, p=pointer)

    model_vocab.add(key=mp.pad_token, p=np.zeros(mp.rep_vocab_dim))
    timings["Vocabulary build"] = perf_counter() - start
    return model_vocab


def build_runtime(
    model_vocab,
    timings,
    opencl_platform_index=None,
    opencl_device_index=None,
):
    """Build the model, select an OpenCL device, and compile the simulator."""
    start = perf_counter()
    model_result = nc.Model(
        sub_lengths=[1, mp.context_length], # model architecture arguments here
        model_vocab=model_vocab,
        strict=mp.strict_vocab,
    )
    timings["Model build"] = perf_counter() - start

    # Device selection is shared with benchmarks so normal runs and profiling
    # runs use the same indexing rules and environment-variable fallbacks.
    opencl_selection = select_opencl_device(
        platform_index=opencl_platform_index,
        device_index=opencl_device_index,
    )
    print_opencl_selection(opencl_selection)
    platform = opencl_selection["platform"]
    device = opencl_selection["device"]
    context = opencl_selection["context"]

    start = perf_counter()
    sim = nengo_ocl.Simulator(
        model_result.model,
        context=context,
        progress_bar=False,
    )
    timings["Simulator compile"] = perf_counter() - start

    runtime = ModelRuntime(
        model_result,
        sim,
        model_vocab,
        step_time=0.02,
    )
    return runtime, model_result, platform, device, opencl_selection


def run_demo_predictions(runtime, testing_set, max_examples, top_k):
    """Print a small qualitative sample of next-token predictions."""
    print("\nSample predictions:\n")

    demo_count = 0
    for tokens in testing_set:
        if len(tokens) < 2:
            continue

        for index in range(len(tokens) - 1):
            prefix = tokens[: index + 1]
            target = tokens[index + 1]
            predictions = runtime.predict_next_sequence(prefix, top_k=top_k)
            prediction_text = ", ".join(
                f"{word} ({score:.3f})" for word, score in predictions
            )
            print(f"{' '.join(prefix)} -> {prediction_text} | target: {target}")

            demo_count += 1
            if demo_count >= max_examples:
                return


def save_run_telemetry(
    runtime,
    model_result,
    platform,
    device,
    opencl_selection,
    timings,
    train_test,
    max_examples,
    training_invocations_before,
    training_invocations_after,
    evaluation_invocations_after,
):
    """Persist the telemetry payload for a normal workflow run."""
    complexity = {
        "network": network_telemetry(model_result.model),
        "operators": operator_telemetry(runtime.sim),
    }
    invocation_estimates = {
        "training": training_invocation_estimate(train_test.training_set),
        "evaluation": evaluation_invocation_estimate(
            train_test.testing_set,
            max_examples=max_examples,
        ),
    }

    # Store enough context to compare runs later without reopening the code:
    # environment, model shape, timings, and both estimated and actual sim use.
    telemetry_path = save_telemetry(
        RESULTS_DIR,
        {
            "kind": "model_run",
            "environment": {
                **environment_telemetry(),
                "opencl_platform": platform.name,
                "opencl_device": device.name,
                "opencl_platform_index": opencl_selection["platform_index"],
                "opencl_device_index": opencl_selection["device_index"],
            },
            "parameters": {
                "sub_lengths": model_result.sub_lengths,
                "context_length": mp.context_length,
                "rep_vocab_dim": mp.rep_vocab_dim,
                "training_restriction": mp.training_restriction,
                "testing_restriction": mp.testing_restriction,
            },
            "timings_seconds": timings,
            "complexity": complexity,
            "invocation_estimates": invocation_estimates,
            "actual_simulator_invocations": {
                "training": invocation_delta(
                    training_invocations_after,
                    training_invocations_before,
                ),
                "evaluation": invocation_delta(
                    evaluation_invocations_after,
                    training_invocations_after,
                ),
                "total": evaluation_invocations_after,
            },
        },
    )
    print(f"\nSaved run telemetry to: {telemetry_path}")


def maybe_save_run_telemetry(
    telemetry_enabled,
    runtime,
    model_result,
    platform,
    device,
    opencl_selection,
    timings,
    train_test,
    max_examples,
    training_invocations_before,
    training_invocations_after,
    evaluation_invocations_after,
):
    """Honor the telemetry toggle while keeping the call site in main.py simple."""
    if not telemetry_enabled:
        print("\nTelemetry recording disabled for this run.")
        return

    save_run_telemetry(
        runtime,
        model_result,
        platform,
        device,
        opencl_selection,
        timings,
        train_test,
        max_examples,
        training_invocations_before,
        training_invocations_after,
        evaluation_invocations_after,
    )
