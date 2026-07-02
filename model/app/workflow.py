"""High-level workflow helpers used by the main model CLI entrypoint."""

import cProfile
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import nengo_ocl
import nengo_spa as spa
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import reuters

import components.net_comp as nc
from components.runtime import ModelRuntime, TRAINING_SEMANTICS_VERSION
from config import model_parameters as mp
from utils import seed_vocab
from utils.calibration import calibrate_token_duration
from utils.eval import iter_next_token_predictions
from utils.input import make_unitary
from utils.opencl import print_opencl_selection, select_opencl_device
from utils.processing import WordsToSPAVocab
from utils.runtime_profile import (
    default_runtime_profile,
    load_runtime_profile,
    save_runtime_profile,
)
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
DEFAULT_STEP_TIME = 0.02
COMPILE_PROFILE_ENV_VARS = (
    "PYOPENCL_NO_CACHE",
    "PYOPENCL_COMPILER_OUTPUT",
    "PYOPENCL_BUILD_OPTIONS",
    "PYOPENCL_CTX",
    "CANVAS_OPENCL_PLATFORM_INDEX",
    "CANVAS_OPENCL_DEVICE_INDEX",
)


def print_timing(label, elapsed):
    """Print one timing line using the same alignment as the rest of the CLI."""
    print(f"{label + ':':<23}{elapsed:.3f} sec")


def invocation_delta(after, before):
    """Compute per-phase deltas from cumulative runtime invocation counters."""
    return {key: after[key] - before[key] for key in after}


def load_seed_vocab_model():
    """Load the cached seed Word2Vec model, generating it once if needed."""
    if SEED_VOCAB_PATH.is_file():
        print("seed_vocab.model exists.")
        return Word2Vec.load(str(SEED_VOCAB_PATH))

    print("seed_vocab.model does NOT exist.")
    print("Generating seed vocabulary model...")
    seed_vocab.generate_seed_vocab(DATASETS)

    if not SEED_VOCAB_PATH.is_file():
        raise FileNotFoundError(f"Failed to generate {SEED_VOCAB_PATH}")

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


def load_requested_runtime_profile(use_runtime_profile):
    """Load the optional local runtime profile when requested by the user."""
    if not use_runtime_profile:
        return None

    profile = load_runtime_profile(BASE_DIR)
    if profile is None:
        print("\nRuntime profile requested, but model/config/runtime_profile.json was not found.")
        return None

    print("\nLoaded runtime profile from model/config/runtime_profile.json")
    return profile


def resolve_training_configuration(args, runtime_profile=None):
    """Resolve training mode and token-duration defaults from CLI/profile."""
    profile_training = runtime_profile.get("training", {}) if runtime_profile else {}
    profile_runtime = runtime_profile.get("runtime", {}) if runtime_profile else {}

    step_time = float(profile_runtime.get("default_step_time", DEFAULT_STEP_TIME))
    training_mode = args.train_mode.replace("-", "_")

    if args.token_duration is not None:
        token_duration = float(args.token_duration)
        token_duration_source = "cli"
    elif runtime_profile is not None and profile_training.get("token_duration") is not None:
        token_duration = float(profile_training["token_duration"])
        token_duration_source = profile_training.get("token_duration_source", "profile")
    else:
        token_duration = step_time
        token_duration_source = "default"

    return {
        "training_mode": training_mode,
        "token_duration": token_duration,
        "token_duration_source": token_duration_source,
        "step_time": step_time,
    }


def compile_profile_artifact_path():
    """Local path for an optional simulator-construction cProfile artifact."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return RESULTS_DIR / f"compile_profile_{timestamp}.prof"


def compile_profile_environment():
    """Capture environment fields that often affect OpenCL build behavior."""
    return {
        name: os.getenv(name)
        for name in COMPILE_PROFILE_ENV_VARS
    }


def construct_simulator(model, context, profile_compile=False):
    """Construct the simulator, optionally wrapping the build in cProfile."""
    profile_output_path = None

    if profile_compile:
        profiler = cProfile.Profile()
        start = perf_counter()
        profiler.enable()
        try:
            simulator = nengo_ocl.Simulator(
                model,
                context=context,
                progress_bar=False,
            )
        finally:
            profiler.disable()
        simulator_construct_seconds = perf_counter() - start
        profile_output_path = compile_profile_artifact_path()
        profiler.dump_stats(str(profile_output_path))
        return simulator, simulator_construct_seconds, profile_output_path

    start = perf_counter()
    simulator = nengo_ocl.Simulator(
        model,
        context=context,
        progress_bar=False,
    )
    simulator_construct_seconds = perf_counter() - start
    return simulator, simulator_construct_seconds, profile_output_path


def run_first_run_warmup(simulator, step_time):
    """Separate first-step backend startup from constructor timing.

    We run one simulator step so kernel startup and lazy backend work can be
    measured explicitly, then reset back to the initial state before normal use.
    """
    run_steps = getattr(simulator, "run_steps", None)
    if callable(run_steps):
        run_steps(1)
    else:
        warmup_dt = float(getattr(simulator, "dt", step_time))
        simulator.run(warmup_dt)

    reset_method = getattr(simulator, "reset", None)
    if callable(reset_method):
        reset_method()


def make_compile_profile(
    backend,
    timings,
    first_run_warmup_enabled,
    profile_compile_enabled,
    profile_output_path,
    platform,
    device,
):
    """Assemble the compile-phase telemetry block for normal workflow runs."""
    return {
        "backend": backend,
        "model_build_seconds": timings.get("Model build"),
        "simulator_construct_seconds": timings.get("Simulator compile"),
        "first_run_warmup_seconds": timings.get("First-run warmup"),
        "first_run_warmup_enabled": first_run_warmup_enabled,
        "profile_compile_enabled": profile_compile_enabled,
        "profile_output_path": (
            str(profile_output_path)
            if profile_output_path is not None
            else None
        ),
        "opencl_platform": platform.name,
        "opencl_device": device.name,
        "environment": compile_profile_environment(),
    }


def build_runtime(
    model_vocab,
    timings,
    opencl_platform_index=None,
    opencl_device_index=None,
    step_time=DEFAULT_STEP_TIME,
    first_run_warmup=False,
    profile_compile=False,
):
    """Build the model, select an OpenCL device, and compile the simulator."""
    start = perf_counter()
    model_result = nc.Model(
        sub_lengths=[1, mp.context_length],
        model_vocab=model_vocab,
        strict=mp.strict_vocab,
    )
    timings["Model build"] = perf_counter() - start

    opencl_selection = select_opencl_device(
        platform_index=opencl_platform_index,
        device_index=opencl_device_index,
    )
    print_opencl_selection(opencl_selection)
    platform = opencl_selection["platform"]
    device = opencl_selection["device"]
    context = opencl_selection["context"]

    sim, simulator_construct_seconds, profile_output_path = construct_simulator(
        model_result.model,
        context,
        profile_compile=profile_compile,
    )
    timings["Simulator compile"] = simulator_construct_seconds

    if first_run_warmup:
        warmup_start = perf_counter()
        run_first_run_warmup(sim, step_time)
        timings["First-run warmup"] = perf_counter() - warmup_start

    runtime = ModelRuntime(
        model_result,
        sim,
        model_vocab,
        step_time=step_time,
    )
    compile_profile = make_compile_profile(
        backend="nengo_ocl",
        timings=timings,
        first_run_warmup_enabled=first_run_warmup,
        profile_compile_enabled=profile_compile,
        profile_output_path=profile_output_path,
        platform=platform,
        device=device,
    )
    return runtime, model_result, platform, device, opencl_selection, compile_profile


def run_demo_predictions(runtime, testing_set, max_examples, top_k):
    """Print a small qualitative sample of streaming next-token predictions."""
    print("\nSample predictions:\n")

    demo_count = 0
    for tokens in testing_set:
        for result in iter_next_token_predictions(runtime, tokens, top_k=top_k):
            prediction_text = ", ".join(
                f"{word} ({score:.3f})"
                for word, score in result["predictions"]
            )
            print(
                f"{' '.join(result['prefix'])} -> {prediction_text} "
                f"| target: {result['target']}"
            )

            demo_count += 1
            if demo_count >= max_examples:
                return


def save_calibrated_runtime_profile(calibration_result, runtime, opencl_selection):
    """Persist a machine-local runtime profile from the latest calibration."""
    profile = default_runtime_profile()
    profile["training"].update(
        {
            "mode": "scheduled",
            "scheduled_training_enabled": True,
            "token_duration": calibration_result["selected_token_duration"],
            "token_duration_source": "calibrated",
            "calibrated": True,
            "training_semantics_version": TRAINING_SEMANTICS_VERSION,
            "calibration": {
                "timestamp": datetime.now().astimezone().isoformat(),
                **calibration_result,
            },
        }
    )
    profile["runtime"]["default_step_time"] = runtime.step_time
    profile["opencl"] = {
        "platform_index": opencl_selection["platform_index"],
        "device_index": opencl_selection["device_index"],
    }

    return save_runtime_profile(BASE_DIR, profile)


def run_token_duration_calibration(runtime, train_test, args, opencl_selection):
    """Calibrate the scheduled-training token duration and save the profile."""
    calibration_result = calibrate_token_duration(
        runtime,
        train_test.training_set,
        train_test.testing_set,
        candidates=args.calibration_candidates,
        baseline_duration=runtime.step_time,
        calibration_train_sequences=args.calibration_train_sequences,
        calibration_eval_examples=args.calibration_eval_examples,
        top_k=args.top_k,
    )
    runtime.configure_training(
        training_mode="scheduled",
        token_duration=calibration_result["selected_token_duration"],
        token_duration_source="calibrated",
    )
    profile_path = save_calibrated_runtime_profile(
        calibration_result,
        runtime,
        opencl_selection,
    )

    print(
        "\nReference-equivalence calibration selected token duration: "
        f"{calibration_result['selected_token_duration']:.3f} sec "
        f"(k={calibration_result['selected_k']}, dt={calibration_result['simulator_dt']:.6f})"
    )
    print(f"Saved runtime profile to: {profile_path}")
    return calibration_result, profile_path


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
    compile_profile,
    evaluation_result=None,
    calibration_result=None,
):
    """Persist the telemetry payload for a normal workflow run."""
    complexity = {
        "network": network_telemetry(model_result.model),
        "operators": operator_telemetry(runtime.sim),
    }
    invocation_estimates = {
        "training": training_invocation_estimate(
            train_test.training_set,
            training_mode=runtime.training_mode,
        ),
        "evaluation": evaluation_invocation_estimate(
            train_test.testing_set,
            max_examples=max_examples,
            evaluation_mode="streaming",
        ),
    }

    payload = {
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
            "sub_lengths_mode": "legacy_deferred",
            "context_length": mp.context_length,
            "rep_vocab_dim": mp.rep_vocab_dim,
            "training_restriction": mp.training_restriction,
            "testing_restriction": mp.testing_restriction,
            "active_context_path": "root_context_module",
            "evaluation_mode": "streaming",
            **runtime.training_configuration(),
        },
        "timings_seconds": timings,
        "compile_profile": compile_profile,
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
    }
    if evaluation_result is not None:
        payload["evaluation"] = evaluation_result
    if calibration_result is not None:
        payload["calibration"] = calibration_result

    telemetry_path = save_telemetry(RESULTS_DIR, payload)
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
    compile_profile,
    evaluation_result=None,
    calibration_result=None,
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
        compile_profile,
        evaluation_result=evaluation_result,
        calibration_result=calibration_result,
    )
