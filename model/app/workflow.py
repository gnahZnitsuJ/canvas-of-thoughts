"""High-level workflow helpers used by the main model CLI entrypoint."""

import cProfile
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import nengo_ocl
import nengo_spa as spa
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import reuters

import components.net_comp as nc
from app.args import resolve_workflow
from app.shell import (
    launch_interactive_prompt,
    launch_runtime_shell,
    run_demo_predictions,
)
from components.runtime import (
    ModelRuntime,
    TRAINING_SEMANTICS_VERSION,
    build_architecture_signature,
    compare_architecture_signatures,
    format_architecture_comparison,
    inspect_checkpoint_metadata,
)
from config import model_parameters as mp
from utils import seed_vocab
from utils.build_config import compile_profile_scope, resolve_compile_profile
from utils.calibration import calibrate_token_duration
from utils.eval import evaluate_model
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


@dataclass(frozen=True)
class CompiledRun:
    """Artifacts that belong to one compiled model runtime."""

    runtime: object
    model_result: object
    opencl_selection: dict
    compile_profile: dict
    compile_fingerprint: dict

    @property
    def platform(self):
        return self.opencl_selection["platform"]

    @property
    def device(self):
        return self.opencl_selection["device"]


@dataclass(frozen=True)
class RunTelemetryContext:
    """Inputs needed to describe one normal runtime execution."""

    compiled: CompiledRun
    timings: dict
    train_test: object
    max_examples: int
    training_invocations_before: dict
    training_invocations_after: dict
    evaluation_invocations_after: dict
    evaluation_result: object = None
    calibration_result: object = None


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


def build_model_result(
    model_vocab,
    timings,
    *,
    probe_mode="debug",
    compile_profile_name="full",
    learned_init_mode="random-function",
    learned_init_seed=None,
    architecture_name="root-context-v1",
):
    """Build the Python-side Nengo model without compiling a simulator."""
    compile_profile_config = resolve_compile_profile(compile_profile_name)

    start = perf_counter()
    with compile_profile_scope(compile_profile_config):
        model_result = nc.Model(
            sub_lengths=[1, mp.context_length],
            model_vocab=model_vocab,
            strict=mp.strict_vocab,
            probe_mode=probe_mode,
            learned_init_mode=learned_init_mode,
            learned_init_seed=learned_init_seed,
            compile_profile_name=compile_profile_config["name"],
            compile_profile_settings=compile_profile_config["settings"],
            architecture_name=architecture_name,
        )
    timings["Model build"] = perf_counter() - start
    return model_result, compile_profile_config


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
    platform=None,
    device=None,
    compile_profile_name="full",
    compile_profile_settings=None,
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
        "opencl_platform": platform.name if platform is not None else None,
        "opencl_device": device.name if device is not None else None,
        "name": compile_profile_name,
        "settings": dict(compile_profile_settings or {}),
        "environment": compile_profile_environment(),
    }


def make_compile_fingerprint(
    model_result,
    compile_profile,
    *,
    opencl_selection=None,
    learned_init_mode="random-function",
    learned_init_seed=None,
):
    """Record the configuration that produced a compile result.

    This scaffolding is broader than checkpoint compatibility. It captures both
    architectural context and workflow/profile knobs so future comparisons do
    not accidentally mix unlike runs.
    """
    opencl_selection = opencl_selection or {}

    return {
        "backend": compile_profile["backend"],
        "opencl_platform": compile_profile["opencl_platform"],
        "opencl_device": compile_profile["opencl_device"],
        "opencl_platform_index": opencl_selection.get("platform_index"),
        "opencl_device_index": opencl_selection.get("device_index"),
        "rep_vocab_dim": mp.rep_vocab_dim,
        "context_length": mp.context_length,
        "strict_vocab": mp.strict_vocab,
        "sub_lengths": model_result.sub_lengths,
        "sub_lengths_mode": "legacy_deferred",
        "architecture_name": model_result.architecture_spec.name,
        "architecture_topology": model_result.architecture_topology_signature,
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "probe_mode": model_result.probe_mode,
        "learned_init_mode": learned_init_mode,
        "learned_init_seed": learned_init_seed,
        "compile_profile": {
            "name": compile_profile["name"],
            "settings": dict(compile_profile.get("settings", {})),
            "first_run_warmup_enabled": compile_profile[
                "first_run_warmup_enabled"
            ],
            "profile_compile_enabled": compile_profile[
                "profile_compile_enabled"
            ],
        },
        "environment": dict(compile_profile["environment"]),
    }


def build_runtime(
    model_vocab,
    timings,
    opencl_platform_index=None,
    opencl_device_index=None,
    step_time=DEFAULT_STEP_TIME,
    first_run_warmup=False,
    profile_compile=False,
    probe_mode="debug",
    compile_profile_name="full",
    learned_init_mode="random-function",
    learned_init_seed=None,
    architecture_name="root-context-v1",
):
    """Build the model, select an OpenCL device, and compile the simulator."""
    model_result, compile_profile_config = build_model_result(
        model_vocab,
        timings,
        probe_mode=probe_mode,
        compile_profile_name=compile_profile_name,
        learned_init_mode=learned_init_mode,
        learned_init_seed=learned_init_seed,
        architecture_name=architecture_name,
    )

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
        compile_profile_name=compile_profile_config["name"],
        compile_profile_settings=compile_profile_config["settings"],
    )
    compile_fingerprint = make_compile_fingerprint(
        model_result,
        compile_profile,
        opencl_selection=opencl_selection,
        learned_init_mode=learned_init_mode,
        learned_init_seed=learned_init_seed,
    )
    runtime.set_compile_fingerprint(compile_fingerprint)
    return CompiledRun(
        runtime=runtime,
        model_result=model_result,
        opencl_selection=opencl_selection,
        compile_profile=compile_profile,
        compile_fingerprint=compile_fingerprint,
    )


def print_dry_run_summary(args, workflow, training_config):
    """Report the resolved workflow plan without building the model."""
    print("\nDry run summary:\n")
    print(f"workflow:                {workflow}")
    print(f"architecture:            {args.architecture}")
    print(f"checkpoint path:         {args.checkpoint_path}")
    print(f"compile profile:         {args.compile_profile}")
    print(f"learned init mode:       {args.learned_init_mode}")
    print(f"learned init seed:       {args.learned_init_seed}")
    print(f"probe mode:              {args.probe_mode}")
    print(f"telemetry enabled:       {not args.no_telemetry}")
    print(f"training mode:           {training_config['training_mode']}")
    print(f"token duration:          {training_config['token_duration']:.6f} sec")
    print(f"token duration source:   {training_config['token_duration_source']}")
    print(f"step time:               {training_config['step_time']:.6f} sec")
    print(f"first-run warmup:        {args.first_run_warmup}")
    print(f"profile compile:         {args.profile_compile}")


def load_checkpoint_metadata(checkpoint_path):
    """Load checkpoint metadata for inspection-oriented workflows."""
    return inspect_checkpoint_metadata(checkpoint_path)


def print_checkpoint_metadata(checkpoint_path, full_path, metadata):
    """Render checkpoint metadata in a readable console summary."""
    print("\nCheckpoint inspection:\n")
    print(f"requested path:          {checkpoint_path}")
    print(f"resolved path:           {full_path}")
    print(f"timestamp:               {metadata.get('timestamp')}")

    architecture = metadata.get("architecture", {})
    compile_fingerprint = metadata.get("compile_fingerprint", {})
    compile_profile = compile_fingerprint.get("compile_profile", {})

    print(f"training semantics:      {architecture.get('training_semantics_version')}")
    print(f"vocab dim:               {architecture.get('vocab_dim')}")
    print(f"context length:          {compile_fingerprint.get('context_length')}")
    print(f"sub_lengths:             {architecture.get('sub_lengths')}")
    print(f"probe mode:              {compile_fingerprint.get('probe_mode')}")
    print(f"compile profile:         {compile_profile.get('name')}")
    print(f"compile settings:        {compile_profile.get('settings')}")
    print(f"learned init mode:       {architecture.get('learned_init_mode')}")
    print(f"learned init seed:       {architecture.get('learned_init_seed')}")
    print(f"backend:                 {compile_fingerprint.get('backend')}")
    print(f"OpenCL platform:         {compile_fingerprint.get('opencl_platform')}")
    print(f"OpenCL device:           {compile_fingerprint.get('opencl_device')}")


def compare_architecture_to_checkpoint(
    model_result,
    model_vocab,
    step_time,
    compile_fingerprint,
    checkpoint_metadata,
):
    """Compare the current no-compile build signature against checkpoint metadata."""
    current_architecture = build_architecture_signature(
        model_result,
        model_vocab,
        step_time,
        compile_fingerprint=compile_fingerprint,
    )
    saved_architecture = checkpoint_metadata.get("architecture", {})
    return compare_architecture_signatures(saved_architecture, current_architecture)


def print_architecture_comparison(comparison):
    """Report whether the current build matches checkpoint architecture metadata."""
    print("\n" + format_architecture_comparison(comparison))


def save_build_only_telemetry(
    model_result,
    model_vocab,
    timings,
    training_config,
    compile_profile_name,
    compile_profile_settings,
    learned_init_mode,
    learned_init_seed,
    checkpoint_comparison=None,
):
    """Persist telemetry for build-only runs that stop before simulator compile."""
    compile_profile = make_compile_profile(
        backend="not_compiled",
        timings=timings,
        first_run_warmup_enabled=False,
        profile_compile_enabled=False,
        profile_output_path=None,
        platform=None,
        device=None,
        compile_profile_name=compile_profile_name,
        compile_profile_settings=compile_profile_settings,
    )
    compile_fingerprint = make_compile_fingerprint(
        model_result,
        compile_profile,
        learned_init_mode=learned_init_mode,
        learned_init_seed=learned_init_seed,
    )

    payload = {
        "kind": "model_build_only",
        "environment": environment_telemetry(),
        "parameters": {
            "architecture_name": model_result.architecture_spec.name,
            "sub_lengths": model_result.sub_lengths,
            "sub_lengths_mode": "legacy_deferred",
            "context_length": mp.context_length,
            "rep_vocab_dim": mp.rep_vocab_dim,
            "probe_mode": model_result.probe_mode,
            "active_context_path": "root_context_module",
            **training_config,
        },
        "probes": {
            "mode": model_result.probe_mode,
            "created_labels": model_result.created_probe_labels,
            "skipped_labels": model_result.skipped_probe_labels,
        },
        "timings_seconds": timings,
        "compile_profile": compile_profile,
        "compile_fingerprint": compile_fingerprint,
        "complexity": {
            "network": network_telemetry(model_result.model),
        },
        "architecture_signature": build_architecture_signature(
            model_result,
            model_vocab,
            training_config["step_time"],
            compile_fingerprint=compile_fingerprint,
        ),
    }
    if checkpoint_comparison is not None:
        payload["checkpoint_comparison"] = checkpoint_comparison

    telemetry_path = save_telemetry(RESULTS_DIR, payload)
    print(f"\nSaved build-only telemetry to: {telemetry_path}")
    return compile_fingerprint


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


def save_run_telemetry(context):
    """Persist the telemetry payload for a normal workflow run."""
    compiled = context.compiled
    runtime = compiled.runtime
    model_result = compiled.model_result
    opencl_selection = compiled.opencl_selection
    complexity = {
        "network": network_telemetry(model_result.model),
        "operators": operator_telemetry(runtime.sim),
    }
    invocation_estimates = {
        "training": training_invocation_estimate(
            context.train_test.training_set,
            training_mode=runtime.training_mode,
        ),
        "evaluation": evaluation_invocation_estimate(
            context.train_test.testing_set,
            max_examples=context.max_examples,
            evaluation_mode="streaming",
        ),
    }

    payload = {
        "kind": "model_run",
        "environment": {
            **environment_telemetry(),
            "opencl_platform": compiled.platform.name,
            "opencl_device": compiled.device.name,
            "opencl_platform_index": opencl_selection["platform_index"],
            "opencl_device_index": opencl_selection["device_index"],
        },
        "parameters": {
            "sub_lengths": model_result.sub_lengths,
            "sub_lengths_mode": "legacy_deferred",
            "context_length": mp.context_length,
            "rep_vocab_dim": mp.rep_vocab_dim,
            "probe_mode": model_result.probe_mode,
            "training_restriction": mp.training_restriction,
            "testing_restriction": mp.testing_restriction,
            "active_context_path": "root_context_module",
            "evaluation_mode": "streaming",
            **runtime.training_configuration(),
        },
        "probes": {
            "mode": model_result.probe_mode,
            "created_labels": model_result.created_probe_labels,
            "skipped_labels": model_result.skipped_probe_labels,
        },
        "timings_seconds": context.timings,
        "compile_profile": compiled.compile_profile,
        "compile_fingerprint": compiled.compile_fingerprint,
        "complexity": complexity,
        "invocation_estimates": invocation_estimates,
        "actual_simulator_invocations": {
            "training": invocation_delta(
                context.training_invocations_after,
                context.training_invocations_before,
            ),
            "evaluation": invocation_delta(
                context.evaluation_invocations_after,
                context.training_invocations_after,
            ),
            "total": context.evaluation_invocations_after,
        },
    }
    if context.evaluation_result is not None:
        payload["evaluation"] = context.evaluation_result
    if context.calibration_result is not None:
        payload["calibration"] = context.calibration_result

    telemetry_path = save_telemetry(RESULTS_DIR, payload)
    print(f"\nSaved run telemetry to: {telemetry_path}")


def maybe_save_run_telemetry(telemetry_enabled, context):
    """Honor the telemetry toggle at the application workflow boundary."""
    if not telemetry_enabled:
        print("\nTelemetry recording disabled for this run.")
        return

    save_run_telemetry(context)


def run_application(args):
    """Execute a resolved non-benchmark CLI workflow."""
    workflow_plan = resolve_workflow(args)
    timings = {}

    runtime_profile = load_requested_runtime_profile(args.use_runtime_profile)
    training_config = resolve_training_configuration(args, runtime_profile)

    if args.calibrate_token_duration:
        if training_config["training_mode"] != "scheduled":
            raise ValueError(
                "--calibrate-token-duration currently requires --train-mode scheduled"
            )
        if not workflow_plan["train"] or not args.force_retrain:
            raise ValueError(
                "--calibrate-token-duration requires an actual retraining run; "
                "use it with --train and --force-retrain"
            )

    if args.dry_run:
        print_dry_run_summary(args, workflow_plan, training_config)
        return

    checkpoint_metadata = None
    if args.inspect_checkpoint:
        try:
            checkpoint_metadata_path, checkpoint_metadata = load_checkpoint_metadata(
                args.checkpoint_path
            )
        except FileNotFoundError:
            if not args.build_only:
                raise
            print(
                f"\nCheckpoint inspection skipped: "
                f"{args.checkpoint_path} does not exist yet."
            )
        else:
            print_checkpoint_metadata(
                args.checkpoint_path,
                checkpoint_metadata_path,
                checkpoint_metadata,
            )
            if not args.build_only:
                return

    seed_vocab_model = load_seed_vocab_model()
    train_test = build_train_test(timings)
    model_vocab = build_model_vocab(seed_vocab_model, train_test.vocab, timings)

    if args.build_only:
        model_result, compile_profile_config = build_model_result(
            model_vocab,
            timings,
            probe_mode=args.probe_mode,
            compile_profile_name=args.compile_profile,
            learned_init_mode=args.learned_init_mode,
            learned_init_seed=args.learned_init_seed,
            architecture_name=args.architecture,
        )
        build_only_fingerprint = {
            "compile_profile": {
                "name": compile_profile_config["name"],
                "settings": compile_profile_config["settings"],
                "first_run_warmup_enabled": False,
                "profile_compile_enabled": False,
            },
            "learned_init_mode": args.learned_init_mode,
            "learned_init_seed": args.learned_init_seed,
        }
        comparison = (
            compare_architecture_to_checkpoint(
                model_result,
                model_vocab,
                training_config["step_time"],
                build_only_fingerprint,
                checkpoint_metadata,
            )
            if checkpoint_metadata is not None and args.compare_current_architecture
            else None
        )
        if not args.no_telemetry:
            save_build_only_telemetry(
                model_result,
                model_vocab,
                timings,
                training_config,
                compile_profile_name=compile_profile_config["name"],
                compile_profile_settings=compile_profile_config["settings"],
                learned_init_mode=args.learned_init_mode,
                learned_init_seed=args.learned_init_seed,
                checkpoint_comparison=comparison,
            )
        else:
            print("\nTelemetry recording disabled for this build-only run.")

        if checkpoint_metadata is not None and args.compare_current_architecture:
            print_architecture_comparison(comparison)

        print("\nRun timings:\n")
        for label, elapsed in timings.items():
            print_timing(label, elapsed)
        return

    compiled = build_runtime(
        model_vocab,
        timings,
        opencl_platform_index=args.opencl_platform_index,
        opencl_device_index=args.opencl_device_index,
        step_time=training_config["step_time"],
        first_run_warmup=args.first_run_warmup,
        profile_compile=args.profile_compile,
        probe_mode=args.probe_mode,
        compile_profile_name=args.compile_profile,
        learned_init_mode=args.learned_init_mode,
        learned_init_seed=args.learned_init_seed,
        architecture_name=args.architecture,
    )
    runtime = compiled.runtime
    runtime.configure_training(
        training_mode=training_config["training_mode"],
        token_duration=training_config["token_duration"],
        token_duration_source=training_config["token_duration_source"],
    )

    calibration_result = None
    if args.calibrate_token_duration:
        start = perf_counter()
        calibration_result, _profile_path = run_token_duration_calibration(
            runtime,
            train_test,
            args,
            compiled.opencl_selection,
        )
        timings["Calibration"] = perf_counter() - start

    training_invocations_before = runtime.simulator_invocation_telemetry()
    training_invocations_after = training_invocations_before

    if workflow_plan["train"]:
        start = perf_counter()
        runtime.train_or_load(
            train_test.training_set,
            checkpoint_path=args.checkpoint_path,
            force_retrain=args.force_retrain,
        )
        timings["Training"] = perf_counter() - start
        training_invocations_after = runtime.simulator_invocation_telemetry()
    elif any(
        workflow_plan[stage]
        for stage in ("eval", "demo", "interactive", "shell")
    ):
        runtime.load_checkpoint(args.checkpoint_path)

    evaluation_result = None
    evaluation_invocations_after = training_invocations_after
    if workflow_plan["eval"]:
        start = perf_counter()
        evaluation_result = evaluate_model(
            runtime,
            train_test.testing_set,
            max_examples=args.max_examples,
            top_k=args.top_k,
        )
        timings["Evaluation"] = perf_counter() - start
        evaluation_invocations_after = runtime.simulator_invocation_telemetry()

    print("\nRun timings:\n")
    for label, elapsed in timings.items():
        print_timing(label, elapsed)

    telemetry_context = RunTelemetryContext(
        compiled=compiled,
        timings=timings,
        train_test=train_test,
        max_examples=args.max_examples,
        training_invocations_before=training_invocations_before,
        training_invocations_after=training_invocations_after,
        evaluation_invocations_after=evaluation_invocations_after,
        evaluation_result=evaluation_result,
        calibration_result=calibration_result,
    )
    maybe_save_run_telemetry(not args.no_telemetry, telemetry_context)

    if workflow_plan["demo"]:
        run_demo_predictions(
            runtime,
            train_test.testing_set,
            max_examples=args.max_demo_examples,
            top_k=args.top_k,
        )

    if workflow_plan["interactive"]:
        launch_interactive_prompt(
            runtime,
            top_k=args.top_k,
            generate=args.generate,
            max_tokens=args.max_tokens,
        )

    if workflow_plan["shell"]:
        launch_runtime_shell(
            runtime,
            train_test.testing_set,
            args.checkpoint_path,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            max_examples=args.max_examples,
            max_demo_examples=args.max_demo_examples,
        )

