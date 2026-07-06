import sys
from pathlib import Path
from time import perf_counter

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.args import BENCHMARK_MODE_MAP, parse_args, resolve_workflow
from app.shell import launch_interactive_prompt, launch_runtime_shell
from app.workflow import (
    build_model_vocab,
    build_model_result,
    build_runtime,
    build_train_test,
    compare_architecture_to_checkpoint,
    load_checkpoint_metadata,
    load_requested_runtime_profile,
    load_seed_vocab_model,
    maybe_save_run_telemetry,
    print_architecture_comparison,
    print_checkpoint_metadata,
    print_dry_run_summary,
    print_timing,
    resolve_training_configuration,
    run_demo_predictions,
    run_token_duration_calibration,
    save_build_only_telemetry,
)
from utils.benchmark_compile import benchmark as run_compile_benchmark
from utils.eval import evaluate_model


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
            repeats=args.benchmark_repeats,
            include_first_run_warmup=args.include_first_run_warmup,
        )
        return

    workflow = resolve_workflow(args)
    timings = {}

    runtime_profile = load_requested_runtime_profile(args.use_runtime_profile)
    training_config = resolve_training_configuration(args, runtime_profile)

    if args.calibrate_token_duration:
        if training_config["training_mode"] != "scheduled":
            raise ValueError(
                "--calibrate-token-duration currently requires --train-mode scheduled"
            )
        if not workflow["train"] or not args.force_retrain:
            raise ValueError(
                "--calibrate-token-duration requires an actual retraining run; "
                "use it with --train and --force-retrain"
            )

    if args.dry_run:
        print_dry_run_summary(args, workflow, training_config)
        return

    checkpoint_metadata = None
    checkpoint_metadata_path = None
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

    (
        runtime,
        model_result,
        platform,
        device,
        opencl_selection,
        compile_profile,
        compile_fingerprint,
    ) = build_runtime(
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
    )
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
            opencl_selection,
        )
        timings["Calibration"] = perf_counter() - start

    training_invocations_before = runtime.simulator_invocation_telemetry()
    training_invocations_after = training_invocations_before

    if workflow["train"]:
        start = perf_counter()
        runtime.train_or_load(
            train_test.training_set,
            checkpoint_path=args.checkpoint_path,
            force_retrain=args.force_retrain,
        )
        timings["Training"] = perf_counter() - start
        training_invocations_after = runtime.simulator_invocation_telemetry()
    elif workflow["eval"] or workflow["demo"] or workflow["interactive"] or workflow["shell"]:
        runtime.load_checkpoint(args.checkpoint_path)

    evaluation_result = None
    evaluation_invocations_after = training_invocations_after
    if workflow["eval"]:
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

    maybe_save_run_telemetry(
        not args.no_telemetry,
        runtime,
        model_result,
        platform,
        device,
        opencl_selection,
        timings,
        train_test,
        args.max_examples,
        training_invocations_before,
        training_invocations_after,
        evaluation_invocations_after,
        compile_profile,
        compile_fingerprint,
        evaluation_result=evaluation_result,
        calibration_result=calibration_result,
    )

    if workflow["demo"]:
        run_demo_predictions(
            runtime,
            train_test.testing_set,
            max_examples=args.max_demo_examples,
            top_k=args.top_k,
        )

    if workflow["interactive"]:
        launch_interactive_prompt(
            runtime,
            top_k=args.top_k,
            generate=args.generate,
            max_tokens=args.max_tokens,
        )

    if workflow["shell"]:
        launch_runtime_shell(
            runtime,
            train_test.testing_set,
            args.checkpoint_path,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            max_examples=args.max_examples,
            max_demo_examples=args.max_demo_examples,
        )


if __name__ == "__main__":
    main()
