import sys
from pathlib import Path
from time import perf_counter

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.args import BENCHMARK_MODE_MAP, parse_args, resolve_workflow
from app.workflow import (
    build_model_vocab,
    build_runtime,
    build_train_test,
    load_seed_vocab_model,
    maybe_save_run_telemetry,
    print_timing,
    run_demo_predictions,
)
from utils.benchmark_compile import benchmark as run_compile_benchmark
from utils.eval import evaluate_model


def main():
    args = parse_args()

    if args.benchmark:
        run_compile_benchmark(BENCHMARK_MODE_MAP[args.benchmark])
        return

    workflow = resolve_workflow(args)
    timings = {}

    seed_vocab_model = load_seed_vocab_model()
    train_test = build_train_test(timings)
    model_vocab = build_model_vocab(seed_vocab_model, train_test.vocab, timings)
    runtime, model_result, platform, device = build_runtime(model_vocab, timings)

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
    elif workflow["eval"] or workflow["demo"] or workflow["interactive"]:
        runtime.load_checkpoint(args.checkpoint_path)

    evaluation_invocations_after = training_invocations_after
    if workflow["eval"]:
        start = perf_counter()
        evaluate_model(
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
        timings,
        train_test,
        args.max_examples,
        training_invocations_before,
        training_invocations_after,
        evaluation_invocations_after,
    )

    if workflow["demo"]:
        run_demo_predictions(
            runtime,
            train_test.testing_set,
            max_examples=args.max_demo_examples,
            top_k=args.top_k,
        )

    if workflow["interactive"]:
        runtime.interactive_loop(
            top_k=args.top_k,
            generate=args.generate,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
