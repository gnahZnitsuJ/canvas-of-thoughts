"""Calibration helpers for scheduled-training token duration."""

from time import perf_counter

from utils.eval import evaluate_model_streaming_metrics

DEFAULT_TOKEN_DURATION_CANDIDATES = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]


def _invocation_delta(after, before):
    return {
        key: after[key] - before[key]
        for key in after
    }


def _subset_sequences(sequences, limit):
    return [tokens for tokens in sequences if len(tokens) > 1][:limit]


def _reset_runtime_for_candidate(runtime, initial_weights):
    """Return the runtime to a repeatable pre-training state."""
    reset_method = getattr(runtime.sim, "reset", None)
    if callable(reset_method):
        try:
            reset_method()
        except Exception:
            pass

    runtime.restore_learning_weights(initial_weights)
    runtime.clear_scheduled_inputs()
    runtime.reset_context()


def calibrate_token_duration(
    runtime,
    train_sequences,
    test_sequences,
    candidates=None,
    baseline_duration=0.02,
    calibration_train_sequences=2,
    calibration_eval_examples=50,
    top_k=3,
    max_relative_similarity_drop=0.10,
    max_absolute_similarity_drop=0.02,
):
    """Choose the fastest scheduled-training token duration that stays stable."""
    candidate_values = list(candidates or DEFAULT_TOKEN_DURATION_CANDIDATES)
    if baseline_duration not in candidate_values:
        candidate_values.append(baseline_duration)
    candidate_values = sorted(set(float(value) for value in candidate_values))

    train_subset = _subset_sequences(train_sequences, calibration_train_sequences)
    if not train_subset:
        raise ValueError("Calibration requires at least one train sequence with length > 1")

    initial_weights = runtime.snapshot_learning_weights()
    previous_config = runtime.training_configuration()
    candidate_results = []

    try:
        for duration in candidate_values:
            _reset_runtime_for_candidate(runtime, initial_weights)
            runtime.configure_training(
                training_mode="scheduled",
                token_duration=duration,
                token_duration_source="calibration",
            )

            before_training = runtime.simulator_invocation_telemetry()
            train_start = perf_counter()
            runtime.train_corpus_scheduled(
                train_subset,
                token_duration=duration,
            )
            training_wall_seconds = perf_counter() - train_start
            after_training = runtime.simulator_invocation_telemetry()

            eval_start = perf_counter()
            metrics = evaluate_model_streaming_metrics(
                runtime,
                test_sequences,
                max_examples=calibration_eval_examples,
                top_k=top_k,
            )
            evaluation_wall_seconds = perf_counter() - eval_start
            after_evaluation = runtime.simulator_invocation_telemetry()

            candidate_results.append(
                {
                    "token_duration": duration,
                    "metrics": metrics,
                    "training_invocations": _invocation_delta(
                        after_training,
                        before_training,
                    ),
                    "evaluation_invocations": _invocation_delta(
                        after_evaluation,
                        after_training,
                    ),
                    "training_wall_seconds": training_wall_seconds,
                    "evaluation_wall_seconds": evaluation_wall_seconds,
                }
            )

        baseline_result = next(
            result
            for result in candidate_results
            if result["token_duration"] == baseline_duration
        )
        baseline_similarity = baseline_result["metrics"]["mean_target_similarity"]
        baseline_topk = baseline_result["metrics"].get(f"top{top_k}_accuracy", 0.0)
        allowed_drop = max(
            max_absolute_similarity_drop,
            abs(baseline_similarity) * max_relative_similarity_drop,
        )

        selected_duration = baseline_duration
        for result in candidate_results:
            duration = result["token_duration"]
            if duration >= baseline_duration:
                break

            candidate_similarity = result["metrics"]["mean_target_similarity"]
            candidate_ok = candidate_similarity >= baseline_similarity - allowed_drop

            if baseline_topk > 0:
                candidate_ok = candidate_ok and (
                    result["metrics"].get(f"top{top_k}_accuracy", 0.0)
                    >= max(0.0, baseline_topk - 0.05)
                )

            if candidate_ok:
                selected_duration = duration
                break

        return {
            "selected_token_duration": selected_duration,
            "baseline_duration": baseline_duration,
            "candidates": candidate_results,
            "selection_rule": {
                "max_relative_similarity_drop": max_relative_similarity_drop,
                "max_absolute_similarity_drop": max_absolute_similarity_drop,
            },
        }
    finally:
        _reset_runtime_for_candidate(runtime, initial_weights)
        runtime.configure_training(
            training_mode=previous_config["training_mode"],
            token_duration=previous_config["token_duration"],
            token_duration_source=previous_config["token_duration_source"],
        )