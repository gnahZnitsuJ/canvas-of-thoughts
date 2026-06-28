"""Calibration helpers for scheduled-training token duration."""

from time import perf_counter

import numpy as np

from utils.eval import evaluate_model_streaming_metrics, iter_next_token_predictions

# Reference-equivalence calibration compares scheduled-training updates against
# the current single-pass stepwise learning behavior. The selector is phrased in
# simulator timesteps so every tested duration is an exact multiple of dt.
MIN_MEAN_DELTA_COSINE_SIMILARITY = 0.98
MAX_RELATIVE_DELTA_ERROR = 0.10
_EPSILON = 1e-12


def _invocation_delta(after, before):
    """Compute a per-phase delta from cumulative simulator counters."""
    return {
        key: after[key] - before[key]
        for key in after
    }


def _subset_sequences(sequences, limit):
    """Pick a stable, order-preserving subset for calibration."""
    return [tokens for tokens in sequences if len(tokens) > 1][:limit]


def _resolve_simulator_dt(runtime):
    """Find the simulator timestep and record where it came from."""
    dt = getattr(runtime.sim, "dt", None)
    if dt is not None:
        return float(dt), "sim.dt"

    model = getattr(runtime.sim, "model", None)
    dt = getattr(model, "dt", None)
    if dt is not None:
        return float(dt), "sim.model.dt"

    return float(runtime.step_time), "runtime.step_time_fallback"


def _candidate_k_values(candidates, baseline_k):
    """Resolve the integer timestep multipliers to test."""
    if baseline_k < 1:
        raise ValueError("baseline_k must be at least 1")

    if candidates is None:
        return list(range(1, baseline_k + 1))

    candidate_k_values = sorted(
        {
            int(value)
            for value in candidates
            if int(value) >= 1 and int(value) <= baseline_k
        }
    )
    if baseline_k not in candidate_k_values:
        candidate_k_values.append(baseline_k)

    return candidate_k_values


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


def _weight_deltas(initial_weights, final_weights):
    """Compute per-connection learning deltas."""
    return [
        final_weights[index] - initial_weights[index]
        for index in range(len(initial_weights))
    ]


def _flattened_cosine_similarity(lhs, rhs):
    """Cosine similarity between flattened arrays with zero-safe handling."""
    lhs_vector = np.ravel(lhs)
    rhs_vector = np.ravel(rhs)
    lhs_norm = float(np.linalg.norm(lhs_vector))
    rhs_norm = float(np.linalg.norm(rhs_vector))

    if lhs_norm <= _EPSILON and rhs_norm <= _EPSILON:
        return 1.0
    if lhs_norm <= _EPSILON or rhs_norm <= _EPSILON:
        return 0.0

    cosine = float(np.dot(lhs_vector, rhs_vector) / (lhs_norm * rhs_norm))
    return float(np.clip(cosine, -1.0, 1.0))


def _relative_delta_error(reference_delta, candidate_delta):
    """Relative error of a candidate delta against the stepwise reference."""
    reference_norm = float(np.linalg.norm(np.ravel(reference_delta)))
    diff_norm = float(np.linalg.norm(np.ravel(candidate_delta - reference_delta)))

    if reference_norm <= _EPSILON:
        return 0.0 if diff_norm <= _EPSILON else diff_norm / _EPSILON

    return diff_norm / reference_norm


def _delta_comparison(reference_deltas, candidate_deltas):
    """Aggregate similarity/error metrics across learning connections."""
    per_connection = []
    for index, (reference_delta, candidate_delta) in enumerate(
        zip(reference_deltas, candidate_deltas)
    ):
        per_connection.append(
            {
                "connection_index": index,
                "delta_cosine_similarity": _flattened_cosine_similarity(
                    reference_delta,
                    candidate_delta,
                ),
                "relative_delta_error": _relative_delta_error(
                    reference_delta,
                    candidate_delta,
                ),
                "reference_delta_norm": float(
                    np.linalg.norm(np.ravel(reference_delta))
                ),
                "candidate_delta_norm": float(
                    np.linalg.norm(np.ravel(candidate_delta))
                ),
            }
        )

    return {
        "mean_delta_cosine_similarity": float(
            sum(item["delta_cosine_similarity"] for item in per_connection)
            / len(per_connection)
        ),
        "max_relative_delta_error": float(
            max(item["relative_delta_error"] for item in per_connection)
        ),
        "per_connection": per_connection,
    }


def _collect_prediction_vectors(runtime, test_sequences, max_examples, top_k):
    """Capture streaming prediction vectors for optional reference matching."""
    vectors = []
    if max_examples <= 0:
        return vectors

    for tokens in test_sequences:
        for _result in iter_next_token_predictions(runtime, tokens, top_k=top_k):
            vectors.append(runtime.current_prediction_vector().copy())
            if len(vectors) >= max_examples:
                return vectors

    return vectors


def _prediction_vector_cosine_similarity(reference_vectors, candidate_vectors):
    """Average cosine similarity over captured streaming prediction vectors."""
    if not reference_vectors or not candidate_vectors:
        return None

    limit = min(len(reference_vectors), len(candidate_vectors))
    if limit == 0:
        return None

    similarities = [
        _flattened_cosine_similarity(reference_vectors[index], candidate_vectors[index])
        for index in range(limit)
    ]
    return float(sum(similarities) / len(similarities))


def _reference_evaluation(runtime, test_sequences, calibration_eval_examples, top_k):
    """Optionally collect reference evaluation metrics and prediction vectors."""
    if calibration_eval_examples <= 0:
        return None, None, None, None

    before_evaluation = runtime.simulator_invocation_telemetry()
    evaluation_start = perf_counter()
    evaluation_metrics = evaluate_model_streaming_metrics(
        runtime,
        test_sequences,
        max_examples=calibration_eval_examples,
        top_k=top_k,
    )
    prediction_vectors = _collect_prediction_vectors(
        runtime,
        test_sequences,
        max_examples=calibration_eval_examples,
        top_k=top_k,
    )
    evaluation_wall_seconds = perf_counter() - evaluation_start
    after_evaluation = runtime.simulator_invocation_telemetry()

    return (
        evaluation_metrics,
        prediction_vectors,
        evaluation_wall_seconds,
        _invocation_delta(after_evaluation, before_evaluation),
    )


def calibrate_token_duration(
    runtime,
    train_sequences,
    test_sequences,
    candidates=None,
    baseline_duration=0.02,
    calibration_train_sequences=2,
    calibration_eval_examples=50,
    top_k=3,
):
    """Choose the smallest scheduled duration that matches stepwise training."""
    train_subset = _subset_sequences(train_sequences, calibration_train_sequences)
    if not train_subset:
        raise ValueError("Calibration requires at least one train sequence with length > 1")

    simulator_dt, simulator_dt_source = _resolve_simulator_dt(runtime)
    if simulator_dt <= 0:
        raise ValueError("Simulator dt must be positive for calibration")

    baseline_k = max(1, int(round(float(baseline_duration) / simulator_dt)))
    baseline_token_duration = baseline_k * simulator_dt
    candidate_k_values = _candidate_k_values(candidates, baseline_k)

    initial_weights = runtime.snapshot_learning_weights()
    previous_config = runtime.training_configuration()
    reference_result = None
    candidate_results = []

    try:
        _reset_runtime_for_candidate(runtime, initial_weights)
        runtime.configure_training(
            training_mode="single_pass",
            token_duration=runtime.step_time,
            token_duration_source="reference",
        )

        before_reference_training = runtime.simulator_invocation_telemetry()
        reference_train_start = perf_counter()
        runtime.train_corpus(train_subset)
        reference_training_wall_seconds = perf_counter() - reference_train_start
        after_reference_training = runtime.simulator_invocation_telemetry()

        reference_weights = runtime.snapshot_learning_weights()
        reference_deltas = _weight_deltas(initial_weights, reference_weights)
        (
            reference_evaluation_metrics,
            reference_prediction_vectors,
            reference_evaluation_wall_seconds,
            reference_evaluation_invocations,
        ) = _reference_evaluation(
            runtime,
            test_sequences,
            calibration_eval_examples,
            top_k,
        )

        reference_result = {
            "training_mode": "single_pass",
            "token_duration": float(runtime.step_time),
            "training_wall_seconds": reference_training_wall_seconds,
            "training_invocations": _invocation_delta(
                after_reference_training,
                before_reference_training,
            ),
            "evaluation_metrics": reference_evaluation_metrics,
            "evaluation_wall_seconds": reference_evaluation_wall_seconds,
            "evaluation_invocations": reference_evaluation_invocations,
            "prediction_vector_count": (
                len(reference_prediction_vectors)
                if reference_prediction_vectors is not None
                else 0
            ),
        }

        for k_value in candidate_k_values:
            token_duration = k_value * simulator_dt
            _reset_runtime_for_candidate(runtime, initial_weights)
            runtime.configure_training(
                training_mode="scheduled",
                token_duration=token_duration,
                token_duration_source="calibration",
            )

            before_training = runtime.simulator_invocation_telemetry()
            train_start = perf_counter()
            runtime.train_corpus_scheduled(
                train_subset,
                token_duration=token_duration,
            )
            training_wall_seconds = perf_counter() - train_start
            after_training = runtime.simulator_invocation_telemetry()

            candidate_weights = runtime.snapshot_learning_weights()
            candidate_deltas = _weight_deltas(initial_weights, candidate_weights)
            delta_comparison = _delta_comparison(reference_deltas, candidate_deltas)

            (
                evaluation_metrics,
                candidate_prediction_vectors,
                evaluation_wall_seconds,
                evaluation_invocations,
            ) = _reference_evaluation(
                runtime,
                test_sequences,
                calibration_eval_examples,
                top_k,
            )
            prediction_vector_cosine_similarity = _prediction_vector_cosine_similarity(
                reference_prediction_vectors,
                candidate_prediction_vectors,
            )

            passes_thresholds = (
                delta_comparison["mean_delta_cosine_similarity"]
                >= MIN_MEAN_DELTA_COSINE_SIMILARITY
                and delta_comparison["max_relative_delta_error"]
                <= MAX_RELATIVE_DELTA_ERROR
            )

            candidate_results.append(
                {
                    "k": k_value,
                    "token_duration": token_duration,
                    "training_wall_seconds": training_wall_seconds,
                    "training_invocations": _invocation_delta(
                        after_training,
                        before_training,
                    ),
                    "simulated_seconds": (
                        after_training["simulated_seconds"]
                        - before_training["simulated_seconds"]
                    ),
                    "mean_delta_cosine_similarity": delta_comparison[
                        "mean_delta_cosine_similarity"
                    ],
                    "max_relative_delta_error": delta_comparison[
                        "max_relative_delta_error"
                    ],
                    "prediction_vector_cosine_similarity": (
                        prediction_vector_cosine_similarity
                    ),
                    "evaluation_metrics": evaluation_metrics,
                    "evaluation_wall_seconds": evaluation_wall_seconds,
                    "evaluation_invocations": evaluation_invocations,
                    "passes_thresholds": passes_thresholds,
                    "per_connection": delta_comparison["per_connection"],
                }
            )

        selected_candidate = next(
            (result for result in candidate_results if result["passes_thresholds"]),
            None,
        )
        fallback_used = selected_candidate is None
        if selected_candidate is None:
            selected_candidate = next(
                result for result in candidate_results if result["k"] == baseline_k
            )

        return {
            "selector": "reference_equivalence",
            "simulator_dt": simulator_dt,
            "simulator_dt_source": simulator_dt_source,
            "baseline_k": baseline_k,
            "baseline_token_duration": baseline_token_duration,
            "reference_training_mode": "single_pass",
            "reference_step_time": float(runtime.step_time),
            "selected_k": selected_candidate["k"],
            "selected_token_duration": selected_candidate["token_duration"],
            "fallback_used": fallback_used,
            "selection_rule": {
                "minimum_mean_delta_cosine_similarity": (
                    MIN_MEAN_DELTA_COSINE_SIMILARITY
                ),
                "maximum_relative_delta_error": MAX_RELATIVE_DELTA_ERROR,
            },
            "reference": reference_result,
            "candidates": candidate_results,
        }
    finally:
        _reset_runtime_for_candidate(runtime, initial_weights)
        runtime.configure_training(
            training_mode=previous_config["training_mode"],
            token_duration=previous_config["token_duration"],
            token_duration_source=previous_config["token_duration_source"],
        )


