# model evaluation utilities

import numpy as np


def iter_next_token_predictions(runtime, tokens, top_k=3):
    """Yield stateful next-token predictions for one sequence.

    This is the streaming path: reset once, present each token once, and score
    the model's current prediction against the following token.
    """
    if len(tokens) < 2:
        return

    runtime.reset_context()
    prefix = []

    for index in range(len(tokens) - 1):
        token = tokens[index]
        target = tokens[index + 1]
        prefix.append(token)
        predictions = runtime.advance_recall(token, top_k=top_k)

        yield {
            "prefix": tuple(prefix),
            "target": target,
            "predictions": predictions,
        }


# Keep the old replay-based evaluator around for comparison and regression checks.
def evaluate_model_prefix_replay(runtime, test_sequences, max_examples=50, top_k=3):
    correct = 0
    total = 0

    print("\nEvaluating model with prefix replay...")

    for tokens in test_sequences:
        if len(tokens) < 2:
            continue

        for index in range(len(tokens) - 1):
            prefix = tokens[: index + 1]
            target = tokens[index + 1]

            preds = runtime.predict_next_sequence(prefix, top_k=top_k)

            if not preds:
                continue

            predicted_word = preds[0][0]

            if predicted_word == target:
                correct += 1

            total += 1

            if total >= max_examples:
                break

        if total >= max_examples:
            break

    accuracy = correct / total if total > 0 else 0.0

    print(f"\nTest Accuracy (top-1, {total} samples): {accuracy:.3f}")
    return accuracy


def evaluate_model_streaming_metrics(
    runtime,
    test_sequences,
    max_examples=50,
    top_k=3,
):
    """Compute richer streaming metrics for calibration and comparisons."""
    correct_top1 = 0
    correct_topk = 0
    total = 0
    target_similarities = []
    top_scores = []

    for tokens in test_sequences:
        for result in iter_next_token_predictions(runtime, tokens, top_k=top_k):
            predictions = result["predictions"]
            if not predictions:
                continue

            target = result["target"]
            predicted_words = [word for word, _ in predictions]

            if predicted_words[0] == target:
                correct_top1 += 1

            if target in predicted_words:
                correct_topk += 1

            top_scores.append(predictions[0][1])

            pred_vec = runtime.current_prediction_vector()
            target_vec = runtime._vector_for(target)
            denominator = max(
                np.linalg.norm(pred_vec) * np.linalg.norm(target_vec),
                1e-12,
            )
            target_similarities.append(float((pred_vec @ target_vec) / denominator))

            total += 1
            if total >= max_examples:
                break

        if total >= max_examples:
            break

    topk_key = f"top{top_k}_accuracy"
    return {
        "total": total,
        "top1_accuracy": correct_top1 / total if total else 0.0,
        topk_key: correct_topk / total if total else 0.0,
        "mean_target_similarity": (
            sum(target_similarities) / len(target_similarities)
            if target_similarities
            else 0.0
        ),
        "mean_top_score": sum(top_scores) / len(top_scores) if top_scores else 0.0,
    }


# Default evaluation now uses the corrected stateful context path.
def evaluate_model_streaming(runtime, test_sequences, max_examples=50, top_k=3):
    """Evaluate the streaming path and return the collected metrics."""
    metrics = evaluate_model_streaming_metrics(
        runtime,
        test_sequences,
        max_examples=max_examples,
        top_k=top_k,
    )

    print("\nEvaluating model with streaming state...")
    print(
        f"\nTest Accuracy (top-1, {metrics['total']} samples): "
        f"{metrics['top1_accuracy']:.3f}"
    )
    return metrics


def evaluate_model(runtime, test_sequences, max_examples=50, top_k=3):
    """Default evaluator used by the main workflow.

    Streaming evaluation is now the default because the root context module is
    the canonical state path.
    """
    return evaluate_model_streaming(
        runtime,
        test_sequences,
        max_examples=max_examples,
        top_k=top_k,
    )
