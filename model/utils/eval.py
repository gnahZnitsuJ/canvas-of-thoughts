# model evaluation utilities


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

        for i in range(len(tokens) - 1):
            prefix = tokens[: i + 1]
            target = tokens[i + 1]

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


# Default evaluation now uses the corrected stateful context path.
def evaluate_model_streaming(runtime, test_sequences, max_examples=50, top_k=3):
    correct = 0
    total = 0

    print("\nEvaluating model with streaming state...")

    for tokens in test_sequences:
        for result in iter_next_token_predictions(runtime, tokens, top_k=top_k):
            preds = result["predictions"]

            if not preds:
                continue

            predicted_word = preds[0][0]
            if predicted_word == result["target"]:
                correct += 1

            total += 1

            if total >= max_examples:
                break

        if total >= max_examples:
            break

    accuracy = correct / total if total > 0 else 0.0

    print(f"\nTest Accuracy (top-1, {total} samples): {accuracy:.3f}")
    return accuracy


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
