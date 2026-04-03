# model evaluation utilities

# evaluate model predictions against test set sequences, with optional top-k accuracy
def evaluate_model(trainer, test_sequences, max_examples=50, top_k=3):
    correct = 0
    total = 0

    print("\nEvaluating model...")
    
    for tokens in test_sequences:
        if len(tokens) < 2:
            continue

        for i in range(len(tokens) - 1):
            prefix = tokens[:i+1]
            target = tokens[i+1]

            preds = trainer.predict_next_sequence(prefix, top_k=top_k)

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