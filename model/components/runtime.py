import numpy as np
from utils.processing import WordsToSPAVocab, SPAVocabToWords
from tqdm import tqdm
import pickle
import os
from datetime import datetime

# helper class for running the model
class ModelRuntime:
    def __init__(self, model_result, sim, model_vocab, step_time=0.02):
        self.model_result = model_result
        self.model = model_result.model
        self.sim = sim
        self.model_vocab = model_vocab
        self.step_time = step_time
        # cached vocabulary arrays for nearest-neighbour decoding of predictions
        self.vocab_keys = list(model_vocab.keys())
        self.vocab_vectors = model_vocab.vectors
        self.vocab_norms = np.linalg.norm(self.vocab_vectors, axis=1)
        self.normalized_vocab_vectors = self.vocab_vectors / np.maximum(
            self.vocab_norms[:, None], 1e-12
        )

    # convert a token into the semantic pointer used by the network
    def _vector_for(self, token):
        token_key = WordsToSPAVocab([token])[0]
        return self.model_vocab[token_key].v

    # translate semantic pointer keys back into readable tokens for demo output
    def _decode_key(self, token_key):
        if token_key.startswith("WV_"):
            return SPAVocabToWords([token_key])[0]
        return token_key

    # choose the closest words in vocabulary space to the current model output
    def _top_predictions(self, vector, top_k=3):
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0:
            return []

        normalized_vector = vector / max(vector_norm, 1e-12)
        similarities = self.normalized_vocab_vectors @ normalized_vector
        top_k = min(top_k, len(similarities))
        if top_k <= 0:
            return []

        top_ids = np.argpartition(similarities, -top_k)[-top_k:]
        top_ids = top_ids[np.argsort(similarities[top_ids])[::-1]]

        return [
            (self._decode_key(self.vocab_keys[idx]), float(similarities[idx]))
            for idx in top_ids
        ]

    # present input and optional target vectors, then run the simulator forward
    def present(self, token, target=None, learn=False):
        vec = self._vector_for(token)
        self.model_result.input_module.set(vec)
        if target is None:
            self.model_result.target_module.set(np.zeros(self.model_vocab.dimensions))
        else:
            self.model_result.target_module.set(self._vector_for(target))
        self.model_result.target_module.is_recall = not learn
        self.sim.run(self.step_time)

    # single next-word training step
    def train_pair(self, token, target):
        self.present(token, target=target, learn=True)
        self.model_result.target_module.is_recall = True

    # training across one token sequence
    def train_sequence(self, tokens):
        self.reset_context()

        for i in range(len(tokens) - 1):
            self.present(tokens[i], learn=False)  # build context
            self.present(tokens[i], target=tokens[i+1], learn=True)

    # training across a corpus of token sequences
    def train_corpus(self, sequences):
        for tokens in tqdm(sequences, desc="Training"):
            if len(tokens) > 1:
                self.train_sequence(tokens)

    # recall-mode prediction for a single token
    def predict_next(self, token, top_k=3):
        self.present(token, learn=False)
        prediction = self.sim.data[self.model_result.p_pred][-1]
        return self._top_predictions(prediction, top_k=top_k)

    # recall-mode prediction for a sequence of tokens (resets context each time)
    def predict_next_sequence(self, tokens, top_k=3, reset_context=True):
        if reset_context:
            self.reset_context()
        
        # reset context implicitly by running fresh sequence
        for token in tokens:
            self.present(token, learn=False)

        prediction = self.sim.data[self.model_result.p_pred][-1]
        return self._top_predictions(prediction, top_k=top_k)
    
    def interactive_predict(self, text, top_k=5, reset_context=True):
        tokens = text.strip().split()

        if len(tokens) == 0:
            return []

        if reset_context:
            self.reset_context()

        predictions = self.predict_next_sequence(tokens, top_k=top_k)

        return predictions

    # autoregressive generation
    def generate(
        self,
        prompt,
        max_tokens=20,
        top_k=5,
        reset_context=True,
        verbose=False
    ):
        tokens = prompt.strip().split()

        if len(tokens) == 0:
            return []

        if reset_context:
            self.reset_context()

        generated = list(tokens)

        # build initial context
        for token in tokens:
            self.present(token, learn=False)

        for _ in range(max_tokens):
            prediction = self.sim.data[self.model_result.p_pred][-1]
            top_predictions = self._top_predictions(prediction, top_k=top_k)

            if len(top_predictions) == 0:
                break

            next_token = top_predictions[0][0]

            generated.append(next_token)

            if verbose:
                prediction_text = ", ".join(
                    f"{word} ({score:.3f})"
                    for word, score in top_predictions
                )
                print(f"next -> {prediction_text}")

            # feed prediction back into network
            self.present(next_token, learn=False)

        return generated

    # realtime console interface
    def interactive_loop(
        self,
        top_k=5,
        generate=False,
        max_tokens=20
    ):
        print("\nRealtime interactive mode")
        print("Type 'exit' to quit")
        print("Type 'reset' to clear context\n")

        self.reset_context()

        while True:
            try:
                text = input(">>> ").strip()

                if text.startswith("/"):
                    command = text.lower()

                    if command in ["/exit", "/quit"]:
                        break

                    elif command == "/reset":
                        self.reset_context()
                        print("[context reset]")

                    elif command == "/help":
                        print("\nCommands:")
                        print("/reset  - clear context memory")
                        print("/exit   - quit interactive mode")
                        print("/help   - show commands")

                    else:
                        print(f"Unknown command: {command}")

                    continue

                if len(text) == 0:
                    continue

                if generate:
                    output = self.generate(
                        text,
                        max_tokens=max_tokens,
                        top_k=top_k,
                        reset_context=False,
                        verbose=False
                    )

                    print("generated:")
                    print(" ".join(output))

                else:
                    predictions = self.interactive_predict(
                        text,
                        top_k=top_k,
                        reset_context=False
                    )

                    if len(predictions) == 0:
                        print("No prediction.")
                        continue

                    prediction_text = ", ".join(
                        f"{word} ({score:.3f})"
                        for word, score in predictions
                    )

                    print(prediction_text)

            except KeyboardInterrupt:
                break

        print("\nExiting interactive mode.")

    # reset context
    def reset_context(self):
        self.model.context_module.reset.output = 1.0
        self.sim.run(self.step_time)
        self.model.context_module.reset.output = 0.0

    # model checkpoint path
    def _resolve_checkpoint_path(self, filename):
        base_dir = os.path.dirname(os.path.dirname(__file__))

        checkpoint_dir = os.path.join(base_dir, "checkpoints")

        os.makedirs(checkpoint_dir, exist_ok=True)

        return os.path.join(checkpoint_dir, filename)

    # validating loaded model architecture
    def _architecture_signature(self):
        return {
            "vocab_dim": self.model_vocab.dimensions,
            "strict_vocab": self.model_result.strict,
            "step_time": self.step_time,
            "num_learning_connections":
                len(self.model_result.learning_connections),
            "learning_shapes": [
                tuple(self.sim.data[conn].weights.shape)
                for conn in self.model_result.learning_connections
            ],
            "sub_lengths": self.model_result.sub_lengths,
        }

    # saving model
    def save_checkpoint(self, path="checkpoint.pkl"):
        checkpoint = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "architecture": self._architecture_signature(),
            },
            "weights": [],
        }

        for conn in self.model_result.learning_connections:
            weights = self.sim.data[conn].weights.copy()
            checkpoint["weights"].append(weights)

        full_path = self._resolve_checkpoint_path(path)

        with open(full_path, "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"\nSaved checkpoint to: {full_path}")

    # loading model
    def _restore_connection_weights(self, conn, weights):
        current_weights = self.sim.data[conn].weights

        if current_weights.shape != weights.shape:
            raise ValueError(
                "Checkpoint weight shape mismatch for "
                f"{conn}: expected {current_weights.shape}, "
                f"found {weights.shape}"
            )

        try:
            current_weights[:] = weights
            return
        except ValueError as exc:
            if "read-only" not in str(exc):
                raise

        weight_signal = self.sim.model.sig[conn]["weights"]
        self.sim.signals[weight_signal] = weights

    def load_checkpoint(self, path="checkpoint.pkl"):
        full_path = self._resolve_checkpoint_path(path)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        with open(full_path, "rb") as f:
            checkpoint = pickle.load(f)

        metadata = checkpoint["metadata"]
        saved_weights = checkpoint["weights"]

        # architecture validation
        saved_arch = metadata["architecture"]
        current_arch = self._architecture_signature()

        if saved_arch != current_arch:
            raise ValueError(
                "\nCheckpoint architecture mismatch.\n"
                f"\nSaved:\n{saved_arch}\n"
                f"\nCurrent:\n{current_arch}"
            )

        expected = len(self.model_result.learning_connections)
        actual = len(saved_weights)

        if expected != actual:
            raise ValueError(
                f"Checkpoint mismatch: "
                f"expected {expected} learning connections, "
                f"found {actual}"
            )

        for conn, weights in zip(
            self.model_result.learning_connections,
            saved_weights
        ):
            self._restore_connection_weights(conn, weights)

        print(f"\nLoaded checkpoint from: {full_path}")
        print(f"Checkpoint timestamp: {metadata['timestamp']}")
    
    # decide whether to train new model or load existing model
    def train_or_load(
        self,
        sequences,
        checkpoint_path="checkpoint.pkl",
        force_retrain=False
    ):
        full_path = self._resolve_checkpoint_path(
            checkpoint_path
        )

        if (
            not force_retrain
            and os.path.isfile(full_path)
        ):
            print("\nCheckpoint found. Loading...")
            self.load_checkpoint(checkpoint_path)

        else:
            print("\nTraining model...")
            self.train_corpus(sequences)

            self.save_checkpoint(checkpoint_path)
