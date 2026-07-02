import os
import pickle
from datetime import datetime
from time import perf_counter

import numpy as np
from tqdm import tqdm

from utils.processing import SPAVocabToWords, WordsToSPAVocab


TRAINING_SEMANTICS_VERSION = "root_context_single_pass_v1"


class ModelRuntime:
    """Run training, recall, checkpoint, and calibration workflows."""

    def __init__(self, model_result, sim, model_vocab, step_time=0.02):
        self.model_result = model_result
        self.model = model_result.model
        self.sim = sim
        self.model_vocab = model_vocab
        self.step_time = float(step_time)
        self.compile_fingerprint = None

        # Cache normalized vocabulary vectors so prediction decoding stays cheap.
        self.vocab_keys = list(model_vocab.keys())
        self.vocab_vectors = model_vocab.vectors
        self.vocab_norms = np.linalg.norm(self.vocab_vectors, axis=1)
        self.normalized_vocab_vectors = self.vocab_vectors / np.maximum(
            self.vocab_norms[:, None],
            1e-12,
        )

        self.sim_run_count = 0
        self.sim_run_seconds = 0.0
        self.simulated_seconds = 0.0
        self.present_calls = 0
        self.reset_context_calls = 0

        self.training_mode = "single_pass"
        self.token_duration = self.step_time
        self.token_duration_source = "default"
        self.scheduled_training_enabled = False

    def _run_sim(self, duration):
        """Advance the simulator while tracking invocation telemetry."""
        start = perf_counter()
        self.sim.run(duration)
        self.sim_run_seconds += perf_counter() - start
        self.sim_run_count += 1
        self.simulated_seconds += duration

    def _set_context_reset(self, active):
        """Toggle reset on every active context module in the current model."""
        value = 1.0 if active else 0.0
        for context_module in self.model_result.active_context_modules:
            context_module.reset_value = value

    def _zero_io_buffers(self):
        """Clear buffered input/target values after scheduled training passes."""
        zeros = np.zeros(self.model_vocab.dimensions)
        self.model_result.input_module.set(zeros)
        self.model_result.target_module.set(zeros)

    def _current_sim_time(self):
        """Read simulator time in a way that works across simulators."""
        return float(getattr(self.sim, "time", 0.0))

    def configure_training(
        self,
        training_mode="single_pass",
        token_duration=None,
        token_duration_source="default",
    ):
        """Set how corpus training should drive the simulator."""
        if training_mode not in ("single_pass", "scheduled"):
            raise ValueError(f"Unknown training mode: {training_mode}")

        if token_duration is None:
            token_duration = self.step_time
        if token_duration <= 0:
            raise ValueError("token_duration must be positive")

        self.training_mode = training_mode
        self.token_duration = float(token_duration)
        self.token_duration_source = token_duration_source
        self.scheduled_training_enabled = training_mode == "scheduled"

    def training_configuration(self):
        """Return the active training configuration for telemetry/reporting."""
        return {
            "training_mode": self.training_mode,
            "training_semantics_version": TRAINING_SEMANTICS_VERSION,
            "scheduled_training_enabled": self.scheduled_training_enabled,
            "token_duration": self.token_duration,
            "token_duration_source": self.token_duration_source,
        }

    def set_compile_fingerprint(self, compile_fingerprint):
        """Attach compile-context metadata for telemetry and checkpoints."""
        self.compile_fingerprint = compile_fingerprint

    def simulator_invocation_telemetry(self):
        return {
            "present_calls": self.present_calls,
            "reset_context_calls": self.reset_context_calls,
            "sim_run_count": self.sim_run_count,
            "sim_run_seconds": self.sim_run_seconds,
            "simulated_seconds": self.simulated_seconds,
        }

    def _weight_signal_for_connection(self, conn):
        """Return the simulator signal that owns the live learned weights."""
        model_signals = getattr(self.sim, "model", None)
        model_signals = getattr(model_signals, "sig", None)
        if model_signals is None or conn not in model_signals:
            return None
        return model_signals[conn].get("weights")

    def _current_connection_weights(self, conn):
        """Read the live connection weights across simulator backends."""
        weight_signal = self._weight_signal_for_connection(conn)
        if weight_signal is not None and hasattr(self.sim, "signals"):
            signal_weights = self.sim.signals.get(weight_signal)
            if signal_weights is not None:
                return np.array(signal_weights, copy=True)

        return np.array(self.sim.data[conn].weights, copy=True)

    def snapshot_learning_weights(self):
        """Capture the current learned weights for calibration or experimentation."""
        return [
            self._current_connection_weights(conn)
            for conn in self.model_result.learning_connections
        ]

    def restore_learning_weights(self, weights_by_connection):
        """Restore a previously captured set of learned weights."""
        for conn, weights in zip(
            self.model_result.learning_connections,
            weights_by_connection,
        ):
            self._restore_connection_weights(conn, weights)

    def clear_scheduled_inputs(self):
        """Remove any active token schedules and restore buffer-driven IO."""
        self.model_result.input_module.clear_schedule()
        self.model_result.target_module.clear_schedule()
        self._zero_io_buffers()
        self.model_result.target_module.is_recall = True

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

    def current_prediction_vector(self):
        """Read the latest decoded prediction state without advancing the sim."""
        prediction_samples = self.sim.data[self.model_result.p_pred]
        if len(prediction_samples) == 0:
            return np.zeros(self.model_vocab.dimensions)
        return prediction_samples[-1]

    def current_predictions(self, top_k=3):
        """Decode the current prediction state into nearest vocabulary items."""
        return self._top_predictions(self.current_prediction_vector(), top_k=top_k)

    # present input and optional target vectors, then run the simulator forward
    def present(self, token, target=None, learn=False):
        self.present_calls += 1
        vec = self._vector_for(token)
        self.model_result.input_module.set(vec)
        if target is None:
            self.model_result.target_module.set(np.zeros(self.model_vocab.dimensions))
        else:
            self.model_result.target_module.set(self._vector_for(target))
        self.model_result.target_module.is_recall = not learn
        self._run_sim(self.step_time)

    def advance_recall(self, token, top_k=None):
        """Feed one token through the stateful recall path and optionally decode it."""
        self.present(token, learn=False)
        if top_k is None:
            return self.current_prediction_vector()
        return self.current_predictions(top_k=top_k)

    # single next-word training step
    def train_pair(self, token, target):
        self.present(token, target=target, learn=True)
        self.model_result.target_module.is_recall = True

    def train_sequence(self, tokens):
        """Train one sequence using single-pass next-token supervision."""
        self.reset_context()

        for index in range(len(tokens) - 1):
            self.present(tokens[index], target=tokens[index + 1], learn=True)

        self.model_result.target_module.is_recall = True

    def train_sequence_scheduled(self, tokens, token_duration=None):
        """Train one sequence using one long scheduled simulator run."""
        if len(tokens) < 2:
            return

        token_duration = self.step_time if token_duration is None else float(token_duration)
        if token_duration <= 0:
            raise ValueError("token_duration must be positive")

        self.reset_context()

        input_vectors = [
            self._vector_for(tokens[index])
            for index in range(len(tokens) - 1)
        ]
        target_vectors = [
            self._vector_for(tokens[index + 1])
            for index in range(len(tokens) - 1)
        ]

        start_time = self._current_sim_time()
        self.model_result.input_module.set_schedule(
            input_vectors,
            start_time=start_time,
            token_duration=token_duration,
        )
        self.model_result.target_module.set_schedule(
            target_vectors,
            start_time=start_time,
            token_duration=token_duration,
        )
        self.model_result.target_module.is_recall = False

        try:
            self._run_sim(len(input_vectors) * token_duration)
        finally:
            self.clear_scheduled_inputs()

    def train_corpus(self, sequences):
        """Train the corpus using single-pass stepwise presentations."""
        for tokens in tqdm(sequences, desc="Training single-pass"):
            if len(tokens) > 1:
                self.train_sequence(tokens)

    def train_corpus_scheduled(self, sequences, token_duration=None):
        """Train the corpus using one scheduled run per sequence."""
        duration = self.token_duration if token_duration is None else token_duration
        for tokens in tqdm(sequences, desc="Training scheduled"):
            if len(tokens) > 1:
                self.train_sequence_scheduled(tokens, token_duration=duration)

    # recall-mode prediction for a single token
    def predict_next(self, token, top_k=3):
        return self.advance_recall(token, top_k=top_k)

    # recall-mode prediction for a sequence of tokens (resets context each time)
    def predict_next_sequence(self, tokens, top_k=3, reset_context=True):
        if reset_context:
            self.reset_context()

        for token in tokens:
            self.present(token, learn=False)

        return self.current_predictions(top_k=top_k)

    def interactive_predict(self, text, top_k=5, reset_context=True):
        tokens = text.strip().split()

        if len(tokens) == 0:
            return []

        return self.predict_next_sequence(
            tokens,
            top_k=top_k,
            reset_context=reset_context,
        )

    # autoregressive generation
    def generate(
        self,
        prompt,
        max_tokens=20,
        top_k=5,
        reset_context=True,
        verbose=False,
    ):
        tokens = prompt.strip().split()

        if len(tokens) == 0:
            return []

        if reset_context:
            self.reset_context()

        generated = list(tokens)

        # Build initial context once, then keep advancing from current state.
        for token in tokens:
            self.present(token, learn=False)

        for _ in range(max_tokens):
            top_predictions = self.current_predictions(top_k=top_k)

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

            # Feed prediction back into the same stateful context path.
            self.present(next_token, learn=False)

        return generated

    # realtime console interface
    def interactive_loop(
        self,
        top_k=5,
        generate=False,
        max_tokens=20,
    ):
        print("\nRealtime interactive mode")
        print("Type '/exit' to quit")
        print("Type '/reset' to clear context")
        print("Type '/help' to show commands\n")

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
                        verbose=False,
                    )

                    print("generated:")
                    print(" ".join(output))

                else:
                    predictions = self.interactive_predict(
                        text,
                        top_k=top_k,
                        reset_context=False,
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
        self.reset_context_calls += 1
        self._set_context_reset(True)
        self._run_sim(self.step_time)
        self._set_context_reset(False)

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
            "training_semantics_version": TRAINING_SEMANTICS_VERSION,
            "num_learning_connections": len(self.model_result.learning_connections),
            "learning_shapes": [
                tuple(self._current_connection_weights(conn).shape)
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
                "compile_fingerprint": self.compile_fingerprint,
            },
            "weights": [],
        }

        for conn in self.model_result.learning_connections:
            checkpoint["weights"].append(self._current_connection_weights(conn))

        full_path = self._resolve_checkpoint_path(path)

        with open(full_path, "wb") as checkpoint_file:
            pickle.dump(checkpoint, checkpoint_file)

        print(f"\nSaved checkpoint to: {full_path}")

    # loading model
    def _restore_connection_weights(self, conn, weights):
        """Write learned weights back to the live simulator state."""
        current_weights = self._current_connection_weights(conn)

        if current_weights.shape != weights.shape:
            raise ValueError(
                "Checkpoint weight shape mismatch for "
                f"{conn}: expected {current_weights.shape}, "
                f"found {weights.shape}"
            )

        weight_signal = self._weight_signal_for_connection(conn)
        if weight_signal is not None and hasattr(self.sim, "signals"):
            signal_weights = self.sim.signals.get(weight_signal)
            if signal_weights is not None:
                try:
                    signal_weights[...] = weights
                except (TypeError, ValueError):
                    self.sim.signals[weight_signal] = np.array(weights, copy=True)

        # Keep the backend-independent data view in sync when it is writable.
        try:
            self.sim.data[conn].weights[...] = weights
        except (TypeError, ValueError):
            pass

    def load_checkpoint(self, path="checkpoint.pkl"):
        full_path = self._resolve_checkpoint_path(path)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(full_path, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        metadata = checkpoint["metadata"]
        saved_weights = checkpoint["weights"]

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
                "Checkpoint mismatch: "
                f"expected {expected} learning connections, "
                f"found {actual}"
            )

        for conn, weights in zip(self.model_result.learning_connections, saved_weights):
            self._restore_connection_weights(conn, weights)

        print(f"\nLoaded checkpoint from: {full_path}")
        print(f"Checkpoint timestamp: {metadata['timestamp']}")

    # decide whether to train new model or load existing model
    def train_or_load(
        self,
        sequences,
        checkpoint_path="checkpoint.pkl",
        force_retrain=False,
    ):
        full_path = self._resolve_checkpoint_path(checkpoint_path)

        if not force_retrain and os.path.isfile(full_path):
            print("\nCheckpoint found. Loading...")
            self.load_checkpoint(checkpoint_path)
            return

        if self.training_mode == "scheduled":
            print("\nTraining model with scheduled corpus input...")
            self.train_corpus_scheduled(sequences, token_duration=self.token_duration)
        else:
            print("\nTraining model with single-pass stepwise input...")
            self.train_corpus(sequences)

        self.save_checkpoint(checkpoint_path)
