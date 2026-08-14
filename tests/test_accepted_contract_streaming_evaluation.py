"""Independent contract tests for stateful streaming evaluation."""

import math
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from utils.eval import (  # noqa: E402
    evaluate_model_prefix_replay,
    evaluate_model_streaming_metrics,
    iter_next_token_predictions,
)


class RecordingRuntime:
    """Small stateful fake whose observations are independent of model code."""

    def __init__(self, predictions, prediction_vectors=None, target_vectors=None):
        self.predictions = predictions
        self.prediction_vectors = prediction_vectors or {}
        self.target_vectors = target_vectors or {}
        self.events = []
        self.last_token = None

    def reset_context(self):
        self.events.append(("reset",))

    def advance_recall(self, token, top_k):
        self.events.append(("advance", token, top_k))
        self.last_token = token
        return self.predictions[token]

    def current_prediction_vector(self):
        self.events.append(("prediction_vector", self.last_token))
        return np.asarray(self.prediction_vectors[self.last_token], dtype=float)

    def _vector_for(self, token):
        self.events.append(("target_vector", token))
        return np.asarray(self.target_vectors[token], dtype=float)


class PrefixReplayRecordingRuntime:
    """Record every legacy prediction call without using model helpers."""

    def __init__(self):
        self.calls = []

    def predict_next_sequence(self, prefix, top_k):
        self.calls.append((tuple(prefix), top_k))
        return [("unused", 1.0)]


class StreamingEvaluationContractTests(unittest.TestCase):
    def test_prefix_replay_zero_maximum_makes_no_predictions(self):
        runtime = PrefixReplayRecordingRuntime()

        with redirect_stdout(StringIO()):
            accuracy = evaluate_model_prefix_replay(
                runtime,
                [["a", "b", "c"]],
                max_examples=0,
                top_k=2,
            )

        self.assertEqual(accuracy, 0.0)
        self.assertEqual(runtime.calls, [])

    def test_iterator_resets_once_and_advances_once_for_each_adjacent_pair(self):
        runtime = RecordingRuntime(
            {
                "alpha": [("beta", 0.9)],
                "beta": [("gamma", 0.8)],
            }
        )

        observed = list(
            iter_next_token_predictions(
                runtime,
                ["alpha", "beta", "gamma"],
                top_k=4,
            )
        )

        self.assertEqual(
            runtime.events,
            [
                ("reset",),
                ("advance", "alpha", 4),
                ("advance", "beta", 4),
            ],
        )
        self.assertEqual(
            [(item["prefix"], item["target"]) for item in observed],
            [(("alpha",), "beta"), (("alpha", "beta"), "gamma")],
        )

    def test_metrics_skip_empty_predictions_and_apply_one_global_limit(self):
        runtime = RecordingRuntime(
            predictions={
                "a": [("b", 0.8), ("x", 0.1)],
                "b": [],
                "c": [("x", 0.6), ("d", 0.3)],
                "e": [("x", 0.4), ("y", 0.2)],
                "f": [("g", 0.9)],
                "h": [("i", 1.0)],
            },
            prediction_vectors={
                "a": [1.0, 0.0],
                "c": [0.0, 1.0],
                "e": [1.0, 0.0],
            },
            target_vectors={
                "b": [1.0, 0.0],
                "d": [1.0, 1.0],
                "f": [-1.0, 0.0],
            },
        )

        metrics = evaluate_model_streaming_metrics(
            runtime,
            [
                ["a", "b", "c", "d"],
                ["e", "f", "g"],
                ["h", "i"],
            ],
            max_examples=3,
            top_k=2,
        )

        self.assertEqual(metrics["total"], 3)
        self.assertAlmostEqual(metrics["top1_accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["top2_accuracy"], 2.0 / 3.0)
        # Hand calculation: (cos([1,0],[1,0])
        #                    + cos([0,1],[1,1])
        #                    + cos([1,0],[-1,0])) / 3.
        self.assertAlmostEqual(
            metrics["mean_target_similarity"],
            (1.0 + 1.0 / math.sqrt(2.0) - 1.0) / 3.0,
        )
        self.assertAlmostEqual(metrics["mean_top_score"], (0.8 + 0.6 + 0.4) / 3.0)

        advances = [event for event in runtime.events if event[0] == "advance"]
        resets = [event for event in runtime.events if event[0] == "reset"]
        self.assertEqual(
            advances,
            [
                ("advance", "a", 2),
                ("advance", "b", 2),
                ("advance", "c", 2),
                ("advance", "e", 2),
            ],
        )
        self.assertEqual(len(resets), 2)
        self.assertNotIn(("advance", "f", 2), advances)
        self.assertNotIn(("advance", "h", 2), advances)
        self.assertNotIn(("prediction_vector", "b"), runtime.events)

    def test_zero_maximum_scores_no_examples(self):
        runtime = RecordingRuntime(
            predictions={"a": [("b", 1.0)]},
            prediction_vectors={"a": [1.0, 0.0]},
            target_vectors={"b": [1.0, 0.0]},
        )

        metrics = evaluate_model_streaming_metrics(
            runtime,
            [["a", "b"]],
            max_examples=0,
            top_k=1,
        )

        self.assertEqual(
            metrics,
            {
                "total": 0,
                "top1_accuracy": 0.0,
                "mean_target_similarity": 0.0,
                "mean_top_score": 0.0,
            },
        )
        self.assertFalse(
            any(event[0] == "advance" for event in runtime.events),
            "max_examples=0 must not advance and score a pair",
        )


if __name__ == "__main__":
    unittest.main()
