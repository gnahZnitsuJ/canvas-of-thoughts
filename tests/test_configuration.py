import json
import sys
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

from config import data_defaults, model_defaults  # noqa: E402
from config.runtime_defaults import DEFAULT_STEP_TIME_SECONDS  # noqa: E402
from utils.runtime_profile import default_runtime_profile  # noqa: E402
from utils.train_partition import data_partition  # noqa: E402


class FakeDataset:
    def fileids(self):
        return [
            "training/one",
            "training/two",
            "training/three",
            "test/one",
            "test/two",
            "test/three",
        ]

    def words(self, file_id):
        return [file_id]


class ConfigurationTests(unittest.TestCase):
    def test_committed_defaults_satisfy_domain_constraints(self):
        self.assertEqual(model_defaults.VOCAB_DIMENSIONS % 16, 0)
        self.assertGreater(model_defaults.PES_LEARNING_RATE, 0)
        self.assertGreater(DEFAULT_STEP_TIME_SECONDS, 0)
        self.assertGreaterEqual(data_defaults.TRAINING_DOCUMENT_LIMIT, 0)
        self.assertGreaterEqual(data_defaults.TESTING_DOCUMENT_LIMIT, 0)
        self.assertEqual(
            len(data_defaults.DATASET_NAMES),
            len(set(data_defaults.DATASET_NAMES)),
        )

    def test_runtime_profile_example_matches_committed_defaults(self):
        example_path = MODEL_DIR / "config" / "runtime_profile.example.json"
        with example_path.open("r", encoding="utf-8") as example_file:
            example = json.load(example_file)

        self.assertEqual(example, default_runtime_profile())
        self.assertEqual(
            example["runtime"]["default_step_time"],
            DEFAULT_STEP_TIME_SECONDS,
        )
        self.assertEqual(
            example["training"]["token_duration"],
            DEFAULT_STEP_TIME_SECONDS,
        )

    def test_partition_uses_explicit_limits_instead_of_global_defaults(self):
        partition = data_partition(
            FakeDataset(),
            training_restriction=1,
            testing_restriction=2,
            strict=True,
        )

        self.assertEqual(partition.training_ids, ["training/one"])
        self.assertEqual(partition.testing_ids, ["test/one", "test/two"])
        self.assertIn(data_defaults.UNKNOWN_TOKEN, partition.vocab)
        self.assertIn(data_defaults.PAD_TOKEN, partition.vocab)


if __name__ == "__main__":
    unittest.main()
