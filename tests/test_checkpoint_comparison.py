import sys
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

from components.runtime import (  # noqa: E402
    compare_architecture_signatures,
    format_architecture_comparison,
)


class CheckpointComparisonTests(unittest.TestCase):
    def test_categorizes_profile_and_structural_mismatches(self):
        saved = {
            "vocab_dim": 256,
            "compile_profile_name": "full",
            "learned_init_mode": "random-function",
        }
        current = {
            "vocab_dim": 128,
            "compile_profile_name": "fast-solver",
            "learned_init_mode": "zero-nosolver",
        }

        comparison = compare_architecture_signatures(saved, current)
        categories = {
            mismatch["field"]: mismatch["category"]
            for mismatch in comparison["mismatches"]
        }
        rendered = format_architecture_comparison(comparison)

        self.assertFalse(comparison["matches"])
        self.assertEqual(categories["vocab_dim"], "structural")
        self.assertEqual(categories["compile_profile_name"], "compile-profile")
        self.assertEqual(categories["learned_init_mode"], "learned-init")
        self.assertIn("saved:   'full'", rendered)

    def test_migrates_legacy_fixed_builder_to_identical_root_context(self):
        saved = {"vocab_dim": 256, "num_learning_connections": 2}
        topology = {
            "architecture_name": "root-context-v1",
            "component_build_order": [
                "tokens",
                "targets",
                "memory",
                "predictor",
                "refiner",
            ],
            "checkpoint_order": ["refiner", "predictor"],
        }
        current = {**saved, "architecture_topology": topology}

        comparison = compare_architecture_signatures(saved, current)
        rendered = format_architecture_comparison(comparison)

        self.assertTrue(comparison["matches"])
        self.assertTrue(comparison["legacy_topology_assumed"])
        self.assertIn("legacy root-context topology migration", rendered)

    def test_rejects_changed_component_build_order(self):
        saved = {
            "architecture_topology": {
                "architecture_name": "root-context-v1",
                "component_build_order": [
                    "tokens",
                    "targets",
                    "memory",
                    "predictor",
                ],
            }
        }
        current = {
            "architecture_topology": {
                "architecture_name": "root-context-v1",
                "component_build_order": [
                    "tokens",
                    "targets",
                    "predictor",
                    "memory",
                ],
            }
        }

        comparison = compare_architecture_signatures(saved, current)

        self.assertFalse(comparison["matches"])
        self.assertEqual(comparison["mismatches"][0]["field"], "architecture_topology")
        self.assertEqual(comparison["mismatches"][0]["category"], "structural")

    def test_rejects_composable_topology_without_component_build_order(self):
        saved = {
            "architecture_topology": {
                "architecture_name": "root-context-v1",
            }
        }
        current = {
            "architecture_topology": {
                "architecture_name": "root-context-v1",
                "component_build_order": ["tokens", "targets", "memory", "predictor"],
            }
        }

        comparison = compare_architecture_signatures(saved, current)

        self.assertFalse(comparison["matches"])
        self.assertFalse(comparison["legacy_topology_assumed"])


if __name__ == "__main__":
    unittest.main()
