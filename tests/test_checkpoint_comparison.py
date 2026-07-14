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


if __name__ == "__main__":
    unittest.main()
