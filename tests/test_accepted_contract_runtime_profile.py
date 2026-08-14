"""Independent contract tests for local runtime-profile persistence."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from utils.runtime_profile import (  # noqa: E402
    default_runtime_profile,
    load_runtime_profile,
    runtime_profile_example_path,
    runtime_profile_path,
    save_runtime_profile,
)


class RuntimeProfileContractTests(unittest.TestCase):
    def test_default_profiles_are_deep_copy_isolated(self):
        first = default_runtime_profile()
        second = default_runtime_profile()

        first["training"]["calibrated"] = True
        first["opencl"]["platform_index"] = 99
        first["new_nested_value"] = {"items": ["changed"]}

        self.assertFalse(second["training"]["calibrated"])
        self.assertIsNone(second["opencl"]["platform_index"])
        self.assertNotIn("new_nested_value", second)
        self.assertEqual(default_runtime_profile(), second)

    def test_missing_profile_returns_none_and_paths_stay_under_base_config(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir)
            local_path = runtime_profile_path(base)
            example_path = runtime_profile_example_path(base)

            self.assertEqual(local_path, base / "config" / "runtime_profile.json")
            self.assertEqual(
                example_path,
                base / "config" / "runtime_profile.example.json",
            )
            self.assertEqual(local_path.parent.resolve(), (base / "config").resolve())
            self.assertEqual(example_path.parent.resolve(), (base / "config").resolve())
            self.assertIsNone(load_runtime_profile(base))

    def test_save_and_load_round_trip_exact_json_document(self):
        document = {
            "schema_version": 17,
            "training": {
                "enabled": True,
                "durations": [0.01, 0.025, None],
            },
            "unicode": "caf\u00e9 \u2603",
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir)
            saved_path = save_runtime_profile(base, document)

            self.assertEqual(saved_path, base / "config" / "runtime_profile.json")
            self.assertEqual(load_runtime_profile(base), document)
            with saved_path.open("r", encoding="utf-8") as saved_file:
                self.assertEqual(json.load(saved_file), document)


if __name__ == "__main__":
    unittest.main()
