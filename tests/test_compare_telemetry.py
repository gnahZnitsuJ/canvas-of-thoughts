import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_telemetry.py"
SPEC = importlib.util.spec_from_file_location("compare_telemetry", SCRIPT_PATH)
compare_telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_telemetry)


def benchmark_document(profile, compile_seconds):
    return {
        "timestamp": "2026-07-13T12:00:00-04:00",
        "kind": "compile_benchmark_current",
        "environment": {
            "source_commit": "abc123",
            "source_dirty": False,
            "source_snapshot": "snapshot123",
            "opencl_platform": "NVIDIA CUDA",
            "opencl_device": "GPU",
        },
        "probe_mode": "minimal",
        "compile_profile": {
            "name": profile,
            "settings": {"ensemble_n_eval_points": None if profile == "full" else 100},
        },
        "learned_init_mode": "random-function",
        "learned_init_seed": None,
        "scaling": [
            {
                "name": "current_configuration",
                "simulator": "nengo_ocl",
                "sub_lengths": [1, 20],
                "context_length": 20,
                "rep_vocab_dim": 256,
                "model_build_seconds": 2.0,
                "simulator_compile_seconds": compile_seconds,
                "first_run_warmup_seconds": None,
                "network": {
                    "network_count": 32,
                    "ensemble_count": 2136,
                    "neuron_count": 308800,
                    "node_count": 58,
                    "connection_count": 6450,
                    "probe_count": 1,
                },
                "operators": {"operator_count": 23730},
            }
        ],
    }


class TelemetryComparisonTests(unittest.TestCase):
    def test_normalizes_benchmark_and_calculates_profile_only_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, (profile, seconds) in enumerate(
                (("full", 280.0), ("fast-solver", 11.0))
            ):
                path = root / f"run{index}.json"
                path.write_text(
                    json.dumps(benchmark_document(profile, seconds)), encoding="utf-8"
                )
                paths.append(path)

            records = compare_telemetry.load_records(paths)
            differences, unexpected, messages = compare_telemetry.validate_controls(
                records, ["compile_profile"]
            )

        self.assertEqual(len(records), 2)
        self.assertIn("compile_profile.name", differences)
        self.assertFalse(unexpected)
        self.assertFalse(messages)
        self.assertEqual(records[1]["simulator_construct_seconds"], 11.0)

    def test_flags_unexpected_device_change_in_strict_comparison(self):
        first = next(
            compare_telemetry.normalize(
                benchmark_document("full", 280.0), Path("first.json")
            )
        )
        changed = benchmark_document("fast-solver", 11.0)
        changed["environment"]["opencl_device"] = "Different GPU"
        second = next(compare_telemetry.normalize(changed, Path("second.json")))

        _, unexpected, messages = compare_telemetry.validate_controls(
            [first, second], ["compile_profile"]
        )

        self.assertEqual(unexpected, {"opencl.device"})
        self.assertTrue(any("Unexpected control differences" in item for item in messages))

    def test_expands_repeat_compile_records(self):
        document = benchmark_document("fast-solver", 10.0)
        case = document.pop("scaling")[0]
        document["repeat_compile"] = [
            {**case, "repeat_index": 0},
            {**case, "repeat_index": 1, "simulator_compile_seconds": 12.0},
        ]

        records = list(compare_telemetry.normalize(document, Path("repeat.json")))

        self.assertEqual([record["repeat_index"] for record in records], [0, 1])
        self.assertEqual(records[1]["simulator_construct_seconds"], 12.0)

    def test_normalizes_build_only_graph_counts(self):
        document = {
            "timestamp": "2026-07-13T12:00:00-04:00",
            "kind": "model_build_only",
            "environment": {"source_commit": "abc123"},
            "parameters": {
                "rep_vocab_dim": 256,
                "context_length": 20,
                "sub_lengths": [1, 20],
                "probe_mode": "minimal",
            },
            "compile_profile": {
                "name": "fast-solver",
                "settings": {"ensemble_n_eval_points": 100},
            },
            "compile_fingerprint": {
                "learned_init_mode": "zero-nosolver",
                "learned_init_seed": None,
            },
            "timings_seconds": {"Model build": 1.5},
            "complexity": {
                "network": {
                    "network_count": 32,
                    "ensemble_count": 2136,
                    "neuron_count": 308800,
                    "node_count": 58,
                    "connection_count": 6450,
                    "probe_count": 1,
                }
            },
            "architecture_signature": {"vocab_dim": 256},
        }

        record = next(compare_telemetry.normalize(document, Path("build.json")))

        self.assertEqual(record["model_build_seconds"], 1.5)
        self.assertEqual(record["connection_count"], 6450)
        self.assertEqual(record["architecture_signature"], '{"vocab_dim":256}')


if __name__ == "__main__":
    unittest.main()
