import sys
import unittest
from pathlib import Path

import nengo_spa as spa


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

import components.net_comp as net_comp  # noqa: E402
from components.runtime import (  # noqa: E402
    ModelRuntime,
    build_architecture_signature,
    compare_architecture_signatures,
)


class ArchitectureAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab = spa.Vocabulary(256, strict=False)
        vocab.populate("POS")
        common = {
            "sub_lengths": [1, 20],
            "model_vocab": vocab,
            "probe_mode": "debug",
            "learned_init_mode": "zero-nosolver",
            "compile_profile_name": "fast-solver",
            "compile_profile_settings": {"ensemble_n_eval_points": 100},
        }
        cls.vocab = vocab
        cls.baseline = net_comp.Model(
            **common,
            architecture_name="root-context-v1",
        )
        cls.variant = net_comp.Model(
            **common,
            architecture_name="no-refiner-v1",
        )

    def test_baseline_reproduces_established_graph_counts(self):
        model = self.baseline.model
        self.assertEqual(len(model.all_networks), 32)
        self.assertEqual(len(model.all_ensembles), 2136)
        self.assertEqual(len(model.all_nodes), 58)
        self.assertEqual(len(model.all_connections), 6450)
        self.assertEqual(len(model.all_probes), 6)
        self.assertEqual(len(self.baseline.learning_connections), 2)

    def test_legacy_facade_is_derived_from_roles(self):
        self.assertIs(
            self.baseline.input_module,
            self.baseline.built_components["tokens"].runtime_handles["input"],
        )
        self.assertIs(
            self.baseline.learning_connections[0],
            self.baseline.built_components["refiner"].learning_connections[0],
        )
        self.assertIs(
            self.baseline.learning_connections[1],
            self.baseline.built_components["predictor"].learning_connections[0],
        )
        self.assertIs(
            self.baseline.context_module,
            self.baseline.built_components["memory"].network,
        )
        self.assertIs(
            self.baseline.active_component,
            self.baseline.built_components["predictor"].network,
        )

    def test_runtime_reset_uses_registered_capability(self):
        runtime = ModelRuntime.__new__(ModelRuntime)
        runtime.model_result = self.baseline
        memory = self.baseline.context_module

        runtime._set_context_reset(True)
        self.assertEqual(memory.reset_value, 1.0)
        runtime._set_context_reset(False)
        self.assertEqual(memory.reset_value, 0.0)

    def test_no_refiner_variant_regenerates_registration(self):
        self.assertNotIn("refiner", self.variant.built_components)
        self.assertEqual(
            self.variant.architecture_spec.roles["prediction"],
            "predictor.prediction",
        )
        self.assertEqual(len(self.variant.learning_connections), 1)
        self.assertEqual(len(self.variant.model.all_probes), 4)

    def test_variant_is_structurally_checkpoint_incompatible(self):
        compile_fingerprint = {
            "compile_profile": {
                "name": "fast-solver",
                "settings": {"ensemble_n_eval_points": 100},
            },
            "learned_init_mode": "zero-nosolver",
            "learned_init_seed": None,
        }
        baseline_signature = build_architecture_signature(
            self.baseline,
            self.vocab,
            0.02,
            compile_fingerprint=compile_fingerprint,
        )
        variant_signature = build_architecture_signature(
            self.variant,
            self.vocab,
            0.02,
            compile_fingerprint=compile_fingerprint,
        )

        comparison = compare_architecture_signatures(
            baseline_signature,
            variant_signature,
        )
        categories = {
            mismatch["field"]: mismatch["category"]
            for mismatch in comparison["mismatches"]
        }
        self.assertFalse(comparison["matches"])
        self.assertEqual(categories["architecture_topology"], "structural")
        self.assertIn("num_learning_connections", categories)


if __name__ == "__main__":
    unittest.main()
