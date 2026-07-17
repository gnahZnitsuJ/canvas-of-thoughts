import json
import sys
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

from architecture import (  # noqa: E402
    ArchitectureSpec,
    ArchitectureValidationError,
    BuiltComponent,
    Port,
    architecture_signature,
    canonical_json,
    validate_architecture,
)


def component(name, ports, *, capabilities=(), handles=None, learning=()):
    return BuiltComponent(
        name=name,
        network=object(),
        ports=ports,
        capabilities=set(capabilities),
        runtime_handles=dict(handles or {}),
        learning_connections=list(learning),
    )


def semantic_port(direction, dimensions=16, vocabulary_id="model_vocab", **kwargs):
    return Port(
        endpoint=object(),
        direction=direction,
        dimensions=dimensions,
        signal_type="semantic_pointer",
        vocabulary_id=vocabulary_id,
        **kwargs,
    )


class ArchitectureContractTests(unittest.TestCase):
    def baseline(self):
        spec = ArchitectureSpec("test_v1")
        spec.add("tokens", "input_source")
        spec.add("targets", "target_source")
        spec.add("memory", "context_memory", alpha=0.99)
        spec.add("predictor", "predictor")
        spec.connect("tokens.output", "memory.token")
        spec.connect("memory.context", "predictor.context")
        spec.connect("targets.output", "predictor.target")
        spec.assign_role("input", "tokens")
        spec.assign_role("target", "targets")
        spec.assign_role("primary_memory", "memory")
        spec.assign_role("prediction", "predictor.prediction")

        built = {
            "tokens": component("tokens", {"output": semantic_port("output")}),
            "targets": component("targets", {"output": semantic_port("output")}),
            "memory": component(
                "memory",
                {
                    "token": semantic_port("input"),
                    "context": semantic_port("output"),
                },
                capabilities={"memory", "resettable"},
                handles={"reset": object()},
            ),
            "predictor": component(
                "predictor",
                {
                    "context": semantic_port("input"),
                    "target": semantic_port("input"),
                    "prediction": semantic_port("output"),
                },
                capabilities={"learnable", "checkpointed"},
                learning=[object()],
            ),
        }
        return spec, built

    def test_validates_semantic_baseline(self):
        spec, built = self.baseline()
        validate_architecture(spec, built)

    def test_rejects_wrong_direction(self):
        spec, built = self.baseline()
        built["tokens"].ports["output"] = semantic_port("input")
        with self.assertRaisesRegex(ArchitectureValidationError, "not an output"):
            validate_architecture(spec, built)

    def test_rejects_dimension_mismatch_without_transform(self):
        spec, built = self.baseline()
        built["memory"].ports["token"] = semantic_port("input", dimensions=8)
        with self.assertRaisesRegex(ArchitectureValidationError, "Dimension mismatch"):
            validate_architecture(spec, built)

    def test_accepts_dimension_mismatch_with_explicit_transform(self):
        spec, built = self.baseline()
        built["memory"].ports["token"] = semantic_port("input", dimensions=8)
        spec.connections[0] = type(spec.connections[0])(
            "tokens.output", "memory.token", transform="explicit-test-transform"
        )
        validate_architecture(spec, built)

    def test_rejects_vocabulary_mismatch(self):
        spec, built = self.baseline()
        built["memory"].ports["token"] = semantic_port(
            "input", vocabulary_id="other_vocab"
        )
        with self.assertRaisesRegex(ArchitectureValidationError, "Vocabulary mismatch"):
            validate_architecture(spec, built)

    def test_requires_runtime_roles(self):
        spec, built = self.baseline()
        del spec.roles["prediction"]
        with self.assertRaisesRegex(ArchitectureValidationError, "prediction"):
            validate_architecture(spec, built)

    def test_requires_input_connections(self):
        spec, built = self.baseline()
        spec.disconnect("memory.context", "predictor.context")
        with self.assertRaisesRegex(ArchitectureValidationError, "predictor.context"):
            validate_architecture(spec, built)

    def test_accepts_recurrent_topology(self):
        spec, built = self.baseline()
        built["memory"].ports["feedback"] = semantic_port(
            "input", required=False
        )
        spec.connect("predictor.prediction", "memory.feedback")
        validate_architecture(spec, built)

    def test_copy_isolates_variant_mutations(self):
        baseline, _ = self.baseline()
        variant = baseline.copy(name="variant").remove("predictor")
        self.assertIn("predictor", baseline.components)
        self.assertNotIn("predictor", variant.components)

    def test_signature_is_deterministic_and_json_serializable(self):
        spec, built = self.baseline()
        first = architecture_signature(spec, built)
        second = architecture_signature(spec.copy(), built)
        self.assertEqual(canonical_json(first), canonical_json(second))
        json.loads(canonical_json(first))

    def test_signature_rejects_unstable_objects(self):
        spec, built = self.baseline()
        spec.components["memory"] = type(spec.components["memory"])(
            name="memory",
            component_type="context_memory",
            parameters={"unstable": object()},
        )
        with self.assertRaisesRegex(TypeError, "deterministic"):
            architecture_signature(spec, built)

    def test_duplicate_component_names_fail_early(self):
        spec = ArchitectureSpec("duplicates")
        spec.add("memory", "context_memory")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            spec.add("memory", "context_memory")


if __name__ == "__main__":
    unittest.main()
