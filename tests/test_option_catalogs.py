import sys
import unittest
from pathlib import Path
from unittest.mock import patch


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

from app import args as app_args  # noqa: E402
from architecture.variants import (  # noqa: E402
    ARCHITECTURE_VARIANTS,
    DEFAULT_ARCHITECTURE_NAME,
    ArchitectureVariant,
    architecture_spec,
    available_architectures,
)
from components.runtime import (  # noqa: E402
    DEFAULT_TRAINING_MODE,
    VALID_TRAINING_MODES,
)
from utils.build_config import (  # noqa: E402
    COMPILE_PROFILE_SETTINGS,
    DEFAULT_COMPILE_PROFILE_NAME,
    DEFAULT_LEARNED_INIT_MODE,
    LEARNED_INIT_MODES,
    available_compile_profiles,
    validate_learned_init_configuration,
)
from utils.probes import DEFAULT_PROBE_MODE, VALID_PROBE_MODES  # noqa: E402


class OptionCatalogTests(unittest.TestCase):
    def test_discovered_architectures_build_fresh_matching_specs(self):
        self.assertIn(DEFAULT_ARCHITECTURE_NAME, available_architectures())

        for name, variant in ARCHITECTURE_VARIANTS.items():
            first = architecture_spec(name)
            second = architecture_spec(name)
            self.assertEqual(first.name, name)
            self.assertEqual(variant.name, name)
            self.assertIsNot(first, second)

    def test_architecture_can_be_hidden_from_cli_without_disabling_builder(self):
        hidden = ArchitectureVariant(
            name="hidden-test-v1",
            builder=lambda: None,
            module_name="test.hidden",
            expose_to_cli=False,
            is_default=False,
        )

        with patch.dict(ARCHITECTURE_VARIANTS, {hidden.name: hidden}):
            self.assertNotIn(hidden.name, available_architectures())

    def test_application_cli_reads_architecture_catalog_at_parse_time(self):
        choices = available_architectures() + ("catalog-test-v1",)
        argv = ["main.py", "--dry-run", "--architecture", "catalog-test-v1"]

        with (
            patch.object(app_args, "available_architectures", return_value=choices),
            patch.object(sys, "argv", argv),
        ):
            parsed = app_args.parse_args()

        self.assertEqual(parsed.architecture, "catalog-test-v1")

    def test_other_cli_choices_match_their_semantic_owners(self):
        self.assertEqual(
            available_compile_profiles(),
            tuple(sorted(COMPILE_PROFILE_SETTINGS)),
        )
        self.assertIn(DEFAULT_COMPILE_PROFILE_NAME, available_compile_profiles())
        self.assertIn(DEFAULT_LEARNED_INIT_MODE, LEARNED_INIT_MODES)
        self.assertIn(DEFAULT_PROBE_MODE, VALID_PROBE_MODES)
        self.assertIn(DEFAULT_TRAINING_MODE, VALID_TRAINING_MODES)
        self.assertEqual(
            app_args.TRAINING_MODE_CHOICES,
            tuple(mode.replace("_", "-") for mode in VALID_TRAINING_MODES),
        )

    def test_learned_init_cross_field_validation_is_shared(self):
        for mode in LEARNED_INIT_MODES:
            seed = 7 if mode == "seeded-nosolver" else None
            validate_learned_init_configuration(mode, seed)

        with self.assertRaisesRegex(ValueError, "explicit learned-init seed"):
            validate_learned_init_configuration("seeded-nosolver", None)
        with self.assertRaisesRegex(ValueError, "Unknown learned init mode"):
            validate_learned_init_configuration("missing", None)


if __name__ == "__main__":
    unittest.main()
