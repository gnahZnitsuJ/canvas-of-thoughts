"""Focused tests for dependency and external-resource diagnostics."""

import ast
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from app import workflow  # noqa: E402
from launcher import cli  # noqa: E402
from launcher.bootstrap import checks  # noqa: E402
from utils import opencl  # noqa: E402


class RequirementManifestTests(unittest.TestCase):
    def test_launcher_package_has_only_stdlib_and_internal_imports(self):
        imported_modules = set()
        launcher_root = ROOT_DIR / "launcher"
        for source_path in launcher_root.rglob("*.py"):
            tree = ast.parse(
                source_path.read_text(encoding="utf-8-sig"),
                filename=str(source_path),
            )
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imported_modules.update(
                        alias.name.split(".", 1)[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported_modules.add(node.module.split(".", 1)[0])

        external_imports = imported_modules - sys.stdlib_module_names - {
            "__future__",
            "launcher",
        }
        self.assertEqual(external_imports, set())

    def test_project_manifest_covers_all_direct_imports(self):
        requirements = checks.load_requirements()
        source_roots = (
            ROOT_DIR / "launcher",
            ROOT_DIR / "model",
            ROOT_DIR / "scripts",
            ROOT_DIR / "tests",
        )
        imported_modules = set()
        for source_root in source_roots:
            for source_path in source_root.rglob("*.py"):
                tree = ast.parse(
                    source_path.read_text(encoding="utf-8-sig"),
                    filename=str(source_path),
                )
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_modules.update(
                            alias.name.split(".", 1)[0] for alias in node.names
                        )
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        imported_modules.add(node.module.split(".", 1)[0])

        local_modules = {"__future__"}
        for source_root in source_roots:
            local_modules.add(source_root.name)
            local_modules.update(path.stem for path in source_root.glob("*.py"))
            local_modules.update(path.name for path in source_root.iterdir() if path.is_dir())
        direct_imports = imported_modules - sys.stdlib_module_names - local_modules

        self.assertEqual(
            direct_imports,
            {
                "gensim",
                "nengo",
                "nengo_ocl",
                "nengo_spa",
                "nltk",
                "numpy",
                "pyopencl",
                "regex",
                "scipy",
                "tqdm",
            },
        )
        self.assertEqual(
            {requirement.import_name for requirement in requirements},
            direct_imports,
        )

    def test_manifest_parser_rejects_unpinned_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_path = Path(temp_dir) / "requirements.txt"
            requirements_path.write_text("nengo>=3\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exact NAME==VERSION pins"):
                checks.load_requirements(requirements_path)

    def test_package_check_reports_import_and_version_failures(self):
        requirements = (
            checks.RequirementSpec("first", "1.0", "first"),
            checks.RequirementSpec("second", "2.0", "second"),
            checks.RequirementSpec("third", "3.0", "third"),
        )
        with (
            patch.object(checks, "load_requirements", return_value=requirements),
            patch.object(
                checks.metadata,
                "version",
                side_effect=("1.0", "1.5", "3.0"),
            ),
            patch.object(
                checks.importlib,
                "import_module",
                side_effect=(None, None, ImportError("broken binary")),
            ),
        ):
            results = checks.check_python_packages()

        self.assertEqual(results[0].status, "OK")
        self.assertEqual(results[1].status, "FAIL")
        self.assertIn("expected 2.0", results[1].detail)
        self.assertEqual(results[2].status, "FAIL")
        self.assertIn("import failed", results[2].detail)

    def test_dependency_graph_check_reports_pip_failure(self):
        completed = SimpleNamespace(
            returncode=1,
            stdout="broken-package requires missing-package",
            stderr="",
        )
        with patch.object(
            checks.subprocess,
            "run",
            return_value=completed,
        ):
            result = checks.check_dependency_graph()

        self.assertEqual(result.status, "FAIL")
        self.assertIn("requires missing-package", result.detail)
        self.assertIn("pip install -r", result.repair)


class ExternalResourceCheckTests(unittest.TestCase):
    def test_reuters_check_reports_document_count(self):
        corpus = SimpleNamespace(fileids=Mock(return_value=["one", "two"]))
        nltk_corpus = SimpleNamespace(reuters=corpus)
        with patch.object(
            checks.importlib,
            "import_module",
            return_value=nltk_corpus,
        ):
            result = checks.check_reuters_corpus()

        self.assertEqual(result.status, "OK")
        self.assertIn("2 documents", result.detail)

    def test_opencl_check_reports_platform_and_device(self):
        device = SimpleNamespace(name="Test GPU")
        platform = SimpleNamespace(
            name="Test Platform",
            get_devices=Mock(return_value=[device]),
        )
        cl = SimpleNamespace(get_platforms=Mock(return_value=[platform]))
        with patch.object(
            checks.importlib,
            "import_module",
            return_value=cl,
        ):
            result = checks.check_opencl()

        self.assertEqual(result.status, "OK")
        self.assertIn("Test Platform / Test GPU", result.detail)

    def test_runtime_opencl_failure_points_to_doctor(self):
        with patch.object(opencl.cl, "get_platforms", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "launcher doctor"):
                opencl.select_opencl_device()

    def test_runtime_reuters_failure_provides_downloader_command(self):
        with patch.object(
            workflow,
            "multiple_data_partition",
            side_effect=LookupError("missing corpus"),
        ):
            with self.assertRaisesRegex(RuntimeError, "nltk.downloader reuters"):
                workflow.build_train_test({})


class BootstrapTests(unittest.TestCase):
    def test_missing_packages_stop_before_model_imports(self):
        missing = (
            checks.RequirementSpec("nengo", "3.2.0", "nengo"),
        )
        stderr = io.StringIO()
        with (
            patch.object(cli, "missing_required_packages", return_value=missing),
            patch.object(cli, "run_model_cli") as run_model_cli,
            contextlib.redirect_stderr(stderr),
        ):
            status = cli.run_command([])

        self.assertEqual(status, 1)
        run_model_cli.assert_not_called()
        self.assertIn("Missing required Python package(s): nengo", stderr.getvalue())
        self.assertIn("pip install -r", stderr.getvalue())

    def test_doctor_is_standalone_bootstrap_mode(self):
        with (
            patch.object(cli, "run_doctor", return_value=0) as doctor,
            patch.object(cli, "missing_required_packages") as missing,
        ):
            status = cli.main(["doctor"])

        self.assertEqual(status, 0)
        doctor.assert_called_once_with()
        missing.assert_not_called()

    def test_run_forwards_all_model_arguments(self):
        with patch.object(cli, "run_command", return_value=0) as run:
            status = cli.main(["run", "--dry-run", "--tokenizer", "bpe-v1"])

        self.assertEqual(status, 0)
        run.assert_called_once_with(
            ["--dry-run", "--tokenizer", "bpe-v1"],
            prog="python -m launcher run",
        )

    def test_legacy_environment_flag_delegates_to_doctor(self):
        with patch.object(cli, "run_doctor", return_value=0) as doctor:
            status = cli.legacy_main(["--check-environment"])

        self.assertEqual(status, 0)
        doctor.assert_called_once_with()

    def test_legacy_environment_flag_rejects_other_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()):
            status = cli.legacy_main(["--check-environment", "--dry-run"])
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
