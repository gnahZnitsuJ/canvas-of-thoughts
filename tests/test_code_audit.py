import ast
import importlib.util
import unittest
from pathlib import Path


AUDIT_PATH = Path(__file__).resolve().parents[1] / "tools" / "code_audit.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("code_audit_under_test", AUDIT_PATH)
code_audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(code_audit)

FunctionCollector = code_audit.FunctionCollector
compare_snapshots = code_audit.compare_snapshots
max_control_nesting = code_audit.max_control_nesting
render_summary = code_audit.render_summary
resolve_dependencies = code_audit.resolve_dependencies
strongly_connected_components = code_audit.strongly_connected_components


class CodeAuditTests(unittest.TestCase):
    def test_symbol_names_and_kinds_are_stable_and_qualified(self):
        tree = ast.parse(
            """
class service:
    async def handle(self, value):
        if value:
            for item in value:
                return item

def outer():
    def inner():
        return 1
    return inner()
"""
        )
        collector = FunctionCollector()
        collector.visit(tree)

        by_name = {item["qualified_name"]: item for item in collector.functions}
        self.assertEqual(by_name["service.handle"]["kind"], "async_method")
        self.assertEqual(by_name["service.handle"]["max_control_nesting"], 2)
        self.assertEqual(by_name["outer.inner"]["kind"], "function")

    def test_nesting_does_not_include_nested_function_body(self):
        function = ast.parse(
            """
def outer():
    if ready:
        def inner():
            if one:
                while two:
                    return three
        return inner()
"""
        ).body[0]
        self.assertEqual(max_control_nesting(function), 1)

    def test_dependency_resolution_and_cycles_use_internal_file_ids(self):
        files = [
            self._file("pkg/a.py", "pkg.a", ["pkg.a"], "pkg.b"),
            self._file("pkg/b.py", "pkg.b", ["pkg.b"], "pkg.a"),
            self._file("pkg/c.py", "pkg.c", ["pkg.c"], "external"),
        ]

        cycles = resolve_dependencies(files)

        self.assertEqual(len(cycles), 1)
        self.assertEqual(
            cycles[0]["file_ids"], ["file:pkg/a.py", "file:pkg/b.py"]
        )
        self.assertEqual(files[2]["dependencies"]["out_degree"], 0)

    def test_tarjan_ignores_acyclic_singletons(self):
        components = strongly_connected_components(
            ["a", "b", "c"], {"a": {"b"}, "b": set(), "c": {"c"}}
        )
        self.assertEqual(components, [["c"]])

    def test_snapshot_comparison_reports_raw_metric_deltas(self):
        before = self._snapshot(cyclomatic=3, cognitive=5, coverage=80.0)
        after = self._snapshot(cyclomatic=5, cognitive=4, coverage=90.0)

        comparison = compare_snapshots(before, after)

        change = comparison["symbol_changes"][0]
        self.assertEqual(change["status"], "changed")
        self.assertEqual(change["deltas"]["cyclomatic_complexity"], 2)
        self.assertEqual(change["deltas"]["cognitive_complexity"], -1)
        self.assertEqual(change["deltas"]["coverage_line_percent"], 10.0)

    def test_summary_names_selected_telemetry_path(self):
        snapshot = {
            "snapshot": {
                "created_at": "2026-01-01T00:00:00Z",
                "git": {"revision": "abc", "is_dirty": False},
            },
            "aggregates": {
                "file_count": 0,
                "function_method_count": 0,
                "sloc": 0,
                "cyclomatic_complexity": {"maximum": None},
                "cognitive_complexity": {"maximum": None},
                "internal_dependency_edge_count": 0,
                "dependency_cycle_count": 0,
            },
            "coverage": {"status": "not_requested", "aggregate": {}},
            "files": [],
            "symbols": [],
            "dependency_cycles": [],
            "diagnostics": [],
        }

        summary = render_summary(snapshot, ".audit/custom.json")

        self.assertIn("- Full telemetry: `.audit/custom.json`", summary)

    @staticmethod
    def _file(path, module, aliases, imported_module):
        return {
            "id": f"file:{path}",
            "path": path,
            "module": module,
            "module_aliases": aliases,
            "imports": [
                {
                    "line": 1,
                    "module": imported_module,
                    "imported_name": None,
                }
            ],
        }

    @staticmethod
    def _snapshot(cyclomatic, cognitive, coverage):
        return {
            "snapshot": {
                "created_at": "2026-01-01T00:00:00Z",
                "git": {"revision": "abc"},
            },
            "symbols": [
                {
                    "id": "python:pkg/a.py::f",
                    "path": "pkg/a.py",
                    "qualified_name": "f",
                    "location": {"start_line": 1, "end_line": 2},
                    "metrics": {
                        "cyclomatic_complexity": cyclomatic,
                        "cognitive_complexity": cognitive,
                        "max_control_nesting": 1,
                        "source_lines": 2,
                        "sloc": 2,
                    },
                    "coverage": {
                        "status": "available",
                        "line_percent": coverage,
                    },
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
