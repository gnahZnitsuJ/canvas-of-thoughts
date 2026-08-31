#!/usr/bin/env python3
"""Prove selected contract tests reject small, plausible source faults.

The script copies only source and tests into a temporary workspace. It never
rewrites the active checkout, which matters for Windows/OneDrive worktrees.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Sentinel:
    """One exact source fault and the narrow test module expected to reject it."""

    name: str
    path: str
    original: str
    mutated: str
    test_pattern: str


SENTINELS = (
    Sentinel(
        "reject-output-as-source",
        "model/architecture/validation.py",
        'if source_port.direction != "output":',
        'if source_port.direction == "output":',
        "test_architecture_contracts.py",
    ),
    Sentinel(
        "require-transform-for-dimension-change",
        "model/architecture/validation.py",
        "and connection.transform is None",
        "and connection.transform is not None",
        "test_architecture_contracts.py",
    ),
    Sentinel(
        "reject-vocabulary-mismatch",
        "model/architecture/validation.py",
        "and source_port.vocabulary_id != target_port.vocabulary_id",
        "and source_port.vocabulary_id == target_port.vocabulary_id",
        "test_architecture_contracts.py",
    ),
    Sentinel(
        "deep-copy-architecture-variants",
        "model/architecture/spec.py",
        "copied = deepcopy(self)",
        "copied = self",
        "test_architecture_contracts.py",
    ),
    Sentinel(
        "preserve-component-build-order",
        "model/architecture/signatures.py",
        '"component_build_order": list(spec.components),',
        '"component_build_order": sorted(spec.components),',
        "test_architecture_contracts.py",
    ),
    Sentinel(
        "flag-undeclared-control-variation",
        "scripts/compare_telemetry.py",
        "unexpected = control_differences - allowed",
        "unexpected = allowed - control_differences",
        "test_compare_telemetry.py",
    ),
    Sentinel(
        "report-removed-connections",
        "scripts/compare_telemetry.py",
        "for encoded in sorted(reference_connections - current_connections):",
        "for encoded in sorted(current_connections - reference_connections):",
        "test_compare_telemetry.py",
    ),
    Sentinel(
        "preserve-shell-quit-alias",
        "model/app/shell.py",
        'if command in ("/exit", "/quit"):',
        'if command == "/exit":',
        "test_shell_commands.py",
    ),
    Sentinel(
        "dry-run-stops-before-data-loading",
        "model/app/workflow.py",
        "    if args.dry_run:\n"
        "        print_dry_run_summary(args, workflow_plan, training_config)\n"
        "        return\n",
        "    if args.dry_run:\n"
        "        print_dry_run_summary(args, workflow_plan, training_config)\n",
        "test_workflow_orchestration.py",
    ),
    Sentinel(
        "render-changed-control-section",
        "scripts/compare_telemetry.py",
        "    if control_differences:\n",
        "    if False and control_differences:\n",
        "test_compare_telemetry.py",
    ),
    Sentinel(
        "render-architecture-change-section",
        "scripts/compare_telemetry.py",
        '    if "architecture_signature" in differences:\n',
        '    if False and "architecture_signature" in differences:\n',
        "test_compare_telemetry.py",
    ),
)


def _copy_test_surface(destination: Path) -> None:
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        "*.model",
        "checkpoints",
        "results",
    )
    for directory in ("model", "scripts", "tests"):
        shutil.copytree(REPO_ROOT / directory, destination / directory, ignore=ignore)


def _run_tests(workspace: Path, pattern: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            pattern,
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _apply_exact_mutation(workspace: Path, sentinel: Sentinel) -> None:
    path = workspace / sentinel.path
    source = path.read_text(encoding="utf-8-sig")
    occurrences = source.count(sentinel.original)
    if occurrences != 1:
        raise RuntimeError(
            f"{sentinel.name}: expected one source fragment, found {occurrences}"
        )
    path.write_text(
        source.replace(sentinel.original, sentinel.mutated, 1),
        encoding="utf-8",
    )


def verify(timeout: float) -> int:
    """Return zero only when clean baselines pass and every sentinel is killed."""
    with tempfile.TemporaryDirectory(prefix="canvas-test-sentinels-") as directory:
        root = Path(directory)
        baseline = root / "baseline"
        _copy_test_surface(baseline)

        patterns = sorted({sentinel.test_pattern for sentinel in SENTINELS})
        for pattern in patterns:
            result = _run_tests(baseline, pattern, timeout)
            if result.returncode != 0:
                print(f"BASELINE FAILED: {pattern}", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                return 2

        killed = 0
        for index, sentinel in enumerate(SENTINELS):
            workspace = root / f"mutant-{index}"
            shutil.copytree(baseline, workspace)
            _apply_exact_mutation(workspace, sentinel)
            result = _run_tests(workspace, sentinel.test_pattern, timeout)
            if result.returncode == 0:
                print(f"SURVIVED: {sentinel.name}")
                continue
            killed += 1
            print(f"KILLED:   {sentinel.name}")

        print(f"Mutation sentinels killed: {killed}/{len(SENTINELS)}")
        return 0 if killed == len(SENTINELS) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Per-test-process timeout in seconds (default: 15).",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return verify(args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
