"""Lightweight dependency, corpus, and OpenCL environment diagnostics."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
_PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")


@dataclass(frozen=True)
class RequirementSpec:
    """One exact direct dependency pin and its top-level import name."""

    distribution: str
    version: str
    import_name: str


@dataclass(frozen=True)
class CheckResult:
    """One human-readable environment-check outcome."""

    label: str
    status: str
    detail: str
    repair: str | None = None

    @property
    def failed(self):
        return self.status == "FAIL"


def _import_name(distribution):
    return distribution.lower().replace("-", "_").replace(".", "_")


def load_requirements(path=DEFAULT_REQUIREMENTS_PATH):
    """Read the project's exact direct pins without importing third parties."""
    path = Path(path)
    requirements = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _PIN_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"Unsupported requirement at {path}:{line_number}: {raw_line!r}. "
                "Environment checks require exact NAME==VERSION pins."
            )
        distribution, version = match.groups()
        requirements.append(
            RequirementSpec(
                distribution=distribution,
                version=version,
                import_name=_import_name(distribution),
            )
        )
    return tuple(requirements)


def pip_install_command(requirements_path=DEFAULT_REQUIREMENTS_PATH):
    """Return a copy/paste-safe install command for the active interpreter."""
    return subprocess.list2cmdline(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(Path(requirements_path)),
        ]
    )


def reuters_install_command():
    """Return the Reuters corpus installation command for this interpreter."""
    return subprocess.list2cmdline(
        [sys.executable, "-m", "nltk.downloader", "reuters"]
    )


def missing_required_packages(requirements_path=DEFAULT_REQUIREMENTS_PATH):
    """Return direct requirements absent from the active interpreter."""
    missing = []
    for requirement in load_requirements(requirements_path):
        try:
            metadata.version(requirement.distribution)
        except metadata.PackageNotFoundError:
            missing.append(requirement)
    return tuple(missing)


def format_missing_package_message(
    missing,
    requirements_path=DEFAULT_REQUIREMENTS_PATH,
):
    """Explain a missing-package startup failure without a Python traceback."""
    names = ", ".join(requirement.distribution for requirement in missing)
    check_command = subprocess.list2cmdline(
        [sys.executable, str(PROJECT_ROOT / "model" / "main.py"), "--check-environment"]
    )
    return "\n".join(
        [
            f"Missing required Python package(s): {names}",
            f"Interpreter: {sys.executable}",
            "Install this project's dependencies with:",
            f"  {pip_install_command(requirements_path)}",
            "Then verify the environment with:",
            f"  {check_command}",
        ]
    )


def check_python_packages(requirements_path=DEFAULT_REQUIREMENTS_PATH):
    """Check exact versions and prove every direct dependency can import."""
    results = []
    for requirement in load_requirements(requirements_path):
        try:
            installed_version = metadata.version(requirement.distribution)
        except metadata.PackageNotFoundError:
            results.append(
                CheckResult(
                    requirement.distribution,
                    "FAIL",
                    f"not installed (expected {requirement.version})",
                    pip_install_command(requirements_path),
                )
            )
            continue

        try:
            importlib.import_module(requirement.import_name)
        except Exception as exc:  # Report binary and API import failures too.
            results.append(
                CheckResult(
                    requirement.distribution,
                    "FAIL",
                    f"{installed_version} is installed but import failed: {exc}",
                    pip_install_command(requirements_path),
                )
            )
            continue

        if installed_version != requirement.version:
            results.append(
                CheckResult(
                    requirement.distribution,
                    "FAIL",
                    f"installed {installed_version}; expected {requirement.version}",
                    pip_install_command(requirements_path),
                )
            )
            continue

        results.append(CheckResult(requirement.distribution, "OK", installed_version))
    return tuple(results)


def check_dependency_graph(requirements_path=DEFAULT_REQUIREMENTS_PATH):
    """Ask pip to validate installed packages' declared dependencies."""
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    detail = (completed.stdout or completed.stderr).strip()
    if completed.returncode:
        return CheckResult(
            "Python dependency graph",
            "FAIL",
            detail or f"pip check exited {completed.returncode}",
            pip_install_command(requirements_path),
        )
    return CheckResult(
        "Python dependency graph",
        "OK",
        detail or "pip check passed",
    )


def check_reuters_corpus():
    """Check that NLTK can load the corpus used by the configured workflow."""
    try:
        reuters = importlib.import_module("nltk.corpus").reuters
    except (ImportError, AttributeError) as exc:
        return CheckResult("Reuters corpus", "SKIP", f"NLTK unavailable: {exc}")

    try:
        document_count = len(reuters.fileids())
    except LookupError:
        return CheckResult(
            "Reuters corpus",
            "FAIL",
            "not downloaded",
            reuters_install_command(),
        )
    return CheckResult(
        "Reuters corpus",
        "OK",
        f"{document_count} documents available",
    )


def check_opencl():
    """Check that PyOpenCL sees at least one usable platform and device."""
    try:
        cl = importlib.import_module("pyopencl")
    except ImportError as exc:
        return CheckResult("OpenCL", "SKIP", f"PyOpenCL unavailable: {exc}")

    try:
        platforms = cl.get_platforms()
        devices = [
            (platform.name, device.name)
            for platform in platforms
            for device in platform.get_devices()
        ]
    except Exception as exc:  # Driver errors use backend-specific PyOpenCL types.
        return CheckResult("OpenCL", "FAIL", f"discovery failed: {exc}")

    if not devices:
        return CheckResult("OpenCL", "FAIL", "no OpenCL devices discovered")

    summary = "; ".join(f"{platform} / {device}" for platform, device in devices)
    return CheckResult("OpenCL", "OK", summary)


def collect_environment_checks(requirements_path=DEFAULT_REQUIREMENTS_PATH):
    """Collect comprehensive checks without coupling them to the model runtime."""
    return (
        *check_python_packages(requirements_path),
        check_dependency_graph(requirements_path),
        check_reuters_corpus(),
        check_opencl(),
    )


def run_environment_check(requirements_path=DEFAULT_REQUIREMENTS_PATH, stream=None):
    """Print the environment report and return a process-style status code."""
    if stream is None:
        stream = sys.stdout

    print("Canvas environment check", file=stream)
    print(f"Interpreter: {sys.executable}", file=stream)
    print(f"Python: {sys.version.split()[0]}", file=stream)
    print(f"Requirements: {Path(requirements_path)}", file=stream)

    try:
        results = collect_environment_checks(requirements_path)
    except (OSError, ValueError) as exc:
        print(f"[FAIL] requirements: {exc}", file=stream)
        return 1

    for result in results:
        print(f"[{result.status}] {result.label}: {result.detail}", file=stream)
        if result.repair:
            print(f"       Repair: {result.repair}", file=stream)

    failed = sum(result.failed for result in results)
    if failed:
        print(f"Environment check failed: {failed} problem(s).", file=stream)
        return 1

    print("Environment check passed.", file=stream)
    return 0
