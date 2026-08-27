"""Deterministic code-complexity, dependency, Git, and coverage telemetry.

The snapshot is factual input for later human or agent interpretation.  This
module deliberately does not label code as good/bad or prescribe refactors.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


SCHEMA_NAME = "canvas.code-audit"
DEFAULT_SNAPSHOT = ".audit/snapshot.json"
DEFAULT_SUMMARY = ".audit/summary.md"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.ExceptHandler,
    ast.comprehension,
)


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _relative(root, path):
    return path.resolve().relative_to(root.resolve()).as_posix()


def _file_id(path):
    return f"file:{path}"


def _symbol_id(path, qualified_name):
    return f"python:{path}::{qualified_name}"


def _percent(covered, total):
    if not total:
        return None
    return round(100.0 * covered / total, 2)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _matches(path, patterns):
    candidate = PurePosixPath(path)
    return any(candidate.match(pattern) for pattern in patterns)


def discover_sources(root, config, diagnostics):
    exclusion_patterns = [item["pattern"] for item in config["exclusions"]]
    paths = []
    for source_root in config["source_roots"]:
        directory = root / source_root
        if not directory.is_dir():
            diagnostics.append(
                {
                    "severity": "warning",
                    "category": "scope",
                    "message": f"Configured source root does not exist: {source_root}",
                }
            )
            continue
        for path in directory.rglob("*.py"):
            relative = _relative(root, path)
            if path.is_file() and not _matches(relative, exclusion_patterns):
                paths.append(path)
    return sorted(set(paths), key=lambda item: _relative(root, item))


class FunctionCollector(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.scope_kinds = []
        self.functions = []

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.scope_kinds.append("class")
        for statement in node.body:
            self.visit(statement)
        self.scope_kinds.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self._visit_function(node, "method" if self._inside_class() else "function")

    def visit_AsyncFunctionDef(self, node):
        self._visit_function(
            node, "async_method" if self._inside_class() else "async_function"
        )

    def _inside_class(self):
        return bool(self.scope_kinds and self.scope_kinds[-1] == "class")

    def _visit_function(self, node, kind):
        qualified_name = ".".join(self.scope + [node.name])
        arguments = node.args
        parameter_count = (
            len(arguments.posonlyargs)
            + len(arguments.args)
            + len(arguments.kwonlyargs)
            + int(arguments.vararg is not None)
            + int(arguments.kwarg is not None)
        )
        self.functions.append(
            {
                "name": node.name,
                "qualified_name": qualified_name,
                "kind": kind,
                "start_line": node.lineno,
                "start_column": node.col_offset,
                "end_line": node.end_lineno,
                "end_column": node.end_col_offset,
                "parameter_count": parameter_count,
                "max_control_nesting": max_control_nesting(node),
                "node": node,
            }
        )
        self.scope.append(node.name)
        self.scope_kinds.append("function")
        for statement in node.body:
            self.visit(statement)
        self.scope_kinds.pop()
        self.scope.pop()


def max_control_nesting(function_node):
    maximum = 0

    def walk(node, depth):
        nonlocal maximum
        if node is not function_node and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return
        nested_depth = depth + 1 if isinstance(node, CONTROL_NODES) else depth
        maximum = max(maximum, nested_depth)
        for child in ast.iter_child_nodes(node):
            walk(child, nested_depth)

    for statement in function_node.body:
        walk(statement, 0)
    return maximum


def _flatten_radon_functions(blocks):
    found = {}

    def add(block):
        fields = getattr(block, "_fields", ())
        if "complexity" in fields:
            found[(block.lineno, block.name)] = block.complexity
        for closure in getattr(block, "closures", ()):
            add(closure)
        for method in getattr(block, "methods", ()):
            add(method)
        for inner_class in getattr(block, "inner_classes", ()):
            add(inner_class)

    for block in blocks:
        add(block)
    return found


def _module_names(relative_path):
    parts = list(PurePosixPath(relative_path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    canonical = ".".join(parts)
    names = {canonical} if canonical else set()
    if parts and parts[0] == "model" and len(parts) > 1:
        names.add(".".join(parts[1:]))
    return canonical, sorted(name for name in names if name)


def _import_requests(tree, module_name, is_package):
    requests = []
    package = module_name if is_package else module_name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                requests.append(
                    {
                        "line": node.lineno,
                        "module": alias.name,
                        "imported_name": None,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package_parts = package.split(".") if package else []
                keep = max(0, len(package_parts) - node.level + 1)
                relative_base = package_parts[:keep]
                if base:
                    relative_base.extend(base.split("."))
                base = ".".join(relative_base)
            for alias in node.names:
                requests.append(
                    {
                        "line": node.lineno,
                        "module": base,
                        "imported_name": alias.name,
                    }
                )
    return sorted(
        requests,
        key=lambda item: (item["line"], item["module"], item["imported_name"] or ""),
    )


def analyze_python_file(root, path, diagnostics):
    relative = _relative(root, path)
    source = path.read_text(encoding="utf-8-sig")
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=relative)

    from radon.complexity import cc_visit
    from radon.raw import analyze as raw_analyze

    raw = raw_analyze(source)
    cyclomatic = _flatten_radon_functions(cc_visit(source))
    cognitive = {}
    cognitive_by_line = {}
    cognitive_status = "available"
    try:
        import complexipy

        result = complexipy.code_complexity(source, no_ignore=True)
        for function in result.functions:
            cognitive_item = {
                "value": function.complexity,
                "contributors": [
                    {"line": item.line, "increment": item.complexity}
                    for item in function.line_complexities
                    if item.complexity
                ],
            }
            cognitive[(function.line_start, function.name)] = cognitive_item
            cognitive_by_line[function.line_start] = cognitive_item
    except (ImportError, RuntimeError, ValueError) as exc:
        cognitive_status = "unavailable"
        diagnostics.append(
            {
                "severity": "warning",
                "category": "cognitive_complexity",
                "path": relative,
                "message": str(exc),
            }
        )

    collector = FunctionCollector()
    collector.visit(tree)
    symbols = []
    for item in collector.functions:
        start = item["start_line"]
        end = item["end_line"]
        segment = "\n".join(source_lines[start - 1 : end]) + "\n"
        function_raw = raw_analyze(segment)
        cognitive_item = cognitive.get((start, item["name"])) or cognitive_by_line.get(start)
        qualified_name = item["qualified_name"]
        symbols.append(
            {
                "id": _symbol_id(relative, qualified_name),
                "file_id": _file_id(relative),
                "path": relative,
                "module": None,
                "name": item["name"],
                "qualified_name": qualified_name,
                "kind": item["kind"],
                "location": {
                    "start_line": start,
                    "start_column": item["start_column"],
                    "end_line": end,
                    "end_column": item["end_column"],
                },
                "content_sha256": hashlib.sha256(
                    segment.encode("utf-8")
                ).hexdigest(),
                "metrics": {
                    "cyclomatic_complexity": cyclomatic.get((start, item["name"])),
                    "cognitive_complexity": (
                        cognitive_item["value"] if cognitive_item else None
                    ),
                    "cognitive_contributors": (
                        cognitive_item["contributors"] if cognitive_item else []
                    ),
                    "max_control_nesting": item["max_control_nesting"],
                    "parameter_count": item["parameter_count"],
                    "source_lines": end - start + 1,
                    "sloc": function_raw.sloc,
                    "logical_lines": function_raw.lloc,
                },
                "coverage": {"status": "not_collected"},
            }
        )

    canonical_module, aliases = _module_names(relative)
    for symbol in symbols:
        symbol["module"] = canonical_module
    return {
        "id": _file_id(relative),
        "path": relative,
        "language": "python",
        "module": canonical_module,
        "module_aliases": aliases,
        "content_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "metrics": {
            "loc": raw.loc,
            "sloc": raw.sloc,
            "logical_lines": raw.lloc,
            "comment_lines": raw.comments,
            "multi_comment_lines": raw.multi,
            "blank_lines": raw.blank,
        },
        "imports": _import_requests(tree, canonical_module, path.name == "__init__.py"),
        "dependencies": {},
        "git": {"status": "not_collected"},
        "coverage": {"status": "not_collected"},
        "analysis": {"cognitive_complexity": cognitive_status},
    }, symbols


def _module_index(files):
    return {
        name: file["id"]
        for file in files
        for name in file["module_aliases"]
    }


def _resolve_import(request, source_id, module_index):
    module = request["module"]
    imported_name = request["imported_name"]
    candidates = []
    if imported_name and imported_name != "*":
        candidates.append(".".join(part for part in (module, imported_name) if part))
    if module:
        candidates.append(module)
    for candidate in candidates:
        target = module_index.get(candidate)
        if target and target != source_id:
            return [target]
    return []


def _dependency_edges(files, module_index):
    outgoing = defaultdict(set)
    for file in files:
        for request in file["imports"]:
            resolved = _resolve_import(request, file["id"], module_index)
            request["resolved_file_ids"] = resolved
            outgoing[file["id"]].update(resolved)
    incoming = defaultdict(set)
    for source, targets in outgoing.items():
        for target in targets:
            incoming[target].add(source)
    return outgoing, incoming


def _dependency_cycles(files, outgoing):
    cycles = strongly_connected_components(
        [file["id"] for file in files], outgoing
    )
    cycle_lookup = {}
    cycle_records = []
    for index, members in enumerate(cycles, start=1):
        cycle_id = f"dependency-cycle:{index}"
        sorted_members = sorted(members)
        cycle_records.append({"id": cycle_id, "file_ids": sorted_members})
        for member in members:
            cycle_lookup[member] = cycle_id
    return cycle_records, cycle_lookup


def resolve_dependencies(files):
    outgoing, incoming = _dependency_edges(files, _module_index(files))
    cycle_records, cycle_lookup = _dependency_cycles(files, outgoing)
    for file in files:
        file_id = file["id"]
        file["dependencies"] = {
            "imports_internal": sorted(outgoing[file_id]),
            "imported_by_internal": sorted(incoming[file_id]),
            "out_degree": len(outgoing[file_id]),
            "in_degree": len(incoming[file_id]),
            "cycle_id": cycle_lookup.get(file_id),
        }
    return cycle_records


def strongly_connected_components(nodes, edges):
    index = 0
    stack = []
    indexes = {}
    lowlinks = {}
    on_stack = set()
    components = []

    def connect(node):
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in edges.get(node, ()):
            if target not in indexes:
                connect(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[target])

        if lowlinks[node] == indexes[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or node in edges.get(node, ()):
                components.append(component)

    for node in nodes:
        if node not in indexes:
            connect(node)
    return sorted(components, key=lambda group: sorted(group))


def _git(root, *arguments, allow_failure=False):
    command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        "-c",
        "core.quotepath=false",
        "-C",
        str(root),
        *arguments,
    ]
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode and not allow_failure:
        raise RuntimeError(result.stderr.strip() or "Git command failed")
    return result


def _repository_git_state(root):
    revision = _git(root, "rev-parse", "HEAD").stdout.strip()
    status_result = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    status_entries = [
        {"status": line[:2], "path": line[3:]}
        for line in status_result.stdout.splitlines()
        if len(line) >= 3
    ]
    version = _git(root, "--version").stdout.strip().removeprefix("git version ")
    return revision, status_entries, version


def _git_unavailable(message):
    return {
        "status": "unavailable",
        "revision": None,
        "is_dirty": None,
        "working_tree": [],
        "history": {"status": "unavailable", "message": message},
    }


def _history_log(root, config, diagnostics):
    max_commits = int(config["git"]["max_commits"])
    log_result = _git(
        root,
        "log",
        f"--max-count={max_commits}",
        "--first-parent",
        "-m",
        "--no-renames",
        "--date=iso-strict",
        "--format=%x1e%H%x1f%cI",
        "--numstat",
        "--",
        *config["source_roots"],
        allow_failure=True,
    )
    if log_result.returncode:
        diagnostics.append(
            {
                "severity": "warning",
                "category": "git_history",
                "message": log_result.stderr.strip(),
            }
        )
    return log_result, max_commits


def _parse_history(log_output, current_paths):
    file_commits = defaultdict(set)
    file_dates = defaultdict(list)
    additions = Counter()
    deletions = Counter()
    commit_files = []
    commit_dates = []
    for chunk in log_output.split("\x1e"):
        chunk = chunk.strip("\r\n")
        if not chunk:
            continue
        lines = chunk.splitlines()
        try:
            commit_hash, date_text = lines[0].split("\x1f", 1)
            commit_date = datetime.fromisoformat(date_text)
        except (ValueError, IndexError):
            continue
        changed = set()
        for line in lines[1:]:
            columns = line.split("\t", 2)
            if len(columns) != 3:
                continue
            added, deleted, path = columns
            normalized = path.replace("\\", "/")
            if normalized not in current_paths:
                continue
            changed.add(normalized)
            file_commits[normalized].add(commit_hash)
            file_dates[normalized].append(commit_date)
            if added.isdigit():
                additions[normalized] += int(added)
            if deleted.isdigit():
                deletions[normalized] += int(deleted)
        if changed:
            commit_files.append(changed)
            commit_dates.append(commit_date)
    return {
        "file_commits": file_commits,
        "file_dates": file_dates,
        "additions": additions,
        "deletions": deletions,
        "commit_files": commit_files,
        "commit_dates": commit_dates,
    }


def _history_window(history):
    commit_dates = history["commit_dates"]
    history_oldest = min(commit_dates) if commit_dates else None
    history_newest = max(commit_dates) if commit_dates else None
    history_days = (
        max(1, (history_newest - history_oldest).days + 1)
        if history_oldest and history_newest
        else None
    )
    return history_oldest, history_newest, history_days


def _working_tree_diff(root, config):
    diff_result = _git(
        root,
        "diff",
        "--numstat",
        "HEAD",
        "--",
        *config["source_roots"],
        allow_failure=True,
    )
    working_diff = {}
    for line in diff_result.stdout.splitlines():
        columns = line.split("\t", 2)
        if len(columns) == 3:
            added, deleted, path = columns
            working_diff[path.replace("\\", "/")] = {
                "lines_added": int(added) if added.isdigit() else None,
                "lines_deleted": int(deleted) if deleted.isdigit() else None,
            }
    return working_diff


def _apply_file_history(files, history, history_days, working_diff, snapshot_time):
    file_commits = history["file_commits"]
    file_dates = history["file_dates"]
    additions = history["additions"]
    deletions = history["deletions"]
    for file in files:
        path = file["path"]
        dates = file_dates[path]
        first_seen = min(dates) if dates else None
        last_changed = max(dates) if dates else None
        commit_count = len(file_commits[path])
        file["git"] = {
            "status": "available",
            "commit_count": commit_count,
            "lines_added": additions[path],
            "lines_deleted": deletions[path],
            "churn": additions[path] + deletions[path],
            "first_seen": _iso(first_seen) if first_seen else None,
            "last_changed": _iso(last_changed) if last_changed else None,
            "age_days": (
                (snapshot_time - first_seen.astimezone(timezone.utc)).days
                if first_seen
                else None
            ),
            "days_since_change": (
                (snapshot_time - last_changed.astimezone(timezone.utc)).days
                if last_changed
                else None
            ),
            "commits_per_year_in_history_window": (
                round(commit_count * 365.25 / history_days, 3)
                if history_days
                else None
            ),
            "working_tree_diff": working_diff.get(path),
        }


def _change_coupling(history, config):
    file_commits = history["file_commits"]
    pair_counts = Counter()
    for changed in history["commit_files"]:
        ordered = sorted(changed)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                pair_counts[(left, right)] += 1
    minimum = int(config["git"]["change_coupling_min_commits"])
    coupling = []
    for (left, right), count in pair_counts.items():
        if count < minimum:
            continue
        union = len(file_commits[left] | file_commits[right])
        coupling.append(
            {
                "left_file_id": _file_id(left),
                "right_file_id": _file_id(right),
                "cochange_commit_count": count,
                "jaccard": round(count / union, 4) if union else None,
            }
        )
    coupling.sort(
        key=lambda item: (-item["cochange_commit_count"], -item["jaccard"], item["left_file_id"])
    )
    return coupling[: int(config["git"]["change_coupling_limit"])]


def collect_git(root, files, config, snapshot_time, diagnostics):
    try:
        revision, status_entries, version = _repository_git_state(root)
    except (OSError, RuntimeError) as exc:
        diagnostics.append(
            {"severity": "warning", "category": "git", "message": str(exc)}
        )
        return _git_unavailable(str(exc)), [], None

    log_result, max_commits = _history_log(root, config, diagnostics)
    history_data = _parse_history(
        log_result.stdout, {file["path"] for file in files}
    )
    history_oldest, history_newest, history_days = _history_window(history_data)
    working_diff = _working_tree_diff(root, config)
    _apply_file_history(
        files, history_data, history_days, working_diff, snapshot_time
    )
    coupling = _change_coupling(history_data, config)
    history = {
        "status": "available" if not log_result.returncode else "degraded",
        "max_commits": max_commits,
        "strategy": config["git"].get("history_strategy", "first-parent"),
        "commits_with_analyzed_files": len(history_data["commit_files"]),
        "oldest_analyzed_commit_time": _iso(history_oldest) if history_oldest else None,
        "newest_analyzed_commit_time": _iso(history_newest) if history_newest else None,
    }
    repository = {
        "status": "available",
        "revision": revision,
        "is_dirty": bool(status_entries),
        "working_tree": status_entries,
        "history": history,
    }
    return repository, coupling, version


def _coverage_command(config):
    command = [sys.executable, "-m", "coverage", "run"]
    if config["coverage"].get("branch", True):
        command.append("--branch")
    command.extend(
        [
            f"--source={','.join(config['source_roots'])}",
            *config["coverage"]["test_command"],
        ]
    )
    return command


def _run_coverage_tests(root, config, data_file, diagnostics):
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["COVERAGE_FILE"] = str(data_file)
    test_result = subprocess.run(
        _coverage_command(config),
        cwd=root,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    status = "available" if test_result.returncode == 0 else "test_failed"
    if test_result.returncode:
        diagnostics.append(
            {
                "severity": "error",
                "category": "coverage_test_run",
                "message": (test_result.stderr or test_result.stdout)[-4000:],
            }
        )
    return test_result, status


def _coverage_measurement(coverage_data, path):
    _, statements, excluded, missing, _ = coverage_data.analysis2(str(path))
    statement_set = set(statements)
    missing_set = set(missing)
    executed_set = statement_set - missing_set
    branch_stats = coverage_data.branch_stats(str(path))
    branch_total = sum(total for total, _ in branch_stats.values())
    branch_covered = sum(taken for _, taken in branch_stats.values())
    record = {
        "status": "available",
        "statement_count": len(statement_set),
        "covered_statement_count": len(executed_set),
        "line_percent": _percent(len(executed_set), len(statement_set)),
        "missing_lines": sorted(missing_set),
        "excluded_lines": sorted(excluded),
        "branch_count": branch_total,
        "covered_branch_count": branch_covered,
        "branch_percent": _percent(branch_covered, branch_total),
    }
    evidence = statement_set, missing_set, branch_stats
    return record, evidence


def _symbol_coverage(symbol, evidence):
    statement_set, missing_set, branch_stats = evidence
    start = symbol["location"]["start_line"]
    end = symbol["location"]["end_line"]
    symbol_statements = {line for line in statement_set if start <= line <= end}
    symbol_missing = {line for line in missing_set if start <= line <= end}
    symbol_branch_stats = {
        line: value for line, value in branch_stats.items() if start <= line <= end
    }
    branch_total = sum(total for total, _ in symbol_branch_stats.values())
    branch_covered = sum(taken for _, taken in symbol_branch_stats.values())
    return {
        "status": "available",
        "statement_count": len(symbol_statements),
        "covered_statement_count": len(symbol_statements - symbol_missing),
        "line_percent": _percent(
            len(symbol_statements - symbol_missing), len(symbol_statements)
        ),
        "missing_lines": sorted(symbol_missing),
        "branch_count": branch_total,
        "covered_branch_count": branch_covered,
        "branch_percent": _percent(branch_covered, branch_total),
    }


def _apply_coverage(coverage_data, root, files, symbols):
    totals = Counter()
    symbols_by_file = defaultdict(list)
    for symbol in symbols:
        symbols_by_file[symbol["file_id"]].append(symbol)

    for file in files:
        try:
            record, evidence = _coverage_measurement(
                coverage_data, root / file["path"]
            )
        except Exception as exc:  # coverage exposes analyzer-specific errors
            file["coverage"] = {"status": "unavailable", "message": str(exc)}
            continue
        file["coverage"] = record
        totals["statements"] += record["statement_count"]
        totals["covered"] += record["covered_statement_count"]
        totals["branches"] += record["branch_count"]
        totals["covered_branches"] += record["covered_branch_count"]
        for symbol in symbols_by_file[file["id"]]:
            symbol["coverage"] = _symbol_coverage(symbol, evidence)
    return {
        "statement_count": totals["statements"],
        "covered_statement_count": totals["covered"],
        "line_percent": _percent(totals["covered"], totals["statements"]),
        "branch_count": totals["branches"],
        "covered_branch_count": totals["covered_branches"],
        "branch_percent": _percent(totals["covered_branches"], totals["branches"]),
    }


def collect_coverage(root, files, symbols, config, data_file, run_tests, diagnostics):
    coverage_version = _package_version("coverage")
    base = {
        "tool_version": coverage_version,
        "test_command": config["coverage"]["test_command"],
    }
    if not run_tests:
        return {"status": "not_requested", **base}
    if coverage_version is None:
        diagnostics.append(
            {
                "severity": "warning",
                "category": "coverage",
                "message": "Coverage.py is not installed; coverage was not collected.",
            }
        )
        return {"status": "tool_unavailable", **base}

    test_result, status = _run_coverage_tests(root, config, data_file, diagnostics)

    try:
        from coverage import Coverage

        coverage_data = Coverage(data_file=str(data_file), branch=True)
        coverage_data.load()
        return {
            "status": status,
            **base,
            "data_file": _relative(root, data_file),
            "test_exit_code": test_result.returncode,
            "aggregate": _apply_coverage(coverage_data, root, files, symbols),
        }
    except Exception as exc:
        diagnostics.append(
            {
                "severity": "error",
                "category": "coverage_read",
                "message": str(exc),
            }
        )
        return {
            "status": "unavailable",
            **base,
            "test_exit_code": test_result.returncode,
        }


def metric_definitions():
    return {
        "cyclomatic_complexity": {
            "scope": "function/method",
            "tool": "radon",
            "definition": "McCabe decision-path count reported by Radon.",
        },
        "cognitive_complexity": {
            "scope": "function/method",
            "tool": "complexipy",
            "definition": "Cognitive complexity reported by Complexipy; contributing lines are retained.",
        },
        "max_control_nesting": {
            "scope": "function/method",
            "tool": "code-audit AST adapter",
            "definition": "Maximum nested Python AST control-flow constructs; a top-level construct has depth 1.",
        },
        "sloc": {
            "scope": "file and function source range",
            "tool": "radon.raw",
            "definition": "Source lines of code reported by Radon.",
        },
        "git.churn": {
            "scope": "file",
            "tool": "git",
            "definition": "Committed lines added plus deleted along first-parent history in the configured window.",
        },
        "git.change_coupling": {
            "scope": "file pair",
            "tool": "git + code-audit adapter",
            "definition": "Commit count touching both current files; Jaccard is intersection/union of their commit sets.",
        },
        "coverage.line_percent": {
            "scope": "file and function source range",
            "tool": "coverage.py + code-audit adapter",
            "definition": "Executed executable statement lines divided by executable statement lines in the source range.",
        },
        "dependency_degree": {
            "scope": "file",
            "tool": "Python AST + code-audit adapter",
            "definition": "Distinct internal files imported (out) or importing this file (in).",
        },
    }


def collect_snapshot(root, config, run_coverage):
    diagnostics = []
    if _package_version("radon") is None:
        raise RuntimeError(
            "Radon is required for collection. Install requirements-audit.txt first."
        )
    snapshot_time = _now()
    source_paths = discover_sources(root, config, diagnostics)
    files = []
    symbols = []
    for path in source_paths:
        try:
            file_record, file_symbols = analyze_python_file(root, path, diagnostics)
            files.append(file_record)
            symbols.extend(file_symbols)
        except (OSError, SyntaxError, UnicodeError) as exc:
            diagnostics.append(
                {
                    "severity": "error",
                    "category": "source_analysis",
                    "path": _relative(root, path),
                    "message": str(exc),
                }
            )

    dependency_cycles = resolve_dependencies(files)
    repository, change_coupling, git_version = collect_git(
        root, files, config, snapshot_time, diagnostics
    )
    coverage_info = collect_coverage(
        root,
        files,
        symbols,
        config,
        root / ".audit" / ".coverage",
        run_coverage,
        diagnostics,
    )

    cyclomatic_values = [
        item["metrics"]["cyclomatic_complexity"]
        for item in symbols
        if item["metrics"]["cyclomatic_complexity"] is not None
    ]
    cognitive_values = [
        item["metrics"]["cognitive_complexity"]
        for item in symbols
        if item["metrics"]["cognitive_complexity"] is not None
    ]
    configuration_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": {"name": SCHEMA_NAME, "version": config["schema_version"]},
        "snapshot": {
            "created_at": _iso(snapshot_time),
            "repository_root": ".",
            "git": repository,
            "configuration": config,
            "configuration_sha256": configuration_hash,
            "scope": {
                "language": "python",
                "source_roots": config["source_roots"],
                "exclusions": config["exclusions"],
                "analyzed_paths": [file["path"] for file in files],
            },
            "tools": {
                "python": {"status": "available", "version": sys.version.split()[0]},
                "radon": {"status": "available", "version": _package_version("radon")},
                "complexipy": {
                    "status": (
                        "available" if _package_version("complexipy") else "unavailable"
                    ),
                    "version": _package_version("complexipy"),
                },
                "coverage": {
                    "status": coverage_info["status"],
                    "version": _package_version("coverage"),
                },
                "git": {
                    "status": repository["status"],
                    "version": git_version,
                },
            },
        },
        "metric_definitions": metric_definitions(),
        "aggregates": {
            "file_count": len(files),
            "function_method_count": len(symbols),
            "sloc": sum(file["metrics"]["sloc"] for file in files),
            "cyclomatic_complexity": {
                "sum": sum(cyclomatic_values),
                "mean": (
                    round(sum(cyclomatic_values) / len(cyclomatic_values), 3)
                    if cyclomatic_values
                    else None
                ),
                "maximum": max(cyclomatic_values) if cyclomatic_values else None,
            },
            "cognitive_complexity": {
                "sum": sum(cognitive_values),
                "mean": (
                    round(sum(cognitive_values) / len(cognitive_values), 3)
                    if cognitive_values
                    else None
                ),
                "maximum": max(cognitive_values) if cognitive_values else None,
            },
            "internal_dependency_edge_count": sum(
                file["dependencies"]["out_degree"] for file in files
            ),
            "dependency_cycle_count": len(dependency_cycles),
        },
        "coverage": coverage_info,
        "files": sorted(files, key=lambda item: item["path"]),
        "symbols": sorted(symbols, key=lambda item: (item["path"], item["location"]["start_line"])),
        "dependency_cycles": dependency_cycles,
        "change_coupling": change_coupling,
        "diagnostics": diagnostics,
    }


def _file_lookup(snapshot):
    return {item["id"]: item for item in snapshot["files"]}


def _symbol_view_item(symbol, file):
    return {
        "id": symbol["id"],
        "path": symbol["path"],
        "qualified_name": symbol["qualified_name"],
        "kind": symbol["kind"],
        "location": symbol["location"],
        "metrics": symbol["metrics"],
        "coverage": symbol["coverage"],
        "file_git": file["git"],
        "file_dependencies": file["dependencies"],
    }


def _file_view_item(file):
    coverage = {
        key: value
        for key, value in file["coverage"].items()
        if key not in {"missing_lines", "excluded_lines"}
    }
    if file["coverage"].get("status") == "available":
        coverage["missing_statement_count"] = len(
            file["coverage"].get("missing_lines", [])
        )
    return {
        "id": file["id"],
        "path": file["path"],
        "module": file["module"],
        "metrics": file["metrics"],
        "coverage": coverage,
        "git": file["git"],
        "dependencies": file["dependencies"],
    }


def _filtered_symbols(snapshot, path, symbol_name):
    files = _file_lookup(snapshot)
    normalized_path = path.replace("\\", "/") if path else None
    return [
        _symbol_view_item(item, files[item["file_id"]])
        for item in snapshot["symbols"]
        if (not normalized_path or item["path"].startswith(normalized_path))
        and (not symbol_name or symbol_name.lower() in item["qualified_name"].lower())
    ]


def _filtered_files(snapshot, path):
    normalized_path = path.replace("\\", "/") if path else None
    return [
        _file_view_item(item)
        for item in snapshot["files"]
        if not normalized_path or item["path"].startswith(normalized_path)
    ]


def _complexity_view(snapshot, symbols, path):
    del snapshot, path
    return (
        sorted(
            symbols,
            key=lambda item: (
                -(item["metrics"]["cyclomatic_complexity"] or -1),
                -(item["metrics"]["cognitive_complexity"] or -1),
                -item["metrics"]["sloc"],
                item["id"],
            ),
        ),
        "Lexicographic: cyclomatic, cognitive, then SLOC descending.",
    )


def _complex_change_view(snapshot, symbols, path):
    del snapshot, path
    for item in symbols:
        commits = item["file_git"].get("commit_count") or 0
        cyclomatic = item["metrics"]["cyclomatic_complexity"] or 0
        item["selection"] = {
            "value": cyclomatic * commits,
            "formula": "cyclomatic_complexity * file.commit_count",
        }
    return (
        sorted(symbols, key=lambda item: (-item["selection"]["value"], item["id"])),
        "Navigation ordering only: cyclomatic complexity multiplied by raw file commit count.",
    )


def _complex_uncovered_view(snapshot, symbols, path):
    del snapshot, path
    available = [
        item for item in symbols if item["coverage"].get("status") == "available"
    ]
    for item in available:
        missing = len(item["coverage"].get("missing_lines", []))
        cyclomatic = item["metrics"]["cyclomatic_complexity"] or 0
        item["selection"] = {
            "value": cyclomatic * missing,
            "formula": "cyclomatic_complexity * uncovered_statement_lines",
        }
    return (
        sorted(
            available,
            key=lambda item: (-item["selection"]["value"], item["id"]),
        ),
        "Navigation ordering only: cyclomatic complexity multiplied by uncovered executable lines in the symbol range.",
    )


def _recent_view(snapshot, symbols, path):
    del snapshot, path
    return (
        sorted(
            symbols,
            key=lambda item: (
                item["file_git"].get("days_since_change")
                if item["file_git"].get("days_since_change") is not None
                else math.inf,
                -(item["metrics"]["cyclomatic_complexity"] or -1),
                item["id"],
            ),
        ),
        "Days since the file's last committed change ascending, then cyclomatic complexity descending.",
    )


def _change_view(snapshot, symbols, path):
    del symbols
    items = _filtered_files(snapshot, path)
    items.sort(
        key=lambda item: (
            -(item["git"].get("commit_count") or 0),
            -(item["git"].get("churn") or 0),
            item["id"],
        )
    )
    return items, "Committed change count, then line churn, descending."


def _dependencies_view(snapshot, symbols, path):
    del symbols
    items = _filtered_files(snapshot, path)
    items.sort(
        key=lambda item: (
            -(item["dependencies"]["in_degree"] + item["dependencies"]["out_degree"]),
            -item["dependencies"]["in_degree"],
            item["id"],
        )
    )
    return (
        items,
        "Internal import total degree, then in-degree, descending; cycles are reported separately.",
    )


def _coupling_view(snapshot, symbols, path):
    del symbols
    normalized_path = path.replace("\\", "/") if path else None
    items = [
        item
        for item in snapshot["change_coupling"]
        if not normalized_path
        or item["left_file_id"].removeprefix("file:").startswith(normalized_path)
        or item["right_file_id"].removeprefix("file:").startswith(normalized_path)
    ]
    return (
        items,
        "Committed co-change count descending, then raw commit-set Jaccard similarity.",
    )


QUERY_BUILDERS = {
    "complexity": _complexity_view,
    "complex-change": _complex_change_view,
    "complex-uncovered": _complex_uncovered_view,
    "recent": _recent_view,
    "change": _change_view,
    "dependencies": _dependencies_view,
    "coupling": _coupling_view,
}


def query_snapshot(snapshot, view, limit=10, path=None, symbol_name=None):
    try:
        builder = QUERY_BUILDERS[view]
    except KeyError as exc:
        raise ValueError(f"Unknown view: {view}")
    symbols = _filtered_symbols(snapshot, path, symbol_name)
    items, derivation = builder(snapshot, symbols, path)

    return {
        "schema": {"name": "canvas.code-audit.query", "version": "1.0.0"},
        "source_snapshot": {
            "revision": snapshot["snapshot"]["git"]["revision"],
            "created_at": snapshot["snapshot"]["created_at"],
        },
        "view": view,
        "derivation": derivation,
        "available_count": len(items),
        "returned_count": min(limit, len(items)),
        "items": items[:limit],
        "dependency_cycles": snapshot["dependency_cycles"] if view == "dependencies" else None,
    }


def compare_snapshots(before, after, limit=20):
    before_symbols = {item["id"]: item for item in before["symbols"]}
    after_symbols = {item["id"]: item for item in after["symbols"]}
    metric_names = (
        "cyclomatic_complexity",
        "cognitive_complexity",
        "max_control_nesting",
        "source_lines",
        "sloc",
    )
    changes = []
    for symbol_id in sorted(before_symbols.keys() | after_symbols.keys()):
        old = before_symbols.get(symbol_id)
        new = after_symbols.get(symbol_id)
        if old is None:
            changes.append({"id": symbol_id, "status": "added", "after": new})
            continue
        if new is None:
            changes.append({"id": symbol_id, "status": "removed", "before": old})
            continue
        deltas = {}
        for name in metric_names:
            left = old["metrics"].get(name)
            right = new["metrics"].get(name)
            deltas[name] = right - left if left is not None and right is not None else None
        old_coverage = old["coverage"].get("line_percent")
        new_coverage = new["coverage"].get("line_percent")
        deltas["coverage_line_percent"] = (
            round(new_coverage - old_coverage, 2)
            if old_coverage is not None and new_coverage is not None
            else None
        )
        if any(value not in (None, 0) for value in deltas.values()):
            changes.append(
                {
                    "id": symbol_id,
                    "status": "changed",
                    "path": new["path"],
                    "qualified_name": new["qualified_name"],
                    "before_location": old["location"],
                    "after_location": new["location"],
                    "deltas": deltas,
                    "before_metrics": old["metrics"],
                    "after_metrics": new["metrics"],
                    "before_coverage": old["coverage"],
                    "after_coverage": new["coverage"],
                }
            )

    def magnitude(change):
        if change["status"] != "changed":
            return (1, change["id"])
        deltas = change["deltas"]
        return (
            -abs(deltas.get("cyclomatic_complexity") or 0)
            - abs(deltas.get("cognitive_complexity") or 0),
            change["id"],
        )

    changes.sort(key=magnitude)
    counts = Counter(item["status"] for item in changes)
    return {
        "schema": {"name": "canvas.code-audit.comparison", "version": "1.0.0"},
        "before": {
            "revision": before["snapshot"]["git"]["revision"],
            "created_at": before["snapshot"]["created_at"],
        },
        "after": {
            "revision": after["snapshot"]["git"]["revision"],
            "created_at": after["snapshot"]["created_at"],
        },
        "derivation": "Stable symbol IDs are matched; raw after-minus-before metric deltas are reported.",
        "summary": {
            "changed": counts["changed"],
            "added": counts["added"],
            "removed": counts["removed"],
            "returned": min(limit, len(changes)),
        },
        "symbol_changes": changes[:limit],
    }


def _md_cell(value):
    return "n/a" if value is None else str(value)


def render_summary(snapshot, telemetry_path=DEFAULT_SNAPSHOT):
    aggregate = snapshot["aggregates"]
    git = snapshot["snapshot"]["git"]
    coverage = snapshot["coverage"]
    complexity = query_snapshot(snapshot, "complexity", 8)["items"]
    changing = query_snapshot(snapshot, "complex-change", 8)["items"]
    uncovered = query_snapshot(snapshot, "complex-uncovered", 8)["items"]
    lines = [
        "# Deterministic code audit",
        "",
        f"- Revision: `{git['revision'] or 'unavailable'}` ({'dirty' if git['is_dirty'] else 'clean' if git['is_dirty'] is False else 'state unavailable'})",
        f"- Scope: {aggregate['file_count']} Python files, {aggregate['function_method_count']} functions/methods, {aggregate['sloc']} SLOC",
        f"- Complexity: maximum cyclomatic {_md_cell(aggregate['cyclomatic_complexity']['maximum'])}; maximum cognitive {_md_cell(aggregate['cognitive_complexity']['maximum'])}",
        f"- Coverage: {coverage['status']}; line {_md_cell(coverage.get('aggregate', {}).get('line_percent'))}% and branch {_md_cell(coverage.get('aggregate', {}).get('branch_percent'))}% when available",
        f"- Dependencies: {aggregate['internal_dependency_edge_count']} internal import edges, {aggregate['dependency_cycle_count']} cycles",
        f"- Full telemetry: `{Path(telemetry_path).as_posix()}`",
        "",
        "Metrics are evidence, not pass/fail thresholds. Combined orderings below are transparent navigation aids.",
        "",
        "## Highest static complexity",
        "",
        "| Symbol | CC | Cognitive | Nesting | SLOC | File commits | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in complexity:
        metrics = item["metrics"]
        lines.append(
            f"| `{item['path']}:{item['location']['start_line']} {item['qualified_name']}` | {_md_cell(metrics['cyclomatic_complexity'])} | {_md_cell(metrics['cognitive_complexity'])} | {metrics['max_control_nesting']} | {metrics['sloc']} | {_md_cell(item['file_git'].get('commit_count'))} | {_md_cell(item['coverage'].get('line_percent'))}% |"
        )

    lines.extend(
        [
            "",
            "## Complex and frequently changed",
            "",
            "Ordered by `cyclomatic_complexity x file.commit_count`; both raw values remain visible.",
            "",
            "| Symbol | CC | File commits | Churn | Selection value |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in changing:
        lines.append(
            f"| `{item['path']}:{item['location']['start_line']} {item['qualified_name']}` | {_md_cell(item['metrics']['cyclomatic_complexity'])} | {_md_cell(item['file_git'].get('commit_count'))} | {_md_cell(item['file_git'].get('churn'))} | {item['selection']['value']} |"
        )

    if uncovered:
        lines.extend(
            [
                "",
                "## Complex and uncovered",
                "",
                "Ordered by `cyclomatic_complexity x uncovered executable lines`; this is not a risk score.",
                "",
                "| Symbol | CC | Uncovered lines | Line coverage | Selection value |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in uncovered:
            lines.append(
                f"| `{item['path']}:{item['location']['start_line']} {item['qualified_name']}` | {_md_cell(item['metrics']['cyclomatic_complexity'])} | {len(item['coverage'].get('missing_lines', []))} | {_md_cell(item['coverage'].get('line_percent'))}% | {item['selection']['value']} |"
            )

    if snapshot["dependency_cycles"]:
        lines.extend(["", "## Dependency cycles", ""])
        for cycle in snapshot["dependency_cycles"]:
            paths = ", ".join(item.removeprefix("file:") for item in cycle["file_ids"])
            lines.append(f"- `{paths}`")
    if snapshot["diagnostics"]:
        lines.extend(["", "## Degraded or failed measurements", ""])
        for diagnostic in snapshot["diagnostics"]:
            lines.append(
                f"- {diagnostic['severity']}: {diagnostic['category']}: {diagnostic['message'].strip()}"
            )
    return "\n".join(lines) + "\n"


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Collect and retrieve deterministic code-audit telemetry."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Write a new telemetry snapshot.")
    collect.add_argument("--config", default="tools/code_audit.json")
    collect.add_argument("--output", default=DEFAULT_SNAPSHOT)
    collect.add_argument("--summary", default=DEFAULT_SUMMARY)
    collect.add_argument(
        "--coverage",
        action="store_true",
        help="Run the configured tests under Coverage.py and associate results with symbols.",
    )

    query = subparsers.add_parser("query", help="Retrieve a compact deterministic view.")
    query.add_argument("snapshot", nargs="?", default=DEFAULT_SNAPSHOT)
    query.add_argument(
        "--view",
        choices=(
            "complexity",
            "change",
            "complex-change",
            "complex-uncovered",
            "recent",
            "dependencies",
            "coupling",
        ),
        default="complexity",
    )
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--path")
    query.add_argument("--symbol")
    query.add_argument("--compact", action="store_true")

    compare = subparsers.add_parser("compare", help="Compare two telemetry snapshots.")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--limit", type=int, default=20)
    compare.add_argument("--compact", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = PROJECT_ROOT
    if args.command == "collect":
        config = _read_json(_resolve(root, args.config))
        snapshot = collect_snapshot(root, config, args.coverage)
        output = _resolve(root, args.output)
        summary = _resolve(root, args.summary)
        _atomic_json(output, snapshot)
        human_summary = render_summary(snapshot, args.output)
        _atomic_text(summary, human_summary)
        print(human_summary)
        print(f"Telemetry: {output}")
        print(f"Summary:   {summary}")
        return 1 if snapshot["coverage"]["status"] == "test_failed" else 0
    if args.command == "query":
        snapshot = _read_json(_resolve(root, args.snapshot))
        result = query_snapshot(
            snapshot, args.view, max(1, args.limit), args.path, args.symbol
        )
        print(json.dumps(result, indent=None if args.compact else 2, ensure_ascii=False))
        return 0
    if args.command == "compare":
        before = _read_json(_resolve(root, args.before))
        after = _read_json(_resolve(root, args.after))
        result = compare_snapshots(before, after, max(1, args.limit))
        print(json.dumps(result, indent=None if args.compact else 2, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
