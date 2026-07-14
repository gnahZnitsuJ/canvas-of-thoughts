#!/usr/bin/env python3
"""Compare compile/build telemetry without importing the Nengo runtime."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


CONTROL_FIELDS = (
    "kind",
    "source.commit",
    "source.dirty",
    "source.snapshot",
    "backend",
    "opencl.platform",
    "opencl.device",
    "dimensions",
    "context_length",
    "sub_lengths",
    "probe_mode",
    "compile_profile.name",
    "compile_profile.settings",
    "learned_init.mode",
    "learned_init.seed",
)

METRIC_FIELDS = (
    "model_build_seconds",
    "simulator_construct_seconds",
    "first_run_warmup_seconds",
    "network_count",
    "ensemble_count",
    "neuron_count",
    "node_count",
    "connection_count",
    "probe_count",
    "operator_count",
)

GROUPS = {
    "compile_profile": {"compile_profile.name", "compile_profile.settings"},
    "learned_init": {"learned_init.mode", "learned_init.seed"},
    "source": {"source.commit", "source.dirty", "source.snapshot"},
    "opencl": {"opencl.platform", "opencl.device"},
    "architecture": {
        "dimensions",
        "context_length",
        "sub_lengths",
        "network_count",
        "ensemble_count",
        "neuron_count",
        "node_count",
        "connection_count",
        "probe_count",
        "operator_count",
        "architecture_signature",
    },
}


def _json_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _first(mapping, *paths, default=None):
    for path in paths:
        value = mapping
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            if value is not None:
                return value
    return default


def _case_records(document):
    for section in ("scaling", "repeat_compile"):
        cases = document.get(section) or []
        if cases:
            for case in cases:
                yield section, case
            return
    yield None, None


def normalize(document, path):
    """Expand one telemetry document into comparable flat records."""
    environment = document.get("environment", {})
    fingerprint = document.get("compile_fingerprint", {})
    parameters = document.get("parameters", {})
    document_profile = document.get("compile_profile", {})

    for section, case in _case_records(document):
        case = case or {}
        profile = case.get("compile_profile") or document_profile or fingerprint.get(
            "compile_profile", {}
        )
        network = case.get("network") or _first(
            document, "complexity.network", default={}
        )
        operators = case.get("operators") or _first(
            document, "complexity.operators", default={}
        )
        timings = document.get("timings_seconds", {})
        repeat_index = case.get("repeat_index")
        suffix = f"#{repeat_index}" if repeat_index is not None else ""

        record = {
            "id": f"{path.name}{suffix}",
            "path": str(path),
            "timestamp": document.get("timestamp"),
            "kind": document.get("kind"),
            "source.commit": environment.get("source_commit"),
            "source.dirty": environment.get("source_dirty"),
            "source.snapshot": environment.get("source_snapshot"),
            "backend": case.get("simulator")
            or fingerprint.get("backend")
            or document_profile.get("backend"),
            "opencl.platform": environment.get("opencl_platform")
            or fingerprint.get("opencl_platform"),
            "opencl.device": environment.get("opencl_device")
            or fingerprint.get("opencl_device"),
            "dimensions": case.get("rep_vocab_dim")
            or parameters.get("rep_vocab_dim")
            or fingerprint.get("rep_vocab_dim"),
            "context_length": case.get("context_length")
            or parameters.get("context_length")
            or fingerprint.get("context_length"),
            "sub_lengths": _json_value(
                case.get("sub_lengths")
                or parameters.get("sub_lengths")
                or fingerprint.get("sub_lengths")
            ),
            "probe_mode": case.get("probe_mode")
            or document.get("probe_mode")
            or parameters.get("probe_mode")
            or fingerprint.get("probe_mode"),
            "compile_profile.name": profile.get("name"),
            "compile_profile.settings": _json_value(profile.get("settings", {})),
            "learned_init.mode": case.get("learned_init_mode")
            or document.get("learned_init_mode")
            or fingerprint.get("learned_init_mode"),
            "learned_init.seed": case.get("learned_init_seed")
            if "learned_init_seed" in case
            else document.get("learned_init_seed", fingerprint.get("learned_init_seed")),
            "model_build_seconds": case.get("model_build_seconds")
            if case
            else timings.get("Model build")
            or document_profile.get("model_build_seconds"),
            "simulator_construct_seconds": case.get("simulator_compile_seconds")
            if case
            else timings.get("Simulator compile")
            or document_profile.get("simulator_construct_seconds"),
            "first_run_warmup_seconds": case.get("first_run_warmup_seconds")
            if case
            else timings.get("First-run warmup")
            or document_profile.get("first_run_warmup_seconds"),
            "network_count": network.get("network_count"),
            "ensemble_count": network.get("ensemble_count"),
            "neuron_count": network.get("neuron_count"),
            "node_count": network.get("node_count"),
            "connection_count": network.get("connection_count"),
            "probe_count": network.get("probe_count"),
            "operator_count": operators.get("operator_count"),
            "warnings": _json_value(document.get("warnings", [])),
            "cache_state": _json_value(
                _first(
                    fingerprint,
                    "environment",
                    default=document_profile.get("environment")
                    or {
                        key: environment.get(key)
                        for key in (
                            "PYOPENCL_NO_CACHE",
                            "PYOPENCL_COMPILER_OUTPUT",
                            "PYOPENCL_BUILD_OPTIONS",
                            "PYOPENCL_CTX",
                            "CANVAS_OPENCL_PLATFORM_INDEX",
                            "CANVAS_OPENCL_DEVICE_INDEX",
                        )
                    },
                )
            ),
            "architecture_signature": _json_value(
                document.get("architecture_signature", {})
            ),
            "section": section,
            "repeat_index": repeat_index,
        }
        yield record


def _parse_time(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed


def _match_where(record, expression):
    if "=" not in expression:
        raise ValueError(f"filter must be FIELD=VALUE: {expression}")
    field, expected = expression.split("=", 1)
    if field not in record:
        raise ValueError(f"unknown filter field: {field}")
    actual = record.get(field)
    if isinstance(actual, bool):
        return str(actual).lower() == expected.lower()
    return str(actual) == expected


def load_records(paths, since=None, until=None, where=()):
    files = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)

    records = []
    since_time = _parse_time(since)
    until_time = _parse_time(until)
    for path in files:
        with path.open(encoding="utf-8") as telemetry_file:
            document = json.load(telemetry_file)
        for record in normalize(document, path):
            timestamp = _parse_time(record["timestamp"])
            if since_time and timestamp < since_time:
                continue
            if until_time and timestamp > until_time:
                continue
            if all(_match_where(record, expression) for expression in where):
                records.append(record)
    return records


def differing_fields(records):
    fields = set(CONTROL_FIELDS) | set(METRIC_FIELDS) | {"architecture_signature"}
    return {
        field
        for field in fields
        if len({_json_value(record.get(field)) for record in records}) > 1
    }


def allowed_variations(vary):
    allowed = set()
    for item in vary:
        allowed.update(GROUPS.get(item, {item}))
    return allowed


def validate_controls(records, vary):
    differences = differing_fields(records)
    control_differences = differences & set(CONTROL_FIELDS)
    allowed = allowed_variations(vary)
    unexpected = control_differences - allowed
    messages = []
    if not control_differences:
        messages.append("No control variable differs across the selected records.")
    if not vary and control_differences:
        groups = []
        for group, fields in GROUPS.items():
            if control_differences <= fields:
                groups.append(group)
        if len(groups) == 1:
            messages.append(f"Inferred independent variable: {groups[0]}.")
            unexpected = set()
        else:
            messages.append(
                "Multiple control variables differ; pass --vary for each intended change."
            )
    elif unexpected:
        messages.append("Unexpected control differences: " + ", ".join(sorted(unexpected)))
    return differences, unexpected, messages


def delta(value, reference):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None, None
    if not isinstance(reference, (int, float)) or isinstance(reference, bool):
        return None, None
    absolute = value - reference
    percentage = None if reference == 0 else absolute / reference * 100
    return absolute, percentage


def _number(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def console_table(records):
    reference = records[0]
    header = (
        f"{'run':<18} {'profile':<12} {'init':<15} {'build_s':>9} {'compile_s':>10} {'delta_s':>9} "
        f"{'delta_%':>9} {'ops':>9} {'probes':>7} {'ens':>7} {'neurons':>10}"
    )
    lines = [header, "-" * len(header)]
    for index, record in enumerate(records):
        absolute, percentage = delta(
            record["simulator_construct_seconds"],
            reference["simulator_construct_seconds"],
        )
        label = f"{index}:{record['id'][-15:]}"
        lines.append(
            f"{label[:18]:<18} {str(record['compile_profile.name'])[:12]:<12} "
            f"{str(record['learned_init.mode'])[:15]:<15} "
            f"{_number(record['model_build_seconds']):>9} "
            f"{_number(record['simulator_construct_seconds']):>10} "
            f"{_number(absolute):>9} {_number(percentage, 1):>9} "
            f"{_number(record['operator_count']):>9} {_number(record['probe_count']):>7} "
            f"{_number(record['ensemble_count']):>7} {_number(record['neuron_count']):>10}"
        )
    return "\n".join(lines)


def markdown_report(records, differences, messages):
    reference = records[0]
    lines = [
        "# Telemetry Comparison",
        "",
        f"Reference: `{reference['id']}`",
        "",
    ]
    for message in messages:
        lines.append(f"> {message}")
    if messages:
        lines.append("")

    lines.extend(
        [
            "| Run | Build (s) | Simulator construction (s) | Abs. delta (s) | Delta (%) | Operators | Probes | Ensembles | Neurons |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        absolute, percentage = delta(
            record["simulator_construct_seconds"],
            reference["simulator_construct_seconds"],
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record['id']}`",
                    _number(record["model_build_seconds"]),
                    _number(record["simulator_construct_seconds"]),
                    _number(absolute),
                    _number(percentage, 1),
                    _number(record["operator_count"]),
                    _number(record["probe_count"]),
                    _number(record["ensemble_count"]),
                    _number(record["neuron_count"]),
                ]
            )
            + " |"
        )

    changed_metrics = sorted(differences & set(METRIC_FIELDS))
    if changed_metrics:
        lines.extend(
            [
                "",
                "## Metric deltas",
                "",
                "| Metric | Run | Value | Absolute delta | Delta (%) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for metric in changed_metrics:
            for index, record in enumerate(records):
                absolute, percentage = delta(record.get(metric), reference.get(metric))
                lines.append(
                    f"| {metric} | {index} | {_number(record.get(metric))} | "
                    f"{_number(absolute)} | {_number(percentage, 1)} |"
                )

    control_differences = sorted(differences & set(CONTROL_FIELDS))
    if control_differences:
        lines.extend(["", "## Changed controls", ""])
        lines.append("| Field | " + " | ".join(f"Run {i}" for i in range(len(records))) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in records) + " |")
        for field in control_differences:
            values = [str(record.get(field, "-")) for record in records]
            lines.append("| " + field + " | " + " | ".join(values) + " |")

    if "architecture_signature" in differences:
        signatures = [json.loads(record["architecture_signature"] or "{}") for record in records]
        signature_fields = sorted(set().union(*(signature.keys() for signature in signatures)))
        changed_signature_fields = [
            field
            for field in signature_fields
            if len({_json_value(signature.get(field)) for signature in signatures}) > 1
        ]
        lines.extend(["", "## Architecture signature changes", ""])
        lines.append("| Field | " + " | ".join(f"Run {i}" for i in range(len(records))) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in records) + " |")
        for field in changed_signature_fields:
            values = [str(_json_value(signature.get(field))) for signature in signatures]
            lines.append("| " + field + " | " + " | ".join(values) + " |")

    lines.extend(["", "## Run fingerprints", ""])
    for index, record in enumerate(records):
        lines.extend(
            [
                f"- Run {index}: `{record['id']}`",
                f"  - timestamp: `{record['timestamp']}`",
                f"  - source commit: `{record['source.commit']}` (dirty: `{record['source.dirty']}`)",
                f"  - source snapshot: `{record['source.snapshot']}`",
                f"  - backend/device: `{record['backend']}` / `{record['opencl.device']}`",
                f"  - warnings: `{record['warnings']}`",
                f"  - cache state: `{record['cache_state']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def write_csv(path, records):
    fields = ["id", "path", "timestamp", *CONTROL_FIELDS, *METRIC_FIELDS]
    reference = records[0]
    fields.extend(
        f"{metric}.{suffix}"
        for metric in METRIC_FIELDS
        for suffix in ("absolute_delta", "percent_delta")
    )
    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            for metric in METRIC_FIELDS:
                absolute, percentage = delta(record.get(metric), reference.get(metric))
                row[f"{metric}.absolute_delta"] = absolute
                row[f"{metric}.percent_delta"] = percentage
            writer.writerow(row)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(Path("model") / "results")],
        help="Telemetry JSON files or directories (default: model/results).",
    )
    parser.add_argument("--since", help="Include timestamps at or after this ISO timestamp.")
    parser.add_argument("--until", help="Include timestamps at or before this ISO timestamp.")
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Filter normalized fingerprint/control fields; repeat as needed.",
    )
    parser.add_argument(
        "--vary",
        action="append",
        default=[],
        help=(
            "Declare an intended independent variable. Groups: compile_profile, "
            "learned_init, source, opencl, architecture; or use an exact field."
        ),
    )
    parser.add_argument("--strict", action="store_true", help="Fail on unexpected control differences.")
    parser.add_argument("--markdown", help="Write a Markdown comparison report.")
    parser.add_argument("--csv", help="Write normalized records and deltas as CSV.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        records = load_records(args.paths, args.since, args.until, args.where)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if len(records) < 2:
        print(f"error: comparison requires at least two records; selected {len(records)}", file=sys.stderr)
        return 2

    differences, unexpected, messages = validate_controls(records, args.vary)
    print(console_table(records))
    for message in messages:
        print(f"WARNING: {message}", file=sys.stderr)

    report = markdown_report(records, differences, messages)
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
        print(f"Wrote Markdown report: {args.markdown}")
    if args.csv:
        write_csv(args.csv, records)
        print(f"Wrote CSV: {args.csv}")
    if args.strict and unexpected:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
