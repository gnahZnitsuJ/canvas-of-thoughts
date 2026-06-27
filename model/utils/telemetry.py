import json
import platform
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _timestamp():
    return datetime.now().astimezone()


def save_telemetry(results_dir, payload, prefix="telemetry"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp()
    result_path = (
        results_dir
        / f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
    )
    document = {
        "timestamp": timestamp.isoformat(),
        **payload,
    }

    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(document, result_file, indent=2)

    return result_path


def save_text_artifact(results_dir, text, prefix="summary", suffix=".md"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp()
    artifact_path = (
        results_dir
        / f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
    )

    with artifact_path.open("w", encoding="utf-8") as artifact_file:
        artifact_file.write(text)

    return artifact_path


def environment_telemetry():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def network_telemetry(network, largest_limit=10):
    ensembles = list(network.all_ensembles)
    largest_ensembles = sorted(
        (
            {
                "label": ensemble.label,
                "neurons": ensemble.n_neurons,
                "dimensions": ensemble.dimensions,
            }
            for ensemble in ensembles
        ),
        key=lambda item: item["neurons"],
        reverse=True,
    )[:largest_limit]

    network_types = Counter(
        type(subnetwork).__name__ for subnetwork in network.all_networks
    )

    return {
        "network_count": len(network.all_networks),
        "network_types": dict(network_types.most_common()),
        "ensemble_count": len(ensembles),
        "neuron_count": sum(ensemble.n_neurons for ensemble in ensembles),
        "node_count": len(network.all_nodes),
        "connection_count": len(network.all_connections),
        "probe_count": len(network.all_probes),
        "largest_ensembles": largest_ensembles,
    }


def _operator_signal_elements(operator):
    signals = []
    for attribute in ("reads", "sets", "incs", "updates"):
        signals.extend(getattr(operator, attribute, ()))
    return sum(getattr(signal, "size", 0) for signal in signals)


def operator_telemetry(simulator, largest_limit=15):
    operators = list(simulator.model.operators)
    counts = Counter(type(operator).__name__ for operator in operators)
    elements_by_type = defaultdict(int)
    largest_operators = []

    for operator in operators:
        operator_type = type(operator).__name__
        signal_elements = _operator_signal_elements(operator)
        elements_by_type[operator_type] += signal_elements
        largest_operators.append(
            {
                "type": operator_type,
                "signal_elements": signal_elements,
                "description": str(operator)[:300],
            }
        )

    largest_operators.sort(
        key=lambda item: item["signal_elements"],
        reverse=True,
    )

    return {
        "operator_count": len(operators),
        "operator_counts_by_type": dict(counts.most_common()),
        "operator_signal_elements_by_type": dict(
            sorted(
                elements_by_type.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "largest_operators": largest_operators[:largest_limit],
    }


def training_invocation_estimate(sequences, training_mode="single_pass"):
    sequence_lengths = [len(tokens) for tokens in sequences if len(tokens) > 1]
    total_pairs = sum(length - 1 for length in sequence_lengths)
    reset_runs = len(sequence_lengths)

    if training_mode == "two_pass":
        presentation_runs = 2 * total_pairs
    elif training_mode == "single_pass":
        presentation_runs = total_pairs
    elif training_mode == "scheduled":
        # Scheduled training still covers every token transition, but it does so
        # with one long simulator run per sequence instead of one run per pair.
        presentation_runs = len(sequence_lengths)
    else:
        raise ValueError(f"Unknown training_mode: {training_mode}")

    return {
        "training_mode": training_mode,
        "total_training_pairs": total_pairs,
        "training_sequence_count": len(sequence_lengths),
        "estimated_training_reset_runs": reset_runs,
        "estimated_training_presentation_runs": presentation_runs,
        "estimated_training_sim_runs": reset_runs + presentation_runs,
    }


def evaluation_invocation_estimate(
    sequences,
    max_examples=50,
    evaluation_mode="streaming",
):
    if evaluation_mode == "prefix_replay":
        prediction_count = 0
        reset_runs = 0
        presentation_runs = 0

        for tokens in sequences:
            for prefix_length in range(1, len(tokens)):
                reset_runs += 1
                presentation_runs += prefix_length
                prediction_count += 1
                if prediction_count >= max_examples:
                    return {
                        "evaluation_mode": evaluation_mode,
                        "total_eval_predictions": prediction_count,
                        "estimated_eval_reset_runs": reset_runs,
                        "estimated_eval_presentation_runs": presentation_runs,
                        "estimated_eval_sim_runs": reset_runs + presentation_runs,
                    }

        return {
            "evaluation_mode": evaluation_mode,
            "total_eval_predictions": prediction_count,
            "estimated_eval_reset_runs": reset_runs,
            "estimated_eval_presentation_runs": presentation_runs,
            "estimated_eval_sim_runs": reset_runs + presentation_runs,
        }

    if evaluation_mode != "streaming":
        raise ValueError(f"Unknown evaluation_mode: {evaluation_mode}")

    prediction_count = 0
    reset_runs = 0
    presentation_runs = 0

    for tokens in sequences:
        if len(tokens) < 2:
            continue

        remaining = max_examples - prediction_count
        sequence_predictions = min(len(tokens) - 1, remaining)
        if sequence_predictions <= 0:
            break

        reset_runs += 1
        presentation_runs += sequence_predictions
        prediction_count += sequence_predictions

        if prediction_count >= max_examples:
            break

    return {
        "evaluation_mode": evaluation_mode,
        "total_eval_predictions": prediction_count,
        "estimated_eval_reset_runs": reset_runs,
        "estimated_eval_presentation_runs": presentation_runs,
        "estimated_eval_sim_runs": reset_runs + presentation_runs,
    }


def _format_seconds(value):
    return f"{value:.3f}"


def _format_int(value):
    return f"{value:,}"


def _row_to_markdown(row):
    return "| " + " | ".join(row) + " |"


def _case_row(case, include_sub_lengths=True):
    network = case["network"]
    operators = case["operators"]
    row = [
        case["name"],
        case["simulator"],
    ]

    if include_sub_lengths:
        sub_lengths = case.get("sub_lengths")
        row.append(
            ",".join(str(length) for length in sub_lengths) if sub_lengths else "-"
        )
        row.append(str(case.get("context_length", "-")))

    row.extend(
        [
            str(case["rep_vocab_dim"]),
            _format_seconds(case["model_build_seconds"]),
            _format_seconds(case["simulator_compile_seconds"]),
            _format_int(operators["operator_count"]),
            _format_int(network["ensemble_count"]),
            _format_int(network["neuron_count"]),
        ]
    )
    return row


def render_compile_benchmark_summary(payload, telemetry_path=None):
    lines = [
        "# Compile Benchmark Summary",
        "",
        f"- Kind: `{payload['kind']}`",
    ]

    if telemetry_path is not None:
        lines.append(f"- Raw telemetry: `{Path(telemetry_path).name}`")

    environment = payload.get("environment", {})
    if environment:
        lines.extend(
            [
                f"- OpenCL platform: `{environment.get('opencl_platform', 'unknown')}`",
                f"- OpenCL device: `{environment.get('opencl_device', 'unknown')}`",
            ]
        )

    sections = [
        ("Scaling", payload.get("scaling", []), True),
        ("Simulator Comparison", payload.get("simulator_comparison", []), True),
        ("Component Costs", payload.get("component_costs", []), False),
    ]

    for title, cases, include_sub_lengths in sections:
        if not cases:
            continue

        lines.extend(["", f"## {title}", ""])

        if include_sub_lengths:
            header = [
                "case",
                "sim",
                "sub_lengths",
                "context_length",
                "dim",
                "model_build_s",
                "compile_s",
                "operators",
                "ensembles",
                "neurons",
            ]
        else:
            header = [
                "case",
                "sim",
                "dim",
                "model_build_s",
                "compile_s",
                "operators",
                "ensembles",
                "neurons",
            ]

        divider = ["---"] * len(header)
        lines.append(_row_to_markdown(header))
        lines.append(_row_to_markdown(divider))
        for case in cases:
            lines.append(_row_to_markdown(_case_row(case, include_sub_lengths)))

    return "\n".join(lines) + "\n"


def print_compile_benchmark_summary(payload):
    sections = [
        ("Scaling", payload.get("scaling", []), True),
        ("Simulator Comparison", payload.get("simulator_comparison", []), True),
        ("Component Costs", payload.get("component_costs", []), False),
    ]

    for title, cases, include_sub_lengths in sections:
        if not cases:
            continue

        print(f"\n{title}:")
        if include_sub_lengths:
            header = (
                f"{'case':<20} {'sim':<10} {'sub_lengths':<14} "
                f"{'ctx':>5} {'dim':>5} {'build_s':>9} {'compile_s':>10} "
                f"{'ops':>10} {'ens':>8}"
            )
            print(header)
            print("-" * len(header))
            for case in cases:
                sub_lengths = ",".join(str(length) for length in case["sub_lengths"])
                print(
                    f"{case['name']:<20} "
                    f"{case['simulator']:<10} "
                    f"{sub_lengths:<14} "
                    f"{case['context_length']:>5} "
                    f"{case['rep_vocab_dim']:>5} "
                    f"{case['model_build_seconds']:>9.3f} "
                    f"{case['simulator_compile_seconds']:>10.3f} "
                    f"{case['operators']['operator_count']:>10,} "
                    f"{case['network']['ensemble_count']:>8,}"
                )
        else:
            header = (
                f"{'case':<20} {'sim':<10} {'dim':>5} {'build_s':>9} "
                f"{'compile_s':>10} {'ops':>10} {'ens':>8}"
            )
            print(header)
            print("-" * len(header))
            for case in cases:
                print(
                    f"{case['name']:<20} "
                    f"{case['simulator']:<10} "
                    f"{case['rep_vocab_dim']:>5} "
                    f"{case['model_build_seconds']:>9.3f} "
                    f"{case['simulator_compile_seconds']:>10.3f} "
                    f"{case['operators']['operator_count']:>10,} "
                    f"{case['network']['ensemble_count']:>8,}"
                )
