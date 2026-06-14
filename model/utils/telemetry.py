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


def training_invocation_estimate(sequences):
    sequence_lengths = [len(tokens) for tokens in sequences if len(tokens) > 1]
    total_pairs = sum(length - 1 for length in sequence_lengths)
    reset_runs = len(sequence_lengths)
    presentation_runs = 2 * total_pairs

    return {
        "total_training_pairs": total_pairs,
        "training_sequence_count": len(sequence_lengths),
        "estimated_training_reset_runs": reset_runs,
        "estimated_training_presentation_runs": presentation_runs,
        "estimated_training_sim_runs": reset_runs + presentation_runs,
    }


def evaluation_invocation_estimate(sequences, max_examples=50):
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
                    "total_eval_predictions": prediction_count,
                    "estimated_eval_reset_runs": reset_runs,
                    "estimated_eval_presentation_runs": presentation_runs,
                    "estimated_eval_sim_runs": reset_runs + presentation_runs,
                }

    return {
        "total_eval_predictions": prediction_count,
        "estimated_eval_reset_runs": reset_runs,
        "estimated_eval_presentation_runs": presentation_runs,
        "estimated_eval_sim_runs": reset_runs + presentation_runs,
    }
