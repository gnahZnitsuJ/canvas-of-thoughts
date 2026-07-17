"""Deterministic serialization for architecture identity and semantic diffs."""

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

from architecture.contracts import BuiltComponent
from architecture.spec import ArchitectureSpec


def normalize_signature_value(value):
    """Convert supported values to deterministic JSON-compatible structures.

    Component factories should put explicit, stable descriptors in their
    signatures. Rejecting unknown objects prevents addresses or unstable reprs
    from silently entering checkpoint compatibility metadata.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return normalize_signature_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return normalize_signature_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): normalize_signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_signature_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [normalize_signature_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "tolist"):
        return normalize_signature_value(value.tolist())
    raise TypeError(
        "Architecture signature values must be deterministic and JSON "
        f"serializable; found {type(value).__name__}"
    )


def _connection_signature(connection):
    return {
        "source": connection.source,
        "target": connection.target,
        "transform": normalize_signature_value(connection.transform),
        "synapse": normalize_signature_value(connection.synapse),
        "label": connection.label,
    }


def architecture_signature(
    spec: ArchitectureSpec,
    built_components: dict[str, BuiltComponent] | None = None,
):
    """Return a canonical semantic topology signature for telemetry/checkpoints."""
    built_components = built_components or {}
    components = []
    for name, component_spec in sorted(spec.components.items()):
        built = built_components.get(name)
        component = {
            "name": name,
            "type": component_spec.component_type,
            "version": component_spec.version,
            "parameters": normalize_signature_value(dict(component_spec.parameters)),
        }
        if built is not None:
            component.update(
                {
                    "capabilities": sorted(built.capabilities),
                    "ports": {
                        port_name: {
                            "direction": port.direction,
                            "dimensions": port.dimensions,
                            "signal_type": port.signal_type,
                            "vocabulary_id": port.vocabulary_id,
                            "required": port.required,
                        }
                        for port_name, port in sorted(built.ports.items())
                    },
                    "implementation": normalize_signature_value(built.signature),
                    "learning_connection_count": len(built.learning_connections),
                }
            )
        components.append(component)

    connections = sorted(
        (_connection_signature(connection) for connection in spec.connections),
        key=lambda item: (
            item["source"],
            item["target"],
            json.dumps(item["transform"], sort_keys=True),
            json.dumps(item["synapse"], sort_keys=True),
            item["label"] or "",
        ),
    )
    return {
        "architecture_name": spec.name,
        "architecture_schema_version": spec.schema_version,
        "components": components,
        "connections": connections,
        "roles": dict(sorted(spec.roles.items())),
        "checkpoint_order": list(spec.checkpoint_order),
    }


def canonical_json(signature):
    """Render a signature with stable ordering and no insignificant whitespace."""
    return json.dumps(
        normalize_signature_value(signature),
        sort_keys=True,
        separators=(",", ":"),
    )
