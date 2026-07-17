"""Fail-fast validation for semantic subsystem architecture specifications."""

from architecture.contracts import BuiltComponent, Port
from architecture.spec import ArchitectureSpec


REQUIRED_ROLES = ("input", "target", "prediction", "primary_memory")


class ArchitectureValidationError(ValueError):
    """Report one or more invalid architecture contracts before compilation."""

    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__("Invalid architecture:\n- " + "\n- ".join(self.errors))


def _split_reference(reference: str):
    parts = reference.split(".")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Endpoint must use 'component.port' syntax: {reference!r}"
        )
    return parts[0], parts[1]


def resolve_port(
    reference: str,
    built_components: dict[str, BuiltComponent],
) -> tuple[BuiltComponent, Port]:
    """Resolve one semantic endpoint reference to its built component and port."""
    component_name, port_name = _split_reference(reference)
    try:
        component = built_components[component_name]
    except KeyError as exc:
        raise KeyError(f"Unknown component in endpoint {reference!r}") from exc
    try:
        port = component.ports[port_name]
    except KeyError as exc:
        raise KeyError(f"Unknown port in endpoint {reference!r}") from exc
    return component, port


def validate_architecture(
    spec: ArchitectureSpec,
    built_components: dict[str, BuiltComponent],
):
    """Validate topology, semantic compatibility, roles, and registrations.

    Recurrent paths are intentionally accepted. Nengo architectures commonly
    contain cycles, so validation reasons about endpoint contracts instead of
    imposing a directed-acyclic-graph restriction.
    """
    errors = []
    if set(spec.components) != set(built_components):
        missing = sorted(set(spec.components) - set(built_components))
        unexpected = sorted(set(built_components) - set(spec.components))
        if missing:
            errors.append("Components were not built: " + ", ".join(missing))
        if unexpected:
            errors.append("Unexpected built components: " + ", ".join(unexpected))

    incoming = set()
    incident = set()
    for connection in spec.connections:
        try:
            source_component, source_port = resolve_port(
                connection.source, built_components
            )
            target_component, target_port = resolve_port(
                connection.target, built_components
            )
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
            continue

        incident.update((source_component.name, target_component.name))
        incoming.add(connection.target)
        if source_port.direction != "output":
            errors.append(f"Connection source is not an output: {connection.source}")
        if target_port.direction != "input":
            errors.append(f"Connection target is not an input: {connection.target}")
        if (
            source_port.dimensions != target_port.dimensions
            and connection.transform is None
        ):
            errors.append(
                f"Dimension mismatch without transform: {connection.source} "
                f"({source_port.dimensions}) -> {connection.target} "
                f"({target_port.dimensions})"
            )
        if (
            source_port.signal_type == "semantic_pointer"
            and target_port.signal_type == "semantic_pointer"
            and source_port.vocabulary_id is not None
            and target_port.vocabulary_id is not None
            and source_port.vocabulary_id != target_port.vocabulary_id
        ):
            errors.append(
                f"Vocabulary mismatch: {connection.source} "
                f"({source_port.vocabulary_id}) -> {connection.target} "
                f"({target_port.vocabulary_id})"
            )

    for component_name, component in built_components.items():
        for port_name, port in component.ports.items():
            reference = f"{component_name}.{port_name}"
            if port.direction == "input" and port.required and reference not in incoming:
                errors.append(f"Required input is not connected: {reference}")

        if "resettable" in component.capabilities and "reset" not in component.runtime_handles:
            errors.append(f"Resettable component lacks a reset handle: {component_name}")
        if (
            {"learnable", "checkpointed"} & component.capabilities
            and not component.learning_connections
        ):
            errors.append(
                f"Learnable/checkpointed component has no learning connection: "
                f"{component_name}"
            )

    checkpointed = {
        name
        for name, component in built_components.items()
        if "checkpointed" in component.capabilities
    }
    if spec.checkpoint_order:
        ordered = set(spec.checkpoint_order)
        if ordered != checkpointed:
            missing = sorted(checkpointed - ordered)
            unexpected = sorted(ordered - checkpointed)
            if missing:
                errors.append(
                    "Checkpoint order omits components: " + ", ".join(missing)
                )
            if unexpected:
                errors.append(
                    "Checkpoint order includes non-checkpointed components: "
                    + ", ".join(unexpected)
                )

    role_components = set()
    for role in REQUIRED_ROLES:
        if role not in spec.roles:
            errors.append(f"Required role is missing: {role}")
    for role, target in spec.roles.items():
        if "." in target:
            try:
                component, port = resolve_port(target, built_components)
                role_components.add(component.name)
                if port.direction != "output":
                    errors.append(f"Role {role!r} must reference an output: {target}")
            except (KeyError, ValueError) as exc:
                errors.append(f"Role {role!r}: {exc}")
        elif target not in built_components:
            errors.append(f"Role {role!r} references unknown component: {target}")
        else:
            role_components.add(target)

    for component_name in built_components:
        if component_name not in incident and component_name not in role_components:
            errors.append(f"Disconnected component is not assigned a role: {component_name}")

    if errors:
        raise ArchitectureValidationError(errors)
