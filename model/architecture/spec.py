"""Python-native authoring surface for subsystem architecture topology."""

from copy import deepcopy
from dataclasses import dataclass, field, replace

from architecture.contracts import ComponentSpec, ConnectionSpec


@dataclass
class ArchitectureSpec:
    """Collect component identities, connections, and explicit runtime roles.

    Specifications are mutable only while an architecture is being authored.
    Variant builders should call :meth:`copy` first so modifying an experiment
    cannot mutate the baseline specification shared by another build.
    """

    name: str
    schema_version: int = 1
    components: dict[str, ComponentSpec] = field(default_factory=dict)
    connections: list[ConnectionSpec] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)
    checkpoint_order: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Architecture name cannot be empty")
        if self.schema_version < 1:
            raise ValueError("Architecture schema version must be positive")

    def add(self, name: str, component_type: str, **parameters):
        """Add a uniquely named component and return this specification."""
        if name in self.components:
            raise ValueError(f"Duplicate component name: {name}")
        self.components[name] = ComponentSpec(
            name=name,
            component_type=component_type,
            parameters=dict(parameters),
        )
        return self

    def remove(self, name: str):
        """Remove a component and every connection or role that refers to it."""
        if name not in self.components:
            raise KeyError(f"Unknown component: {name}")
        del self.components[name]
        prefix = f"{name}."
        self.connections = [
            connection
            for connection in self.connections
            if not connection.source.startswith(prefix)
            and not connection.target.startswith(prefix)
        ]
        self.roles = {
            role: target
            for role, target in self.roles.items()
            if target != name and not target.startswith(prefix)
        }
        self.checkpoint_order = [
            component_name
            for component_name in self.checkpoint_order
            if component_name != name
        ]
        return self

    def replace(self, name: str, component_type: str, **parameters):
        """Replace one implementation while preserving topology and roles."""
        if name not in self.components:
            raise KeyError(f"Unknown component: {name}")
        current = self.components[name]
        self.components[name] = replace(
            current,
            component_type=component_type,
            parameters=dict(parameters),
        )
        return self

    def connect(
        self,
        source: str,
        target: str,
        *,
        transform=None,
        synapse=None,
        label: str | None = None,
    ):
        """Add one directed semantic-port connection."""
        self.connections.append(
            ConnectionSpec(
                source=source,
                target=target,
                transform=transform,
                synapse=synapse,
                label=label,
            )
        )
        return self

    def disconnect(self, source: str, target: str):
        """Remove one exact connection and fail if it was not present."""
        retained = [
            connection
            for connection in self.connections
            if not (connection.source == source and connection.target == target)
        ]
        if len(retained) == len(self.connections):
            raise KeyError(f"Unknown connection: {source} -> {target}")
        self.connections = retained
        return self

    def assign_role(self, role: str, target: str):
        """Assign a unique runtime-facing role to a component or output port."""
        if not role:
            raise ValueError("Role name cannot be empty")
        self.roles[role] = target
        return self

    def set_checkpoint_order(self, *component_names: str):
        """Set the stable component order used to serialize learned weights."""
        if len(component_names) != len(set(component_names)):
            raise ValueError("Checkpoint order cannot contain duplicate components")
        self.checkpoint_order = list(component_names)
        return self

    def copy(self, *, name: str | None = None):
        """Return an independent copy suitable for a derived variant."""
        copied = deepcopy(self)
        if name is not None:
            copied.name = name
        return copied
