"""Data contracts shared by architecture specifications and component adapters.

The contracts deliberately describe subsystem boundaries, not low-level Nengo
objects. Concrete endpoints remain opaque so the specification and validation
layers can be tested without constructing a simulator.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


PortDirection = Literal["input", "output"]
SignalType = Literal["semantic_pointer", "scalar", "control", "neurons"]


@dataclass(frozen=True)
class Port:
    """Expose one semantically named component endpoint to the assembler."""

    endpoint: Any = field(compare=False, repr=False)
    direction: PortDirection
    dimensions: int
    signal_type: SignalType
    vocabulary_id: str | None = None
    required: bool = True

    def __post_init__(self):
        if self.direction not in {"input", "output"}:
            raise ValueError(f"Unknown port direction: {self.direction}")
        if self.signal_type not in {
            "semantic_pointer",
            "scalar",
            "control",
            "neurons",
        }:
            raise ValueError(f"Unknown signal type: {self.signal_type}")
        if self.dimensions < 1:
            raise ValueError("Port dimensions must be positive")


@dataclass
class BuiltComponent:
    """Describe a built subsystem and all architecture-relevant registrations."""

    name: str
    network: Any
    ports: dict[str, Port]
    capabilities: set[str] = field(default_factory=set)
    runtime_handles: dict[str, Any] = field(default_factory=dict)
    learning_connections: list[Any] = field(default_factory=list)
    probes: dict[str, Any] = field(default_factory=dict)
    signature: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchitectureBuildContext:
    """Hold settings shared by every component factory in one model build."""

    vocab: Any
    dimensions: int
    seed: int
    strict_vocab: bool
    probe_registry: Any
    compile_profile: Mapping[str, Any]
    learned_init_mode: str
    learned_init_seed: int | None = None
    vocabulary_id: str = "model_vocab"


@dataclass(frozen=True)
class ComponentSpec:
    """Select one registered component implementation and its stable parameters."""

    name: str
    component_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        if not self.name:
            raise ValueError("Component name cannot be empty")
        if not self.component_type:
            raise ValueError("Component type cannot be empty")
        if self.version < 1:
            raise ValueError("Component version must be positive")


@dataclass(frozen=True)
class ConnectionSpec:
    """Describe one architecture-level connection between semantic ports."""

    source: str
    target: str
    transform: Any = None
    synapse: Any = None
    label: str | None = None

