"""Composable, subsystem-level architecture specifications for the Nengo model."""

from architecture.contracts import (
    ArchitectureBuildContext,
    BuiltComponent,
    ComponentSpec,
    ConnectionSpec,
    Port,
)
from architecture.signatures import architecture_signature, canonical_json
from architecture.spec import ArchitectureSpec
from architecture.validation import ArchitectureValidationError, validate_architecture

__all__ = [
    "ArchitectureBuildContext",
    "ArchitectureSpec",
    "ArchitectureValidationError",
    "BuiltComponent",
    "ComponentSpec",
    "ConnectionSpec",
    "Port",
    "architecture_signature",
    "canonical_json",
    "validate_architecture",
]
