"""Build a validated architecture specification into a NengoSPA network."""

from dataclasses import dataclass
from typing import Any

import nengo
import nengo_spa as spa

from architecture.contracts import ArchitectureBuildContext, BuiltComponent
from architecture.registry import ComponentRegistry
from architecture.signatures import architecture_signature
from architecture.spec import ArchitectureSpec
from architecture.validation import resolve_port, validate_architecture


CAPABILITY_HANDLE_KEYS = {
    "resettable": "reset",
    "schedulable": "schedule",
    "recall_controlled": "recall_control",
}


@dataclass
class AssembledModelResult:
    """Expose assembled topology plus the legacy attributes used by workflows."""

    model: Any
    architecture_spec: ArchitectureSpec
    built_components: dict[str, BuiltComponent]
    roles: dict[str, Any]
    capabilities: dict[str, list[Any]]
    learning_connections: list[Any]
    probes: dict[str, Any]
    architecture_topology_signature: dict[str, Any]
    strict: bool
    sub_lengths: list[int] | None
    learned_init_mode: str
    learned_init_seed: int | None
    compile_profile_name: str
    compile_profile_settings: dict[str, Any]
    probe_mode: str
    created_probe_labels: list[str]
    skipped_probe_labels: list[str]

    def install_compatibility_facade(self):
        """Derive the historical ModelResult surface from roles/capabilities."""
        input_component = self.built_components[self.architecture_spec.roles["input"]]
        target_component = self.built_components[self.architecture_spec.roles["target"]]
        memory_component = self.built_components[
            self.architecture_spec.roles["primary_memory"]
        ]
        active_name = self.architecture_spec.roles.get("active_component")

        self.input_module = input_component.runtime_handles["input"]
        self.target_module = target_component.runtime_handles["target"]
        self.context_module = memory_component.network
        self.primary_context_module = memory_component.network
        self.active_context_modules = [
            component.network
            for component in self.built_components.values()
            if "memory" in component.capabilities
        ]
        self.legacy_context_modules = []
        self.all_context_modules = list(self.active_context_modules)

        if active_name is None:
            learnable = [
                component
                for component in self.built_components.values()
                if "learnable" in component.capabilities
            ]
            self.active_component = learnable[0].network if learnable else None
        else:
            self.active_component = self.built_components[active_name].network
        self.active_components = [
            component.network
            for component in self.built_components.values()
            if "learnable" in component.capabilities
        ]

        self.p_pred = self.probes["prediction"]
        self.p_error = next(
            (probe for probe in self.model.probes if probe.label == "error"),
            None,
        )
        self.p_post_state = next(
            (probe for probe in self.model.probes if probe.label == "post_state"),
            None,
        )

        # Preserve model attributes used by interactive debugging and older
        # callers while keeping roles/capabilities authoritative internally.
        self.model.input_module = self.input_module
        self.model.target_module = self.target_module
        self.model.context_module = self.context_module
        self.model.primary_context_module = self.primary_context_module
        self.model.active_component = self.active_component
        self.model.active_context_modules = self.active_context_modules
        return self


def _resolve_role(target: str, built_components: dict[str, BuiltComponent]):
    if "." in target:
        _, port = resolve_port(target, built_components)
        return port.endpoint
    return built_components[target]


def _aggregate_capabilities(built_components):
    capabilities: dict[str, list[Any]] = {}
    for component in built_components.values():
        for capability in sorted(component.capabilities):
            handle_key = CAPABILITY_HANDLE_KEYS.get(capability)
            handle = component.runtime_handles.get(handle_key, component)
            capabilities.setdefault(capability, []).append(handle)
    return capabilities


def assemble_architecture(
    spec: ArchitectureSpec,
    context: ArchitectureBuildContext,
    registry: ComponentRegistry,
    *,
    sub_lengths=None,
):
    """Build, validate, wire, and register one subsystem architecture."""
    built_components: dict[str, BuiltComponent] = {}
    with spa.Network(seed=context.seed) as model:
        for component_spec in spec.components.values():
            built_components[component_spec.name] = registry.build(
                context,
                component_spec,
                built_components,
            )

        # Factories expose concrete ports, so semantic incompatibilities are
        # rejected before architecture-level connections are created.
        validate_architecture(spec, built_components)

        for connection in spec.connections:
            _, source_port = resolve_port(connection.source, built_components)
            _, target_port = resolve_port(connection.target, built_components)
            kwargs = {"synapse": connection.synapse}
            if connection.transform is not None:
                kwargs["transform"] = connection.transform
            if connection.label is not None:
                kwargs["label"] = connection.label
            nengo.Connection(source_port.endpoint, target_port.endpoint, **kwargs)

        roles = {
            role: _resolve_role(target, built_components)
            for role, target in spec.roles.items()
        }
        _, prediction_port = resolve_port(spec.roles["prediction"], built_components)
        prediction_probe = context.probe_registry.required(
            prediction_port.endpoint,
            label="prediction",
            synapse=0.01,
        )

    checkpoint_components = (
        [built_components[name] for name in spec.checkpoint_order]
        if spec.checkpoint_order
        else [
            component
            for component in built_components.values()
            if "checkpointed" in component.capabilities
        ]
    )
    # Checkpoint order is architecture state: changing it without changing
    # shapes could silently swap weights between otherwise identical learners.
    learning_connections = [
        connection
        for component in checkpoint_components
        for connection in component.learning_connections
    ]
    probes = {
        f"{component.name}.{name}": probe
        for component in built_components.values()
        for name, probe in component.probes.items()
        if probe is not None
    }
    probes["prediction"] = prediction_probe

    result = AssembledModelResult(
        model=model,
        architecture_spec=spec,
        built_components=built_components,
        roles=roles,
        capabilities=_aggregate_capabilities(built_components),
        learning_connections=learning_connections,
        probes=probes,
        architecture_topology_signature=architecture_signature(spec, built_components),
        strict=context.strict_vocab,
        sub_lengths=sub_lengths,
        learned_init_mode=context.learned_init_mode,
        learned_init_seed=context.learned_init_seed,
        compile_profile_name=str(context.compile_profile.get("name", "full")),
        compile_profile_settings=dict(context.compile_profile.get("settings", {})),
        probe_mode=context.probe_registry.mode,
        created_probe_labels=list(context.probe_registry.created_probe_labels),
        skipped_probe_labels=list(context.probe_registry.skipped_probe_labels),
    )
    return result.install_compatibility_facade()
