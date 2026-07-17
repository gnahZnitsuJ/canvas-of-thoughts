"""Local registry mapping architecture component types to Python factories."""

from collections.abc import Callable

from architecture.contracts import ArchitectureBuildContext, BuiltComponent, ComponentSpec


ComponentFactory = Callable[
    [ArchitectureBuildContext, ComponentSpec, dict[str, BuiltComponent]],
    BuiltComponent,
]


class ComponentRegistry:
    """Resolve explicit component type names without plugin discovery."""

    def __init__(self):
        self._factories: dict[str, ComponentFactory] = {}

    def register(self, component_type: str, factory: ComponentFactory):
        if not component_type:
            raise ValueError("Component type cannot be empty")
        if component_type in self._factories:
            raise ValueError(f"Component type already registered: {component_type}")
        self._factories[component_type] = factory

    def build(
        self,
        context: ArchitectureBuildContext,
        spec: ComponentSpec,
        built_components: dict[str, BuiltComponent],
    ) -> BuiltComponent:
        try:
            factory = self._factories[spec.component_type]
        except KeyError as exc:
            raise KeyError(f"Unknown component type: {spec.component_type}") from exc
        built = factory(context, spec, built_components)
        if built.name != spec.name:
            raise ValueError(
                f"Factory for {spec.component_type} returned component "
                f"{built.name!r}; expected {spec.name!r}"
            )
        return built

    def __contains__(self, component_type: str):
        return component_type in self._factories

