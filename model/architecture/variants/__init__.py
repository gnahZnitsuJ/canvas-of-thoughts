"""Discover named architecture specifications supported by the model builder."""

from dataclasses import dataclass
from importlib import import_module
from pkgutil import iter_modules


@dataclass(frozen=True)
class ArchitectureVariant:
    """Discovered architecture builder and its selection policy."""

    name: str
    builder: object
    module_name: str
    expose_to_cli: bool
    is_default: bool


def _discover_architecture_variants():
    """Import variant modules and collect their explicitly marked builders."""
    discovered = {}
    package_prefix = f"{__name__}."

    for module_info in sorted(iter_modules(__path__), key=lambda item: item.name):
        if module_info.name.startswith("_"):
            continue

        module = import_module(f"{package_prefix}{module_info.name}")
        builder = getattr(module, "ARCHITECTURE_BUILDER", None)
        if builder is None:
            continue
        if not callable(builder):
            raise TypeError(
                f"{module.__name__}.ARCHITECTURE_BUILDER must be callable"
            )

        spec = builder()
        if spec.name in discovered:
            first_module = discovered[spec.name].module_name
            raise ValueError(
                f"Duplicate architecture name {spec.name!r} in "
                f"{first_module!r} and {module.__name__!r}"
            )

        discovered[spec.name] = ArchitectureVariant(
            name=spec.name,
            builder=builder,
            module_name=module.__name__,
            expose_to_cli=bool(getattr(module, "ARCHITECTURE_EXPOSE_TO_CLI", True)),
            is_default=bool(getattr(module, "ARCHITECTURE_DEFAULT", False)),
        )

    if not discovered:
        raise RuntimeError("No architecture variant modules were discovered")

    defaults = [variant.name for variant in discovered.values() if variant.is_default]
    if len(defaults) != 1:
        raise RuntimeError(
            "Exactly one architecture variant must set ARCHITECTURE_DEFAULT=True; "
            f"found: {defaults}"
        )
    if not discovered[defaults[0]].expose_to_cli:
        raise RuntimeError("The default architecture must be exposed to the CLI")

    return discovered


ARCHITECTURE_VARIANTS = _discover_architecture_variants()
ARCHITECTURE_BUILDERS = {
    name: variant.builder for name, variant in ARCHITECTURE_VARIANTS.items()
}
DEFAULT_ARCHITECTURE_NAME = next(
    variant.name for variant in ARCHITECTURE_VARIANTS.values() if variant.is_default
)


def available_architectures():
    """Return sorted architecture names intentionally exposed to CLI users."""
    return tuple(
        sorted(
            variant.name
            for variant in ARCHITECTURE_VARIANTS.values()
            if variant.expose_to_cli
        )
    )


def architecture_spec(name):
    """Construct a fresh specification for one discovered architecture name."""
    try:
        builder = ARCHITECTURE_BUILDERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(ARCHITECTURE_BUILDERS))
        raise ValueError(f"Unknown architecture {name!r}; choose one of: {choices}") from exc
    return builder()


__all__ = [
    "ARCHITECTURE_BUILDERS",
    "ARCHITECTURE_VARIANTS",
    "DEFAULT_ARCHITECTURE_NAME",
    "ArchitectureVariant",
    "architecture_spec",
    "available_architectures",
]
