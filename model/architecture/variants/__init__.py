"""Named architecture specifications supported by the model builder."""

from architecture.variants.no_refiner import no_refiner_v1
from architecture.variants.root_context_v1 import root_context_v1

ARCHITECTURE_BUILDERS = {
    "root-context-v1": root_context_v1,
    "no-refiner-v1": no_refiner_v1,
}


def architecture_spec(name):
    """Construct a fresh specification for one supported architecture name."""
    try:
        builder = ARCHITECTURE_BUILDERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(ARCHITECTURE_BUILDERS))
        raise ValueError(f"Unknown architecture {name!r}; choose one of: {choices}") from exc
    return builder()


__all__ = [
    "ARCHITECTURE_BUILDERS",
    "architecture_spec",
    "no_refiner_v1",
    "root_context_v1",
]
