"""Concrete Nengo components and the compatibility model facade.

Submodules are intentionally not imported eagerly. Architecture adapters import
``components.net_classes``, while ``components.net_comp`` imports the
architecture registry; eager imports here would make that valid import order
circular.
"""

__all__ = ["net_comp", "net_classes"]
