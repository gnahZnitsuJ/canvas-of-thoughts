"""Committed defaults grouped by their change and compatibility domain.

Import the owning module directly instead of relying on package-level re-exports;
this keeps dependencies on model, data, and runtime policy visible at call sites.
"""

__all__ = ["data_defaults", "model_defaults", "runtime_defaults"]
