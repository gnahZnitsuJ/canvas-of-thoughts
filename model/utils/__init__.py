"""Utility package with deliberately lazy submodule imports.

Callers should import the utility they use. Eagerly importing every utility made
lightweight helpers such as runtime-profile loading require the full Nengo stack.
"""

__all__ = [
    "processing",
    "train_partition",
    "input",
    "seed_vocab",
    "benchmark_compile",
    "calibration",
    "runtime_profile",
]
