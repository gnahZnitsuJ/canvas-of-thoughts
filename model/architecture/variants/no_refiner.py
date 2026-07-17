"""Mechanical proof variant that bypasses the top-level prediction refiner."""

from architecture.variants.root_context_v1 import root_context_v1


def no_refiner_v1():
    """Remove the second learner and expose the context predictor directly."""
    spec = root_context_v1().copy(name="no-refiner-v1")
    spec.remove("refiner")
    spec.replace("targets", "target_source", output_count=1)
    spec.assign_role("prediction", "predictor.prediction")
    return spec
