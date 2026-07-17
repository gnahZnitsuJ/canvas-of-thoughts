"""Specification reproducing the established root-context architecture."""

from architecture.spec import ArchitectureSpec


def root_context_v1():
    """Return a fresh baseline spec with two sequential learned predictors."""
    spec = ArchitectureSpec("root-context-v1")
    spec.add("tokens", "input_source")
    spec.add("targets", "target_source", output_count=2)
    spec.add("memory", "context_memory", alpha=0.99)
    spec.add(
        "predictor",
        "context_predictor",
        label="RootContextComponent",
        recall_component="targets",
        learned_init_seed_offset=1,
    )
    spec.add("refiner", "prediction_refiner", recall_component="targets")

    spec.connect("tokens.output", "memory.token", synapse=None)
    spec.connect("memory.context", "predictor.context", synapse=None)
    spec.connect("targets.output", "predictor.target", synapse=None)
    spec.connect("predictor.prediction", "refiner.input", synapse=0.005)
    spec.connect("targets.refiner_output", "refiner.target", synapse=None)

    spec.assign_role("input", "tokens")
    spec.assign_role("target", "targets")
    spec.assign_role("primary_memory", "memory")
    spec.assign_role("active_component", "predictor")
    spec.assign_role("prediction", "refiner.prediction")
    # Preserve the fixed builder's historical checkpoint weight order.
    spec.set_checkpoint_order("refiner", "predictor")
    return spec
