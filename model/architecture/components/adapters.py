"""Contract adapters for the model's existing input, memory, and learners."""

import numpy as np
import nengo
import nengo_spa as spa

import components.net_classes as ncls
from architecture.contracts import BuiltComponent, Port
from architecture.registry import ComponentRegistry
from config import model_parameters as mp
from utils.build_config import make_learned_connection
from utils.input import InputModule


class AttributeResetHandle:
    """Adapt ContextModule's reset value to a small runtime capability method."""

    def __init__(self, module):
        self.module = module

    def set_reset(self, active):
        self.module.reset_value = 1.0 if active else 0.0


def _semantic_port(ctx, endpoint, direction, *, required=True):
    return Port(
        endpoint=endpoint,
        direction=direction,
        dimensions=ctx.dimensions,
        signal_type="semantic_pointer",
        vocabulary_id=ctx.vocabulary_id,
        required=required,
    )


def build_input_source(ctx, spec, built_components):
    module = InputModule(dim=ctx.dimensions)
    node = module.node()
    return BuiltComponent(
        name=spec.name,
        network=node,
        ports={"output": _semantic_port(ctx, node, "output")},
        capabilities={"schedulable", "recall_controlled"},
        runtime_handles={
            "input": module,
            "schedule": module,
            "recall_control": module,
        },
        signature={"type": "input_source"},
    )


def build_target_source(ctx, spec, built_components):
    module = InputModule(dim=ctx.dimensions)
    output_count = int(spec.parameters.get("output_count", 1))
    if output_count not in {1, 2}:
        raise ValueError("target_source output_count must be 1 or 2")
    nodes = [module.node() for _ in range(output_count)]
    ports = {"output": _semantic_port(ctx, nodes[0], "output")}
    if output_count == 2:
        # The established graph used separate nodes for the predictor and
        # refiner even though both read the same target buffer. Keeping the
        # second endpoint makes baseline reproduction graph-identical.
        ports["refiner_output"] = _semantic_port(ctx, nodes[1], "output")
    return BuiltComponent(
        name=spec.name,
        network=nodes[0] if len(nodes) == 1 else tuple(nodes),
        ports=ports,
        capabilities={"schedulable", "recall_controlled"},
        runtime_handles={
            "target": module,
            "schedule": module,
            "recall_control": module,
        },
        signature={"type": "target_source", "output_count": output_count},
    )


def build_context_memory(ctx, spec, built_components):
    alpha = float(spec.parameters.get("alpha", 0.99))
    module = ncls.ContextModule(
        ctx.vocab,
        alpha=alpha,
        label=spec.parameters.get("label"),
        seed=spec.parameters.get("seed"),
    )
    return BuiltComponent(
        name=spec.name,
        network=module,
        ports={
            "token": _semantic_port(ctx, module.token_in, "input"),
            "context": _semantic_port(ctx, module.output, "output"),
        },
        capabilities={"memory", "resettable"},
        runtime_handles={"reset": AttributeResetHandle(module), "memory": module},
        signature={"type": "context_memory", "alpha": alpha},
    )


def _required_component_handle(built_components, name, handle):
    try:
        return built_components[name].runtime_handles[handle]
    except KeyError as exc:
        raise ValueError(
            f"Component dependency {name!r} with handle {handle!r} must be built first"
        ) from exc


def build_context_predictor(ctx, spec, built_components):
    recall_component = spec.parameters.get("recall_component", "targets")
    recall_source = _required_component_handle(
        built_components, recall_component, "recall_control"
    )
    seed_offset = int(spec.parameters.get("learned_init_seed_offset", 1))
    learned_seed = (
        None
        if ctx.learned_init_seed is None
        else ctx.learned_init_seed + seed_offset
    )
    component = ncls.BaseComponent(
        label=spec.parameters.get("label", "RootContextComponent"),
        seed=spec.parameters.get("seed", ctx.seed),
        context_in=None,
        target_in=None,
        recall_source=recall_source,
        model_vocab=ctx.vocab,
        probe_registry=ctx.probe_registry,
        strict=ctx.strict_vocab,
        learned_init_mode=ctx.learned_init_mode,
        learned_init_seed=learned_seed,
    )
    return BuiltComponent(
        name=spec.name,
        network=component,
        ports={
            "context": _semantic_port(ctx, component.context_input, "input"),
            "target": _semantic_port(ctx, component.target_input, "input"),
            "prediction": _semantic_port(ctx, component.prediction.output, "output"),
        },
        capabilities={"learnable", "checkpointed", "predictor"},
        learning_connections=[component.learning_connection],
        probes={
            "error": component.p_error,
            "post_state": component.p_post_state,
            "context": component.p_context,
        },
        signature={
            "type": "context_predictor",
            "learning_rate": mp.model_lr * 0.5,
            "learned_init_seed_offset": seed_offset,
        },
    )


def build_prediction_refiner(ctx, spec, built_components):
    recall_component = spec.parameters.get("recall_component", "targets")
    recall_source = _required_component_handle(
        built_components, recall_component, "recall_control"
    )

    pre_state = spa.State(
        ctx.vocab,
        subdimensions=ctx.dimensions,
        represent_cc_identity=False,
    )
    post_state = spa.State(
        ctx.vocab,
        subdimensions=ctx.dimensions,
        represent_cc_identity=False,
    )
    error = spa.State(ctx.vocab)

    -post_state >> error
    learning_connection = make_learned_connection(
        pre_state.all_ensembles[0],
        post_state.all_ensembles[0],
        dimensions=ctx.dimensions,
        learning_rate=mp.model_lr * 0.5,
        init_mode=ctx.learned_init_mode,
        init_seed=ctx.learned_init_seed,
    )
    nengo.Connection(error.output, learning_connection.learning_rule, transform=-1)

    is_recall_node = nengo.Node(lambda t: recall_source.is_recall, size_out=1)
    for ensemble in error.all_ensembles:
        nengo.Connection(
            is_recall_node,
            ensemble.neurons,
            transform=-100 * np.ones((ensemble.n_neurons, 1)),
        )

    p_error = ctx.probe_registry.debug(error.output, label="error")
    p_post_state = ctx.probe_registry.debug(post_state.output, label="post_state")
    return BuiltComponent(
        name=spec.name,
        network=post_state,
        ports={
            "input": _semantic_port(ctx, pre_state.input, "input"),
            "target": _semantic_port(ctx, error.input, "input"),
            "prediction": _semantic_port(ctx, post_state.output, "output"),
        },
        capabilities={"learnable", "checkpointed", "prediction_refiner"},
        learning_connections=[learning_connection],
        probes={"error": p_error, "post_state": p_post_state},
        signature={
            "type": "prediction_refiner",
            "learning_rate": mp.model_lr * 0.5,
        },
    )


def default_component_registry():
    """Return the explicit built-in registry used by supported architectures."""
    registry = ComponentRegistry()
    registry.register("input_source", build_input_source)
    registry.register("target_source", build_target_source)
    registry.register("context_memory", build_context_memory)
    registry.register("context_predictor", build_context_predictor)
    registry.register("prediction_refiner", build_prediction_refiner)
    return registry
