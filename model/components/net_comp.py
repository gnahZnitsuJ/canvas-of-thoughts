# composition of the network model from various parts

from utils.input import InputModule
from config import model_parameters as mp
import numpy as np
import nengo
import nengo_spa as spa
import components.net_classes as ncls
from utils.probes import ProbeRegistry


class ModelResult:
    """Bundle the built model together with the probes and active modules.

    The root `context_module` is the canonical model memory. Legacy
    per-component contexts are tracked separately so we can document the old
    architecture without keeping it on the active prediction path.
    """

    def __init__(
        self,
        model,
        context_module,
        active_component,
        probe_registry,
        strict=None,
        learning_connections=None,
        sub_lengths=None,
        legacy_context_modules=None,
    ):
        self.model = model
        self.probe_mode = probe_registry.mode
        self.created_probe_labels = list(probe_registry.created_probe_labels)
        self.skipped_probe_labels = list(probe_registry.skipped_probe_labels)
        self.p_error = next((t for t in model.probes if t.label == "error"), None)
        self.p_post_state = next(
            (t for t in model.probes if t.label == "post_state"),
            None,
        )
        self.p_pred = [t for t in model.probes if t.label == "prediction"][0]
        self.input_module = model.input_module
        self.target_module = model.target_module
        self.context_module = context_module
        self.primary_context_module = context_module
        self.active_component = active_component
        self.active_components = [active_component]
        self.active_context_modules = [context_module]
        self.legacy_context_modules = legacy_context_modules or []
        self.all_context_modules = [
            *self.active_context_modules,
            *self.legacy_context_modules,
        ]
        self.strict = strict
        self.learning_connections = learning_connections or []
        self.sub_lengths = sub_lengths

        model.context_module = context_module
        model.primary_context_module = context_module
        model.active_component = active_component
        model.active_context_modules = self.active_context_modules


# function that returns a model result object containing the desired model
def Model(sub_lengths, model_vocab, strict=mp.strict_vocab, probe_mode="debug"):
    """Build the model around one active root-context prediction path.

    `sub_lengths` is retained for compatibility and telemetry, but the old
    per-component context fan-out is intentionally deferred/legacy.
    """
    probe_registry = ProbeRegistry(mode=probe_mode)
    with spa.Network(seed=mp.seed) as model:
        input_module = InputModule(dim=mp.rep_vocab_dim)
        target_module = InputModule(dim=mp.rep_vocab_dim)
        context_module = ncls.ContextModule(model_vocab)

        # The root context is the canonical memory and the only active context
        # driving prediction. `sub_lengths` is kept only as a legacy/deferred
        # configuration knob for future architectural work.
        legacy_context_modules = []
        active_component = ncls.BaseComponent(
            label="RootContextComponent",
            seed=mp.seed,
            context_in=context_module,
            target_in=target_module,
            model_vocab=model_vocab,
            probe_registry=probe_registry,
            strict=mp.strict_vocab,
        )

        target_node = target_module.node()

        # State (ensembles) for learning
        pre_state = spa.State(
            model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False
        )
        post_state = spa.State(
            model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False
        )
        error = spa.State(model_vocab)

        active_component.prediction >> pre_state

        -post_state >> error
        nengo.Connection(target_node, error.input, synapse=None)

        # learning between ensembles
        # assert len(pre_state.all_ensembles) == 1
        # assert len(post_state.all_ensembles) == 1
        learning_connection = nengo.Connection(
            pre_state.all_ensembles[0],
            post_state.all_ensembles[0],
            function=lambda x: np.random.random(model_vocab.dimensions),
            learning_rule_type=nengo.PES(mp.model_lr * 0.5),  # Prescribed Error Sensitivity
        )
        nengo.Connection(error.output, learning_connection.learning_rule, transform=-1)

        # Raw token input enters the root context memory directly.
        nengo.Connection(input_module.node(), context_module.token_in, synapse=None)

        # Suppress learning in the final iteration to test
        is_recall_node = nengo.Node(lambda t: target_module.is_recall, size_out=1)
        for ens in error.all_ensembles:
            nengo.Connection(
                is_recall_node, ens.neurons, transform=-100 * np.ones((ens.n_neurons, 1))
            )

        # Probes to record simulation data
        probe_registry.debug(error.output, label="error")
        probe_registry.debug(post_state.output, label="post_state")

        # prediction and probe
        prediction = post_state
        probe_registry.required(
            prediction.output,
            label="prediction",
            synapse=0.01,
        )

        model.input_module = input_module
        model.target_module = target_module

    all_learning_connections = [learning_connection, active_component.learning_connection]

    return ModelResult(
        model,
        context_module=context_module,
        active_component=active_component,
        probe_registry=probe_registry,
        strict=strict,
        learning_connections=all_learning_connections,
        sub_lengths=sub_lengths,
        legacy_context_modules=legacy_context_modules,
    )
