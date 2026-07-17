"""Reusable Nengo network implementations adapted by architecture factories."""

import nengo
import nengo_spa as spa
from nengo_spa.network import Network
import numpy as np
from config import model_parameters as mp
from utils.build_config import make_learned_connection
from utils.probes import ProbeRegistry


class BaseComponent(Network):
    """Learn a next-token mapping from the active context state."""

    def __init__(
        self,
        model_vocab,
        context_in=None,
        target_in=None,
        recall_source=None,
        probe_registry=None,
        label=None,
        seed=None,
        context_sub_length=mp.context_length,
        strict=mp.strict_vocab,
        learned_init_mode="random-function",
        learned_init_seed=None,
    ):
        super().__init__(label=label, seed=seed)
        probe_registry = probe_registry or ProbeRegistry()

        with self:
            # Legacy callers may still pass concrete upstream modules. The
            # architecture assembler instead wires directly into the semantic
            # input endpoints exposed below, avoiding proxy nodes and preserving
            # the existing graph size.
            if context_in is None:
                context_source = None
            elif hasattr(context_in, "node"):
                context_source = context_in.node()
            else:
                context_source = context_in.output

            if target_in is None:
                target_source = None
            elif hasattr(target_in, "node"):
                target_source = target_in.node()
            else:
                target_source = target_in.output

            # State (ensembles) for learning.
            self.pre_state = spa.State(
                model_vocab,
                subdimensions=model_vocab.dimensions,
                represent_cc_identity=False,
            )
            self.post_state = spa.State(
                model_vocab,
                subdimensions=model_vocab.dimensions,
                represent_cc_identity=False,
            )
            self.error = spa.State(model_vocab)

            # Normalize context to keep the learned mapping from being dominated
            # by context magnitude growth over longer sequences.
            self.norm_node = nengo.Node(
                lambda t, x: x / (np.linalg.norm(x) + 1e-8),
                size_in=model_vocab.dimensions,
                size_out=model_vocab.dimensions,
            )
            self.context_input = self.norm_node
            self.target_input = self.error.input
            if context_source is not None:
                nengo.Connection(context_source, self.context_input, synapse=None)
            nengo.Connection(self.norm_node, self.pre_state.input, synapse=None)

            # The learning signal is target minus current prediction.
            if target_source is not None:
                nengo.Connection(target_source, self.target_input, synapse=None)
            -self.post_state >> self.error

            assert len(self.pre_state.all_ensembles) == 1
            assert len(self.post_state.all_ensembles) == 1
            self.learning_connection = make_learned_connection(
                self.pre_state.all_ensembles[0],
                self.post_state.all_ensembles[0],
                dimensions=model_vocab.dimensions,
                learning_rate=mp.model_lr * 0.5,
                init_mode=learned_init_mode,
                init_seed=learned_init_seed,
            )
            nengo.Connection(
                self.error.output,
                self.learning_connection.learning_rule,
                transform=-1,
            )

            # Suppress learning during recall mode; training toggles this on the
            # target module so the same network can switch between train/recall.
            if recall_source is None:
                if hasattr(target_in, "is_recall"):
                    recall_source = target_in
                elif hasattr(context_in, "is_recall"):
                    recall_source = context_in
            if recall_source is None:
                raise ValueError("BaseComponent requires a recall-control source")
            self.is_recall_node = nengo.Node(
                lambda t: recall_source.is_recall,
                size_out=1,
            )
            for ens in self.error.all_ensembles:
                nengo.Connection(
                    self.is_recall_node,
                    ens.neurons,
                    transform=-100 * np.ones((ens.n_neurons, 1)),
                )

            # These probes are useful for inspection and debugging, but normal
            # prediction/evaluation only requires the top-level prediction probe.
            self.p_error = probe_registry.debug(self.error.output, label="error")
            self.p_post_state = probe_registry.debug(
                self.post_state.output,
                label="post_state",
            )
            # In assembled builds the normalization node is the public context
            # input endpoint. Probing it retains the same instrumentation count,
            # while reporting the signal actually consumed by the learner.
            self.context = context_source or self.context_input
            self.target = target_source or self.target_input
            self.p_context = probe_registry.debug(self.context, label="context")

            # Component prediction output.
            self.prediction = self.post_state


class ContextModule(spa.Network):
    """Maintain the rolling root context that feeds prediction."""

    def __init__(self, vocab, alpha=0.99, label=None, seed=None):
        super().__init__(label=label, seed=seed)

        dimensions = vocab.dimensions

        with self:
            self.token_in = nengo.Node(size_in=dimensions)
            self.pos = spa.State(vocab)

            self.pos_init = nengo.Node(output=vocab["POS"].v)
            nengo.Connection(self.pos_init, self.pos.input, synapse=None)

            self.context = spa.State(vocab, feedback=alpha)

            self.bind = spa.Bind(vocab)
            nengo.Connection(self.token_in, self.bind.input_left)
            nengo.Connection(self.pos.output, self.bind.input_right)
            nengo.Connection(self.bind.output, self.context.input)

            # Keep rolling the position pointer forward after each token.
            self.pos_update = spa.Bind(vocab)
            self.pos_step = nengo.Node(output=vocab["POS"].v)
            nengo.Connection(self.pos.output, self.pos_update.input_left)
            nengo.Connection(self.pos_step, self.pos_update.input_right, synapse=None)
            nengo.Connection(self.pos_update.output, self.pos.input)

            self.reset_value = 0.0
            self.reset = nengo.Node(lambda t: self.reset_value)

            for ens in self.context.all_ensembles:
                nengo.Connection(
                    self.reset,
                    ens.neurons,
                    transform=-100 * np.ones((ens.n_neurons, 1)),
                )

            for ens in self.pos.all_ensembles:
                nengo.Connection(
                    self.reset,
                    ens.neurons,
                    transform=-100 * np.ones((ens.n_neurons, 1)),
                )

            self.output = self.context.output
