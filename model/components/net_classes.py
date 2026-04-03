# reusable generic network classes that may be useful later

import nengo
import nengo_spa as spa
from nengo_spa.network import Network
import numpy as np
from config import model_parameters as mp

# class for single processing step from context
class BaseComponent(Network):
    def __init__(self, model_vocab, context_in, target_in, 
                 label=None, seed=None, 
                 context_sub_length=mp.context_length, 
                 strict=mp.strict_vocab):
        super().__init__(
            label=label, seed=seed
        )

        with self:
            # transcoding training into semantic pointers
            # allow either InputModule (has .node()) or ContextModule (has .output)
            if hasattr(context_in, "node"):
                self.context = context_in.node()
            else:
                self.context = context_in.output
            self.target = target_in.node()

            # State (ensembles) for learning
            self.pre_state = spa.State(model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False)
            self.post_state = spa.State(model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False)
            self.error = spa.State(model_vocab)

            # signal connections between objects see report for connection logic
            # normalize context to stabilize learning
            self.norm_node = nengo.Node(
                lambda t, x: x / (np.linalg.norm(x) + 1e-8),
                size_in=model_vocab.dimensions,
                size_out=model_vocab.dimensions,
            )
            nengo.Connection(self.context, self.norm_node, synapse=None)
            nengo.Connection(self.norm_node, self.pre_state.input, synapse=None)

            # input and error
            nengo.Connection(self.target, self.error.input, synapse=None)
            -self.post_state >> self.error
            nengo.Connection(self.target, self.error.input, synapse=None)


            # learning between ensembles
            assert len(self.pre_state.all_ensembles) == 1
            assert len(self.post_state.all_ensembles) == 1
            self.learning_connection = nengo.Connection(
                self.pre_state.all_ensembles[0],
                self.post_state.all_ensembles[0],
                function=lambda x: np.random.random(model_vocab.dimensions),
                learning_rule_type=nengo.PES(mp.model_lr*0.5), # Prescribed Error Sensitivity
            )
            nengo.Connection(self.error.output, self.learning_connection.learning_rule, transform=-1)

            # Suppress learning during recall mode; training toggles this on the target module.
            recall_source = target_in if hasattr(target_in, "is_recall") else context_in
            self.is_recall_node = nengo.Node(lambda t: recall_source.is_recall, size_out=1)
            for ens in self.error.all_ensembles:
                nengo.Connection(
                    self.is_recall_node, ens.neurons, transform=-100 * np.ones((ens.n_neurons, 1))
                )

            # Probes to record simulation data
            # self.p_target = nengo.Probe(self.target.output, label="target")
            self.p_error = nengo.Probe(self.error.output, label="error")
            self.p_post_state = nengo.Probe(self.post_state.output, label="post_state")
            self.p_context = nengo.Probe(self.context, label="context")

            # component prediction output
            self.prediction = self.post_state

# module that stores incoming tokens as future context to feed into model
class ContextModule(spa.Network):
    def __init__(self, vocab, alpha=0.9, label=None, seed=None):
        super().__init__(label=label, seed=seed)

        D = vocab.dimensions

        with self:
            # input token
            self.token_in = nengo.Node(size_in=D)

            # position state
            self.pos = spa.State(vocab)

            # initialize position
            self.pos_init = nengo.Node(output=vocab["POS"].v)
            nengo.Connection(self.pos_init, self.pos.input, synapse=None)

            # context memory
            self.context = spa.State(vocab, feedback=alpha)

            # binding pos and token
            self.bind = spa.Bind(vocab)

            nengo.Connection(self.token_in, self.bind.input_left)
            nengo.Connection(self.pos.output, self.bind.input_right)

            # add into context
            nengo.Connection(self.bind.output, self.context.input)

            # update position: we pind pos to itself
            self.pos_update = spa.Bind(vocab)
            self.pos_step = nengo.Node(output=vocab["POS"].v)
            nengo.Connection(self.pos.output, self.pos_update.input_left)
            nengo.Connection(self.pos_step, self.pos_update.input_right, synapse=None)
            nengo.Connection(self.pos_update.output, self.pos.input)

            # reset signal to clear context and position (if needed)
            self.reset_value = 0.0
            self.reset = nengo.Node(lambda t: self.reset_value)

            # reset context
            for ens in self.context.all_ensembles:
                nengo.Connection(
                    self.reset,
                    ens.neurons,
                    transform=-100 * np.ones((ens.n_neurons, 1))
                )

            # reset position
            for ens in self.pos.all_ensembles:
                nengo.Connection(
                    self.reset,
                    ens.neurons,
                    transform=-100 * np.ones((ens.n_neurons, 1))
                )
            
            # expose output
            self.output = self.context.output
