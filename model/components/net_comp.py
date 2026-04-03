# composition of the network model from various parts

from utils.input import InputModule
from config import model_parameters as mp
import numpy as np
import nengo
import nengo_spa as spa
import components.net_classes as ncls

# helper class to store model and probes
class ModelResult:
    def __init__(self, model, context_module, strict=None):
        self.model = model
        self.p_error = [t for t in model.probes if t.label == "error"][0]
        self.p_post_state = [t for t in model.probes if t.label == "post_state"][0]
        self.p_pred = [t for t in model.probes if t.label == "prediction"][0]
        self.input_module = model.input_module
        self.target_module = model.target_module
        self.context_module = context_module
        self.strict = strict
        model.context_module = context_module

# function that returns a model result object containing the desired model
def Model(sub_lengths, model_vocab, strict=mp.strict_vocab):
    with spa.Network(seed=mp.seed) as model:
        input_module = InputModule(dim=mp.rep_vocab_dim)
        target_module = InputModule(dim=mp.rep_vocab_dim)
        context_module = ncls.ContextModule(model_vocab)
        
        # subsystems
        subs = [ncls.BaseComponent(
            label=f"Component_{t}",
            seed=mp.seed,
            context_in=context_module,
            target_in=target_module,
            model_vocab=model_vocab,
            context_sub_length=t,
            strict=mp.strict_vocab) 
            for t in sub_lengths]

        target_node = target_module.node()

        # State (ensembles) for learning
        pre_state = spa.State(
            model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False
        )
        post_state = spa.State(
            model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False
        )
        error = spa.State(model_vocab)

        for i in subs:
            i.prediction >> pre_state
        
        -post_state >> error
        nengo.Connection(target_node, error.input, synapse=None)


        # learning between ensembles
        # assert len(pre_state.all_ensembles) == 1
        # assert len(post_state.all_ensembles) == 1
        learning_connection = nengo.Connection(
            pre_state.all_ensembles[0],
            post_state.all_ensembles[0],
            function=lambda x: np.random.random(model_vocab.dimensions),
            learning_rule_type=nengo.PES(mp.model_lr*0.5), # Prescribed Error Sensitivity
        )
        nengo.Connection(error.output, learning_connection.learning_rule, transform=-1)
        nengo.Connection(input_module.node(),context_module.token_in, synapse=None)

        # Suppress learning in the final iteration to test
        is_recall_node = nengo.Node(lambda t: target_module.is_recall, size_out=1)
        for ens in error.all_ensembles:
            nengo.Connection(
                is_recall_node, ens.neurons, transform=-100 * np.ones((ens.n_neurons, 1))
            )

        # Probes to record simulation data
        # p_target = nengo.Probe(target.output, label="target")
        p_error = nengo.Probe(error.output, label="error")
        p_post_state = nengo.Probe(post_state.output, label="post_state")
        
        # prediction and probe
        prediction = post_state
        pred_probe = nengo.Probe(prediction.output, label="prediction", synapse=0.01)

        model.input_module = input_module
        model.target_module = target_module

    return ModelResult(
        model,
        context_module=context_module,
        strict=strict
    )