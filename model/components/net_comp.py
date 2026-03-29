from utils.input import InputModule
from config import model_parameters as mp
import numpy as np
import nengo
import nengo_spa as spa
import components.net_classes as ncls

class ModelResult:
    def __init__(self, model):
        self.model = model
        # self.p_target = [t for t in model.probes if t.label == "target"][0]
        self.p_error = [t for t in model.probes if t.label == "error"][0]
        self.p_post_state = [t for t in model.probes if t.label == "post_state"][0]
        # self.p_target_word = [t for t in model.probes if t.label == "target_word"][0]
        # self.p_result_word = [t for t in model.probes if t.label == "result_word"][0]
        self.p_pred = [t for t in model.probes if t.label == "prediction"][0]
        self.input_module = model.input_module
        self.target_module = model.target_module

def aggregate(sub_lengths, model_vocab, training_set=[], testing_set=[], strict=False, vocab=[]):
    with spa.Network(seed=mp.seed) as model:
        input_module = InputModule(dim=mp.rep_vocab_dim)
        target_module = InputModule(dim=mp.rep_vocab_dim)

        # subsystems
        subs = [ncls.BaseComponent(
            label=f"Component_{t}",
            seed=mp.seed,
            context_in=input_module,
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
            learning_rule_type=nengo.PES(mp.model_lr), # Prescribed Error Sensitivity
        )
        nengo.Connection(error.output, learning_connection.learning_rule, transform=-1)

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
        
        # sampling more consistently for word data
        # p_target_word = nengo.Probe(target.output, sample_every=mp.tr_impression/2, label="target_word")
        # p_result_word = nengo.Probe(post_state.output, sample_every=mp.tr_impression/2, label="result_word")

        # prediction and probe
        prediction = post_state
        pred_probe = nengo.Probe(prediction.output, label="prediction", synapse=0.01)

        model.input_module = input_module
        model.target_module = target_module

    return ModelResult(model)
