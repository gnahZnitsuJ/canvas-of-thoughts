from utils.input import context_in, find_target, is_recall
from config import model_parameters as mp
import numpy as np
import nengo
import nengo_spa as spa

class ModelResult:
    def __init__(self, model):
        self.model = model
        self.p_target = [t for t in model.probes if t.label == "target"][0]
        self.p_error = [t for t in model.probes if t.label == "error"][0]
        self.p_post_state = [t for t in model.probes if t.label == "post_state"][0]
        self.p_target_word = [t for t in model.probes if t.label == "target_word"][0]
        self.p_result_word = [t for t in model.probes if t.label == "result_word"][0]

# model
def single(model_vocab, training_set=[], testing_set=[], strict=False, vocab=[]):
    with spa.Network(seed=mp.seed) as model:
        # transcoding training into semantic pointers
        context = spa.Transcode(lambda t : context_in(t, training_set, testing_set, strict, vocab), output_vocab=model_vocab)
        target = spa.Transcode(lambda t: find_target(t, training_set, testing_set, strict, vocab), output_vocab=model_vocab)

        # State (ensembles) for learning
        pre_state = spa.State(
            model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False
        )
        post_state = spa.State(
            model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False
        )
        error = spa.State(model_vocab)

        # signal connections between objects see report for connection logic
        # input and error
        context >> pre_state
        -post_state >> error
        target >> error

        # learning between ensembles
        assert len(pre_state.all_ensembles) == 1
        assert len(post_state.all_ensembles) == 1
        learning_connection = nengo.Connection(
            pre_state.all_ensembles[0],
            post_state.all_ensembles[0],
            function=lambda x: np.random.random(model_vocab.dimensions),
            learning_rule_type=nengo.PES(mp.model_lr), # Prescribed Error Sensitivity
        )
        nengo.Connection(error.output, learning_connection.learning_rule, transform=-1)

        # Suppress learning in the final iteration to test
        is_recall_node = nengo.Node(lambda t: is_recall(t, len(training_set)*mp.tr_impression), size_out=1)
        for ens in error.all_ensembles:
            nengo.Connection(
                is_recall_node, ens.neurons, transform=-100 * np.ones((ens.n_neurons, 1))
            )

        # Probes to record simulation data
        p_target = nengo.Probe(target.output, label="target")
        p_error = nengo.Probe(error.output, label="error")
        p_post_state = nengo.Probe(post_state.output, label="post_state")
        
        # sampling more consistently for word data
        p_target_word = nengo.Probe(target.output, sample_every=mp.tr_impression/2, label="target_word")
        p_result_word = nengo.Probe(post_state.output, sample_every=mp.tr_impression/2, label="result_word")

    return ModelResult(model)
