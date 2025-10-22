import nengo
import nengo_spa as spa
import numpy as np
from utils.input import context_in, find_target, is_recall
from config import model_parameters as mp

# class for single processing step from context
class BaseComponent(spa.Network):
    def __init__(self, model_vocab, label=None, seed=None, training_set=[], testing_set=[], 
                 context_sub_length=mp.context_length, strict=mp.strict_vocab, vocab=[]):
        super(BaseComponent, self).__init__(
            label=label, seed=seed
        )

        with self:
            # transcoding training into semantic pointers
            self.context = spa.Transcode(
                lambda t : context_in(
                    t=t, training_set=training_set, testing_set=testing_set, 
                    sub_length=context_sub_length, strict=strict, vocab=vocab), 
                output_vocab=model_vocab
            )
            self.target = spa.Transcode(
                lambda t: find_target(
                    t=t, training_set=training_set, testing_set=testing_set, 
                    strict=strict, vocab=vocab), 
                output_vocab=model_vocab
            )

            # State (ensembles) for learning
            self.pre_state = spa.State(model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False)
            self.post_state = spa.State(model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False)
            self.error = spa.State(model_vocab)

            # signal connections between objects see report for connection logic
            # input and error
            self.context >> self.pre_state
            -self.post_state >> self.error
            self.target >> self.error

            # learning between ensembles
            assert len(self.pre_state.all_ensembles) == 1
            assert len(self.post_state.all_ensembles) == 1
            self.learning_connection = nengo.Connection(
                self.pre_state.all_ensembles[0],
                self.post_state.all_ensembles[0],
                function=lambda x: np.random.random(model_vocab.dimensions),
                learning_rule_type=nengo.PES(mp.model_lr), # Prescribed Error Sensitivity
            )
            nengo.Connection(self.error.output, self.learning_connection.learning_rule, transform=-1)

            # Suppress learning in the final iteration to test
            self.is_recall_node = nengo.Node(lambda t: is_recall(t, len(training_set)*mp.tr_impression), size_out=1) 
            for ens in self.error.all_ensembles:
                nengo.Connection(
                    self.is_recall_node, ens.neurons, transform=-100 * np.ones((ens.n_neurons, 1))
                )

            # Probes to record simulation data
            self.p_target = nengo.Probe(self.target.output, label="target")
            self.p_error = nengo.Probe(self.error.output, label="error")
            self.p_post_state = nengo.Probe(self.post_state.output, label="post_state")
            
            # sampling more consistently for word data
            self.p_target_word = nengo.Probe(self.target.output, sample_every=mp.tr_impression/2, label="target_word")
            self.p_result_word = nengo.Probe(self.post_state.output, sample_every=mp.tr_impression/2, label="result_word")

            # component prediction output
            self.prediction = self.post_state