import numpy as np
from utils.processing import WordsToSPAVocab, SPAVocabToWords
from tqdm import tqdm

# helper class for driving training and simple recall tests
class Trainer:
    def __init__(self, model_result, sim, model_vocab, step_time=0.02):
        self.model_result = model_result
        self.model = model_result.model
        self.sim = sim
        self.model_vocab = model_vocab
        self.step_time = step_time
        # cached vocabulary arrays for nearest-neighbour decoding of predictions
        self.vocab_keys = list(model_vocab.keys())
        self.vocab_vectors = model_vocab.vectors
        self.vocab_norms = np.linalg.norm(self.vocab_vectors, axis=1)
        self.normalized_vocab_vectors = self.vocab_vectors / np.maximum(
            self.vocab_norms[:, None], 1e-12
        )

    # convert a token into the semantic pointer used by the network
    def _vector_for(self, token):
        token_key = WordsToSPAVocab([token])[0]
        return self.model_vocab[token_key].v

    # translate semantic pointer keys back into readable tokens for demo output
    def _decode_key(self, token_key):
        if token_key.startswith("WV_"):
            return SPAVocabToWords([token_key])[0]
        return token_key

    # choose the closest words in vocabulary space to the current model output
    def _top_predictions(self, vector, top_k=3):
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0:
            return []

        normalized_vector = vector / max(vector_norm, 1e-12)
        similarities = self.normalized_vocab_vectors @ normalized_vector
        top_k = min(top_k, len(similarities))
        if top_k <= 0:
            return []

        top_ids = np.argpartition(similarities, -top_k)[-top_k:]
        top_ids = top_ids[np.argsort(similarities[top_ids])[::-1]]

        return [
            (self._decode_key(self.vocab_keys[idx]), float(similarities[idx]))
            for idx in top_ids
        ]

    # present input and optional target vectors, then run the simulator forward
    def present(self, token, target=None, learn=False):
        vec = self._vector_for(token)
        self.model_result.input_module.set(vec)
        if target is None:
            self.model_result.target_module.set(np.zeros(self.model_vocab.dimensions))
        else:
            self.model_result.target_module.set(self._vector_for(target))
        self.model_result.target_module.is_recall = not learn
        self.sim.run(self.step_time)

    # single next-word training step
    def train_pair(self, token, target):
        self.present(token, target=target, learn=True)
        self.model_result.target_module.is_recall = True

    # training across one token sequence
    def train_sequence(self, tokens):
        self.reset_context()

        for i in range(len(tokens) - 1):
            self.present(tokens[i], learn=False)  # build context
            self.present(tokens[i], target=tokens[i+1], learn=True)

    # training across a corpus of token sequences
    def train_corpus(self, sequences):
        for tokens in tqdm(sequences, desc="Training"):
            if len(tokens) > 1:
                self.train_sequence(tokens)

    # recall-mode prediction for a single token
    def predict_next(self, token, top_k=3):
        self.present(token, learn=False)
        prediction = self.sim.data[self.model_result.p_pred][-1]
        return self._top_predictions(prediction, top_k=top_k)

    # recall-mode prediction for a sequence of tokens (resets context each time)
    def predict_next_sequence(self, tokens, top_k=3):
        # reset context implicitly by running fresh sequence
        for token in tokens:
            self.present(token, learn=False)

        prediction = self.sim.data[self.model_result.p_pred][-1]
        return self._top_predictions(prediction, top_k=top_k)
    
    # reset context
    def reset_context(self):
        self.model.context_module.reset.output = 1.0
        self.sim.run(self.step_time)
        self.model.context_module.reset.output = 0.0
