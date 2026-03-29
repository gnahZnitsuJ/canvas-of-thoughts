import os

# for preprocessing of natural language
# important nltk requisite downloads include: tokenize, reuters
import nltk
# reuters articles corpus for training
from nltk.corpus import reuters

# nengo: see www.nengo.ai
import nengo
import nengo_spa as spa
import matplotlib.pyplot as plt
# import nengo_dl
import nengo_ocl

# gensim for seed word embedding
import gensim
from gensim.models import Word2Vec

# numpy
import numpy as np

# other string processing
import re

# data processing functions
from utils import input, train_partition, seed_vocab
from utils.processing import WordsToSPAVocab, SPAVocabToWords
from utils.train_partition import multiple_data_partition, data_partition
# config
from config import model_parameters as mp
# components
import components.net_comp as nc
# from components.net_comp import test_model

# datasets
datasets = [reuters]

# seed vocab data using training sets of all datasets
if os.path.isfile("canvas-of-thoughts/model/utils/seed_vocab.model"):
    print("seed_vocab.model exists in the current directory.")
    seed_vocab_model = Word2Vec.load("canvas-of-thoughts/model/utils/seed_vocab.model")
else:
    print("seed_vocab.model does NOT exist in the current directory.")
    print("Generating seed vocabulary model...")
    from utils import seed_vocab
    seed_vocab.generate_seed_vocab(datasets)
    seed_vocab_model = Word2Vec.load("canvas-of-thoughts/model/utils/seed_vocab.model")
 
# unique words in training set

train_test = multiple_data_partition(datasets, 
                                     training_restriction=mp.training_restriction, 
                                     testing_restriction=mp.testing_restriction, 
                                     strict=mp.strict_vocab)
vocab = train_test.vocab

# translated words that can be used in the model
spa_vocab = WordsToSPAVocab(vocab)

# vectors from seed_vocab
if mp.strict_vocab == False:
    # non-strict case: removing pad_token from spa_vocab in seeding vocab
    seed_vocab_vectors = {i: seed_vocab_model.wv.get_vector(i) 
                          for i in spa_vocab if i != mp.pad_token}
else:
    # strict case: removing pad_token and unknown_token from spa_vocab in seeding vocab
    seed_vocab_vectors = {i: seed_vocab_model.wv.get_vector(i) 
                          for i in spa_vocab if (i != mp.pad_token and i != mp.unknown_token)}

# vocabulary for our model: store of semantic pointers 
model_vocab = spa.Vocabulary(dimensions=mp.rep_vocab_dim, strict=mp.strict_vocab, pointer_gen=None, max_similarity=mp.rep_vocab_max_sim)

# creating pointers
for i,j in seed_vocab_vectors.items():
    model_vocab.add(key = i, p = j)
# padding and unknown characters
model_vocab.add(key = mp.pad_token, p = np.zeros(mp.rep_vocab_dim))

model_result = nc.aggregate(
    sub_lengths=[1], # sub_lengths=[1,mp.context_length],
    model_vocab=model_vocab,
    training_set=train_test.training_set,
    testing_set=train_test.testing_set,
    strict=mp.strict_vocab,
    vocab=vocab
)

demo_training_set = train_test.training_set[:2]
demo_testing_set = train_test.testing_set[:1]

# probes
# p_target = model_result.p_target
# p_error = model_result.p_error
# p_post_state = model_result.p_post_state
# p_target_word = model_result.p_target_word
# p_result_word = model_result.p_result_word

# simulator object
sim = nengo_ocl.Simulator(model_result.model)

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

        similarities = self.vocab_vectors @ vector
        similarities /= np.maximum(self.vocab_norms * vector_norm, 1e-12)
        top_ids = np.argsort(similarities)[-top_k:][::-1]

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
        for i in range(len(tokens) - 1):
            self.train_pair(tokens[i], tokens[i + 1])

    # training across a corpus of token sequences
    def train_corpus(self, sequences):
        for tokens in sequences:
            if len(tokens) > 1:
                self.train_sequence(tokens)

    # recall-mode prediction for a single token
    def predict_next(self, token, top_k=3):
        self.present(token, learn=False)
        prediction = self.sim.data[self.model_result.p_pred][-1]
        return self._top_predictions(prediction, top_k=top_k)

    # formatted console demo for a few next-word predictions
    def demo_predictions(self, sequences, max_examples=8, top_k=3):
        lines = ["", "Next-word prediction demo", "-" * 72]

        shown = 0
        for tokens in sequences:
            for i in range(len(tokens) - 1):
                candidates = self.predict_next(tokens[i], top_k=top_k)
                if not candidates:
                    continue

                top_word, top_score = candidates[0]
                candidate_text = ", ".join(
                    f"{word} ({score:.3f})" for word, score in candidates
                )
                lines.append(
                    f"input={tokens[i]!r:>16} | predicted={top_word!r:>16} "
                    f"| expected={tokens[i + 1]!r:>16} | top={candidate_text}"
                )

                shown += 1
                if shown >= max_examples:
                    return lines

        return lines

trainer = Trainer(model_result, sim, model_vocab, step_time=0.02)

summary_line = (
    f"Training demo model on {len(demo_training_set)} article(s); "
    f"testing on {len(demo_testing_set)} article(s)."
)
print(summary_line)
trainer.train_corpus(demo_training_set)
demo_lines = [summary_line] + trainer.demo_predictions(demo_testing_set)

for line in demo_lines:
    print(line)

# print(f"Amount of simulated training time: {training_time}")
# print(f"Amount of simulated testing time: {testing_time}")

# # plotting metrics for testing
# plt.figure(figsize=(8,8))
# plt.plot(sim.trange(), np.linalg.norm(spa.similarity(sim.data[p_post_state], model_vocab) - spa.similarity(sim.data[p_target], model_vocab), axis = 1))
# plt.ylim(bottom=0)
# plt.title("Training")
# plt.ylabel("Norm of Difference in Vocab Similarity between Result and Target")
# plt.xlabel(f"Time (Testing Starts After: {training_time})")

# # example text
# print(" ".join(SPAVocabToWords(["WV_" + spa.text(sim.data[p_result_word][i], vocab=model_vocab, maximum_count=1).split("WV_",1)[-1] 
#  for i in range(len(train_test.training_set), len(train_test.training_set)+100)])[::2]))
# print(" ".join(SPAVocabToWords(["WV_" + spa.text(sim.data[p_target_word][i], vocab=model_vocab, maximum_count=1).split("WV_",1)[-1] 
#  for i in range(len(train_test.training_set), len(train_test.training_set)+100)])[::2]))

# # end
# print("End")

# real time simulation

# with nengo.Simulator(model_result.model) as sim:
#     # Use a while loop to keep the simulation running indefinitely for live input
#     print("Simulation running in real-time. Press Ctrl+C to stop.")
#     try:
#         while True:
#             sim.step()
            
#     except KeyboardInterrupt:
#         print("Simulation stopped by user.")
