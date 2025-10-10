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
import nengo_dl

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

model_result = nc.single(
    model_vocab,
    training_set=train_test.training_set,
    testing_set=train_test.testing_set,
    strict=mp.strict_vocab,
    vocab=vocab
)

# preliminary simulation

# probes
p_target = model_result.p_target
p_error = model_result.p_error
p_post_state = model_result.p_post_state
p_target_word = model_result.p_target_word
p_result_word = model_result.p_result_word

# training time length
training_time = len(train_test.training_set)*mp.tr_impression
# testing time length
testing_time = (len(train_test.testing_set)-1)*mp.tr_impression

simulation_length = training_time + testing_time

with nengo_dl.Simulator(model_result.model) as sim:
    sim.run(simulation_length)

print(f"Amount of simulated training time: {training_time}")
print(f"Amount of simulated testing time: {testing_time}")

# plotting metrics for testing
plt.figure(figsize=(8,8))
plt.plot(sim.trange(), np.linalg.norm(spa.similarity(sim.data[p_post_state], model_vocab) - spa.similarity(sim.data[p_target], model_vocab), axis = 1))
plt.ylim(bottom=0)
plt.title("Training")
plt.ylabel("Norm of Difference in Vocab Similarity between Result and Target")
plt.xlabel(f"Time (Testing Starts After: {training_time})")

# example text
print(" ".join(SPAVocabToWords(["WV_" + spa.text(sim.data[p_result_word][i], vocab=model_vocab, maximum_count=1).split("WV_",1)[-1] 
 for i in range(len(train_test.training_set), len(train_test.training_set)+100)])[::2]))
print(" ".join(SPAVocabToWords(["WV_" + spa.text(sim.data[p_target_word][i], vocab=model_vocab, maximum_count=1).split("WV_",1)[-1] 
 for i in range(len(train_test.training_set), len(train_test.training_set)+100)])[::2]))

# end
print("End")