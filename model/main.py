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

# train test split
from sklearn.model_selection import train_test_split

# numpy
import numpy as np

# other string processing
import re

# data processing functions
from utils import input, processing, train_partition, seed_vocab

# config
from utils.config import model_parameters as mp

# spa_vocab = WordsToSPAVocab(vocab)

# # Seed vocab data
# # seed_vocab_vectors = {i: seed_vocab_model.wv.get_vector(i) for i in spa_vocab[0:-2]} # removing pad_token and unknown_token
# seed_vocab_vectors = {i: seed_vocab_model.wv.get_vector(i) for i in spa_vocab[0:-1]} # removing pad_token

# # vocabulary for our model: store of semantic pointers 
# model_vocab = spa.Vocabulary(dimensions=mp.rep_vocab_dim, strict=False, pointer_gen=None, max_similarity=rep_vocab_max_sim)
# # for random creation, model_vocab.populate(";".join(spa_vocab))
# # for random normalized creation, model_vocab.populate(".normalized();".join(spa_vocab))
# # for random unitary creation, model_vocab.populate(".unitary();".join(spa_vocab))

# # creating pointers
# for i,j in seed_vocab_vectors.items():
#     model_vocab.add(key = i, p = j)
# # padding and unknown characters
# # model_vocab.populate(";".join([pad_token, unknown_token])) # strict case
# model_vocab.add(key = pad_token, p = np.zeros(rep_vocab_dim))