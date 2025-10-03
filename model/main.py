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
                                     strict=False)
vocab = train_test.vocab

# translated words that can be used in the model
spa_vocab = WordsToSPAVocab(vocab)

# vectors from seed_vocab
seed_vocab_vectors = {i: seed_vocab_model.wv.get_vector(i) for i in spa_vocab[0:-1]} # removing pad_token

# vocabulary for our model: store of semantic pointers 
model_vocab = spa.Vocabulary(dimensions=mp.rep_vocab_dim, strict=False, pointer_gen=None, max_similarity=mp.rep_vocab_max_sim)

# creating pointers
for i,j in seed_vocab_vectors.items():
    model_vocab.add(key = i, p = j)
# padding and unknown characters
model_vocab.add(key = mp.pad_token, p = np.zeros(mp.rep_vocab_dim))

# end
print("End")