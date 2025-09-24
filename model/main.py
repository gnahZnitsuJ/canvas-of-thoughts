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
from config import model_parameters as mp

# spa_vocab = WordsToSPAVocab(vocab)
