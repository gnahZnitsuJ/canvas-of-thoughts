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

# opencl configuration
import pyopencl as cl
import nengo_ocl
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])

# gensim for seed word embedding
import gensim
from gensim.models import Word2Vec

# numpy
import numpy as np

# other string processing
import re

# data processing functions
from utils import seed_vocab
from utils.input import make_unitary
from utils.processing import WordsToSPAVocab, SPAVocabToWords
from utils.train_partition import multiple_data_partition, data_partition
# config
from config import model_parameters as mp
# components
import components.net_comp as nc
from components.trainer import Trainer
# evaluation
from utils.eval import evaluate_model

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
# add unitary vectors for position encoding if using context subsystems
pos_vec = make_unitary(dim=mp.rep_vocab_dim)
model_vocab.add("POS", pos_vec)

# creating pointers
for i,j in seed_vocab_vectors.items():
    model_vocab.add(key = i, p = j)
# padding and unknown characters
model_vocab.add(key = mp.pad_token, p = np.zeros(mp.rep_vocab_dim))

model_result = nc.Model(
    sub_lengths=[1], # sub_lengths=[1,mp.context_length],
    model_vocab=model_vocab,
    strict=mp.strict_vocab
)

# simulator object
sim = nengo_ocl.Simulator(model_result.model, context=ctx, progress_bar=False)

trainer = Trainer(model_result, sim, model_vocab, step_time=0.02)

print(f"\nTraining on {len(train_test.training_set)} sequences...")
trainer.train_corpus(train_test.training_set)

evaluate_model(trainer, train_test.testing_set)

print("\nSample predictions:\n")

demo_lines = trainer.demo_predictions(train_test.testing_set, max_examples=10)

for line in demo_lines:
    print(line)
# real time simulation

# with nengo.Simulator(model_result.model) as sim:
#     # Use a while loop to keep the simulation running indefinitely for live input
#     print("Simulation running in real-time. Press Ctrl+C to stop.")
#     try:
#         while True:
#             sim.step()
            
#     except KeyboardInterrupt:
#         print("Simulation stopped by user.")
