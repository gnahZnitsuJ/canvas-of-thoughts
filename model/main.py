# file handling
import os
from pathlib import Path

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
from components.runtime import ModelRuntime
# evaluation
from utils.eval import evaluate_model

# datasets
datasets = [reuters]

# seed vocab data using training sets of all datasets
# project root directory
BASE_DIR = Path(__file__).resolve().parent

# seed vocab model path
SEED_VOCAB_PATH = (
    BASE_DIR
    / "utils"
    / "seed_vocab.model"
)

# seed vocab data using training sets of all datasets
if SEED_VOCAB_PATH.is_file():
    print("seed_vocab.model exists.")

    seed_vocab_model = Word2Vec.load(
        str(SEED_VOCAB_PATH)
    )

else:
    print("seed_vocab.model does NOT exist.")
    print("Generating seed vocabulary model...")

    from utils import seed_vocab

    seed_vocab.generate_seed_vocab(datasets)

    if not SEED_VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f"Failed to generate {SEED_VOCAB_PATH}"
        )

    seed_vocab_model = Word2Vec.load(
        str(SEED_VOCAB_PATH)
    )
 
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

# creating vocab keys
for i,j in seed_vocab_vectors.items():
    model_vocab.add(key = i, p = j)

# padding and unknown characters
model_vocab.add(key = mp.pad_token, p = np.zeros(mp.rep_vocab_dim))

model_result = nc.Model(
    sub_lengths=[2,4,8,16,32,64,128], # sub_lengths=[1,mp.context_length],
    model_vocab=model_vocab,
    strict=mp.strict_vocab
)

# simulator object
sim = nengo_ocl.Simulator(model_result.model, context=ctx, progress_bar=False)

runtime = ModelRuntime(model_result, sim, model_vocab, step_time=0.02)

# simple test
runtime.train_or_load(
    train_test.training_set,
    checkpoint_path="reuters_checkpoint.pkl"
)

evaluate_model(runtime, train_test.testing_set)

print("\nSample predictions:\n")

demo_count = 0
max_demo_examples = 10

for tokens in train_test.testing_set:
    if len(tokens) < 2:
        continue

    for i in range(len(tokens) - 1):
        prefix = tokens[:i+1]
        target = tokens[i+1]
        predictions = runtime.predict_next_sequence(prefix, top_k=3)
        prediction_text = ", ".join(
            f"{word} ({score:.3f})" for word, score in predictions
        )

        print(f"{' '.join(prefix)} -> {prediction_text} | target: {target}")

        demo_count += 1
        if demo_count >= max_demo_examples:
            break

    if demo_count >= max_demo_examples:
        break

# interactive component
runtime.interactive_loop(
    top_k=5,
    generate=True,
    max_tokens=15
)
