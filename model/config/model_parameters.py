import numpy as np

# parameters; change as needed

# random seeding 
seed = 42
rng = np.random.RandomState(seed + 1) # 43 no reason in particular

# dimension and max similarity of semantic pointer vocabulary
# chosen using Johnson-Lindenstrauss Lemma to represent n vectors with
# (arbitrarily chosen) given max similarity angles and dimensions
rep_vocab_dim = 128 # must be divisible by 16 for spa.State to work.
rep_vocab_max_sim = 0.8 # I think this only matters for .populate() ?

# context length: number of words looked at before
context_length = 20

# training impression
tr_impression = 0.01 # an unrealistic actual impression for time's sake
# dataset restrictions, number of articles in reuters corpus
training_restriction = 20
testing_restriction = 2

# model
model_lr = 0.005 # learning rate for PES rule.

# special tokens
unknown_token = "CV_UNK"
pad_token = "CV_PAD"

# strict vocab: include unknown token in vocab
strict_vocab = False