# semantic pointer vocabulary generation for model

from gensim.models import Word2Vec
from utils.processing import WordsToSPAVocab
# import utils.train_partition as tp
# from nltk.corpus import reuters
# from utils.config import model_parameters as mp

# # seed vocab parameters
# seed_epochs = 50

# # relevant datasets
# pt_rt = tp.data_partition(reuters, 
#                           training_restriction=mp.training_restriction, 
#                           testing_restriction=mp.testing_restriction, 
#                           strict=False) # reuters partition


# # initial generation

# vocab = vocab = [
#     t
#     for x in pt_rt.training_ids
#     for t in reuters.words(x)
# ]
# vocab = list(set(vocab))
# spa_vocab = WordsToSPAVocab(vocab)

# # attempt at adding a basic "seed" word embedding in here
# # not sure how useful it is to have this learned further
# seed_vocab_data = [WordsToSPAVocab(i) for x in pt_rt.training_ids for i in reuters.sents(x)]
# seed_vocab_model = Word2Vec(sentences=seed_vocab_data, min_count=1, vector_size=mp.rep_vocab_dim, window = mp.context_length, epochs = seed_epochs)

# seed_vocab_model.save("seed_vocab.model")