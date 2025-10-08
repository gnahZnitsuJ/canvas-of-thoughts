# semantic pointer vocabulary generation for model

from gensim.models import Word2Vec
from utils.processing import WordsToSPAVocab
import utils.train_partition as tp
from nltk.corpus import reuters
from config import model_parameters as mp

# seed vocab parameters
seed_epochs = 50

def generate_seed_vocab(dataset_list=[]):
    """Generate and save a seed vocabulary model using Word2Vec."""
    """Returns the unique vocabulary used in the training sets of the provided datasets."""

    if dataset_list==[]: # exit if no datasets provided
        print("No datasets provided for seed vocabulary generation.")
        return None
    
    # initialize vocab and seed_vocab_data
    vocab = []
    seed_vocab_data = []

    for ds in dataset_list:
        pt = tp.data_partition(ds, 
                               training_restriction=mp.training_restriction, 
                               testing_restriction=mp.testing_restriction, 
                               strict=mp.strict_vocab) # partition of dataset
        
        # appending the vocab and the sentence data of partitions in the dataset list
        vocab += [
            t
            for x in pt.training_ids
            for t in ds.words(x)
        ]

        seed_vocab_data += [WordsToSPAVocab(i) for x in pt.training_ids for i in reuters.sents(x)]
    
    vocab = list(set(vocab))

    # spa_vocab = WordsToSPAVocab(vocab)

    # attempt at adding a basic "seed" word embedding in here
    # not sure how useful it is to have this learned further
    
    seed_vocab_model = Word2Vec(sentences=seed_vocab_data, 
                                min_count=1, 
                                vector_size=mp.rep_vocab_dim, 
                                window = mp.context_length, 
                                epochs = seed_epochs)

    seed_vocab_model.save("canvas-of-thoughts/model/utils/seed_vocab.model")

    return vocab