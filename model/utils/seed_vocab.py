"""Generate the cached Word2Vec seed vocabulary used by the model."""

import re
from pathlib import Path

from gensim.models import Word2Vec
from nltk.corpus import reuters

from config import data_defaults, model_defaults
from utils.processing import WordsToSPAVocab
import utils.train_partition as tp


def generate_seed_vocab(dataset_list=None, output_path=None):
    """Generate, save, and return the vocabulary for the supplied datasets."""

    if not dataset_list:
        print("No datasets provided for seed vocabulary generation.")
        return None

    if output_path is None:
        output_path = Path(__file__).with_name("seed_vocab.model")
    
    # initialize vocab and seed_vocab_data
    vocab = []
    seed_vocab_data = []

    for ds in dataset_list:
        pt = tp.data_partition(
            ds,
            training_restriction=data_defaults.TRAINING_DOCUMENT_LIMIT,
            testing_restriction=data_defaults.TESTING_DOCUMENT_LIMIT,
            strict=data_defaults.STRICT_VOCAB,
        )
        
        # appending the vocab and the sentence data of partitions in the dataset list
        vocab += [
            i
            for x in pt.training_ids
            for t in ds.words(x)
            for i in re.split(r'([^a-zA-Z0-9])', t) if i.strip()
        ]

        # seed_vocab_data += [WordsToSPAVocab(i) for x in pt.training_ids for i in reuters.sents(x)]
        seed_vocab_data += [WordsToSPAVocab([i for t in sent for i in re.split(r'([^a-zA-Z0-9])', t) if i.strip()]) for x in pt.training_ids for sent in reuters.sents(x)]

    vocab = list(set(vocab))

    # spa_vocab = WordsToSPAVocab(vocab)

    # attempt at adding a basic "seed" word embedding in here
    # not sure how useful it is to have this learned further
    
    seed_vocab_model = Word2Vec(
        sentences=seed_vocab_data,
        min_count=1,
        vector_size=model_defaults.VOCAB_DIMENSIONS,
        window=model_defaults.CONTEXT_LENGTH,
        epochs=data_defaults.SEED_VOCAB_EPOCHS,
    )

    seed_vocab_model.save(str(output_path))

    return vocab
