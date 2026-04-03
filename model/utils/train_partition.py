# for partitioning of training and testing data

from config import model_parameters as mp

class ds_part:
    def __init__(self, training_ids, testing_ids, training_set, testing_set, vocab):
        self.training_ids = training_ids
        self.testing_ids = testing_ids
        self.training_set = training_set
        self.testing_set = testing_set
        self.vocab = vocab

def data_partition(ds, training_restriction=0, testing_restriction=0, strict=False):
    """Partition the provided dataset into training and testing sets."""
    # distinguishing training and testing data
    training_ids = [x for x in ds.fileids() if "training/" in x]
    testing_ids = [x for x in ds.fileids() if "test/" in x]

    # restriction of training set for time's sake
    if training_restriction > 0:
        training_ids = training_ids[:mp.training_restriction]
    if testing_restriction > 0:
        testing_ids = testing_ids[:mp.testing_restriction]

    # train and validation split

    # training set vocabulary
    vocab = [
        t
        for x in training_ids
        for t in ds.words(x)
    ] # all words in training
    
    # print(len(vocab))
    vocab = list(set(vocab)) # unique words

    if strict:
        vocab += [mp.unknown_token] # unknown placeholder (in strict case)
    
    vocab += [mp.pad_token] # padding character
    # print(len(vocab))

    # training set generation
    training_set = [
        ds.words(x)
        for x in training_ids
    ]

    # testing set generation
    testing_set = [
        ds.words(x)
        for x in testing_ids
    ]

    return ds_part(training_ids=training_ids, 
                   testing_ids=testing_ids, 
                   training_set=training_set, 
                   testing_set=testing_set, 
                   vocab=vocab)

def multiple_data_partition(datasets=[], context_length=mp.context_length, training_restriction=0, testing_restriction=0, strict=False):
    """data_partition, but for multiple datasets"""
    
    training_ids = []
    testing_ids = []
    training_set = []
    testing_set = []
    vocab = []

    for ds in datasets:
        pt = data_partition(ds, training_restriction, testing_restriction, strict)
        training_ids += pt.training_ids
        testing_ids += pt.testing_ids
        training_set += pt.training_set
        testing_set += pt.testing_set
        vocab += pt.vocab

    return ds_part(training_ids=training_ids, 
                   testing_ids=testing_ids,
                   training_set=training_set,
                   testing_set=testing_set,
                   vocab=list(set(vocab))) # unique vocab
