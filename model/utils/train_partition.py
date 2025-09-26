# for partitioning of training and testing data

from .config import model_parameters as mp

class ds_part:
    def __init__(self, training_ids, testing_ids, training_set, testing_set, vocab):
        self.training_ids = training_ids
        self.testing_ids = testing_ids
        self.training_set = training_set
        self.testing_set = testing_set
        self.vocab = vocab

def data_partition(ds, training_restriction=0, testing_restriction=0, strict=False):
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
        [ds.words(x)[max(0,i-mp.context_length):max(0,i)], [ds.words(x)[i]]]
        for x in training_ids
        for i in range(len(ds.words(x)))
    ]

    # padding for training set
    training_set = [
        [[mp.pad_token] * (mp.context_length - len(x[0])) + x[0], x[1]]
        for x in training_set
    ]

    # testing set generation
    testing_set = [
        [ds.words(x)[max(0,i-mp.context_length):max(0,i)], [ds.words(x)[i]]]
        for x in testing_ids
        for i in range(len(ds.words(x)))
    ]

    # padding for testing set
    testing_set = [
        [[mp.pad_token] * (mp.context_length - len(x[0])) + x[0], x[1]]
        for x in testing_set
    ]

    return ds_part(training_ids=training_ids, 
                   testing_ids=testing_ids, 
                   training_set=training_set, 
                   testing_set=testing_set, 
                   vocab=vocab)