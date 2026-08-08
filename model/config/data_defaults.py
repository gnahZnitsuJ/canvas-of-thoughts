"""Committed defaults for corpus selection and vocabulary preparation."""

# Names are resolved through the workflow's dataset registry so configuration
# stays declarative rather than importing corpus loader objects.
DATASET_NAMES = ("reuters",)

# These document limits keep routine Reuters development runs short. Positive
# values bound each partition; zero means use the complete partition.
TRAINING_DOCUMENT_LIMIT = 2
TESTING_DOCUMENT_LIMIT = 2

# Strict vocabularies represent unseen input with UNKNOWN_TOKEN. PAD_TOKEN is
# always present so sequence construction can use an explicit zero vector.
UNKNOWN_TOKEN = "CV_UNK"
PAD_TOKEN = "CV_PAD"
STRICT_VOCAB = False

# Word2Vec epochs used when regenerating the cached seed vocabulary.
SEED_VOCAB_EPOCHS = 50
