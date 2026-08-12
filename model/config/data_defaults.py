"""Committed defaults for corpus selection and vocabulary preparation."""

# Names are resolved through the workflow's dataset registry so configuration
# stays declarative rather than importing corpus loader objects.
DATASET_NAMES = ("reuters",)

# Tokenizer profiles are versioned because their output determines the neural
# timestep, vocabulary, seed-vector cache, checkpoint compatibility, and model
# quality. Learned subword profiles use the size and maximum-piece limits below.
TOKENIZER_NAME = "word-v1"
TOKENIZER_NORMALIZATION = "NFC"
TOKENIZER_VOCAB_SIZE = 512
TOKENIZER_MAX_SUBWORD_LENGTH = 12

# These document limits keep routine Reuters development runs short. Positive
# values bound each partition; zero means use the complete partition.
TRAINING_DOCUMENT_LIMIT = 2
TESTING_DOCUMENT_LIMIT = 2

# UNKNOWN_TOKEN is always present as the runtime's deterministic unseen-input
# fallback. STRICT_VOCAB separately controls NengoSPA's implicit-addition mode;
# runtime lookups never depend on implicit growth. PAD_TOKEN is explicit zero.
UNKNOWN_TOKEN = "CV_UNK"
PAD_TOKEN = "CV_PAD"
STRICT_VOCAB = False

# The profile is part of cache identity: bump it whenever initialization
# semantics change. A stable token hash and one worker make regeneration
# process-independent; epochs remain an experiment-facing training control.
SEED_VOCAB_PROFILE = "word2vec-deterministic-v1"
SEED_VOCAB_EPOCHS = 50
