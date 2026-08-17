"""Committed scientific defaults for model construction and learned state."""

# Model construction is deterministic unless a caller explicitly supplies a
# different seed through a supported build interface.
MODEL_SEED = 42

# PES connections initialized through decoded random functions must use a
# stable seed or Nengo's exact decoder-cache key changes on every process. This
# default makes compatible scientific builds reusable across sessions while
# keeping an explicit CLI override for controlled experiments.
LEARNED_INIT_SEED = MODEL_SEED

# The current predictor derives its independently identifiable initializer by
# adding one to the base seed. Reserve that offset during CLI validation so the
# effective seed always remains a valid NumPy RandomState seed.
LEARNED_INIT_MAX_SEED_OFFSET = 1

# The representation size and similarity threshold were selected with the
# Johnson-Lindenstrauss lemma in mind for approximately 100,000 vectors.
# VOCAB_DIMENSIONS must remain divisible by 16 for the current nengo_spa State
# construction.
VOCAB_DIMENSIONS = 256

# When nengo_spa randomly creates a semantic pointer, its cosine similarity to
# every existing pointer should remain below this threshold. Explicitly supplied
# seed-vocabulary vectors are not regenerated to satisfy it.
VOCAB_MAX_SIMILARITY = 0.6

# This began as the intended number of preceding words in context. It currently
# controls the Word2Vec window and is recorded as architecture metadata; the
# active ContextModule memory horizon is governed separately by feedback alpha.
CONTEXT_LENGTH = 20

# Base learning rate for PES rules. The current predictor and refiner each apply
# their component-specific 0.5 scale when constructing learned connections.
PES_LEARNING_RATE = 0.005
