"""Train cached corpus-derived vectors from canonical token sequences."""

import hashlib
from pathlib import Path

from gensim.models import Word2Vec

from config import data_defaults, model_defaults


def _stable_token_hash(token):
    """Give Gensim process-independent initial vectors for identical tokens."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def generate_seed_vocab(token_sequences, output_path=None):
    """Train and persist Word2Vec vectors for already-tokenized sequences."""
    sentences = [list(sequence) for sequence in token_sequences if sequence]
    if not sentences:
        raise ValueError("At least one non-empty token sequence is required")

    if output_path is None:
        output_path = Path(__file__).with_name("seed_vocab.model")

    # Word2Vec is only an initialization policy.  Keeping segmentation outside
    # this module ensures it cannot drift from training or interactive input.
    seed_vocab_model = Word2Vec(
        sentences=sentences,
        min_count=1,
        vector_size=model_defaults.VOCAB_DIMENSIONS,
        window=model_defaults.CONTEXT_LENGTH,
        epochs=data_defaults.SEED_VOCAB_EPOCHS,
        seed=model_defaults.MODEL_SEED,
        workers=1,
        hashfxn=_stable_token_hash,
    )
    seed_vocab_model.save(str(output_path))
    return seed_vocab_model
