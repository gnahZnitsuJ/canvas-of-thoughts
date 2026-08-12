"""Partition corpora through one fitted tokenizer and vocabulary contract."""

from dataclasses import dataclass

from config import data_defaults
from utils.tokenization import build_tokenizer


@dataclass(frozen=True)
class DatasetPartition:
    """Tokenized corpus partitions plus the exact tokenizer that produced them."""

    training_ids: list
    testing_ids: list
    training_set: list
    testing_set: list
    vocab: list
    tokenizer: object


# Preserve the historical result name for callers outside the repository while
# moving new code to a descriptive type name.
ds_part = DatasetPartition


def _partition_ids(dataset, training_restriction, testing_restriction):
    training_ids = [file_id for file_id in dataset.fileids() if "training/" in file_id]
    testing_ids = [file_id for file_id in dataset.fileids() if "test/" in file_id]

    if training_restriction > 0:
        training_ids = training_ids[:training_restriction]
    if testing_restriction > 0:
        testing_ids = testing_ids[:testing_restriction]
    return training_ids, testing_ids


def _document_text(dataset, file_id):
    """Prefer source text and retain a controlled fallback for simple adapters."""
    raw = getattr(dataset, "raw", None)
    if callable(raw):
        return raw(file_id)

    # Test adapters and some NLTK readers expose tokens but no raw document.
    # Joining is necessarily lossy, so production corpus adapters should expose
    # raw() whenever punctuation and whitespace boundaries matter.
    return " ".join(dataset.words(file_id))


def _build_partition(
    datasets,
    *,
    training_restriction,
    testing_restriction,
    tokenizer,
):
    training_ids = []
    testing_ids = []
    training_documents = []
    testing_documents = []

    for dataset in datasets:
        dataset_training_ids, dataset_testing_ids = _partition_ids(
            dataset,
            training_restriction,
            testing_restriction,
        )
        training_ids.extend(dataset_training_ids)
        testing_ids.extend(dataset_testing_ids)
        training_documents.extend(
            _document_text(dataset, file_id) for file_id in dataset_training_ids
        )
        testing_documents.extend(
            _document_text(dataset, file_id) for file_id in dataset_testing_ids
        )

    tokenizer.fit(training_documents)
    training_set = [tokenizer.encode(text) for text in training_documents]
    testing_set = [tokenizer.encode(text) for text in testing_documents]

    # Stable ordering prevents set/hash iteration from changing seed-vocabulary
    # construction and cache identity between otherwise identical runs.
    vocab = sorted({token for sequence in training_set for token in sequence})
    # UNKNOWN_TOKEN is always present so runtime lookup has a deterministic
    # fallback. SPA vocabulary strictness is a separate model-layer concern.
    if data_defaults.UNKNOWN_TOKEN not in vocab:
        vocab.append(data_defaults.UNKNOWN_TOKEN)
    if data_defaults.PAD_TOKEN not in vocab:
        vocab.append(data_defaults.PAD_TOKEN)

    return DatasetPartition(
        training_ids=training_ids,
        testing_ids=testing_ids,
        training_set=training_set,
        testing_set=testing_set,
        vocab=vocab,
        tokenizer=tokenizer,
    )


def data_partition(
    dataset,
    training_restriction=0,
    testing_restriction=0,
    tokenizer=None,
):
    """Partition one corpus using the configured or supplied tokenizer."""
    if tokenizer is None:
        tokenizer = build_tokenizer(
            data_defaults.TOKENIZER_NAME,
            normalization=data_defaults.TOKENIZER_NORMALIZATION,
            vocab_size=data_defaults.TOKENIZER_VOCAB_SIZE,
            max_subword_length=data_defaults.TOKENIZER_MAX_SUBWORD_LENGTH,
        )
    return _build_partition(
        [dataset],
        training_restriction=training_restriction,
        testing_restriction=testing_restriction,
        tokenizer=tokenizer,
    )


def multiple_data_partition(
    datasets,
    training_restriction=0,
    testing_restriction=0,
    tokenizer=None,
):
    """Fit once across multiple corpora and return one aligned partition."""
    if tokenizer is None:
        tokenizer = build_tokenizer(
            data_defaults.TOKENIZER_NAME,
            normalization=data_defaults.TOKENIZER_NORMALIZATION,
            vocab_size=data_defaults.TOKENIZER_VOCAB_SIZE,
            max_subword_length=data_defaults.TOKENIZER_MAX_SUBWORD_LENGTH,
        )
    return _build_partition(
        datasets,
        training_restriction=training_restriction,
        testing_restriction=testing_restriction,
        tokenizer=tokenizer,
    )
