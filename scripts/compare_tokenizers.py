#!/usr/bin/env python3
"""Compare Canvas tokenizer profiles without building or compiling Nengo."""

import argparse
import json
import sys
import unicodedata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from config import data_defaults  # noqa: E402
from utils.tokenization import available_tokenizers, build_tokenizer  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fit and compare tokenizers on supplied representative text."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Literal text to tokenize and use for fitting.")
    source.add_argument("--file", type=Path, help="UTF-8 text file to tokenize.")
    source.add_argument(
        "--reuters-docs",
        type=int,
        help="Use the first N Reuters training documents from the NLTK corpus.",
    )
    parser.add_argument(
        "--tokenizer",
        action="append",
        choices=available_tokenizers(),
        help="Profile to compare; repeat as needed. Defaults to every profile.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=data_defaults.TOKENIZER_VOCAB_SIZE,
        help="Vocabulary budget used by learned subword tokenizers.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the compact table.",
    )
    return parser.parse_args()


def compare(texts, profile_names, vocab_size):
    """Return comparable segmentation and round-trip facts for each profile."""
    records = []
    for name in profile_names:
        tokenizer = build_tokenizer(
            name,
            normalization=data_defaults.TOKENIZER_NORMALIZATION,
            vocab_size=vocab_size,
            max_subword_length=data_defaults.TOKENIZER_MAX_SUBWORD_LENGTH,
        ).fit(texts)
        tokenized = [tokenizer.encode(text) for text in texts]
        tokens = [token for sequence in tokenized for token in sequence]
        decoded = [tokenizer.decode(sequence) for sequence in tokenized]
        normalized = [
            unicodedata.normalize(tokenizer.normalization, text) for text in texts
        ]
        records.append(
            {
                "name": name,
                "token_count": len(tokens),
                "unique_tokens": len(set(tokens)),
                "round_trip_exact": decoded == normalized,
                "fingerprint": tokenizer.fingerprint(),
                "tokens": tokens,
                "decoded": decoded,
            }
        )
    return records


def main():
    args = _parse_args()
    if args.text is not None:
        texts = [args.text]
    elif args.file is not None:
        texts = [args.file.read_text(encoding="utf-8")]
    else:
        if args.reuters_docs < 1:
            raise SystemExit("--reuters-docs must be at least 1")
        from nltk.corpus import reuters

        training_ids = [
            file_id for file_id in reuters.fileids() if file_id.startswith("training/")
        ][: args.reuters_docs]
        texts = [reuters.raw(file_id) for file_id in training_ids]
    profiles = args.tokenizer or available_tokenizers()
    records = compare(texts, profiles, args.vocab_size)

    if args.json:
        print(json.dumps(records, ensure_ascii=True, indent=2))
        return

    print(f"{'profile':<16} {'tokens':>8} {'unique':>8} {'round-trip':>12} preview")
    for record in records:
        # ASCII escaping keeps the tool usable in Windows consoles whose active
        # code page cannot render private markers or arbitrary Unicode tokens.
        preview = ascii(record["tokens"][:12])
        print(
            f"{record['name']:<16} {record['token_count']:>8} "
            f"{record['unique_tokens']:>8} "
            f"{str(record['round_trip_exact']):>12} {preview}"
        )


if __name__ == "__main__":
    main()
