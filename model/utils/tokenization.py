"""Versioned, interchangeable text tokenizers for corpus and runtime input.

Tokenization changes the unit advanced through the Nengo model, so every
implementation exposes deterministic metadata and a fingerprint.  Callers use
the registry and the common ``TextTokenizer`` contract instead of reproducing
segmentation rules in data preparation or interactive runtime code.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter

import regex


_BOUNDARY = "\ue000"
_ESCAPE = "\ue001"
_LITERAL_TOKEN = "\ue002"
_RESERVED_TOKENS = frozenset({"CV_PAD", "CV_UNK"})
_TOKENIZER_TYPES = {}


def _protect_token(token):
    """Keep corpus text distinct from model-reserved vocabulary entries."""
    if token in _RESERVED_TOKENS or token.startswith(_LITERAL_TOKEN):
        return _LITERAL_TOKEN + token
    return token


def _unprotect_token(token):
    if token.startswith(_LITERAL_TOKEN):
        return token[len(_LITERAL_TOKEN) :]
    return token


def _escape_subword_text(text):
    """Reserve private-use markers without rejecting arbitrary Unicode input."""
    return text.replace(_ESCAPE, _ESCAPE + "E").replace(
        _BOUNDARY,
        _ESCAPE + "B",
    )


def _unescape_subword_text(text):
    output = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == _BOUNDARY:
            output.append(" ")
            index += 1
            continue
        if char == _ESCAPE and index + 1 < len(text):
            escaped = text[index + 1]
            if escaped == "E":
                output.append(_ESCAPE)
                index += 2
                continue
            if escaped == "B":
                output.append(_BOUNDARY)
                index += 2
                continue
        output.append(char)
        index += 1
    return "".join(output).lstrip(" ")


def _subword_sequences(text):
    """Represent normalized whitespace as a marker on each surface segment."""
    collapsed = regex.sub(r"\s+", " ", text.strip())
    if not collapsed:
        return []

    sequences = []
    for segment in collapsed.split(" "):
        graphemes = regex.findall(r"\X", _escape_subword_text(segment))
        if graphemes:
            graphemes[0] = _BOUNDARY + graphemes[0]
            sequences.append(tuple(graphemes))
    return sequences


def register_tokenizer(cls):
    """Register one tokenizer implementation under its stable profile name."""
    name = cls.name
    if not name or name in _TOKENIZER_TYPES:
        raise ValueError(f"Duplicate or empty tokenizer profile: {name!r}")
    _TOKENIZER_TYPES[name] = cls
    return cls


def available_tokenizers():
    """Return the authoritative tokenizer profile catalog."""
    return tuple(sorted(_TOKENIZER_TYPES))


def build_tokenizer(
    name,
    *,
    normalization="NFC",
    vocab_size=512,
    max_subword_length=12,
):
    """Construct a tokenizer from the registry and validate its shared policy."""
    try:
        tokenizer_type = _TOKENIZER_TYPES[name]
    except KeyError as exc:
        choices = ", ".join(available_tokenizers())
        raise ValueError(f"Unknown tokenizer {name!r}; expected one of: {choices}") from exc

    return tokenizer_type(
        normalization=normalization,
        vocab_size=vocab_size,
        max_subword_length=max_subword_length,
    )


class TextTokenizer(ABC):
    """Common contract for fixed and corpus-trained tokenization strategies."""

    name = None

    def __init__(self, *, normalization="NFC", vocab_size=512, max_subword_length=12):
        if normalization not in {"NFC", "NFKC"}:
            raise ValueError("normalization must be NFC or NFKC")
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if max_subword_length < 1:
            raise ValueError("max_subword_length must be positive")

        self.normalization = normalization
        self.vocab_size = int(vocab_size)
        self.max_subword_length = int(max_subword_length)
        self._fitted = False

    def normalize(self, text):
        """Apply the declared Unicode normalization without hiding bad types."""
        if not isinstance(text, str):
            raise TypeError("tokenizer input must be str")
        return unicodedata.normalize(self.normalization, text)

    def fit(self, texts):
        """Fit corpus-dependent state; fixed tokenizers only record readiness."""
        for text in texts:
            self.normalize(text)
        self._fitted = True
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError(f"Tokenizer {self.name!r} must be fitted before use")

    @abstractmethod
    def encode(self, text):
        """Convert text to the tokens advanced through the model."""

    @abstractmethod
    def decode(self, tokens):
        """Convert model tokens into human-readable normalized text."""

    def display_token(self, token):
        """Remove only the internal escape used for reserved literal text."""
        return _unprotect_token(token)

    def state(self):
        """Return deterministic tokenizer state used by caches and checkpoints."""
        return {
            "name": self.name,
            "normalization": self.normalization,
        }

    def fingerprint(self):
        """Hash every setting or learned value that can change token identity."""
        payload = json.dumps(
            self.state(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def metadata(self):
        return {**self.state(), "fingerprint": self.fingerprint()}


@register_tokenizer
class WordTokenizer(TextTokenizer):
    """Segment Unicode words, numbers, and remaining grapheme-like symbols."""

    name = "word-v1"
    _pattern = regex.compile(
        r"\p{L}[\p{L}\p{M}\p{N}_]*(?:['’]\p{L}[\p{L}\p{M}\p{N}_]*)*"
        r"|\p{N}+(?:[.,]\p{N}+)*"
        r"|(?!\s)\X"
    )

    def encode(self, text):
        self._require_fitted()
        return [
            _protect_token(match.group(0))
            for match in self._pattern.finditer(self.normalize(text))
        ]

    def decode(self, tokens):
        # Word tokenization intentionally omits whitespace.  This readable
        # rendering is not represented as a lossless source-text round trip.
        return " ".join(_unprotect_token(token) for token in tokens)


@register_tokenizer
class CharacterTokenizer(TextTokenizer):
    """Emit Unicode extended grapheme clusters, including whitespace clusters."""

    name = "character-v1"

    def encode(self, text):
        self._require_fitted()
        return [
            _protect_token(grapheme)
            for grapheme in regex.findall(r"\X", self.normalize(text))
        ]

    def decode(self, tokens):
        return "".join(_unprotect_token(token) for token in tokens)


@register_tokenizer
class ByteTokenizer(TextTokenizer):
    """Emit stable UTF-8 byte labels for tokenizer-free coverage experiments."""

    name = "byte-v1"
    _prefix = "BYTE_"

    def encode(self, text):
        self._require_fitted()
        return [f"{self._prefix}{value:02X}" for value in self.normalize(text).encode("utf-8")]

    def decode(self, tokens):
        output = []
        values = bytearray()

        def flush_bytes():
            if values:
                output.append(bytes(values).decode("utf-8", errors="replace"))
                values.clear()

        for token in tokens:
            token = _unprotect_token(token)
            if token in _RESERVED_TOKENS:
                flush_bytes()
                output.append(token)
                continue
            if regex.fullmatch(r"BYTE_[0-9A-F]{2}", token) is None:
                raise ValueError(f"Invalid byte token: {token!r}")
            values.append(int(token[len(self._prefix) :], 16))
        flush_bytes()
        return "".join(output)


class _SubwordTokenizer(TextTokenizer):
    """Shared boundary-marker and rendering behavior for learned subwords."""

    def decode(self, tokens):
        joined = "".join(_unprotect_token(token) for token in tokens)
        return _unescape_subword_text(joined)

    def _normalized_sequences(self, text):
        return _subword_sequences(self.normalize(text))


@register_tokenizer
class BpeTokenizer(_SubwordTokenizer):
    """Learn deterministic byte-pair merges over Unicode grapheme sequences."""

    name = "bpe-v1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.merges = []

    @staticmethod
    def _merge(sequence, pair):
        merged = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and sequence[index : index + 2] == pair:
                merged.append(pair[0] + pair[1])
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return tuple(merged)

    def fit(self, texts):
        sequence_counts = Counter()
        for text in texts:
            for sequence in self._normalized_sequences(text):
                sequence_counts[sequence] += 1

        symbol_count = len({symbol for seq in sequence_counts for symbol in seq})
        if symbol_count > self.vocab_size:
            raise ValueError(
                "bpe vocab_size is smaller than the required symbol inventory"
            )
        max_merges = max(self.vocab_size - symbol_count, 0)
        self.merges = []

        for _ in range(max_merges):
            pair_counts = Counter()
            for sequence, frequency in sequence_counts.items():
                for pair in zip(sequence, sequence[1:]):
                    pair_counts[pair] += frequency
            if not pair_counts:
                break

            # Lexical tie-breaking makes learned artifacts stable across runs.
            best_pair, best_count = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if best_count < 2:
                break

            self.merges.append(best_pair)
            updated = Counter()
            for sequence, frequency in sequence_counts.items():
                updated[self._merge(sequence, best_pair)] += frequency
            sequence_counts = updated

        self._fitted = True
        return self

    def encode(self, text):
        self._require_fitted()
        output = []
        for sequence in self._normalized_sequences(text):
            for pair in self.merges:
                sequence = self._merge(sequence, pair)
            output.extend(_protect_token(piece) for piece in sequence)
        return output

    def state(self):
        return {
            **super().state(),
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
        }


@register_tokenizer
class UnigramTokenizer(_SubwordTokenizer):
    """Learn a deterministic substring vocabulary and Viterbi segmentation.

    This is a compact unigram-language-model implementation for controlled
    Canvas experiments.  It uses observed substring frequencies rather than
    SentencePiece's iterative EM pruning, which is recorded in the versioned
    profile name and must not be treated as a SentencePiece result.
    """

    name = "unigram-v1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.piece_scores = {}
        self.unknown_score = -100.0

    def fit(self, texts):
        counts = Counter()
        required = set()
        for text in texts:
            for sequence in self._normalized_sequences(text):
                required.update(sequence)
                length = len(sequence)
                for start in range(length):
                    stop_limit = min(length, start + self.max_subword_length)
                    for stop in range(start + 1, stop_limit + 1):
                        counts["".join(sequence[start:stop])] += 1

        if len(required) > self.vocab_size:
            raise ValueError(
                "unigram vocab_size is smaller than the required symbol inventory"
            )

        optional = [piece for piece in counts if piece not in required]
        optional.sort(key=lambda piece: (-counts[piece], -len(piece), piece))
        pieces = list(sorted(required)) + optional[: self.vocab_size - len(required)]
        total = sum(counts[piece] for piece in pieces) or 1
        self.piece_scores = {
            piece: math.log(counts[piece] / total)
            for piece in pieces
        }
        self.unknown_score = math.log(1.0 / (total * 1000.0))
        self._fitted = True
        return self

    def _segment(self, sequence):
        length = len(sequence)
        best = [None] * (length + 1)
        best[0] = (0.0, ())
        for stop in range(1, length + 1):
            start_min = max(0, stop - self.max_subword_length)
            for start in range(start_min, stop):
                previous = best[start]
                if previous is None:
                    continue
                piece = "".join(sequence[start:stop])
                score = self.piece_scores.get(piece)
                # A single unseen grapheme remains an explicit surface token.
                # The fixed model vocabulary will route it to UNKNOWN_TOKEN;
                # tokenization itself must remain total for arbitrary input.
                if score is None and stop - start == 1:
                    score = self.unknown_score
                if score is None:
                    continue
                candidate = (previous[0] + score, previous[1] + (piece,))
                current = best[stop]
                if current is None or candidate[0] > current[0] or (
                    candidate[0] == current[0] and candidate[1] < current[1]
                ):
                    best[stop] = candidate

        if best[length] is None:
            raise ValueError("Unigram vocabulary cannot represent normalized input")
        return best[length][1]

    def encode(self, text):
        self._require_fitted()
        output = []
        for sequence in self._normalized_sequences(text):
            output.extend(_protect_token(piece) for piece in self._segment(sequence))
        return output

    def state(self):
        return {
            **super().state(),
            "vocab_size": self.vocab_size,
            "max_subword_length": self.max_subword_length,
            "piece_scores": [
                [piece, self.piece_scores[piece]]
                for piece in sorted(self.piece_scores)
            ],
            "unknown_score": self.unknown_score,
        }
