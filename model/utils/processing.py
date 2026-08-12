"""Encode arbitrary model tokens as reversible NengoSPA vocabulary keys."""

import base64
import binascii

from config import data_defaults


_TOKEN_KEY_PREFIX = "WV_B32_"


def token_to_spa_key(token):
    """Return a collision-free Python identifier for one already-delimited token."""
    if token in {data_defaults.PAD_TOKEN, data_defaults.UNKNOWN_TOKEN}:
        return token
    if not isinstance(token, str) or token == "":
        raise ValueError("SPA vocabulary tokens must be non-empty strings")

    encoded = base64.b32encode(token.encode("utf-8")).decode("ascii").rstrip("=")
    return _TOKEN_KEY_PREFIX + encoded


def spa_key_to_token(key):
    """Decode a key produced by :func:`token_to_spa_key`."""
    if key in {data_defaults.PAD_TOKEN, data_defaults.UNKNOWN_TOKEN}:
        return key
    if not key.startswith(_TOKEN_KEY_PREFIX):
        raise ValueError(f"Not a token vocabulary key: {key!r}")

    payload = key[len(_TOKEN_KEY_PREFIX) :]
    padding = "=" * (-len(payload) % 8)
    try:
        return base64.b32decode(payload + padding).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid token vocabulary key: {key!r}") from exc


def WordsToSPAVocab(words):
    """Compatibility wrapper that encodes a sequence of model tokens."""
    return [token_to_spa_key(token) for token in words]


def SPAVocabToWords(keys):
    """Compatibility wrapper that decodes a sequence of vocabulary keys."""
    return [spa_key_to_token(key) for key in keys]
