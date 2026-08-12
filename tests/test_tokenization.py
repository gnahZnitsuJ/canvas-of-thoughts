import sys
import unittest
import unicodedata
from pathlib import Path

import nengo_spa as spa
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

from config import data_defaults  # noqa: E402
from components.runtime import ModelRuntime  # noqa: E402
from utils.processing import spa_key_to_token, token_to_spa_key  # noqa: E402
from utils.tokenization import available_tokenizers, build_tokenizer  # noqa: E402
from utils.train_partition import data_partition  # noqa: E402


SAMPLE = "Café can't cost 12.50€ 👩🏽‍💻."


class RawDataset:
    def fileids(self):
        return ["training/one", "test/one"]

    def raw(self, file_id):
        if file_id.startswith("training/"):
            return "Hello, world! CV_PAD"
        return "Hello, uncertain-token!"


class TokenizationTests(unittest.TestCase):
    def _fitted(self, name, texts=None, vocab_size=128):
        texts = texts or [SAMPLE]
        return build_tokenizer(name, vocab_size=vocab_size).fit(texts)

    def test_registry_exposes_every_versioned_strategy(self):
        self.assertEqual(
            available_tokenizers(),
            ("bpe-v1", "byte-v1", "character-v1", "unigram-v1", "word-v1"),
        )

    def test_word_tokenizer_handles_unicode_punctuation_and_reserved_literals(self):
        tokenizer = self._fitted("word-v1")
        tokens = tokenizer.encode("Café, 👩🏽‍💻 CV_PAD")

        self.assertEqual(tokens[0:2], ["Café", ","])
        self.assertIn("👩🏽‍💻", tokens)
        self.assertNotIn(data_defaults.PAD_TOKEN, tokens)
        self.assertIn("CV_PAD", [tokenizer.display_token(token) for token in tokens])

    def test_character_tokenizer_uses_graphemes_and_round_trips(self):
        text = "e\u0301 👩🏽‍💻\n"
        tokenizer = self._fitted("character-v1", [text])
        tokens = tokenizer.encode(text)

        self.assertIn("é", tokens)
        self.assertIn("👩🏽‍💻", tokens)
        self.assertEqual(tokenizer.decode(tokens), unicodedata.normalize("NFC", text))

    def test_byte_tokenizer_round_trips_arbitrary_unicode(self):
        tokenizer = self._fitted("byte-v1")
        tokens = tokenizer.encode(SAMPLE)

        self.assertTrue(all(token.startswith("BYTE_") for token in tokens))
        self.assertEqual(tokenizer.decode(tokens), unicodedata.normalize("NFC", SAMPLE))
        self.assertEqual(
            tokenizer.decode(["BYTE_41", data_defaults.UNKNOWN_TOKEN, "BYTE_42"]),
            "ACV_UNKB",
        )
        for malformed in ("BYTE_0", "BYTE_000", "BYTE_GG", "not-a-byte"):
            with self.assertRaises(ValueError):
                tokenizer.decode([malformed])

    def test_fixed_tokenizer_fingerprint_ignores_irrelevant_subword_limits(self):
        first = build_tokenizer(
            "character-v1",
            vocab_size=32,
            max_subword_length=4,
        ).fit([SAMPLE])
        second = build_tokenizer(
            "character-v1",
            vocab_size=4096,
            max_subword_length=64,
        ).fit([SAMPLE])

        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_learned_subword_tokenizers_are_deterministic_and_decodable(self):
        training = ["lower lowest newer wider", "low lower newest"]
        for name in ("bpe-v1", "unigram-v1"):
            first = self._fitted(name, training, vocab_size=48)
            second = self._fitted(name, training, vocab_size=48)

            tokens = first.encode("lower newest")
            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertEqual(tokens, second.encode("lower newest"))
            self.assertEqual(first.decode(tokens), "lower newest")

    def test_spa_key_encoding_is_reversible_and_collision_free(self):
        tokens = [".", "CV_PERIOD", "Café", "👩🏽‍💻", " "]
        keys = [token_to_spa_key(token) for token in tokens]

        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(key.isidentifier() for key in keys))
        self.assertEqual([spa_key_to_token(key) for key in keys], tokens)

    def test_partition_uses_one_tokenizer_for_sequences_and_vocabulary(self):
        tokenizer = self._fitted("word-v1", ["placeholder"])
        partition = data_partition(
            RawDataset(),
            tokenizer=tokenizer,
        )

        training_tokens = partition.training_set[0]
        self.assertTrue(set(training_tokens).issubset(set(partition.vocab)))
        self.assertEqual(
            partition.testing_set[0],
            tokenizer.encode("Hello, uncertain-token!"),
        )
        self.assertIn(data_defaults.UNKNOWN_TOKEN, partition.vocab)
        self.assertIn(data_defaults.PAD_TOKEN, partition.vocab)

    def test_runtime_routes_unseen_tokens_without_growing_vocabulary(self):
        vocab = spa.Vocabulary(16, strict=False)
        vocab.add(data_defaults.UNKNOWN_TOKEN, np.ones(16) / 4.0)
        known_key = token_to_spa_key("known")
        vocab.add(known_key, np.arange(16, dtype=float))
        runtime = ModelRuntime.__new__(ModelRuntime)
        runtime.model_vocab = vocab
        runtime.vocab_key_set = set(vocab.keys())
        keys_before = list(vocab.keys())

        result = runtime._vector_for("never-seen")

        np.testing.assert_array_equal(result, vocab[data_defaults.UNKNOWN_TOKEN].v)
        self.assertEqual(list(vocab.keys()), keys_before)

    def test_every_strategy_is_total_for_empty_and_mixed_text(self):
        for name in available_tokenizers():
            tokenizer = self._fitted(name, ["plain ASCII training", ""])
            self.assertEqual(tokenizer.encode(""), [])
            self.assertIsInstance(tokenizer.encode(SAMPLE), list)


if __name__ == "__main__":
    unittest.main()
