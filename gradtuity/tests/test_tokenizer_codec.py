"""
Tests for tokenizer encode/decode: toy artifacts, round-trips, fixed encode outputs.
"""

import os

import pytest

from gradtuity.tokenizer import Tokenizer


def _toy_vocab_hello() -> dict[str, int]:
    """Vocab for encoding 'hello' with BPE: h,e,l,o + he,hel,hell,hello."""
    return {
        "h": 0,
        "e": 1,
        "l": 2,
        "o": 3,
        "he": 4,
        "hel": 5,
        "hell": 6,
        "hello": 7,
    }


def _toy_merges_hello() -> str:
    """Merges that produce 'hello' as single token from h,e,l,l,o."""
    return "h e\nhe l\nhel l\nhell o\n"


@pytest.fixture
def toy_hello_dir(tmp_path):
    """Write toy vocab and merges for 'hello' encode/decode."""
    import json

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps(_toy_vocab_hello(), ensure_ascii=False),
        encoding="utf-8",
    )
    merges_path = tmp_path / "merges.txt"
    merges_path.write_text(_toy_merges_hello(), encoding="utf-8")
    return tmp_path


class TestToyCodec:
    """Encode/decode with toy vocab and merges."""

    def test_decode_encode_hello_round_trip(self, toy_hello_dir):
        tok = Tokenizer.from_files(
            str(toy_hello_dir / "vocab.json"),
            str(toy_hello_dir / "merges.txt"),
        )
        text = "hello"
        assert tok.decode(tok.encode(text)) == text

    def test_encode_hello_fixed_output(self, toy_hello_dir):
        tok = Tokenizer.from_files(
            str(toy_hello_dir / "vocab.json"),
            str(toy_hello_dir / "merges.txt"),
        )
        ids = tok.encode("hello")
        assert ids == [7]


class TestRealArtifactsOptional:
    """Tests with real artifacts from the GPT-2 tokenizer."""
    def test_vocab_size_and_round_trip_hello(self):
        vocab_path = "gradtuity/tests/tokenizer/vocab.json"
        merges_path = "gradtuity/tests/tokenizer/merges.txt"
        tok = Tokenizer.from_files(vocab_path, merges_path)
        assert tok.vocab_size == 50257
        assert tok.decode(tok.encode("Hello")) == "Hello"
