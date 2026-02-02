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


VOCAB_PATH = "gradtuity/tests/tokenizer/vocab.json"
MERGES_PATH = "gradtuity/tests/tokenizer/merges.txt"


@pytest.fixture
def real_tokenizer():
    """Load tokenizer once for tests that use real artifacts."""
    return Tokenizer.from_files(VOCAB_PATH, MERGES_PATH)


class TestRealArtifactsOptional:
    """Tests with real artifacts from the GPT-2 tokenizer."""

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "\t\n", " ", "a"],
        ids=["empty", "spaces", "tab_newline", "single_space", "single_letter"],
    )
    def test_round_trip_edge_cases(self, real_tokenizer, text):
        assert real_tokenizer.decode(real_tokenizer.encode(text)) == text

    @pytest.mark.parametrize(
        "text",
        [
            "encode → ids → decode",
            "café, naïve, naïve",
            "Señor Niño",
            "€50 • £30 • ¥100",
            "… — “quoted” — …",
            "München, Zürich",
            "日本語",
            "α → β ⇒ γ",
        ],
        ids=[
            "arrow_phrase",
            "accented_e_naive",
            "spanish_n_tilde",
            "currency_bullet",
            "ellipsis_em_dash_quotes",
            "umlaut_cities",
            "japanese",
            "greek_arrows",
        ],
    )
    def test_no_character_drop_invariant(self, real_tokenizer, text):
        """Decode(encode(text)) == text for strings with non-ASCII characters."""
        assert real_tokenizer.decode(real_tokenizer.encode(text)) == text

    @pytest.mark.parametrize("c", [chr(i) for i in range(32, 127)])
    def test_round_trip_each_printable_ascii_char(self, real_tokenizer, c):
        """Each printable ASCII character round-trips alone."""
        assert real_tokenizer.decode(real_tokenizer.encode(c)) == c
