"""
Tests for tokenizer I/O: vocab contiguity, merges parsing, file-not-found.
"""

import pytest

from gradtuity.tokenizer import Tokenizer


class TestVocabOrderingContiguity:
    """Vocab ordering and contiguity validation."""

    def test_vocab_size_and_decode_no_crash(self, tmp_path):
        vocab = tmp_path / "vocab.json"
        vocab.write_text('{"a": 0, "b": 1, "<|endoftext|>": 2}', encoding="utf-8")
        merges = tmp_path / "merges.txt"
        merges.write_text("# empty\n", encoding="utf-8")
        tok = Tokenizer.from_files(str(vocab), str(merges))
        assert tok.vocab_size == 3
        tok.decode([0, 1, 2])

    def test_non_contiguous_ids_raises(self, tmp_path):
        vocab = tmp_path / "vocab.json"
        vocab.write_text('{"a": 0, "b": 2}', encoding="utf-8")
        merges = tmp_path / "merges.txt"
        merges.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="contiguous|missing"):
            Tokenizer.from_files(str(vocab), str(merges))


class TestMergesParsing:
    """Merges file parsing."""

    def test_merges_comment_and_two_lines(self, tmp_path):
        vocab = tmp_path / "vocab.json"
        vocab.write_text(
            '{"a": 0, "b": 1, "ab": 2, "c": 3, "d": 4, "cd": 5}',
            encoding="utf-8",
        )
        merges = tmp_path / "merges.txt"
        merges.write_text("# comment\na b\nc d\n", encoding="utf-8")
        tok = Tokenizer.from_files(str(vocab), str(merges))
        assert tok.vocab_size == 6
        ids = tok.encode("ab")
        assert ids == [2]
        ids_cd = tok.encode("cd")
        assert ids_cd == [5]


class TestFileNotFound:
    """Missing files raise FileNotFoundError."""

    def test_missing_vocab_raises(self, tmp_path):
        merges = tmp_path / "merges.txt"
        merges.write_text("", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="vocab"):
            Tokenizer.from_files(str(tmp_path / "nonexistent_vocab.json"), str(merges))

    def test_missing_merges_raises(self, tmp_path):
        vocab = tmp_path / "vocab.json"
        vocab.write_text('{"a": 0}', encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="merges"):
            Tokenizer.from_files(str(vocab), str(tmp_path / "nonexistent_merges.txt"))
