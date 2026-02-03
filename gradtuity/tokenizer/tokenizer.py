"""
Generic Tokenizer: vocab.json + merges.txt (Option-B).

Loads from disk; provides encode/decode and vocab_size. Byte-level BPE
semantics compatible with common artifact formats.
"""

from __future__ import annotations

import json
import re

from .bpe import apply_bpe, parse_merges_lines
from .bytes import get_byte_decoder, get_byte_encoder

# Pre-tokenization: split into pieces (contractions, letters, numbers, other, whitespace).
# Stdlib re only; ASCII-oriented approximation (letters/underscore via [a-zA-Z_], digits via [0-9]).
_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z_]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.ASCII,
)


def _load_vocab(vocab_path: str) -> dict[str, int]:
    with open(vocab_path, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {vocab_path!r}: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(
            f"vocab must be a JSON object (token -> id), got {type(raw).__name__}"
        )
    return {str(k): int(v) for k, v in raw.items()}


def _validate_contiguous(token_to_id: dict[str, int]) -> int:
    """Validate IDs are unique and contiguous 0..N-1. Return vocab_size. Raise ValueError if not."""
    ids = set(token_to_id.values())
    n = len(ids)
    if n != len(token_to_id):
        raise ValueError("vocab must have unique ids")
    missing = [i for i in range(n) if i not in ids]
    if missing:
        sample = missing[:5]
        raise ValueError(
            f"vocab ids must be contiguous 0..vocab_size-1; missing (sample): {sample}"
        )
    return n


def _load_merges(merges_path: str) -> dict[tuple[str, str], int]:
    with open(merges_path, encoding="utf-8") as f:
        lines = f.readlines()
    return parse_merges_lines(lines)


class Tokenizer:
    """
    Generic tokenizer built from vocab.json and merges.txt.

    Use Tokenizer.from_files(vocab_path, merges_path) to construct.
    """

    def __init__(
        self,
        token_to_id: dict[str, int],
        id_to_token: list[str],
        bpe_ranks: dict[tuple[str, str], int],
        decode_errors: str = "replace",
        **kwargs: object,
    ) -> None:
        self._token_to_id = token_to_id
        self._id_to_token = id_to_token
        self._bpe_ranks = bpe_ranks
        self._decode_errors = decode_errors
        self._byte_encoder = get_byte_encoder()
        self._byte_decoder = get_byte_decoder()
        self._bpe_cache: dict[str, str] = {}

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        *,
        decode_errors: str = "replace",
        **kwargs: object,
    ) -> Tokenizer:
        """
        Build a Tokenizer from vocab.json and merges.txt.

        Raises FileNotFoundError if either path is missing; ValueError on invalid JSON
        or non-contiguous vocab ids.
        """
        try:
            token_to_id = _load_vocab(vocab_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"vocab file not found: {vocab_path!r}") from None
        except OSError as e:
            raise FileNotFoundError(
                f"cannot read vocab file {vocab_path!r}: {e}"
            ) from e

        vocab_size = _validate_contiguous(token_to_id)
        id_to_token = [""] * vocab_size
        for token, i in token_to_id.items():
            id_to_token[i] = token

        try:
            bpe_ranks = _load_merges(merges_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"merges file not found: {merges_path!r}") from None
        except OSError as e:
            raise FileNotFoundError(
                f"cannot read merges file {merges_path!r}: {e}"
            ) from e

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            bpe_ranks=bpe_ranks,
            decode_errors=decode_errors,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def encode(self, text: str) -> list[int]:
        """
        Encode text to a list of token ids.

        Raises ValueError if a BPE symbol is not in the vocab (with context).
        """
        if not self._bpe_ranks:
            raise ValueError(
                "merges required for encode (merges file was empty or had no valid lines)"
            )
        ids: list[int] = []
        for piece in _PAT.findall(text):
            token_bytes = piece.encode("utf-8")
            word = "".join(self._byte_encoder[b] for b in token_bytes)
            cached = self._bpe_cache.get(word)
            if cached is not None:
                merged = cached
            else:
                merged = apply_bpe(word, self._bpe_ranks)
                self._bpe_cache[word] = merged
            for symbol in merged.split():
                if symbol not in self._token_to_id:
                    raise ValueError(
                        f"symbol {symbol!r} not in vocab (context: {text[:80]!r}...)"
                    )
                ids.append(self._token_to_id[symbol])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids to a string."""
        tokens = [self._id_to_token[i] for i in ids]
        text = "".join(tokens)
        byte_array = bytearray(self._byte_decoder[c] for c in text)
        return byte_array.decode("utf-8", errors=self._decode_errors)
