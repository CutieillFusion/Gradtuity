"""
BPE merge ranks and apply loop (stateless).

Parses merges from lines (e.g. merges.txt) and provides apply_bpe(word, bpe_ranks).
Cache is owned by Tokenizer; this module is stateless.
"""

from __future__ import annotations


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return set of adjacent symbol pairs in word."""
    if len(word) < 2:
        return set()
    pairs: set[tuple[str, str]] = set()
    prev = word[0]
    for char in word[1:]:
        pairs.add((prev, char))
        prev = char
    return pairs


def parse_merges_lines(lines: list[str]) -> dict[tuple[str, str], int]:
    """
    Build bpe_ranks from merge file lines.

    Skips comment lines (e.g. starting with #). Each non-comment line is
    "a b" -> pair (a, b) with rank = 0-based line index.
    Lower rank = merge earlier.
    """
    bpe_ranks: dict[tuple[str, str], int] = {}
    rank = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]
        bpe_ranks[(a, b)] = rank
        rank += 1
    return bpe_ranks


def apply_bpe(word: str, bpe_ranks: dict[tuple[str, str], int]) -> str:
    """
    Apply BPE merges to a pre-token string (after byte-encoding).

    Returns the merged string with BPE symbols separated by spaces, so the
    caller can split and map to ids. Uses same algorithm as standard byte-level BPE.
    """
    if not word:
        return ""
    if not bpe_ranks:
        return word
    word_tuple: tuple[str, ...] = tuple(word)
    while True:
        pairs = get_pairs(word_tuple)
        if not pairs:
            break
        bigram = min(pairs, key=lambda p: bpe_ranks.get(p, float("inf")))
        if bigram not in bpe_ranks:
            break
        first, second = bigram
        new_word: list[str] = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == (
                first,
                second,
            ):
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        word_tuple = tuple(new_word)
        if len(word_tuple) == 1:
            break
    return " ".join(word_tuple)
