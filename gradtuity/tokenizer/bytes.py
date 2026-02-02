"""
Byte-level BPE: reversible mapping between bytes (0-255) and Unicode characters.

Provides deterministic byte_encoder and byte_decoder tables so arbitrary UTF-8
byte sequences can be represented as strings for BPE without control/whitespace
issues. Used for encode/decode only.
"""

from __future__ import annotations


def _bytes_to_unicode_tables() -> tuple[dict[int, str], dict[str, int]]:
    """
    Build byte_encoder (byte -> char) and byte_decoder (char -> byte).

    Uses printable ASCII, then Latin-1 supplement ranges, then remaining bytes
    mapped to chr(256+n), so all 256 bytes have a unique reversible character.
    """
    # Ranges that map byte value to same codepoint (printable, no control chars)
    safe_byte_ranges: list[tuple[int, int]] = [
        (ord("!"), ord("~") + 1),   # printable ASCII
        (ord("¡"), ord("¬") + 1),   # Latin-1 supplement
        (ord("®"), ord("ÿ") + 1),   # Latin-1 supplement
    ]
    bs: list[int] = []
    for lo, hi in safe_byte_ranges:
        bs.extend(range(lo, hi))
    cs: list[int] = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    char_list = [chr(c) for c in cs]
    byte_encoder = dict(zip(bs, char_list))
    byte_decoder = {ch: b for b, ch in byte_encoder.items()}
    return byte_encoder, byte_decoder


_BYTE_ENCODER: dict[int, str] | None = None
_BYTE_DECODER: dict[str, int] | None = None


def get_byte_encoder() -> dict[int, str]:
    """Return the byte -> unicode character table (lazy singleton)."""
    global _BYTE_ENCODER, _BYTE_DECODER
    if _BYTE_ENCODER is None:
        _BYTE_ENCODER, _BYTE_DECODER = _bytes_to_unicode_tables()
    return _BYTE_ENCODER


def get_byte_decoder() -> dict[str, int]:
    """Return the unicode character -> byte table (lazy singleton)."""
    global _BYTE_ENCODER, _BYTE_DECODER
    if _BYTE_DECODER is None:
        _BYTE_ENCODER, _BYTE_DECODER = _bytes_to_unicode_tables()
    return _BYTE_DECODER
