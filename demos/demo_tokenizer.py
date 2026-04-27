#!/usr/bin/env python3
"""
Gradtuity Tokenizer Demo

Demonstrates the generic tokenizer: load from vocab.json + merges.txt,
encode text to ids, decode ids to text, and show vocab_size.

Usage:
    uv run python demos/demo_tokenizer.py
"""

import os
import sys

from gradtuity import Tokenizer

print("=" * 60)
print("Gradtuity Tokenizer Demo")
print("=" * 60)
print()

vocab_path = "demos/tokenizer/vocab.json"
merges_path = "demos/tokenizer/merges.txt"

if not os.path.isfile(vocab_path):
    print(f"ERROR: vocab file not found at {vocab_path}.")
    sys.exit(1)
if not os.path.isfile(merges_path):
    print(f"ERROR: merges file not found at {merges_path}.")
    sys.exit(1)

print("Loading tokenizer...")
tok = Tokenizer.from_files(vocab_path, merges_path)
print(f"  vocab_size: {tok.vocab_size}")
print()

examples = [
    "".join(chr(i) for i in range(32, 127)),  # All printable ASCII characters
    "encode → ids → decode",  # Special characters (→)
    "€50 • £30 • ¥100",
    "日本語",
    "α → β ⇒ γ",
]

print("Encode / Decode examples:")
print("-" * 60)
for text in examples:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"  input:  {text!r}")
    print(f"  ids:    {ids[:20]}{'...' if len(ids) > 20 else ''} (len={len(ids)})")
    print(f"  decode: {decoded!r}")
    print()

print("LM training slice (input_ids = ids[:-1], labels = ids[1:]):")
print("-" * 60)
text = "Hello World, This project is awesome!"
ids = tok.encode(text)
input_ids = ids[:-1]
labels = ids[1:]
print(f"  text:       {text!r}")
print(f"  ids:        {ids}")
print(f"  input_ids:  {input_ids}")
print(f"  labels:     {labels}")
print()

print("=" * 60)
print("Demo complete!")
print("=" * 60)
