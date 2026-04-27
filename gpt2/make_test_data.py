#!/usr/bin/env python3
"""
Write a tokens.bin + meta.json for smoke-testing the training pipeline without
running the full HF dataset prep. Tokens are random uint16 in [0, vocab_size);
loss will hover near ln(vocab_size) (~10.8) — useful for verifying the pipeline
runs end-to-end, NOT for measuring convergence. For a convergence smoke test,
use a repeating-pattern dataset instead (see README).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

EOS_TOKEN_ID = 50256
VOCAB_SIZE = 50257
DEFAULT_NUM_TOKENS = 200_000  # ~few steps with seq_len=1024, batch_size=8, world_size=2


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate random uint16 tokens for pipeline testing")
    ap.add_argument("--num_tokens", type=int, default=DEFAULT_NUM_TOKENS)
    ap.add_argument("--out_dir", type=str, default=None, help="Defaults to <repo>/gpt2/data/test-1M")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "test-1M"
    )
    os.makedirs(out_dir, exist_ok=True)
    tokens_path = os.path.join(out_dir, "tokens.bin")
    meta_path = os.path.join(out_dir, "meta.json")

    rng = np.random.default_rng(args.seed)
    print(f"Writing {args.num_tokens:,} uint16 tokens to {tokens_path}...")
    tokens = rng.integers(0, VOCAB_SIZE, size=args.num_tokens, dtype=np.uint16)
    tokens.tofile(tokens_path)

    meta = {
        "num_tokens": int(args.num_tokens),
        "vocab_size": VOCAB_SIZE,
        "dtype": "uint16",
        "eos_token_id": EOS_TOKEN_ID,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
