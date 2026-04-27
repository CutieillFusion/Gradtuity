"""
Inline data pipeline for GPT-2 training: read tokens.bin + meta.json, pack segments,
step-aligned DDP sharding, yield (input_ids, labels) as Python lists for Tensor(...).

No HuggingFace at train time. Indices are converted to float32 exact integers
before being passed to Gradtuity Tensor (embedding expects integer-valued float32).
"""

from __future__ import annotations

import json
from typing import Iterator

import numpy as np


def _load_meta(meta_path: str) -> dict:
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def iter_batches(
    tokens_path: str,
    meta_path: str,
    seq_len: int,
    batch_size: int,
    total_steps: int,
    rank: int,
    world_size: int,
) -> Iterator[tuple[list[list[float]], list[list[float]]]]:
    """
    Yield exactly total_steps batches for the given rank (step-aligned DDP sharding).

    Each batch is (input_ids, labels) where each is a list of lists of float (exact integers)
    with shape (batch_size, seq_len) for Gradtuity Tensor(...).

    Step alignment: global_batch_id = step * world_size + rank; each batch uses
    batch_size consecutive segments so every rank has the same number of steps.
    """
    meta = _load_meta(meta_path)
    num_tokens = meta["num_tokens"]
    dtype = meta.get("dtype", "uint16")
    if dtype != "uint16":
        raise ValueError(f"Only uint16 dtype supported, got {dtype!r}")

    num_segments = (num_tokens - 1) // seq_len
    if num_segments < 1:
        raise ValueError(
            f"Not enough tokens for one segment: num_tokens={num_tokens}, seq_len={seq_len}"
        )

    segments_needed = total_steps * world_size * batch_size
    if segments_needed > num_segments:
        raise ValueError(
            f"Not enough segments: need {segments_needed} (total_steps={total_steps} * world_size={world_size} * batch_size={batch_size}), have {num_segments}"
        )

    tokens = np.fromfile(tokens_path, dtype=np.uint16, count=num_tokens)

    for step in range(total_steps):
        global_batch_id = step * world_size + rank
        seg0 = global_batch_id * batch_size

        # (batch_size, seq_len + 1) view: each row is one segment of consecutive tokens
        start = seg0 * seq_len
        end = start + batch_size * seq_len + 1
        flat = tokens[start:end].astype(np.float32, copy=False)
        # Build a (B, S+1) window via stride tricks would be clearer but this is fine:
        # rows[i] = tokens[seg0*S + i*S : seg0*S + (i+1)*S + 1]
        input_ids_batch: list[list[float]] = []
        labels_batch: list[list[float]] = []
        for i in range(batch_size):
            row_start = i * seq_len
            input_ids_batch.append(flat[row_start : row_start + seq_len].tolist())
            labels_batch.append(flat[row_start + 1 : row_start + seq_len + 1].tolist())
        yield (input_ids_batch, labels_batch)


def get_steps_per_epoch(
    meta_path: str,
    seq_len: int,
    batch_size: int,
    world_size: int,
) -> int:
    """
    Number of batches (steps) per rank in one full pass over the data (one epoch).
    """
    meta = _load_meta(meta_path)
    num_tokens = meta["num_tokens"]
    num_segments = (num_tokens - 1) // seq_len
    return max(0, num_segments // (world_size * batch_size))
