#!/usr/bin/env python3
"""
GPT-2 training with DDP: inline data pipeline (tokens.bin + meta.json),
epoch-based sharding, sync_grads then clip_grad_norm_ then step.

Usage:
  uv run python -m gradtuity.launch --nproc 1 gpt2/train.py --data_dir gpt2/data/test-1M --seq_len 512 --batch_size 2 --epochs 2
  (If OOM: use --batch_size 1 and/or --seq_len 256.)

Requires RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT (set by launch).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# gpt2/ is not an installed package; the launcher invokes this script by path,
# so sys.path[0] is gpt2/, not the repo root. Insert repo root for `gpt2.*` imports.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from gpt2.data_pipeline import get_steps_per_epoch, iter_batches
from gpt2.model import GPT2_SMALL, GPT2Small
from gradtuity import AdamW, Tensor, clip_grad_norm_, save_safetensors
from gradtuity.dist import (
    destroy_process_group,
    get_rank,
    get_world_size,
    init,
    init_sync,
    print_rank,
    sync_grads,
)


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float = 1e-5,
) -> float:
    """Linear warmup then cosine decay to min_lr (per-step)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train GPT-2 small with DDP")
    ap.add_argument("--data_dir", type=str, required=True, help="Dir containing tokens.bin and meta.json")
    ap.add_argument("--seq_len", type=int, default=512, help="Sequence length (lower if OOM, e.g. 256)")
    ap.add_argument("--batch_size", type=int, default=2, help="Per-rank batch size (lower if OOM, e.g. 1)")
    ap.add_argument("--epochs", type=int, default=2, help="Number of full passes over the data")
    ap.add_argument("--lr", type=float, default=3e-4, help="Max learning rate")
    ap.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps (per-step schedule)")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    ap.add_argument("--max_norm", type=float, default=1.0, help="Gradient clip norm")
    ap.add_argument("--checkpoint_dir", type=str, default=None, help="Save checkpoints here (rank 0)")
    ap.add_argument("--save_every", type=int, default=1, help="Save every N epochs")
    ap.add_argument("--log_every", type=int, default=10, help="Log loss every N steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_checkpoint", action="store_true", help="Activation checkpointing per block (lower memory, ~30%% slower)")
    args = ap.parse_args()

    init()
    rank = get_rank()
    world_size = get_world_size()

    tokens_path = os.path.join(args.data_dir, "tokens.bin")
    meta_path = os.path.join(args.data_dir, "meta.json")
    if not os.path.isfile(tokens_path) or not os.path.isfile(meta_path):
        if rank == 0:
            print(f"Missing {tokens_path} or {meta_path}. Run gpt2/dataset.py first.", file=sys.stderr)
        destroy_process_group()
        sys.exit(1)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    seq_len = args.seq_len
    batch_size = args.batch_size

    steps_per_epoch = get_steps_per_epoch(meta_path, seq_len, batch_size, world_size)
    if steps_per_epoch < 1:
        if rank == 0:
            print("Not enough segments for one epoch. Reduce seq_len/batch_size or add data.", file=sys.stderr)
        destroy_process_group()
        sys.exit(1)
    total_steps = args.epochs * steps_per_epoch
    print_rank(
        f"epochs={args.epochs} steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
        f"batch_size={batch_size} seq_len={seq_len} world_size={world_size}"
    )

    model_cfg = {k: v for k, v in GPT2_SMALL.items() if k not in ("vocab_size", "n_positions")}
    model = GPT2Small(vocab_size=vocab_size, n_positions=seq_len, use_checkpoint=args.use_checkpoint, **model_cfg)
    init_sync(model)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.checkpoint_dir and rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        for input_ids_batch, labels_batch in iter_batches(
            tokens_path, meta_path, seq_len, batch_size, steps_per_epoch, rank, world_size
        ):
            lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr)
            optimizer.lr = lr

            input_ids_t = Tensor(input_ids_batch)
            labels_flat = [x for row in labels_batch for x in row]
            labels_t = Tensor(labels_flat)

            logits = model(input_ids_t)
            B, S, V = logits.shape
            loss = logits.view((B * S, V)).cross_entropy(labels_t, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            sync_grads(model.parameters())
            clip_grad_norm_(model.parameters(), args.max_norm, eps=1e-6)
            optimizer.step()

            global_step += 1
            if args.log_every and global_step % args.log_every == 0:
                print_rank(
                    f"epoch {epoch + 1}/{args.epochs} step {global_step}/{total_steps} "
                    f"loss={loss.item():.4f} lr={lr:.2e}"
                )

        if (
            args.checkpoint_dir
            and args.save_every
            and (epoch + 1) % args.save_every == 0
            and rank == 0
        ):
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_epoch_{epoch + 1}.safetensors")
            save_safetensors(model.state_dict(), ckpt_path)
            print_rank(f"Saved {ckpt_path}")

    print_rank("Training done.")
    destroy_process_group()


if __name__ == "__main__":
    main()
