#!/usr/bin/env python3
"""
Gradtuity Training Demo with SafeTensors Checkpointing

Trains a small MLP, saving the model to a SafeTensors file. If a checkpoint
already exists, it is loaded and training continues from those weights.

Usage:
    uv run python demos/demo_checkpoint.py

Run twice to see load-then-train: first run saves ckpt.safetensors,
second run loads it and trains further.
"""

import os

from gradtuity import MLP, SGD, Tensor, load_safetensors, randn, save_safetensors

# Checkpoint path next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "ckpt.safetensors")


def main() -> None:
    print("=" * 60)
    print("Gradtuity Training Demo (SafeTensors checkpoint)")
    print("=" * 60)
    print()

    # Same architecture as moons: 2 -> 16 -> 16 -> 1
    model = MLP(2, [16, 16, 1])
    num_params = sum(p.numel for p in model.parameters())
    print(f"Model: {model}")
    print(f"Parameters: {num_params}")
    print()

    # Load checkpoint if it exists
    if os.path.exists(CKPT_PATH):
        print(f"Loading checkpoint: {CKPT_PATH}")
        state = load_safetensors(CKPT_PATH, requires_grad=True)
        model.load_state_dict(state)
        print("Checkpoint loaded. Resuming from saved weights.")
    else:
        print(f"No checkpoint at {CKPT_PATH}; training from scratch.")
    print()

    # Synthetic data (small batch, same input size as moons)
    batch_size = 32
    num_iters = 50
    learning_rate = 0.01

    print("Configuration:")
    print(f"  Batch size:   {batch_size}")
    print(f"  Iterations:   {num_iters}")
    print(f"  Learning rate: {learning_rate}")
    print()

    # Fixed random inputs/targets for reproducibility
    X = randn((batch_size, 2), seed=42)
    # Fake targets: sign of sum of coords (simple separable task)
    Y_raw = [
        [1.0] if (X.to_list()[i][0] + X.to_list()[i][1] > 0) else [-1.0]
        for i in range(batch_size)
    ]
    Y = Tensor(Y_raw)

    print("Training...")
    print("-" * 40)

    optimizer = SGD(model.parameters(), lr=learning_rate)
    for step in range(num_iters):
        # Forward
        scores = model(X)
        # Hinge-like loss: mean relu(1 - y * score)
        margins = Tensor([[1.0]] * batch_size) - Y * scores
        loss = margins.relu().sum() * (1.0 / batch_size)

        loss_val = loss.item()
        if step % 10 == 0 or step == num_iters - 1:
            print(f"  step {step:3d}: loss = {loss_val:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("-" * 40)
    print()

    # Save checkpoint
    state_dict = model.state_dict()
    save_safetensors(
        CKPT_PATH,
        state_dict,
        metadata={"demo": "demo_checkpoint", "steps": str(num_iters)},
    )
    print(f"Checkpoint saved: {CKPT_PATH}")
    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
