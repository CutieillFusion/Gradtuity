#!/usr/bin/env python3
"""
Gradtuity MLP Training Demo

This demonstrates a complete training loop using the from-scratch
tensor autograd engine with Triton kernels.

Forward: Z = relu(X @ W + b)
Loss: loss = sum(Z)
Backward: Compute gradients via autograd
Update: SGD step

Usage:
    uv run python -m gradtuity.demo_train
"""

from gradtuity import SGD, Tensor, randn


def main():
    """Train a single-layer MLP and demonstrate loss decrease."""
    print("=" * 60)
    print("Gradtuity MLP Training Demo")
    print("=" * 60)
    print()

    # Hyperparameters
    batch_size = 32
    in_features = 64
    out_features = 16
    learning_rate = 0.001
    num_iterations = 100

    print(f"Configuration:")
    print(f"  Batch size:    {batch_size}")
    print(f"  In features:   {in_features}")
    print(f"  Out features:  {out_features}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iterations:    {num_iterations}")
    print()

    # Initialize tensors (all from scratch, no external libraries)
    print("Initializing tensors...")
    X = randn((batch_size, in_features), seed=42)  # Input (not trainable)
    W = randn((in_features, out_features), requires_grad=True, seed=123)  # Weights
    b = randn((out_features,), requires_grad=True, seed=456)  # Bias

    print(f"  X shape: {X.shape}")
    print(f"  W shape: {W.shape}")
    print(f"  b shape: {b.shape}")
    print()

    # Training loop
    print("Training...")
    print("-" * 40)

    optimizer = SGD([W, b], lr=learning_rate)
    losses = []
    for i in range(num_iterations):
        # Forward pass: Z = relu(X @ W + b)
        h = X.matmul(W)  # Linear: (batch, in) @ (in, out) -> (batch, out)
        y = h.add_bias(b)  # Add bias: (batch, out) + (out,) -> (batch, out)
        z = y.relu()  # ReLU activation
        loss = z.sum()  # Sum all activations as loss

        # Get scalar loss value
        loss_val = loss.item()
        losses.append(loss_val)

        # Print progress
        if i % 10 == 0 or i == num_iterations - 1:
            print(f"  iter {i:3d}: loss = {loss_val:.4f}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()  # Compute gradients

        # SGD update
        optimizer.step()

    print("-" * 40)
    print()

    # Summary
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_decrease = initial_loss - final_loss
    percent_decrease = (loss_decrease / initial_loss) * 100

    print("Training Summary:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Decrease:     {loss_decrease:.4f} ({percent_decrease:.1f}%)")

    # Verify loss decreased
    if final_loss < initial_loss:
        print()
        print("SUCCESS: Loss decreased during training!")
    else:
        print()
        print("WARNING: Loss did not decrease. Check gradients.")

    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return losses


if __name__ == "__main__":
    main()
