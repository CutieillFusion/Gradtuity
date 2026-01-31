#!/usr/bin/env python3
"""
Gradtuity Moons Demo - Micrograd Style

This script demonstrates Gradtuity's neural network capabilities on the classic
"moons" dataset, closely mirroring the micrograd demo notebook structure.

Key differences from micrograd:
- Uses batched GPU operations instead of scalar Python loops
- Single forward/backward pass for the entire batch
- Much faster execution due to GPU parallelism

Usage:
    uv run python demos/demo_moons_notebook.py
"""

import os
import random

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for SSH/headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from gradtuity import MLP, SGD, Tensor, ones

# Create plots directory
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Set seeds for reproducibility (same as micrograd demo)
np.random.seed(1337)
random.seed(1337)

print("=" * 60)
print("Gradtuity Moons Demo (Micrograd Style)")
print("=" * 60)
print()

# Make up a dataset
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # make y be -1 or 1 (same as micrograd)

# Visualize in 2D
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
plt.title("Moons Dataset")
plt.savefig(os.path.join(PLOT_DIR, "moons_1_dataset.png"), dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/moons_1_dataset.png")

# Initialize a model (same architecture as micrograd: 2 -> 16 -> 16 -> 1)
model = MLP(2, [16, 16, 1])
print(model)
print(f"number of parameters {sum(p.numel for p in model.parameters())}")
print()


def loss(batch_size=None):
    """
    Compute loss and accuracy, matching micrograd's loss function.

    Uses SVM "max-margin" hinge loss: (1 + -yi*scorei).relu()
    """
    # Inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]

    # Convert to Gradtuity tensors
    # X_tensor shape: (N, 2), y_tensor shape: (N, 1)
    X_tensor = Tensor(Xb.tolist())
    y_tensor = Tensor([[yi] for yi in yb.tolist()])  # Shape (N, 1)

    # Forward the model to get scores
    scores = model(X_tensor)  # Shape (N, 1)

    # SVM "max-margin" loss: (1 + -yi*scorei).relu()
    # = (1 - yi*scorei).relu()
    # = relu(1 - y * scores)

    # Create ones tensor for the margin
    ones_tensor = ones((len(yb), 1))

    # Compute: 1 - y * scores = ones - (y * scores)
    y_times_scores = y_tensor * scores  # Elementwise multiply
    margins = ones_tensor - y_times_scores  # 1 - y*score
    losses = margins.relu()  # max(0, 1 - y*score)

    # Mean loss
    data_loss = losses.sum() * (1.0 / len(yb))

    # L2 regularization (same alpha as micrograd)
    alpha = 1e-4
    reg_loss_val = 0.0
    for p in model.parameters():
        p_squared = p * p  # Elementwise square
        reg_loss_val += p_squared.sum().item()

    # Note: We compute reg_loss as scalar since it's simpler
    # and doesn't need to be part of the autograd graph for this demo
    # (the main gradients come from data_loss)

    # Also get accuracy
    scores_np = np.array(scores.to_list()).flatten()
    accuracy = ((yb > 0) == (scores_np > 0)).mean()

    return data_loss, accuracy, alpha * reg_loss_val


# Test initial state
total_loss, acc, reg = loss()
print(f"Initial: loss {total_loss.item() + reg:.6f}, accuracy {acc * 100:.1f}%")
print()

# Optimization (same as micrograd: 100 steps)
print("Training...")
print("-" * 40)

optimizer = SGD(model.parameters(), lr=1.0)
losses_history = []
acc_history = []

for k in range(100):
    # Forward
    total_loss, acc, reg = loss()

    # Record for plotting
    loss_val = total_loss.item() + reg
    losses_history.append(loss_val)
    acc_history.append(acc)

    # Backward
    optimizer.zero_grad()
    total_loss.backward()

    # Update (SGD) - same learning rate schedule as micrograd
    learning_rate = 1.0 - 0.9 * k / 100
    optimizer.lr = learning_rate
    optimizer.step()

    if k % 1 == 0:
        print(f"step {k} loss {loss_val:.6f}, accuracy {acc * 100:.1f}%")

print("-" * 40)
print()

# Plot training progress
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(losses_history)
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")

ax2.plot([a * 100 for a in acc_history])
ax2.set_xlabel("Step")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training Accuracy")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "moons_2_training.png"), dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/moons_2_training.png")

# Visualize decision boundary (same code as micrograd)
h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]

# Forward pass on mesh (batched)
X_mesh_tensor = Tensor(Xmesh.tolist())
scores_mesh = model(X_mesh_tensor)
scores_mesh_np = np.array(scores_mesh.to_list()).flatten()
Z = (scores_mesh_np > 0).astype(int)
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors="black")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"Decision Boundary (Final Accuracy: {acc_history[-1] * 100:.1f}%)")
plt.savefig(os.path.join(PLOT_DIR, "moons_3_boundary.png"), dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/moons_3_boundary.png")

print()
print("=" * 60)
print("Demo complete!")
print(f"Final accuracy: {acc_history[-1] * 100:.1f}%")
print("=" * 60)
