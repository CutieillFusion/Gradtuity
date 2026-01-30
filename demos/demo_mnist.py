#!/usr/bin/env python3
"""
Gradtuity MNIST Demo

This script demonstrates training an MLP on the MNIST handwritten digit dataset
using Gradtuity's from-scratch tensor autodiff engine with Triton GPU kernels.

Architecture: 784 -> 128 -> 64 -> 10
Loss: MSE with one-hot encoded targets (+1 for correct class, -1 for others)

Usage:
    uv run python demos/demo_mnist.py
"""

import os
import random
import time

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for SSH/headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from gradtuity import MLP, Tensor, sgd_step

# Create plots directory
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 60)
print("Gradtuity MNIST Demo")
print("=" * 60)
print()

# -----------------------------------------------------------------------------
# Load MNIST Dataset
# -----------------------------------------------------------------------------
print("Loading MNIST dataset...")
start_time = time.time()

# Fetch MNIST from OpenML (cached after first download)
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_all, y_all = mnist.data, mnist.target.astype(int)

# Normalize pixel values to [0, 1]
X_all = X_all / 255.0

# Use a subset for faster training (full MNIST is 70K samples)
# For demo purposes, use 10K train + 2K test
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, train_size=10000, test_size=2000, random_state=42, stratify=y_all
)

print(f"Loaded in {time.time() - start_time:.1f}s")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Input features: {X_train.shape[1]}")
print(f"Classes: 0-9 (10 digits)")
print()

# Visualize some samples
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.suptitle("Sample MNIST Images")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "mnist_1_samples.png"), dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/mnist_1_samples.png")

# -----------------------------------------------------------------------------
# Create Model
# -----------------------------------------------------------------------------
print()
print("Creating model...")

# MLP: 784 -> 128 -> 64 -> 10
model = MLP(784, [128, 64, 10])
print(model)
num_params = sum(p.numel for p in model.parameters())
print(f"Number of parameters: {num_params:,}")
print()

# -----------------------------------------------------------------------------
# Training Setup
# -----------------------------------------------------------------------------

# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 15
INITIAL_LR = 0.01


def create_one_hot(labels, num_classes=10):
    """
    Create one-hot encoded targets with +1 for correct class, -1 for others.

    This encoding works well with MSE loss for classification.
    """
    batch_size = len(labels)
    targets = []
    for label in labels:
        row = [-1.0] * num_classes
        row[int(label)] = 1.0
        targets.append(row)
    return targets


def compute_loss_and_accuracy(X_batch_np, y_batch_np):
    """
    Compute MSE loss and accuracy for a batch.

    Loss: MSE between model outputs and one-hot targets
    Accuracy: Percentage of correct predictions (argmax of outputs)
    """
    batch_size = len(y_batch_np)

    # Convert to Gradtuity tensors
    X_tensor = Tensor(X_batch_np.tolist())
    y_onehot = Tensor(create_one_hot(y_batch_np))

    # Forward pass
    scores = model(X_tensor)  # Shape: (batch, 10)

    # MSE Loss: ((scores - targets)^2).sum() / (batch_size * num_classes)
    diff = scores - y_onehot
    squared = diff * diff  # Elementwise square
    loss = squared.sum() * (1.0 / (batch_size * 10))

    # Compute accuracy (on CPU using numpy)
    scores_np = np.array(scores.to_list())
    predictions = np.argmax(scores_np, axis=1)
    accuracy = (predictions == y_batch_np).mean()

    return loss, accuracy


def evaluate(X_data, y_data, batch_size=256):
    """Evaluate model on a dataset, returns average accuracy."""
    num_samples = len(X_data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    total_correct = 0
    total_samples = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        X_batch = X_data[start_idx:end_idx]
        y_batch = y_data[start_idx:end_idx]

        # Forward pass
        X_tensor = Tensor(X_batch.tolist())
        scores = model(X_tensor)

        # Compute predictions
        scores_np = np.array(scores.to_list())
        predictions = np.argmax(scores_np, axis=1)

        total_correct += (predictions == y_batch).sum()
        total_samples += len(y_batch)

    return total_correct / total_samples


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
print("Training...")
print("-" * 60)

num_batches = len(X_train) // BATCH_SIZE
train_losses = []
train_accuracies = []
test_accuracies = []

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Learning rate decay
    lr = INITIAL_LR * (1.0 - 0.5 * epoch / NUM_EPOCHS)

    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx in range(num_batches):
        # Get batch
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]

        # Forward + loss
        loss, acc = compute_loss_and_accuracy(X_batch, y_batch)

        epoch_loss += loss.item()
        epoch_acc += acc

        # Backward
        model.zero_grad()
        loss.backward()

        # SGD update
        sgd_step(model.parameters(), lr=lr)

    # Average metrics
    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches

    # Evaluate on test set
    test_acc = evaluate(X_test, y_test)

    train_losses.append(avg_loss)
    train_accuracies.append(avg_acc)
    test_accuracies.append(test_acc)

    epoch_time = time.time() - epoch_start
    print(
        f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Train Acc: {avg_acc * 100:.1f}% | "
        f"Test Acc: {test_acc * 100:.1f}% | "
        f"LR: {lr:.4f} | "
        f"Time: {epoch_time:.1f}s"
    )

total_time = time.time() - start_time
print("-" * 60)
print(f"Training complete in {total_time:.1f}s")
print()

# -----------------------------------------------------------------------------
# Final Evaluation
# -----------------------------------------------------------------------------
print("Final Evaluation:")
final_train_acc = evaluate(X_train, y_train)
final_test_acc = evaluate(X_test, y_test)
print(f"  Train Accuracy: {final_train_acc * 100:.2f}%")
print(f"  Test Accuracy:  {final_test_acc * 100:.2f}%")
print()

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, "b-", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(
    range(1, NUM_EPOCHS + 1),
    [a * 100 for a in train_accuracies],
    "b-",
    linewidth=2,
    label="Train",
)
ax2.plot(
    range(1, NUM_EPOCHS + 1),
    [a * 100 for a in test_accuracies],
    "r-",
    linewidth=2,
    label="Test",
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training and Test Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "mnist_2_training.png"), dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/mnist_2_training.png")

# Visualize predictions on test samples
fig, axes = plt.subplots(3, 5, figsize=(12, 8))

# Get predictions for first 15 test samples
X_sample = X_test[:15]
y_sample = y_test[:15]

X_tensor = Tensor(X_sample.tolist())
scores = model(X_tensor)
scores_np = np.array(scores.to_list())
predictions = np.argmax(scores_np, axis=1)

for i, ax in enumerate(axes.flat):
    ax.imshow(X_sample[i].reshape(28, 28), cmap="gray")
    color = "green" if predictions[i] == y_sample[i] else "red"
    ax.set_title(f"Pred: {predictions[i]} (True: {y_sample[i]})", color=color)
    ax.axis("off")

plt.suptitle(f"Sample Predictions (Test Accuracy: {final_test_acc * 100:.1f}%)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "mnist_3_predictions.png"), dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/mnist_3_predictions.png")

# Confusion matrix visualization (simplified)
print()
print("Per-class accuracy:")
for digit in range(10):
    mask = y_test == digit
    if mask.sum() > 0:
        X_digit = X_test[mask]
        y_digit = y_test[mask]

        X_tensor = Tensor(X_digit.tolist())
        scores = model(X_tensor)
        scores_np = np.array(scores.to_list())
        preds = np.argmax(scores_np, axis=1)

        acc = (preds == y_digit).mean()
        print(f"  Digit {digit}: {acc * 100:.1f}% ({mask.sum()} samples)")

print()
print("=" * 60)
print("Demo complete!")
print(f"Final test accuracy: {final_test_acc * 100:.2f}%")
print("=" * 60)
