#!/usr/bin/env python3
"""
Gradtuity MNIST Multi-GPU Demo (CNN w/DDP)

This script trains a convolutional neural network (CNN) on the MNIST handwritten digit dataset
using Gradtuity's from-scratch tensor autograd engine with Triton GPU kernels.

It is designed for distributed multi-GPU training: each process (rank) works on a unique shard
of the training data, gradients are synchronized across all GPUs with AllReduce after every
backward pass, ensuring all model replicas stay in sync throughout training.

Architecture: CNN (Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Linear -> ReLU -> Linear)
Input: (N, 1, 28, 28). Output: (N, 10).
Loss: MSE with one-hot encoded targets (+1 for correct class, -1 for others)

Required env vars (set by gradtuity.launch or manually):
  RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT

Usage:
  # 2 GPUs on one node (recommended)
  uv run python -m gradtuity.launch --nproc 2 demos/demo_mnist_dist.py

  # Single GPU (still need all env vars)
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \\
    uv run python demos/demo_mnist_dist.py
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

from gradtuity import CNN, SGD, Tensor, one_hot
from gradtuity.dist import (
    distributed_indices,
    get_rank,
    get_world_size,
    init,
    init_sync,
    print_rank,
)

init()
rank = get_rank()
world_size = get_world_size()
is_rank_0 = rank == 0

# Create plots directory (CNN-specific subdir or prefix)
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
PLOT_PREFIX = "mnist_dist_cnn_"  # e.g. mnist_dist_cnn_1_samples.png

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# This will only print on rank 0 (by Default)
print_rank("=" * 60)
print_rank("Gradtuity MNIST Multi-GPU Demo (CNN w/DDP)")
print_rank("=" * 60)
print_rank("")

# -----------------------------------------------------------------------------
# Load MNIST Dataset
# -----------------------------------------------------------------------------
print_rank("Loading MNIST dataset...")
start_time = time.time()

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_all, y_all = mnist.data, mnist.target.astype(int)
X_all = X_all / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, train_size=60000, test_size=10000, random_state=42, stratify=y_all
)

print_rank(f"Loaded in {time.time() - start_time:.1f}s")
print_rank(f"Training samples: {len(X_train)}")
print_rank(f"Test samples: {len(X_test)}")
print_rank("Input shape for CNN: (batch, 1, 28, 28)")
print_rank("Classes: 0-9 (10 digits)")
print_rank("")

# Visualize some samples
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.suptitle("Sample MNIST Images (CNN Demo)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{PLOT_PREFIX}1_samples.png"), dpi=150)
plt.close()
print_rank(f"Saved: {PLOT_DIR}/{PLOT_PREFIX}1_samples.png")

# -----------------------------------------------------------------------------
# Create Model
# -----------------------------------------------------------------------------
print_rank("")
print_rank("Creating CNN model...")

model = CNN()
init_sync(model)
print_rank(model)
num_params = sum(p.numel for p in model.parameters())
print_rank(f"Number of parameters: {num_params:,}")
print_rank("")

# -----------------------------------------------------------------------------
# Training Setup
# -----------------------------------------------------------------------------

BATCH_SIZE = 64
NUM_EPOCHS = 10
INITIAL_LR = 0.01 * world_size**0.5

optimizer = SGD(model.parameters(), lr=INITIAL_LR)


# -----------------------------------------------------------------------------
# Pre-convert data to Tensors (4D for CNN)
# -----------------------------------------------------------------------------
print_rank("Pre-converting data to Tensors (4D for CNN)...")
preconv_start = time.time()

train_indices = list(distributed_indices(len(X_train)))
num_train_batches = max(1, len(train_indices) // BATCH_SIZE)
train_X_batches = []
train_Y_batches = []
train_Y_labels = []

for i in range(num_train_batches):
    start = i * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(train_indices))
    idx = train_indices[start:end]
    X_batch = X_train[idx].reshape(-1, 1, 28, 28)
    y_batch = y_train[idx]
    train_X_batches.append(Tensor(X_batch.tolist()))
    train_Y_batches.append(
        one_hot(Tensor(y_batch.astype(np.float32).tolist()), num_classes=10)
    )
    train_Y_labels.append(y_batch)

TEST_BATCH_SIZE = 256
test_indices = list(distributed_indices(len(X_test)))
num_test_batches = max(1, len(test_indices) // TEST_BATCH_SIZE)
test_X_batches = []
test_Y_labels = []

for i in range(num_test_batches):
    start = i * TEST_BATCH_SIZE
    end = min(start + TEST_BATCH_SIZE, len(test_indices))
    idx = test_indices[start:end]
    X_batch = X_test[idx].reshape(-1, 1, 28, 28)
    y_batch = y_test[idx]
    test_X_batches.append(Tensor(X_batch.tolist()))
    test_Y_labels.append(y_batch)

print_rank(
    f"Pre-converted {num_train_batches} train batches + {num_test_batches} test batches in {time.time() - preconv_start:.1f}s"
)
print_rank("")


def evaluate_preconverted():
    total_correct = 0
    total_samples = 0
    for X_tensor, y_labels in zip(test_X_batches, test_Y_labels):
        scores = model(X_tensor)
        pred_indices = scores.argmax(dim=1)
        predictions = np.array(pred_indices.to_list(), dtype=int)
        total_correct += (predictions == y_labels).sum()
        total_samples += len(y_labels)
    return total_correct / total_samples


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
print_rank("Training...")
print_rank("-" * 60)

train_losses = []
train_accuracies = []
test_accuracies = []
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    lr = INITIAL_LR * (1.0 - 0.5 * epoch / NUM_EPOCHS)
    optimizer.lr = lr

    batch_indices = np.random.permutation(num_train_batches)
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx in batch_indices:
        X_tensor = train_X_batches[batch_idx]
        y_onehot = train_Y_batches[batch_idx]
        y_labels = train_Y_labels[batch_idx]

        scores = model(X_tensor)
        loss = scores.mse_loss(y_onehot)
        epoch_loss += loss.item()

        pred_indices = scores.argmax(dim=1)
        predictions = np.array(pred_indices.to_list(), dtype=int)
        acc = (predictions == y_labels).mean()
        epoch_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / num_train_batches
    avg_acc = epoch_acc / num_train_batches
    test_acc = evaluate_preconverted()

    train_losses.append(avg_loss)
    train_accuracies.append(avg_acc)
    test_accuracies.append(test_acc)
    epoch_time = time.time() - epoch_start
    print_rank(
        f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc * 100:.1f}% | Test Acc: {test_acc * 100:.1f}% | LR: {lr:.4f} | Time: {epoch_time:.1f}s"
    )

total_time = time.time() - start_time
print_rank("-" * 60)
print_rank(f"Training complete in {total_time:.1f}s")
print_rank("")

# -----------------------------------------------------------------------------
# Final Evaluation
# -----------------------------------------------------------------------------
print_rank("Final Evaluation:")


def evaluate_train_preconverted():
    total_correct = 0
    total_samples = 0
    for X_tensor, y_labels in zip(train_X_batches, train_Y_labels):
        scores = model(X_tensor)
        pred_indices = scores.argmax(dim=1)
        predictions = np.array(pred_indices.to_list(), dtype=int)
        total_correct += (predictions == y_labels).sum()
        total_samples += len(y_labels)
    return total_correct / total_samples


final_train_acc = evaluate_train_preconverted()
final_test_acc = evaluate_preconverted()
print_rank(f"  Train Accuracy: {final_train_acc * 100:.2f}%")
print_rank(f"  Test Accuracy:  {final_test_acc * 100:.2f}%")
print_rank("")

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
if is_rank_0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, "b-", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss (CNN)")
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
    ax2.set_title("Training and Test Accuracy (CNN)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{PLOT_PREFIX}2_training.png"), dpi=150)
    plt.close()
    print_rank(f"Saved: {PLOT_DIR}/{PLOT_PREFIX}2_training.png")

    # Sample predictions (4D input for CNN)
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    X_sample = X_test[:15]
    y_sample = y_test[:15]
    X_sample_4d = X_sample.reshape(-1, 1, 28, 28)
    X_tensor = Tensor(X_sample_4d.tolist())
    scores = model(X_tensor)
    pred_indices = scores.argmax(dim=1)
    predictions = np.array(pred_indices.to_list(), dtype=int)
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_sample[i].reshape(28, 28), cmap="gray")
        color = "green" if predictions[i] == y_sample[i] else "red"
        ax.set_title(f"Pred: {predictions[i]} (True: {y_sample[i]})", color=color)
        ax.axis("off")
    plt.suptitle(f"Sample Predictions CNN (Test Accuracy: {final_test_acc * 100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{PLOT_PREFIX}3_predictions.png"), dpi=150)
    plt.close()
    print_rank(f"Saved: {PLOT_DIR}/{PLOT_PREFIX}3_predictions.png")

    print_rank()
    print_rank("Per-class accuracy (CNN):")
    for digit in range(10):
        mask = y_test == digit
        if mask.sum() > 0:
            X_digit = X_test[mask]
            y_digit = y_test[mask]
            X_digit_4d = X_digit.reshape(-1, 1, 28, 28)
            X_tensor = Tensor(X_digit_4d.tolist())
            scores = model(X_tensor)
            pred_indices = scores.argmax(dim=1)
            preds = np.array(pred_indices.to_list(), dtype=int)
            acc = (preds == y_digit).mean()
            print_rank(f"  Digit {digit}: {acc * 100:.1f}% ({mask.sum()} samples)")

    print_rank()
    print_rank("=" * 60)
    print_rank("CNN demo complete!")
    print_rank(f"Final test accuracy: {final_test_acc * 100:.2f}%")
    print_rank("=" * 60)
