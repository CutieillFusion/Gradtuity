#!/usr/bin/env python3
"""
Gradtuity MNIST CNN Demo

This script demonstrates training a CNN on the MNIST handwritten digit dataset
using Gradtuity's from-scratch tensor autodiff engine with Triton GPU kernels.

Architecture: CNN (Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Linear -> ReLU -> Linear)
Input: (N, 1, 28, 28). Output: (N, 10).
Loss: MSE with one-hot encoded targets (+1 for correct class, -1 for others)

Usage:
    uv run python demos/demo_mnist_cnn.py
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

from gradtuity import CNN, Tensor, sgd_step

# Create plots directory (CNN-specific subdir or prefix)
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
PLOT_PREFIX = "mnist_cnn_"  # e.g. mnist_cnn_1_samples.png

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 60)
print("Gradtuity MNIST CNN Demo")
print("=" * 60)
print()

# -----------------------------------------------------------------------------
# Load MNIST Dataset
# -----------------------------------------------------------------------------
print("Loading MNIST dataset...")
start_time = time.time()

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_all, y_all = mnist.data, mnist.target.astype(int)
X_all = X_all / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, train_size=60000, test_size=10000, random_state=42, stratify=y_all
)

print(f"Loaded in {time.time() - start_time:.1f}s")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Input shape for CNN: (batch, 1, 28, 28)")
print(f"Classes: 0-9 (10 digits)")
print()

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
print(f"Saved: {PLOT_DIR}/{PLOT_PREFIX}1_samples.png")

# -----------------------------------------------------------------------------
# Create Model
# -----------------------------------------------------------------------------
print()
print("Creating CNN model...")

model = CNN()
print(model)
num_params = sum(p.numel for p in model.parameters())
print(f"Number of parameters: {num_params:,}")
print()

# -----------------------------------------------------------------------------
# Training Setup
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 15
INITIAL_LR = 0.01


def create_one_hot(labels, num_classes=10):
    targets = []
    for label in labels:
        row = [-1.0] * num_classes
        row[int(label)] = 1.0
        targets.append(row)
    return targets


# -----------------------------------------------------------------------------
# Pre-convert data to Tensors (4D for CNN)
# -----------------------------------------------------------------------------
print("Pre-converting data to Tensors (4D for CNN)...")
preconv_start = time.time()

num_train_batches = len(X_train) // BATCH_SIZE
train_X_batches = []
train_Y_batches = []
train_Y_labels = []

for i in range(num_train_batches):
    start_idx = i * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    X_batch_np = X_train[start_idx:end_idx]
    y_batch_np = y_train[start_idx:end_idx]
    X_batch_4d = X_batch_np.reshape(-1, 1, 28, 28)
    train_X_batches.append(Tensor(X_batch_4d.tolist()))
    train_Y_batches.append(Tensor(create_one_hot(y_batch_np)))
    train_Y_labels.append(y_batch_np)

TEST_BATCH_SIZE = 256
num_test_batches = (len(X_test) + TEST_BATCH_SIZE - 1) // TEST_BATCH_SIZE
test_X_batches = []
test_Y_labels = []

for i in range(num_test_batches):
    start_idx = i * TEST_BATCH_SIZE
    end_idx = min(start_idx + TEST_BATCH_SIZE, len(X_test))
    X_batch_np = X_test[start_idx:end_idx]
    y_batch_np = y_test[start_idx:end_idx]
    X_batch_4d = X_batch_np.reshape(-1, 1, 28, 28)
    test_X_batches.append(Tensor(X_batch_4d.tolist()))
    test_Y_labels.append(y_batch_np)

print(f"Pre-converted {num_train_batches} train batches + {num_test_batches} test batches in {time.time() - preconv_start:.1f}s")
print()


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
print("Training...")
print("-" * 60)

train_losses = []
train_accuracies = []
test_accuracies = []
TIMING_DEBUG = True
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    time_forward = 0.0
    time_loss_calc = 0.0
    time_zero_grad = 0.0
    time_backward = 0.0
    time_sgd = 0.0
    time_accuracy = 0.0
    batch_indices = np.random.permutation(num_train_batches)
    lr = INITIAL_LR * (1.0 - 0.5 * epoch / NUM_EPOCHS)
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx in batch_indices:
        X_tensor = train_X_batches[batch_idx]
        y_onehot = train_Y_batches[batch_idx]
        y_labels = train_Y_labels[batch_idx]

        t0 = time.perf_counter()
        scores = model(X_tensor)
        time_forward += time.perf_counter() - t0

        t0 = time.perf_counter()
        loss = scores.mse_loss(y_onehot)
        time_loss_calc += time.perf_counter() - t0
        epoch_loss += loss.item()

        t0 = time.perf_counter()
        pred_indices = scores.argmax(dim=1)
        predictions = np.array(pred_indices.to_list(), dtype=int)
        acc = (predictions == y_labels).mean()
        epoch_acc += acc
        time_accuracy += time.perf_counter() - t0

        t0 = time.perf_counter()
        model.zero_grad()
        time_zero_grad += time.perf_counter() - t0

        t0 = time.perf_counter()
        loss.backward()
        time_backward += time.perf_counter() - t0

        t0 = time.perf_counter()
        sgd_step(model.parameters(), lr=lr)
        time_sgd += time.perf_counter() - t0

    avg_loss = epoch_loss / num_train_batches
    avg_acc = epoch_acc / num_train_batches
    t0 = time.perf_counter()
    test_acc = evaluate_preconverted()
    time_eval = time.perf_counter() - t0

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
    if TIMING_DEBUG:
        total_train = time_forward + time_loss_calc + time_zero_grad + time_backward + time_sgd + time_accuracy
        print(f"  Timing breakdown (train loop {total_train:.2f}s + eval {time_eval:.2f}s):")
        print(f"    Forward:     {time_forward:6.3f}s ({100*time_forward/total_train:5.1f}%)")
        print(f"    Loss calc:   {time_loss_calc:6.3f}s ({100*time_loss_calc/total_train:5.1f}%)")
        print(f"    Accuracy:    {time_accuracy:6.3f}s ({100*time_accuracy/total_train:5.1f}%)")
        print(f"    Zero grad:   {time_zero_grad:6.3f}s ({100*time_zero_grad/total_train:5.1f}%)")
        print(f"    Backward:    {time_backward:6.3f}s ({100*time_backward/total_train:5.1f}%)")
        print(f"    SGD update:  {time_sgd:6.3f}s ({100*time_sgd/total_train:5.1f}%)")

total_time = time.time() - start_time
print("-" * 60)
print(f"Training complete in {total_time:.1f}s")
print()

# -----------------------------------------------------------------------------
# Final Evaluation
# -----------------------------------------------------------------------------
print("Final Evaluation:")


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
print(f"  Train Accuracy: {final_train_acc * 100:.2f}%")
print(f"  Test Accuracy:  {final_test_acc * 100:.2f}%")
print()

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, "b-", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss (CNN)")
ax1.grid(True, alpha=0.3)
ax2.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in train_accuracies], "b-", linewidth=2, label="Train")
ax2.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in test_accuracies], "r-", linewidth=2, label="Test")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training and Test Accuracy (CNN)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{PLOT_PREFIX}2_training.png"), dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/{PLOT_PREFIX}2_training.png")

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
print(f"Saved: {PLOT_DIR}/{PLOT_PREFIX}3_predictions.png")

print()
print("Per-class accuracy (CNN):")
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
        print(f"  Digit {digit}: {acc * 100:.1f}% ({mask.sum()} samples)")

print()
print("=" * 60)
print("CNN demo complete!")
print(f"Final test accuracy: {final_test_acc * 100:.2f}%")
print("=" * 60)
