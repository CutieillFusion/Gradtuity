"""
Tensor factory functions for creating tensors with specific values.

All functions create tensors on GPU using CUDA memory operations
and Triton kernels where appropriate.
"""

import random
import struct

import triton

from .cuda_mem import cuda_malloc, cuda_memset, cuda_memcpy_htod
from .kernels.optim_kernels import fill_kernel, sgd_update_kernel
from .kernels.one_hot_kernels import one_hot_kernel
from .tensor import Tensor


def zeros(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """
    Allocate a zero-initialized GPU tensor.

    Args:
        shape: Shape of the tensor (rank 1, 2, 3, or 4).
        requires_grad: Whether to track gradients.

    Returns:
        Tensor filled with zeros.
    """
    return Tensor._zeros(shape, requires_grad=requires_grad)


def zeros_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """
    Create a zero tensor with the same shape as the input.

    Args:
        t: Reference tensor for shape.
        requires_grad: Whether to track gradients.

    Returns:
        Tensor filled with zeros, same shape as t.
    """
    return Tensor._zeros_like(t, requires_grad=requires_grad)


def ones(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """
    Allocate a tensor filled with 1.0 via Triton fill kernel.

    Args:
        shape: Shape of the tensor (rank 1, 2, 3, or 4).
        requires_grad: Whether to track gradients.

    Returns:
        Tensor filled with ones.
    """
    return Tensor._ones(shape, requires_grad=requires_grad)


def ones_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """
    Create a tensor of ones with the same shape as the input.

    Args:
        t: Reference tensor for shape.
        requires_grad: Whether to track gradients.

    Returns:
        Tensor filled with ones, same shape as t.
    """
    return Tensor._ones_like(t, requires_grad=requires_grad)


def randn(
    shape: tuple[int, ...],
    requires_grad: bool = False,
    seed: int | None = None,
    std: float = 1.0,
) -> Tensor:
    """
    Generate a tensor with random normal values (mean=0, std=std).

    Uses Python's random module for generation, then transfers to GPU.

    Args:
        shape: Shape of the tensor (rank 1, 2, 3, or 4).
        requires_grad: Whether to track gradients.
        seed: Optional random seed for reproducibility.
        std: Standard deviation of the normal distribution (default 1.0).

    Returns:
        Tensor with random normal values.
    """
    # Validate shape
    if len(shape) not in (1, 2, 3, 4):
        raise ValueError(
            f"Only rank 1, 2, 3, or 4 tensors supported, got rank {len(shape)}"
        )

    # Set seed if provided
    if seed is not None:
        random.seed(seed)

    # Compute total elements
    numel = 1
    for s in shape:
        numel *= s

    # Generate random normal values on CPU with specified std
    data = [random.gauss(0, std) for _ in range(numel)]

    # Pack to bytes and copy to GPU
    host_bytes = struct.pack(f"{numel}f", *data)
    ptr = cuda_malloc(numel * 4)
    cuda_memcpy_htod(ptr, host_bytes)

    return Tensor._from_ptr(ptr, shape, owns_memory=True, requires_grad=requires_grad)


def full(
    shape: tuple[int, ...],
    fill_value: float,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor filled with a specific value.

    Args:
        shape: Shape of the tensor (rank 1, 2, 3, or 4).
        fill_value: Value to fill the tensor with.
        requires_grad: Whether to track gradients.

    Returns:
        Tensor filled with fill_value.
    """
    # Start with zeros (allocates memory)
    t = zeros(shape, requires_grad=requires_grad)

    # Fill with the specified value using Triton kernel
    grid = lambda meta: (triton.cdiv(t.numel, meta["BLOCK"]),)
    fill_kernel[grid](t.ptr, fill_value, t.numel, BLOCK=256)

    return t


def full_like(t: Tensor, fill_value: float, requires_grad: bool = False) -> Tensor:
    """
    Create a tensor filled with a specific value, same shape as input.

    Args:
        t: Reference tensor for shape.
        fill_value: Value to fill the tensor with.
        requires_grad: Whether to track gradients.

    Returns:
        Tensor filled with fill_value, same shape as t.
    """
    return full(t.shape, fill_value, requires_grad=requires_grad)


def one_hot(
    labels,
    num_classes: int,
    on_value: float = 1.0,
    off_value: float = -1.0,
) -> Tensor:
    """
    One-hot encode labels on GPU. Output shape (B, num_classes) with on_value at the
    label index and off_value elsewhere (default 1 and -1 for MSE).

    Args:
        labels: Tensor of shape (B,) float32 (class indices on GPU).
        num_classes: Number of classes.
        on_value: Value at the correct class index (default 1.0).
        off_value: Value elsewhere (default -1.0 for MSE).

    Returns:
        Tensor of shape (B, num_classes), float32, no requires_grad.
    """
    if not isinstance(labels, Tensor):
        raise ValueError(
            f"one_hot labels must be a Tensor of shape (B,) float32, got {type(labels).__name__}"
        )
    labels_t = labels
    B = labels_t.shape[0]
    if len(labels_t.shape) != 1:
        raise ValueError(f"one_hot labels must be 1D, got shape {labels_t.shape}")
    out = Tensor._zeros((B, num_classes), requires_grad=False)
    grid = (num_classes, B)
    one_hot_kernel[grid](
        labels_t.ptr,
        out.ptr,
        B=B,
        num_classes=num_classes,
        ON_VALUE=on_value,
        OFF_VALUE=off_value,
    )
    return out


# -------------------------------------------------------------------------
# Gradient management functions
# -------------------------------------------------------------------------


def zero_grad(params: list[Tensor]) -> None:
    """
    Zero out gradients for all parameters.

    For each parameter:
    - If grad is None, allocate a zero-filled gradient tensor
    - If grad exists, zero it with cudaMemset (faster than many small Triton kernels)

    This should be called before loss.backward() in training loops.

    Args:
        params: List of parameter tensors to zero gradients for.
    """
    for p in params:
        if not p.requires_grad:
            continue

        if p.grad is None:
            # Allocate zero-initialized gradient
            p.grad = zeros(p.shape)
        else:
            # Zero existing gradient with cudaMemset (byte 0)
            cuda_memset(p.grad.ptr, 0, p.grad.nbytes)


def sgd_step(params: list[Tensor], lr: float) -> None:
    """
    Perform an SGD update step on parameters.

    Updates each parameter in-place: param -= lr * param.grad

    This uses a Triton kernel to keep all math on GPU.

    Args:
        params: List of parameter tensors to update.
        lr: Learning rate.

    Raises:
        RuntimeError: If any parameter's grad is None.
    """
    for p in params:
        if not p.requires_grad:
            continue

        if p.grad is None:
            raise RuntimeError(
                f"Cannot perform SGD step: gradient is None for tensor {p.name or 'unnamed'}. "
                "Call backward() before sgd_step()."
            )

        grid = lambda meta: (triton.cdiv(p.numel, meta["BLOCK"]),)
        sgd_update_kernel[grid](p.ptr, p.grad.ptr, lr, p.numel, BLOCK=256)
