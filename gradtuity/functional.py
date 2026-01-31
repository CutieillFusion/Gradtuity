"""
Tensor factory functions for creating tensors with specific values.

All functions create tensors on GPU using CUDA memory operations
and Triton kernels where appropriate.
"""

import random
import struct

import triton

from .cuda_mem import cuda_malloc, cuda_memset, cuda_memcpy_htod
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
    # Validate shape
    if len(shape) not in (1, 2, 3, 4):
        raise ValueError(f"Only rank 1, 2, 3, or 4 tensors supported, got rank {len(shape)}")

    # Compute total elements and bytes
    numel = 1
    for s in shape:
        numel *= s
    nbytes = numel * 4  # float32

    # Allocate and zero-initialize
    ptr = cuda_malloc(nbytes)
    cuda_memset(ptr, 0, nbytes)

    return Tensor._from_ptr(ptr, shape, owns_memory=True, requires_grad=requires_grad)


def zeros_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """
    Create a zero tensor with the same shape as the input.

    Args:
        t: Reference tensor for shape.
        requires_grad: Whether to track gradients.

    Returns:
        Tensor filled with zeros, same shape as t.
    """
    return zeros(t._shape, requires_grad=requires_grad)


def ones(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """
    Allocate a tensor filled with 1.0 via Triton fill kernel.

    Args:
        shape: Shape of the tensor (rank 1, 2, 3, or 4).
        requires_grad: Whether to track gradients.

    Returns:
        Tensor filled with ones.
    """
    # Start with zeros (allocates memory)
    t = zeros(shape, requires_grad=requires_grad)

    # Fill with 1.0 using Triton kernel
    from .kernels.optim_kernels import fill_kernel

    grid = lambda meta: (triton.cdiv(t._numel, meta["BLOCK"]),)
    fill_kernel[grid](t._ptr, 1.0, t._numel, BLOCK=256)

    return t


def ones_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """
    Create a tensor of ones with the same shape as the input.

    Args:
        t: Reference tensor for shape.
        requires_grad: Whether to track gradients.

    Returns:
        Tensor filled with ones, same shape as t.
    """
    return ones(t._shape, requires_grad=requires_grad)


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
        raise ValueError(f"Only rank 1, 2, 3, or 4 tensors supported, got rank {len(shape)}")

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
    from .kernels.optim_kernels import fill_kernel

    grid = lambda meta: (triton.cdiv(t._numel, meta["BLOCK"]),)
    fill_kernel[grid](t._ptr, fill_value, t._numel, BLOCK=256)

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
    return full(t._shape, fill_value, requires_grad=requires_grad)


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
    from .cuda_mem import cuda_memset

    for p in params:
        if not p.requires_grad:
            continue

        if p.grad is None:
            # Allocate zero-initialized gradient
            p.grad = zeros(p._shape)
        else:
            # Zero existing gradient with cudaMemset (byte 0)
            cuda_memset(p.grad._ptr, 0, p.grad._nbytes)


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
    from .kernels.optim_kernels import sgd_update_kernel

    for p in params:
        if not p.requires_grad:
            continue

        if p.grad is None:
            raise RuntimeError(
                f"Cannot perform SGD step: gradient is None for tensor {p.name or 'unnamed'}. "
                "Call backward() before sgd_step()."
            )

        grid = lambda meta: (triton.cdiv(p._numel, meta["BLOCK"]),)
        sgd_update_kernel[grid](p._ptr, p.grad._ptr, lr, p._numel, BLOCK=256)
