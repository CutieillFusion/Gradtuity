"""
Triton kernels for optimization and utility operations.

Includes:
- fill_kernel: Fill a tensor with a constant value
- sgd_update_kernel: In-place SGD parameter update
"""

import triton
import triton.language as tl


@triton.jit
def fill_kernel(
    dst_ptr: tl.pointer_type(tl.float32),
    value: tl.float32,
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Fill a GPU buffer with a constant value.

    Args:
        dst_ptr: Destination GPU pointer (float32*).
        value: The scalar value to fill with (float32).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    # Compute the block's starting position
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # Mask for out-of-bounds elements
    mask = offsets < numel

    # Store the value
    tl.store(dst_ptr + offsets, value, mask=mask)


@triton.jit
def sgd_update_kernel(
    param_ptr: tl.pointer_type(tl.float32),
    grad_ptr: tl.pointer_type(tl.float32),
    lr: tl.float32,
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    In-place SGD update: param -= lr * grad

    Args:
        param_ptr: Parameter GPU pointer (float32*).
        grad_ptr: Gradient GPU pointer (float32*).
        lr: Learning rate (float32).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    # Compute the block's starting position
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # Mask for out-of-bounds elements
    mask = offsets < numel

    # Load param and grad
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)

    # Update: param -= lr * grad
    param = param - lr * grad

    # Store back
    tl.store(param_ptr + offsets, param, mask=mask)
