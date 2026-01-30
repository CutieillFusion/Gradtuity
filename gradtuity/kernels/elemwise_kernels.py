"""
Triton kernels for elementwise operations.

Includes:
- add_kernel: Elementwise addition (C = A + B)
- relu_kernel: ReLU forward (Z = max(Y, 0))
- relu_backward_kernel: ReLU backward with accumulation (dY += dZ * (Y > 0))
- add_inplace_kernel: In-place addition for gradient accumulation (A += B)
"""

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Elementwise addition: C = A + B

    Args:
        a_ptr: First input tensor GPU pointer (float32*).
        b_ptr: Second input tensor GPU pointer (float32*).
        c_ptr: Output tensor GPU pointer (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def relu_kernel(
    y_ptr: tl.pointer_type(tl.float32),
    z_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    ReLU forward: Z = max(Y, 0)

    Args:
        y_ptr: Input tensor GPU pointer (float32*).
        z_ptr: Output tensor GPU pointer (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.maximum(y, 0.0)

    tl.store(z_ptr + offsets, z, mask=mask)


@triton.jit
def relu_backward_kernel(
    dy_ptr: tl.pointer_type(tl.float32),
    dz_ptr: tl.pointer_type(tl.float32),
    y_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    ReLU backward with accumulation: dY += dZ * (Y > 0)

    Computes the gradient mask from the original input Y (not the output Z),
    since Z >= 0 always, so Z > 0 is not equivalent to Y > 0.

    Args:
        dy_ptr: Gradient w.r.t. input Y, accumulated in-place (float32*).
        dz_ptr: Gradient w.r.t. output Z (float32*).
        y_ptr: Original input Y for mask computation (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    # Load values
    dy = tl.load(dy_ptr + offsets, mask=mask)
    dz = tl.load(dz_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute relu mask: 1.0 where Y > 0, else 0.0
    relu_mask = (y > 0.0).to(tl.float32)

    # Accumulate: dY += dZ * mask
    dy = dy + dz * relu_mask

    tl.store(dy_ptr + offsets, dy, mask=mask)


@triton.jit
def add_inplace_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    In-place addition for gradient accumulation: A += B

    Args:
        a_ptr: Destination tensor GPU pointer (float32*), modified in-place.
        b_ptr: Source tensor GPU pointer (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    a = a + b

    tl.store(a_ptr + offsets, a, mask=mask)
