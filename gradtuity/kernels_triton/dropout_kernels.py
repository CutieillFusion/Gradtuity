"""
Triton kernels for dropout (inverted dropout with deterministic RNG).

- dropout_forward_kernel: y = x * mask / (1-p), mask ~ Bernoulli(1-p) from (seed, offset)
- dropout_backward_kernel: dx = dy * mask / (1-p), same mask regenerated
"""

import triton
import triton.language as tl


@triton.jit
def dropout_forward_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    y_ptr: tl.pointer_type(tl.float32),
    n: tl.int32,
    p: tl.float32,
    seed: tl.int32,
    offset: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Inverted dropout forward: y = x * mask / (1-p), mask = (r >= p) with r ~ U(0,1).

    Args:
        x_ptr: Input tensor (float32*).
        y_ptr: Output tensor (float32*).
        n: Total number of elements.
        p: Drop probability (keep with prob 1-p).
        seed: RNG seed.
        offset: Base offset for this dropout call (counter value at call time).
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    tensor_offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tensor_offsets < n
    rng_offsets = offset + tensor_offsets

    x = tl.load(x_ptr + tensor_offsets, mask=mask, other=0.0)
    r = tl.rand(seed, rng_offsets)
    keep = r >= p
    scale = 1.0 / (1.0 - p)
    y = tl.where(keep, x * scale, 0.0)
    tl.store(y_ptr + tensor_offsets, y, mask=mask)


@triton.jit
def dropout_backward_kernel(
    dx_ptr: tl.pointer_type(tl.float32),
    dy_ptr: tl.pointer_type(tl.float32),
    n: tl.int32,
    p: tl.float32,
    seed: tl.int32,
    offset: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Dropout backward: dx = dy * mask / (1-p) with same mask as forward.

    No dependency on x; mask is regenerated from (seed, offset).
    Caller must ensure dx_ptr is zero-initialized if accumulating.

    Args:
        dx_ptr: Gradient for input (float32*), written (or accumulated).
        dy_ptr: Upstream gradient (float32*).
        n: Total number of elements.
        p: Drop probability.
        seed: RNG seed (same as forward).
        offset: Base offset (same as forward).
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    tensor_offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tensor_offsets < n
    rng_offsets = offset + tensor_offsets

    dy = tl.load(dy_ptr + tensor_offsets, mask=mask, other=0.0)
    r = tl.rand(seed, rng_offsets)
    keep = r >= p
    scale = 1.0 / (1.0 - p)
    dx_val = tl.where(keep, dy * scale, 0.0)
    # Accumulate into dx (grad may already exist from other uses)
    dx_old = tl.load(dx_ptr + tensor_offsets, mask=mask, other=0.0)
    tl.store(dx_ptr + tensor_offsets, dx_old + dx_val, mask=mask)
