"""
Triton kernels for embedding row-gather and scatter-add backward.

- embedding_gather_kernel: out[n, d] = W[int(idx[n]), d]; weight (V, D), indices (N,), out (N, D).
- embedding_scatter_add_kernel: dW[row, d] += dOut[n, d] where row = idx[n]; uses atomic_add for accumulation.
"""

import triton
import triton.language as tl


@triton.jit
def embedding_gather_kernel(
    W_ptr: tl.pointer_type(tl.float32),
    idx_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    N: tl.int32,
    D: tl.int32,
    V: tl.int32,
    BLOCK_D: tl.constexpr,
):
    """
    Row gather: out[n, d] = W[int(idx[n]), d].

    Grid: (N, cdiv(D, BLOCK_D)). Each program handles one row n and a block of columns d.

    Args:
        W_ptr: Weight matrix (V, D) float32.
        idx_ptr: Indices (N,) float32, integer-valued.
        out_ptr: Output (N, D) float32.
        N, D, V: Dimensions.
        BLOCK_D: Column block size.
    """
    n = tl.program_id(0)
    d_block = tl.program_id(1)

    if n >= N:
        return

    row_val = tl.load(idx_ptr + n)
    row = tl.cast(row_val, tl.int32)

    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offsets < D

    # W[row, d] -> row-major: row * D + d_offsets
    w_offsets = row * D + d_offsets
    w_vals = tl.load(W_ptr + w_offsets, mask=mask, other=0.0)

    out_offsets = n * D + d_offsets
    tl.store(out_ptr + out_offsets, w_vals, mask=mask)


@triton.jit
def embedding_scatter_add_kernel(
    dOut_ptr: tl.pointer_type(tl.float32),
    idx_ptr: tl.pointer_type(tl.float32),
    dW_ptr: tl.pointer_type(tl.float32),
    N: tl.int32,
    D: tl.int32,
    V: tl.int32,
    BLOCK_D: tl.constexpr,
):
    """
    Scatter-add backward: for each n, add dOut[n, :] into dW[row, :] where row = idx[n].
    dW_ptr must be zero-initialized. Uses atomic_add for correct accumulation with repeated indices.

    Grid: (N, cdiv(D, BLOCK_D)). Each program handles one (n) and a block of d; atomic_add into dW.

    Args:
        dOut_ptr: Upstream gradient (N, D) float32.
        idx_ptr: Indices (N,) float32, integer-valued.
        dW_ptr: Gradient w.r.t. weight (V, D), accumulated via atomics.
        N, D, V: Dimensions.
        BLOCK_D: Column block size.
    """
    n = tl.program_id(0)
    d_block = tl.program_id(1)

    if n >= N:
        return

    row_val = tl.load(idx_ptr + n)
    row = tl.cast(row_val, tl.int32)
    in_bounds = (row >= 0) & (row < V)

    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offsets < D

    dout_offsets = n * D + d_offsets
    dout_vals = tl.load(dOut_ptr + dout_offsets, mask=mask, other=0.0)

    dw_offsets = row * D + d_offsets
    tl.atomic_add(dW_ptr + dw_offsets, dout_vals, mask=in_bounds & mask)
