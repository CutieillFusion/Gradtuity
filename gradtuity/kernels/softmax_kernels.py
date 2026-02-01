"""
Triton kernels for softmax over the last dimension.

- softmax_forward_kernel: Numerically stable softmax per row (max-subtract then exp/sum).
- softmax_backward_kernel: dx = y * (dy - sum(dy * y)) per row.
"""

import triton
import triton.language as tl


@triton.jit
def softmax_forward_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    y_ptr: tl.pointer_type(tl.float32),
    rows: tl.int32,
    cols: tl.int32,
    BLOCK_COLS: tl.constexpr,
):
    """
    Softmax over last dimension: treat input as (rows, cols), softmax per row.

    One program per row. For each row: row_max, sum_exp, then y_j = exp(x_j - row_max) / sum_exp.

    Args:
        x_ptr: Input GPU pointer (float32*), shape (rows, cols).
        y_ptr: Output GPU pointer (float32*), shape (rows, cols).
        rows: Number of rows.
        cols: Number of columns (last dimension).
        BLOCK_COLS: Column block size (compile-time constant).
    """
    row_idx = tl.program_id(0)

    if row_idx >= rows:
        return

    # Row max (numerically stable)
    m = -float("inf")
    for col_start in range(0, cols, BLOCK_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < cols
        indices = row_idx * cols + col_offsets
        x = tl.load(x_ptr + indices, mask=mask, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        m = tl.maximum(m, local_max)

    # sumexp = sum(exp(x - m))
    sumexp = 0.0
    for col_start in range(0, cols, BLOCK_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < cols
        indices = row_idx * cols + col_offsets
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)
        exp_val = tl.where(mask, tl.exp(x - m), 0.0)
        sumexp += tl.sum(exp_val, axis=0)

    # Write y = exp(x - m) / sumexp
    for col_start in range(0, cols, BLOCK_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < cols
        indices = row_idx * cols + col_offsets
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)
        y_val = tl.where(mask, tl.exp(x - m) / sumexp, 0.0)
        tl.store(y_ptr + indices, y_val, mask=mask)


@triton.jit
def softmax_backward_kernel(
    dx_ptr: tl.pointer_type(tl.float32),
    dy_ptr: tl.pointer_type(tl.float32),
    y_ptr: tl.pointer_type(tl.float32),
    rows: tl.int32,
    cols: tl.int32,
    BLOCK_COLS: tl.constexpr,
):
    """
    Softmax backward: dx = y * (dy - sum(dy * y)) per row.

    One program per row. Reads y from forward output.

    Args:
        dx_ptr: Gradient w.r.t. input (float32*), written (not accumulated).
        dy_ptr: Upstream gradient (float32*).
        y_ptr: Forward output softmax (float32*).
        rows: Number of rows.
        cols: Number of columns.
        BLOCK_COLS: Column block size (compile-time constant).
    """
    row_idx = tl.program_id(0)

    if row_idx >= rows:
        return

    # dot = sum(dy * y) for this row
    dot = 0.0
    for col_start in range(0, cols, BLOCK_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < cols
        indices = row_idx * cols + col_offsets
        dy = tl.load(dy_ptr + indices, mask=mask, other=0.0)
        y = tl.load(y_ptr + indices, mask=mask, other=0.0)
        dot += tl.sum(tl.where(mask, dy * y, 0.0), axis=0)

    # dx_j = y_j * (dy_j - dot)
    for col_start in range(0, cols, BLOCK_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < cols
        indices = row_idx * cols + col_offsets
        dy = tl.load(dy_ptr + indices, mask=mask, other=0.0)
        y = tl.load(y_ptr + indices, mask=mask, other=0.0)
        dx_val = tl.where(mask, y * (dy - dot), 0.0)
        tl.store(dx_ptr + indices, dx_val, mask=mask)
