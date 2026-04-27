"""
Triton kernels for 2D convolution via im2col + matmul.

Includes:
- im2col_kernel: Unfold input (N, C_in, H, W) to (N*H_out*W_out, C_in*kH*kW)
- col2im_kernel: Accumulate unfolded gradient back to (N, C_in, H, W) (backward)
"""

import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    col_ptr: tl.pointer_type(tl.float32),
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    kH: tl.int32,
    kW: tl.int32,
    stride_h: tl.int32,
    stride_w: tl.int32,
    pad_h: tl.int32,
    pad_w: tl.int32,
    H_out: tl.int32,
    W_out: tl.int32,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    """
    Im2col: unfold input (N, C, H, W) into (N*H_out*W_out, C*kH*kW).

    Each output row corresponds to one spatial position (n, h_out, w_out);
    the row contains the flattened patch from x at that position (with padding = 0).

    Args:
        x_ptr: Input (N, C, H, W) row-major.
        col_ptr: Output (N*H_out*W_out, C*kH*kW) row-major.
        N, C, H, W: Input dimensions.
        kH, kW: Kernel size.
        stride_h, stride_w: Stride.
        pad_h, pad_w: Padding (each side).
        H_out, W_out: Output spatial size.
        BLOCK_ROW, BLOCK_COL: Block sizes for grid.
    """
    num_rows = N * H_out * W_out
    num_cols = C * kH * kW

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    for i in range(BLOCK_ROW):
        for j in range(BLOCK_COL):
            row_idx = pid_row * BLOCK_ROW + i
            col_idx = pid_col * BLOCK_COL + j
            if row_idx >= num_rows or col_idx >= num_cols:
                continue

            # row_idx = n * (H_out*W_out) + h_out * W_out + w_out
            n = row_idx // (H_out * W_out)
            rest = row_idx % (H_out * W_out)
            h_out = rest // W_out
            w_out = rest % W_out

            # col_idx = c * (kH*kW) + kh * kW + kw
            c = col_idx // (kH * kW)
            rest_c = col_idx % (kH * kW)
            kh = rest_c // kW
            kw = rest_c % kW

            h_in = h_out * stride_h - pad_h + kh
            w_in = w_out * stride_w - pad_w + kw

            out_offset = row_idx * num_cols + col_idx
            if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                # x index: n*C*H*W + c*H*W + h_in*W + w_in
                x_offset = n * (C * H * W) + c * (H * W) + h_in * W + w_in
                val = tl.load(x_ptr + x_offset)
            else:
                val = 0.0
            tl.store(col_ptr + out_offset, val)


@triton.jit
def im2col_kernel_2d(
    x_ptr: tl.pointer_type(tl.float32),
    col_ptr: tl.pointer_type(tl.float32),
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    kH: tl.int32,
    kW: tl.int32,
    stride_h: tl.int32,
    stride_w: tl.int32,
    pad_h: tl.int32,
    pad_w: tl.int32,
    H_out: tl.int32,
    W_out: tl.int32,
    num_cols: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Im2col: 2D grid. pid(0)=row_idx, pid(1)=col_block. Vectorized over BLOCK cols.
    """
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    num_rows = N * H_out * W_out
    if row_idx >= num_rows:
        return

    n = row_idx // (H_out * W_out)
    rest = row_idx % (H_out * W_out)
    h_out = rest // W_out
    w_out = rest % W_out
    h_start = h_out * stride_h - pad_h
    w_start = w_out * stride_w - pad_w

    col_offs = col_block * BLOCK + tl.arange(0, BLOCK)
    mask = col_offs < num_cols

    c = col_offs // (kH * kW)
    rest_c = col_offs % (kH * kW)
    kh = rest_c // kW
    kw = rest_c % kW
    h_in = h_start + kh
    w_in = w_start + kw

    in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
    load_mask = mask & in_bounds
    val = tl.where(
        in_bounds,
        tl.load(
            x_ptr + n * (C * H * W) + c * (H * W) + h_in * W + w_in,
            mask=load_mask,
        ),
        0.0,
    )
    tl.store(col_ptr + row_idx * num_cols + col_offs, val, mask=mask)


@triton.jit
def col2im_kernel(
    col_ptr: tl.pointer_type(tl.float32),
    x_ptr: tl.pointer_type(tl.float32),
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    kH: tl.int32,
    kW: tl.int32,
    stride_h: tl.int32,
    stride_w: tl.int32,
    pad_h: tl.int32,
    pad_w: tl.int32,
    H_out: tl.int32,
    W_out: tl.int32,
    num_cols: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Col2im: 2D grid. Accumulate col into x; atomic add for overlapping positions.
    """
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    num_rows = N * H_out * W_out
    if row_idx >= num_rows:
        return

    n = row_idx // (H_out * W_out)
    rest = row_idx % (H_out * W_out)
    h_out = rest // W_out
    w_out = rest % W_out
    h_start = h_out * stride_h - pad_h
    w_start = w_out * stride_w - pad_w

    col_offs = col_block * BLOCK + tl.arange(0, BLOCK)
    mask = col_offs < num_cols

    c = col_offs // (kH * kW)
    rest_c = col_offs % (kH * kW)
    kh = rest_c // kW
    kw = rest_c % kW
    h_in = h_start + kh
    w_in = w_start + kw

    val = tl.load(
        col_ptr + row_idx * num_cols + col_offs,
        mask=mask,
        other=0.0,
    )
    in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
    x_offset = n * (C * H * W) + c * (H * W) + h_in * W + w_in
    tl.atomic_add(
        x_ptr + x_offset,
        tl.where(in_bounds, val, 0.0),
        mask=mask & in_bounds,
    )
