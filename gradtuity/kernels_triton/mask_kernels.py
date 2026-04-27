"""
Triton kernels for causal attention masking and 4D transpose.

- transpose4d_12_kernel: Swap axes 1 and 2 of a 4D tensor (B, A, C, D) -> (B, C, A, D).
- causal_mask_inplace_kernel: Set upper-triangular positions to NEG_INF (in-place).
- causal_mask_backward_kernel: Zero gradients at masked positions (j > i).
"""

import triton
import triton.language as tl


@triton.jit
def transpose4d_12_kernel(
    src_ptr: tl.pointer_type(tl.float32),
    dst_ptr: tl.pointer_type(tl.float32),
    B: tl.int32,
    A: tl.int32,
    C: tl.int32,
    D: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Transpose 4D tensor: dst[b, c, a, d] = src[b, a, c, d].
    Swaps axes 1 and 2. Layout: row-major contiguous.

    Grid: 1D over total elements. Each program handles BLOCK elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    numel = B * A * C * D
    mask = offsets < numel

    # Decode flat index into (b, a, c, d); flat = b*(A*C*D) + a*(C*D) + c*D + d
    d = offsets % D
    idx = offsets // D
    c = idx % C
    idx = idx // C
    a = idx % A
    b = idx // A

    # Source index: src[b, a, c, d]
    src_idx = b * (A * C * D) + a * (C * D) + c * D + d
    val = tl.load(src_ptr + src_idx, mask=mask)

    # Destination index: dst[b, c, a, d]
    dst_idx = b * (C * A * D) + c * (A * D) + a * D + d
    tl.store(dst_ptr + dst_idx, val, mask=mask)


@triton.jit
def causal_mask_inplace_kernel(
    scores_ptr: tl.pointer_type(tl.float32),
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,
    NEG_INF: tl.float32,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """
    In-place causal mask: for each (b, h, i, j) set scores[b,h,i,j] = NEG_INF if j > i.

    scores_ptr: 4D (B, H, S, S) contiguous.
    Grid: (B*H, cdiv(S, BLOCK_I), cdiv(S, BLOCK_J)).
    """
    bh = tl.program_id(0)
    tile_i = tl.program_id(1)
    tile_j = tl.program_id(2)

    if bh >= B * H:
        return

    i_offsets = tile_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = tile_j * BLOCK_J + tl.arange(0, BLOCK_J)

    i_mask = i_offsets < S
    j_mask = j_offsets < S
    # (BLOCK_I, BLOCK_J) tile mask
    mask_ij = tl.expand_dims(i_mask, 1) & tl.expand_dims(j_mask, 0)

    # (BLOCK_I, BLOCK_J) tile of indices
    i_vals = tl.expand_dims(i_offsets, 1)
    j_vals = tl.expand_dims(j_offsets, 0)
    causal_mask = j_vals > i_vals  # j > i -> mask out

    # Linear offset for this (b,h) slice: bh * S * S
    base = bh * S * S
    # For each (i, j) in tile: index = base + i * S + j
    indices = base + tl.expand_dims(i_offsets, 1) * S + tl.expand_dims(j_offsets, 0)
    load_mask = indices < (B * H * S * S)  # redundant but safe
    load_mask = load_mask & mask_ij

    vals = tl.load(scores_ptr + indices, mask=load_mask, other=0.0)
    new_vals = tl.where(causal_mask & mask_ij, NEG_INF, vals)
    tl.store(scores_ptr + indices, new_vals, mask=load_mask)


@triton.jit
def causal_mask_backward_kernel(
    dout_ptr: tl.pointer_type(tl.float32),
    dscores_ptr: tl.pointer_type(tl.float32),
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Backward of causal mask: dscores[b,h,i,j] = dout[b,h,i,j] if j <= i else 0.

    Grid: 1D over B*H*S*S elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    numel = B * H * S * S
    mask = offsets < numel

    # Within each (S, S) matrix: offset % (S*S) = i*S + j, so j = rest % S, i = rest // S
    rest = offsets % (S * S)
    j = rest % S
    i = rest // S

    dout_val = tl.load(dout_ptr + offsets, mask=mask, other=0.0)
    # j <= i: pass through; j > i: zero
    write_val = tl.where(j <= i, dout_val, 0.0)
    tl.store(dscores_ptr + offsets, write_val, mask=mask)
