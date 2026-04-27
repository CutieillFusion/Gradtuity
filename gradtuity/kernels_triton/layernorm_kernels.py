"""
Triton kernels for LayerNorm over the last dimension.

- layernorm_fwd_kernel: y = (x - mean) * rstd * gamma + beta per row; saves xhat, rstd.
- layernorm_bwd_kernel: dx, dgamma, dbeta from dy, xhat, rstd, gamma (atomics for dgamma/dbeta).
"""

import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    gamma_ptr: tl.pointer_type(tl.float32),
    beta_ptr: tl.pointer_type(tl.float32),
    y_ptr: tl.pointer_type(tl.float32),
    xhat_ptr: tl.pointer_type(tl.float32),
    rstd_ptr: tl.pointer_type(tl.float32),
    N: tl.int32,
    H: tl.int32,
    eps: tl.float32,
    BLOCK_H: tl.constexpr,
):
    """
    LayerNorm forward over last dimension: one program per row.

    Per row: Welford mean/var, then rstd = 1/sqrt(var+eps), xhat = (x-mean)*rstd,
    y = xhat*gamma + beta. Saves xhat (N,H) and rstd (N,) for backward.

    Args:
        x_ptr: Input (N, H).
        gamma_ptr, beta_ptr: (H,) affine params.
        y_ptr: Output (N, H).
        xhat_ptr: Normalized (x-mean)*rstd (N, H), saved for backward.
        rstd_ptr: 1/sqrt(var+eps) per row (N,).
        N, H: Row and column size.
        eps: Epsilon for variance.
        BLOCK_H: Column block size.
    """
    row_idx = tl.program_id(0)

    if row_idx >= N:
        return

    # Welford: n, mean, M2 over row (n as float for Triton loop-carried consistency)
    n = 0.0
    mean = 0.0
    M2 = 0.0

    for col_start in range(0, H, BLOCK_H):
        col_offsets = col_start + tl.arange(0, BLOCK_H)
        mask = col_offsets < H
        indices = row_idx * H + col_offsets
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)

        # Count and local mean for this block
        block_count = tl.sum(tl.where(mask, 1.0, 0.0))
        block_sum = tl.sum(tl.where(mask, x, 0.0))
        block_mean = tl.where(block_count > 0, block_sum / block_count, 0.0)

        # Local M2 = sum((x - block_mean)^2) for masked elements
        diff = tl.where(mask, x - block_mean, 0.0)
        block_M2 = tl.sum(diff * diff)

        # Merge with running stats (Welford merge)
        n1 = n
        n2 = block_count
        n_new = n1 + n2
        delta = block_mean - mean
        mean = tl.where(
            n_new > 0,
            mean + delta * n2 / tl.where(n_new > 0, n_new, 1.0),
            mean,
        )
        M2 = tl.where(
            n_new > 0,
            M2 + block_M2 + delta * delta * n1 * n2 / n_new,
            M2,
        )
        n = tl.where(n_new > 0, n_new, n)

    var = M2 / tl.where(n > 0, n, 1.0)
    rstd_val = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row_idx, rstd_val)

    # Second pass: xhat = (x - mean) * rstd, y = xhat * gamma + beta
    for col_start in range(0, H, BLOCK_H):
        col_offsets = col_start + tl.arange(0, BLOCK_H)
        mask = col_offsets < H
        indices = row_idx * H + col_offsets
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.0)
        beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)

        xhat_val = tl.where(mask, (x - mean) * rstd_val, 0.0)
        y_val = tl.where(mask, xhat_val * gamma + beta, 0.0)

        tl.store(xhat_ptr + indices, xhat_val, mask=mask)
        tl.store(y_ptr + indices, y_val, mask=mask)


@triton.jit
def layernorm_bwd_kernel(
    dx_ptr: tl.pointer_type(tl.float32),
    dgamma_ptr: tl.pointer_type(tl.float32),
    dbeta_ptr: tl.pointer_type(tl.float32),
    dy_ptr: tl.pointer_type(tl.float32),
    xhat_ptr: tl.pointer_type(tl.float32),
    rstd_ptr: tl.pointer_type(tl.float32),
    gamma_ptr: tl.pointer_type(tl.float32),
    N: tl.int32,
    H: tl.int32,
    BLOCK_H: tl.constexpr,
):
    """
    LayerNorm backward: one program per row.

    dx = (1/H) * rstd * (H*dxhat - sum1 - xhat*sum2) with dxhat = dy*gamma.
    dgamma and dbeta via atomic_add (v1). dgamma_ptr and dbeta_ptr must be zero-initialized.

    Args:
        dx_ptr: Gradient w.r.t. x (N, H), written.
        dgamma_ptr: Gradient w.r.t. gamma (H), accumulated via atomics.
        dbeta_ptr: Gradient w.r.t. beta (H), accumulated via atomics.
        dy_ptr: Upstream gradient (N, H).
        xhat_ptr, rstd_ptr: Saved from forward.
        gamma_ptr: gamma (H).
        N, H: Shape.
        BLOCK_H: Column block size.
    """
    row_idx = tl.program_id(0)

    if row_idx >= N:
        return

    rstd_val = tl.load(rstd_ptr + row_idx)

    # sum1 = sum(dxhat), sum2 = sum(dxhat * xhat) where dxhat = dy * gamma
    sum1 = 0.0
    sum2 = 0.0
    for col_start in range(0, H, BLOCK_H):
        col_offsets = col_start + tl.arange(0, BLOCK_H)
        mask = col_offsets < H
        indices = row_idx * H + col_offsets
        dy = tl.load(dy_ptr + indices, mask=mask, other=0.0)
        xhat = tl.load(xhat_ptr + indices, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.0)
        dxhat = tl.where(mask, dy * gamma, 0.0)
        sum1 += tl.sum(dxhat, axis=0)
        sum2 += tl.sum(dxhat * xhat, axis=0)

    # dx = (1/H) * rstd * (H*dxhat - sum1 - xhat*sum2)
    inv_H = 1.0 / (H * 1.0)  # float divisor for Triton
    for col_start in range(0, H, BLOCK_H):
        col_offsets = col_start + tl.arange(0, BLOCK_H)
        mask = col_offsets < H
        indices = row_idx * H + col_offsets
        dy = tl.load(dy_ptr + indices, mask=mask, other=0.0)
        xhat = tl.load(xhat_ptr + indices, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.0)
        dxhat = tl.where(mask, dy * gamma, 0.0)
        dx_val = tl.where(
            mask,
            rstd_val * inv_H * (H * 1.0 * dxhat - sum1 - xhat * sum2),
            0.0,
        )
        tl.store(dx_ptr + indices, dx_val, mask=mask)

    # dgamma += dy * xhat, dbeta += dy (atomics per element)
    for col_start in range(0, H, BLOCK_H):
        col_offsets = col_start + tl.arange(0, BLOCK_H)
        mask = col_offsets < H
        indices = row_idx * H + col_offsets
        dy = tl.load(dy_ptr + indices, mask=mask, other=0.0)
        xhat = tl.load(xhat_ptr + indices, mask=mask, other=0.0)
        dg = tl.where(mask, dy * xhat, 0.0)
        db = tl.where(mask, dy, 0.0)
        tl.atomic_add(dgamma_ptr + col_offsets, dg, mask=mask)
        tl.atomic_add(dbeta_ptr + col_offsets, db, mask=mask)
