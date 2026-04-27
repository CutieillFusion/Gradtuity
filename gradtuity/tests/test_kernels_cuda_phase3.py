"""
Phase 3 parity tests: softmax, layernorm, cross-entropy, maxpool.

Looser tolerances than Phase 2 because these all involve reductions over
floating-point values and the two backends use different reduction order
(Triton tile reduction vs CUDA shared-mem tree).
"""

import math
import struct

import pytest
import triton

from ._helpers import alloc_floats, alloc_zeros, close_enough, max_abs_diff, read_floats

from gradtuity.cuda_mem import cuda_free, cuda_malloc, cuda_memset
from gradtuity.tensor import grid1d

pytestmark = [pytest.mark.requires_cuda, pytest.mark.kernel_backend_agnostic]


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------
class TestSoftmax:
    BLOCK_COLS = 128

    def _run(self, rows, cols, T_kernel, C_kernel, **kwargs):
        N = rows * cols
        x_vals = [(((i * 37) % 97) - 50) / 20.0 for i in range(N)]
        x = alloc_floats(x_vals)
        y_t, y_c = cuda_malloc(N * 4), cuda_malloc(N * 4)
        T_kernel[(rows,)](x, y_t, rows, cols, BLOCK_COLS=self.BLOCK_COLS)
        C_kernel((rows,), x, y_c, rows, cols, self.BLOCK_COLS)
        return x, y_t, y_c, N

    def test_forward(self):
        from gradtuity.kernels_triton.softmax_kernels import softmax_forward_kernel as TK
        from gradtuity.kernels_cuda.softmax_kernels import softmax_forward_kernel_launch as CK
        x, y_t, y_c, N = self._run(8, 64, TK, CK)
        assert close_enough(read_floats(y_t, N), read_floats(y_c, N), rtol=1e-5, atol=1e-6)
        # Each row sums to 1 (within float)
        out_c = read_floats(y_c, N)
        for r in range(8):
            row_sum = sum(out_c[r * 64:(r + 1) * 64])
            assert abs(row_sum - 1.0) < 1e-5
        for p in (x, y_t, y_c): cuda_free(p)

    def test_backward(self):
        from gradtuity.kernels_triton.softmax_kernels import (
            softmax_backward_kernel as TK_b,
            softmax_forward_kernel as TK_f,
        )
        from gradtuity.kernels_cuda.softmax_kernels import (
            softmax_backward_kernel_launch as CK_b,
            softmax_forward_kernel_launch as CK_f,
        )
        rows, cols = 8, 64
        N = rows * cols
        x_vals = [(((i * 37) % 97) - 50) / 20.0 for i in range(N)]
        dy_vals = [(((i * 11) % 31) - 15) / 50.0 for i in range(N)]
        x = alloc_floats(x_vals)
        dy = alloc_floats(dy_vals)
        # Use Triton's softmax forward to produce y; this tests backward in isolation.
        y = cuda_malloc(N * 4)
        TK_f[(rows,)](x, y, rows, cols, BLOCK_COLS=self.BLOCK_COLS)
        dx_t, dx_c = cuda_malloc(N * 4), cuda_malloc(N * 4)
        TK_b[(rows,)](dx_t, dy, y, rows, cols, BLOCK_COLS=self.BLOCK_COLS)
        CK_b((rows,), dx_c, dy, y, rows, cols, self.BLOCK_COLS)
        assert close_enough(read_floats(dx_t, N), read_floats(dx_c, N), rtol=1e-5, atol=1e-6)
        for p in (x, dy, y, dx_t, dx_c): cuda_free(p)

    def test_with_causal_mask_forward(self):
        from gradtuity.kernels_triton.softmax_kernels import softmax_with_causal_mask_forward_kernel as TK
        from gradtuity.kernels_cuda.softmax_kernels import softmax_with_causal_mask_forward_kernel_launch as CK
        # Caller passes rows = B*H*S, cols = S.
        B, H, S = 1, 2, 16
        rows = B * H * S
        cols = S
        N = rows * cols
        x_vals = [(((i * 7) % 41) - 20) / 10.0 for i in range(N)]
        x = alloc_floats(x_vals)
        y_t, y_c = cuda_malloc(N * 4), cuda_malloc(N * 4)
        TK[(rows,)](x, y_t, rows, cols, BLOCK_COLS=self.BLOCK_COLS)
        CK((rows,), x, y_c, rows, cols, self.BLOCK_COLS)
        assert close_enough(read_floats(y_t, N), read_floats(y_c, N), rtol=1e-5, atol=1e-6)
        # Sanity: positions j > i within each (S,S) block must be exactly 0.
        out_c = read_floats(y_c, N)
        for r in range(rows):
            i_in_block = r % S
            for j in range(S):
                if j > i_in_block:
                    assert out_c[r * S + j] == 0.0
        for p in (x, y_t, y_c): cuda_free(p)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------
class TestLayerNorm:
    BLOCK_H = 64

    def test_forward(self):
        from gradtuity.kernels_triton.layernorm_kernels import layernorm_fwd_kernel as TK
        from gradtuity.kernels_cuda.layernorm_kernels import layernorm_fwd_kernel_launch as CK
        N, H = 8, 64
        eps = 1e-5
        x_vals = [(((i * 23) % 73) - 36) / 10.0 for i in range(N * H)]
        gamma_vals = [1.0 + 0.01 * i for i in range(H)]
        beta_vals = [0.5 - 0.02 * i for i in range(H)]
        x = alloc_floats(x_vals)
        gamma, beta = alloc_floats(gamma_vals), alloc_floats(beta_vals)
        y_t, y_c = cuda_malloc(N * H * 4), cuda_malloc(N * H * 4)
        xh_t, xh_c = cuda_malloc(N * H * 4), cuda_malloc(N * H * 4)
        rstd_t, rstd_c = cuda_malloc(N * 4), cuda_malloc(N * 4)
        TK[(N,)](x, gamma, beta, y_t, xh_t, rstd_t, N, H, eps, BLOCK_H=self.BLOCK_H)
        CK((N,), x, gamma, beta, y_c, xh_c, rstd_c, N, H, eps, self.BLOCK_H)
        # Welford vs 2-pass differ slightly numerically; ~1e-4 abs is plenty.
        assert close_enough(read_floats(y_t, N * H), read_floats(y_c, N * H), 1e-4, 1e-5)
        assert close_enough(read_floats(xh_t, N * H), read_floats(xh_c, N * H), 1e-4, 1e-5)
        assert close_enough(read_floats(rstd_t, N), read_floats(rstd_c, N), 1e-4, 1e-5)
        # Sanity: each row of xhat has ~0 mean and ~1 var.
        xh_vals = read_floats(xh_c, N * H)
        for r in range(N):
            row = xh_vals[r * H:(r + 1) * H]
            mean = sum(row) / H
            var = sum((v - mean) ** 2 for v in row) / H
            assert abs(mean) < 1e-4
            assert abs(var - 1.0) < 1e-3
        for p in (x, gamma, beta, y_t, y_c, xh_t, xh_c, rstd_t, rstd_c):
            cuda_free(p)

    def test_backward(self):
        from gradtuity.kernels_triton.layernorm_kernels import (
            layernorm_bwd_kernel as TK_b,
            layernorm_fwd_kernel as TK_f,
        )
        from gradtuity.kernels_cuda.layernorm_kernels import (
            layernorm_bwd_kernel_launch as CK_b,
        )
        N, H = 8, 64
        eps = 1e-5
        x_vals = [(((i * 23) % 73) - 36) / 10.0 for i in range(N * H)]
        dy_vals = [(((i * 11) % 31) - 15) / 50.0 for i in range(N * H)]
        gamma_vals = [1.0 + 0.01 * i for i in range(H)]
        beta_vals = [0.5 - 0.02 * i for i in range(H)]
        x = alloc_floats(x_vals)
        dy = alloc_floats(dy_vals)
        gamma, beta = alloc_floats(gamma_vals), alloc_floats(beta_vals)
        y = cuda_malloc(N * H * 4)
        xhat = cuda_malloc(N * H * 4)
        rstd = cuda_malloc(N * 4)
        # Use Triton fwd to produce a shared (xhat, rstd) so backward is what we're testing.
        TK_f[(N,)](x, gamma, beta, y, xhat, rstd, N, H, eps, BLOCK_H=self.BLOCK_H)
        dx_t, dx_c = cuda_malloc(N * H * 4), cuda_malloc(N * H * 4)
        dg_t, dg_c = alloc_zeros(H), alloc_zeros(H)
        db_t, db_c = alloc_zeros(H), alloc_zeros(H)
        TK_b[(N,)](dx_t, dg_t, db_t, dy, xhat, rstd, gamma, N, H, BLOCK_H=self.BLOCK_H)
        CK_b((N,), dx_c, dg_c, db_c, dy, xhat, rstd, gamma, N, H, self.BLOCK_H)
        # Atomics on dgamma/dbeta - tolerance for reduction order.
        assert close_enough(read_floats(dx_t, N * H), read_floats(dx_c, N * H), 1e-4, 1e-5)
        assert close_enough(read_floats(dg_t, H), read_floats(dg_c, H), 1e-4, 1e-5)
        assert close_enough(read_floats(db_t, H), read_floats(db_c, H), 1e-4, 1e-5)
        for p in (x, dy, gamma, beta, y, xhat, rstd, dx_t, dx_c, dg_t, dg_c, db_t, db_c):
            cuda_free(p)


# ---------------------------------------------------------------------------
# Cross-entropy
# ---------------------------------------------------------------------------
class TestCrossEntropy:
    BLOCK_C = 64

    def test_forward(self):
        from gradtuity.kernels_triton.loss_kernels import cross_entropy_forward_kernel as TK
        from gradtuity.kernels_cuda.loss_kernels import cross_entropy_forward_kernel_launch as CK
        B, C = 16, 32
        logit_vals = [(((i * 17) % 53) - 26) / 10.0 for i in range(B * C)]
        target_vals = [float(i % C) for i in range(B)]
        logits = alloc_floats(logit_vals)
        targets = alloc_floats(target_vals)
        out_t, out_c = alloc_zeros(1), alloc_zeros(1)
        TK[(B,)](logits, targets, out_t, B, C, BLOCK_C=self.BLOCK_C)
        CK((B,), logits, targets, out_c, B, C, self.BLOCK_C)
        rt, rc = read_floats(out_t, 1)[0], read_floats(out_c, 1)[0]
        assert abs(rt - rc) < 1e-3 + 1e-4 * abs(rt)
        for p in (logits, targets, out_t, out_c): cuda_free(p)

    def test_backward(self):
        from gradtuity.kernels_triton.loss_kernels import cross_entropy_backward_kernel as TK
        from gradtuity.kernels_cuda.loss_kernels import cross_entropy_backward_kernel_launch as CK
        B, C = 16, 32
        N = B * C
        logit_vals = [(((i * 17) % 53) - 26) / 10.0 for i in range(N)]
        target_vals = [float(i % C) for i in range(B)]
        logits = alloc_floats(logit_vals)
        targets = alloc_floats(target_vals)
        seed = [0.01 * i for i in range(N)]
        dl_t, dl_c = alloc_floats(seed), alloc_floats(seed)
        TK[(B,)](dl_t, logits, targets, 1.0 / B, B, C, BLOCK_C=self.BLOCK_C)
        CK((B,), dl_c, logits, targets, 1.0 / B, B, C, self.BLOCK_C)
        assert close_enough(read_floats(dl_t, N), read_floats(dl_c, N), 1e-5, 1e-6)
        for p in (logits, targets, dl_t, dl_c): cuda_free(p)


# ---------------------------------------------------------------------------
# MaxPool
# ---------------------------------------------------------------------------
class TestMaxPool:
    BLOCK_ELEMS = 128

    def test_forward(self):
        from gradtuity.kernels_triton.pool_kernels import maxpool2d_forward_kernel as TK
        from gradtuity.kernels_cuda.pool_kernels import maxpool2d_forward_kernel_launch as CK
        N_, C_, H, W = 1, 2, 8, 8
        kH, kW = 2, 2
        sh, sw = 2, 2
        H_out = (H - kH) // sh + 1
        W_out = (W - kW) // sw + 1
        numel_out = N_ * C_ * H_out * W_out
        x_vals = [float(i % 13) for i in range(N_ * C_ * H * W)]
        x = alloc_floats(x_vals)
        out_t, out_c = cuda_malloc(numel_out * 4), cuda_malloc(numel_out * 4)
        idx_t, idx_c = cuda_malloc(numel_out * 4), cuda_malloc(numel_out * 4)
        grid_size = triton.cdiv(numel_out, self.BLOCK_ELEMS)
        TK[(grid_size,)](
            x, out_t, idx_t, N=N_, C=C_, H=H, W=W,
            H_out=H_out, W_out=W_out, stride_h=sh, stride_w=sw,
            BLOCK_KH=kH, BLOCK_KW=kW, BLOCK_ELEMS=self.BLOCK_ELEMS,
        )
        CK(
            (grid_size,), x, out_c, idx_c, N_, C_, H, W,
            H_out, W_out, sh, sw, kH, kW, self.BLOCK_ELEMS,
        )
        assert read_floats(out_t, numel_out) == read_floats(out_c, numel_out)
        # Max-index may differ if there are ties (different scan order); check
        # that the indexed value matches the max either way.
        idx_t_v, idx_c_v = read_floats(idx_t, numel_out), read_floats(idx_c, numel_out)
        # Both implementations pick the FIRST occurrence in row-major scan, so
        # they should agree exactly.
        assert idx_t_v == idx_c_v
        for p in (x, out_t, out_c, idx_t, idx_c): cuda_free(p)

    def test_backward(self):
        from gradtuity.kernels_triton.pool_kernels import (
            maxpool2d_backward_kernel as TK_b,
            maxpool2d_forward_kernel as TK_f,
        )
        from gradtuity.kernels_cuda.pool_kernels import maxpool2d_backward_kernel_launch as CK_b
        N_, C_, H, W = 1, 2, 8, 8
        kH, kW = 2, 2
        sh, sw = 2, 2
        H_out = (H - kH) // sh + 1
        W_out = (W - kW) // sw + 1
        numel_out = N_ * C_ * H_out * W_out
        numel_in = N_ * C_ * H * W
        x_vals = [float(i % 13) for i in range(numel_in)]
        x = alloc_floats(x_vals)
        out = cuda_malloc(numel_out * 4)
        idx = cuda_malloc(numel_out * 4)
        grid_size = triton.cdiv(numel_out, self.BLOCK_ELEMS)
        # Use Triton fwd to produce shared (out, idx).
        TK_f[(grid_size,)](
            x, out, idx, N=N_, C=C_, H=H, W=W,
            H_out=H_out, W_out=W_out, stride_h=sh, stride_w=sw,
            BLOCK_KH=kH, BLOCK_KW=kW, BLOCK_ELEMS=self.BLOCK_ELEMS,
        )
        grad_out_vals = [0.1 * i + 0.5 for i in range(numel_out)]
        grad_out = alloc_floats(grad_out_vals)
        gi_t, gi_c = alloc_zeros(numel_in), alloc_zeros(numel_in)
        TK_b[(grid_size,)](
            grad_out, idx, gi_t, N=N_, C=C_, H=H, W=W,
            H_out=H_out, W_out=W_out, stride_h=sh, stride_w=sw,
            BLOCK_KW=kW, BLOCK_ELEMS=self.BLOCK_ELEMS,
        )
        CK_b(
            (grid_size,), grad_out, idx, gi_c, N_, C_, H, W,
            H_out, W_out, sh, sw, kW, self.BLOCK_ELEMS,
        )
        assert close_enough(read_floats(gi_t, numel_in), read_floats(gi_c, numel_in), 1e-5, 1e-6)
        for p in (x, out, idx, grad_out, gi_t, gi_c): cuda_free(p)
