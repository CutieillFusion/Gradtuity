"""
Parity tests: every CUDA kernel matches its Triton twin on the same inputs.

For each kernel we:
  1. Allocate identical inputs in two output buffers (or in-place targets).
  2. Run the Triton kernel into one buffer, the CUDA kernel into the other.
  3. Assert numerical equivalence (bit-exact for elementwise; small tolerance
     for transcendentals and atomic-reduction kernels).

Dropout uses a different RNG between backends, so it has its own statistical
checks (drop fraction + forward/backward mask consistency within a backend).
"""

import math
import struct

import pytest
import triton

from ._helpers import alloc_floats, alloc_zeros, close_enough, max_abs_diff, read_floats

from gradtuity.cuda_mem import cuda_free, cuda_malloc, cuda_memcpy_dtod, cuda_memset
from gradtuity.tensor import grid1d

pytestmark = [pytest.mark.requires_cuda, pytest.mark.kernel_backend_agnostic]


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------
class TestElemwise:
    BLOCK = 256
    N = 4096

    def _two_inputs(self):
        a_vals = [(i % 17) * 0.5 - 3.0 for i in range(self.N)]
        b_vals = [(i % 13) * 0.7 + 1.0 for i in range(self.N)]
        return a_vals, b_vals

    def test_add(self):
        from gradtuity.kernels_triton.elemwise_kernels import add_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import add_kernel_launch as CK
        a_vals, b_vals = self._two_inputs()
        a, b = alloc_floats(a_vals), alloc_floats(b_vals)
        c_t, c_c = cuda_malloc(self.N * 4), cuda_malloc(self.N * 4)
        TK[grid1d(self.N)](a, b, c_t, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), a, b, c_c, self.N, self.BLOCK)
        assert read_floats(c_t, self.N) == read_floats(c_c, self.N)
        for p in (a, b, c_t, c_c): cuda_free(p)

    def test_mul(self):
        from gradtuity.kernels_triton.elemwise_kernels import mul_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import mul_kernel_launch as CK
        a_vals, b_vals = self._two_inputs()
        a, b = alloc_floats(a_vals), alloc_floats(b_vals)
        c_t, c_c = cuda_malloc(self.N * 4), cuda_malloc(self.N * 4)
        TK[grid1d(self.N)](a, b, c_t, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), a, b, c_c, self.N, self.BLOCK)
        assert read_floats(c_t, self.N) == read_floats(c_c, self.N)
        for p in (a, b, c_t, c_c): cuda_free(p)

    def test_mul_scalar(self):
        from gradtuity.kernels_triton.elemwise_kernels import mul_scalar_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import mul_scalar_kernel_launch as CK
        a_vals, _ = self._two_inputs()
        a = alloc_floats(a_vals)
        c_t, c_c = cuda_malloc(self.N * 4), cuda_malloc(self.N * 4)
        scalar = 0.375
        TK[grid1d(self.N)](a, scalar, c_t, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), a, scalar, c_c, self.N, self.BLOCK)
        assert read_floats(c_t, self.N) == read_floats(c_c, self.N)
        for p in (a, c_t, c_c): cuda_free(p)

    def test_mul_scalar_inplace(self):
        from gradtuity.kernels_triton.elemwise_kernels import mul_scalar_inplace_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import mul_scalar_inplace_kernel_launch as CK
        a_vals, _ = self._two_inputs()
        x_t, x_c = alloc_floats(a_vals), alloc_floats(a_vals)
        TK[grid1d(self.N)](x_t, 0.5, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), x_c, 0.5, self.N, self.BLOCK)
        assert read_floats(x_t, self.N) == read_floats(x_c, self.N)
        for p in (x_t, x_c): cuda_free(p)

    def test_add_inplace(self):
        from gradtuity.kernels_triton.elemwise_kernels import add_inplace_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import add_inplace_kernel_launch as CK
        a_vals, b_vals = self._two_inputs()
        a_t, a_c = alloc_floats(a_vals), alloc_floats(a_vals)
        b = alloc_floats(b_vals)
        TK[grid1d(self.N)](a_t, b, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), a_c, b, self.N, self.BLOCK)
        assert read_floats(a_t, self.N) == read_floats(a_c, self.N)
        for p in (a_t, a_c, b): cuda_free(p)

    def test_mul_backward(self):
        from gradtuity.kernels_triton.elemwise_kernels import mul_backward_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import mul_backward_kernel_launch as CK
        a_vals, b_vals = self._two_inputs()
        out_grad = alloc_floats(a_vals)
        other = alloc_floats(b_vals)
        # Pre-existing grads (accumulation must be respected)
        seed = [0.1 * i for i in range(self.N)]
        g_t, g_c = alloc_floats(seed), alloc_floats(seed)
        TK[grid1d(self.N)](g_t, out_grad, other, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), g_c, out_grad, other, self.N, self.BLOCK)
        assert read_floats(g_t, self.N) == read_floats(g_c, self.N)
        for p in (out_grad, other, g_t, g_c): cuda_free(p)

    def test_scale_backward(self):
        from gradtuity.kernels_triton.elemwise_kernels import scale_backward_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import scale_backward_kernel_launch as CK
        a_vals, _ = self._two_inputs()
        out_grad = alloc_floats(a_vals)
        seed = [0.1 * i for i in range(self.N)]
        g_t, g_c = alloc_floats(seed), alloc_floats(seed)
        TK[grid1d(self.N)](g_t, out_grad, 0.25, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), g_c, out_grad, 0.25, self.N, self.BLOCK)
        assert read_floats(g_t, self.N) == read_floats(g_c, self.N)
        for p in (out_grad, g_t, g_c): cuda_free(p)

    def test_relu(self):
        from gradtuity.kernels_triton.elemwise_kernels import relu_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import relu_kernel_launch as CK
        a_vals = [(i % 7) - 3.0 for i in range(self.N)]
        y = alloc_floats(a_vals)
        z_t, z_c = cuda_malloc(self.N * 4), cuda_malloc(self.N * 4)
        TK[grid1d(self.N)](y, z_t, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), y, z_c, self.N, self.BLOCK)
        assert read_floats(z_t, self.N) == read_floats(z_c, self.N)
        for p in (y, z_t, z_c): cuda_free(p)

    def test_relu_backward(self):
        from gradtuity.kernels_triton.elemwise_kernels import relu_backward_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import relu_backward_kernel_launch as CK
        y_vals = [(i % 7) - 3.0 for i in range(self.N)]
        dz_vals = [(i % 11) * 0.3 for i in range(self.N)]
        seed = [0.1 * i for i in range(self.N)]
        y, dz = alloc_floats(y_vals), alloc_floats(dz_vals)
        dy_t, dy_c = alloc_floats(seed), alloc_floats(seed)
        TK[grid1d(self.N)](dy_t, dz, y, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), dy_c, dz, y, self.N, self.BLOCK)
        assert read_floats(dy_t, self.N) == read_floats(dy_c, self.N)
        for p in (y, dz, dy_t, dy_c): cuda_free(p)

    def test_relu_mask_mul(self):
        from gradtuity.kernels_triton.elemwise_kernels import relu_mask_mul_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import relu_mask_mul_kernel_launch as CK
        a_vals, b_vals = self._two_inputs()
        a = alloc_floats(a_vals)
        y = alloc_floats([(i % 7) - 3.0 for i in range(self.N)])
        c_t, c_c = cuda_malloc(self.N * 4), cuda_malloc(self.N * 4)
        TK[grid1d(self.N)](c_t, a, y, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), c_c, a, y, self.N, self.BLOCK)
        assert read_floats(c_t, self.N) == read_floats(c_c, self.N)
        for p in (a, y, c_t, c_c): cuda_free(p)

    def test_gelu(self):
        from gradtuity.kernels_triton.elemwise_kernels import gelu_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import gelu_kernel_launch as CK
        x_vals = [((i % 41) - 20) * 0.1 for i in range(self.N)]
        x = alloc_floats(x_vals)
        y_t, y_c = cuda_malloc(self.N * 4), cuda_malloc(self.N * 4)
        TK[grid1d(self.N)](x, y_t, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), x, y_c, self.N, self.BLOCK)
        # Transcendentals: fast-math approximations differ slightly between
        # NVRTC and Triton's compiler. 1e-5 absolute tolerance is generous.
        assert close_enough(read_floats(y_t, self.N), read_floats(y_c, self.N), rtol=1e-5, atol=1e-5)
        for p in (x, y_t, y_c): cuda_free(p)

    def test_gelu_backward(self):
        from gradtuity.kernels_triton.elemwise_kernels import gelu_backward_kernel as TK
        from gradtuity.kernels_cuda.elemwise_kernels import gelu_backward_kernel_launch as CK
        x_vals = [((i % 41) - 20) * 0.1 for i in range(self.N)]
        dy_vals = [(i % 13) * 0.07 - 0.5 for i in range(self.N)]
        x, dy = alloc_floats(x_vals), alloc_floats(dy_vals)
        dx_t, dx_c = cuda_malloc(self.N * 4), cuda_malloc(self.N * 4)
        TK[grid1d(self.N)](dx_t, dy, x, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), dx_c, dy, x, self.N, self.BLOCK)
        assert close_enough(read_floats(dx_t, self.N), read_floats(dx_c, self.N), rtol=1e-5, atol=1e-5)
        for p in (x, dy, dx_t, dx_c): cuda_free(p)


# ---------------------------------------------------------------------------
# Optim
# ---------------------------------------------------------------------------
class TestOptim:
    BLOCK = 256
    N = 1024

    def test_fill(self):
        from gradtuity.kernels_triton.optim_kernels import fill_kernel as TK
        from gradtuity.kernels_cuda.optim_kernels import fill_kernel_launch as CK
        d_t, d_c = cuda_malloc(self.N * 4), cuda_malloc(self.N * 4)
        TK[grid1d(self.N)](d_t, 3.5, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), d_c, 3.5, self.N, self.BLOCK)
        assert read_floats(d_t, self.N) == read_floats(d_c, self.N)
        for p in (d_t, d_c): cuda_free(p)

    def test_sgd_update(self):
        from gradtuity.kernels_triton.optim_kernels import sgd_update_kernel as TK
        from gradtuity.kernels_cuda.optim_kernels import sgd_update_kernel_launch as CK
        p_vals = [0.01 * i for i in range(self.N)]
        g_vals = [0.001 * (i % 17 - 8) for i in range(self.N)]
        p_t, p_c = alloc_floats(p_vals), alloc_floats(p_vals)
        g = alloc_floats(g_vals)
        TK[grid1d(self.N)](p_t, g, 0.05, self.N, BLOCK=self.BLOCK)
        CK(grid1d(self.N), p_c, g, 0.05, self.N, self.BLOCK)
        assert read_floats(p_t, self.N) == read_floats(p_c, self.N)
        for p in (p_t, p_c, g): cuda_free(p)

    def test_adamw_step(self):
        from gradtuity.kernels_triton.optim_kernels import adamw_step_kernel as TK
        from gradtuity.kernels_cuda.optim_kernels import adamw_step_kernel_launch as CK
        p_vals = [0.01 * (i % 19 - 9) for i in range(self.N)]
        g_vals = [0.005 * (i % 23 - 11) for i in range(self.N)]
        p_t, p_c = alloc_floats(p_vals), alloc_floats(p_vals)
        g = alloc_floats(g_vals)
        m_t, m_c = alloc_zeros(self.N), alloc_zeros(self.N)
        v_t, v_c = alloc_zeros(self.N), alloc_zeros(self.N)
        kw = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                  weight_decay=0.01, bc1=10.0, bc2=1000.0)
        TK[grid1d(self.N)](
            p_t, g, m_t, v_t, self.N,
            kw["lr"], kw["beta1"], kw["beta2"], kw["eps"],
            kw["weight_decay"], kw["bc1"], kw["bc2"],
            BLOCK=self.BLOCK,
        )
        CK(
            grid1d(self.N), p_c, g, m_c, v_c, self.N,
            kw["lr"], kw["beta1"], kw["beta2"], kw["eps"],
            kw["weight_decay"], kw["bc1"], kw["bc2"], self.BLOCK,
        )
        # sqrt is fast-math: tolerate a few ULPs.
        assert close_enough(read_floats(p_t, self.N), read_floats(p_c, self.N), 1e-6, 1e-7)
        assert read_floats(m_t, self.N) == read_floats(m_c, self.N)
        assert read_floats(v_t, self.N) == read_floats(v_c, self.N)
        for p in (p_t, p_c, g, m_t, m_c, v_t, v_c): cuda_free(p)


# ---------------------------------------------------------------------------
# One-hot
# ---------------------------------------------------------------------------
class TestOneHot:
    def test_one_hot(self):
        from gradtuity.kernels_triton.one_hot_kernels import one_hot_kernel as TK
        from gradtuity.kernels_cuda.one_hot_kernels import one_hot_kernel_launch as CK
        B, C = 8, 10
        labels = [float(i % C) for i in range(B)]
        lp = alloc_floats(labels)
        out_t, out_c = cuda_malloc(B * C * 4), cuda_malloc(B * C * 4)
        grid = (C, B)
        TK[grid](lp, out_t, B=B, num_classes=C, ON_VALUE=1.0, OFF_VALUE=-1.0)
        CK(grid, lp, out_c, B, C, 1.0, -1.0)
        assert read_floats(out_t, B * C) == read_floats(out_c, B * C)
        for p in (lp, out_t, out_c): cuda_free(p)


# ---------------------------------------------------------------------------
# Reductions / mask / transpose
# ---------------------------------------------------------------------------
class TestReduce:
    BLOCK = 256

    def test_sum_all(self):
        from gradtuity.kernels_triton.reduce_kernels import sum_all_kernel as TK
        from gradtuity.kernels_cuda.reduce_kernels import sum_all_kernel_launch as CK
        N = 4096
        x_vals = [0.001 * (i % 31) for i in range(N)]
        x = alloc_floats(x_vals)
        out_t, out_c = alloc_zeros(1), alloc_zeros(1)
        TK[grid1d(N)](x, out_t, N, BLOCK=self.BLOCK)
        CK(grid1d(N), x, out_c, N, self.BLOCK)
        # Different reduction order between Triton's tl.sum and our shared-mem
        # tree reduction; tolerate ~1e-3 relative on a sum of ~4k floats.
        rt, rc = read_floats(out_t, 1)[0], read_floats(out_c, 1)[0]
        assert abs(rt - rc) < 1e-3 + 1e-4 * abs(rt)
        for p in (x, out_t, out_c): cuda_free(p)

    def test_sum_axis0(self):
        from gradtuity.kernels_triton.reduce_kernels import sum_axis0_kernel as TK
        from gradtuity.kernels_cuda.reduce_kernels import sum_axis0_kernel_launch as CK
        rows, cols = 64, 32
        x_vals = [0.01 * ((i % 13) - 6) for i in range(rows * cols)]
        x = alloc_floats(x_vals)
        out_t, out_c = alloc_zeros(cols), alloc_zeros(cols)
        TK[(cols,)](x, out_t, rows, cols, BLOCK_ROWS=16)
        CK((cols,), x, out_c, rows, cols, 16)
        assert close_enough(read_floats(out_t, cols), read_floats(out_c, cols), 1e-5, 1e-6)
        for p in (x, out_t, out_c): cuda_free(p)

    def test_add_scalar_inplace(self):
        from gradtuity.kernels_triton.reduce_kernels import add_scalar_inplace_kernel as TK
        from gradtuity.kernels_cuda.reduce_kernels import add_scalar_inplace_kernel_launch as CK
        N = 1024
        x_vals = [0.1 * i for i in range(N)]
        scalar = alloc_floats([2.5])
        x_t, x_c = alloc_floats(x_vals), alloc_floats(x_vals)
        TK[grid1d(N)](x_t, scalar, N, BLOCK=self.BLOCK)
        CK(grid1d(N), x_c, scalar, N, self.BLOCK)
        assert read_floats(x_t, N) == read_floats(x_c, N)
        for p in (x_t, x_c, scalar): cuda_free(p)

    def test_add_bias(self):
        from gradtuity.kernels_triton.reduce_kernels import add_bias_kernel as TK
        from gradtuity.kernels_cuda.reduce_kernels import add_bias_kernel_launch as CK
        rows, cols = 32, 16
        N = rows * cols
        x_vals = [0.1 * (i % 17) for i in range(N)]
        b_vals = [0.5 * (i % 5) for i in range(cols)]
        x, b = alloc_floats(x_vals), alloc_floats(b_vals)
        y_t, y_c = cuda_malloc(N * 4), cuda_malloc(N * 4)
        TK[grid1d(N)](x, b, y_t, rows, cols, BLOCK=self.BLOCK)
        CK(grid1d(N), x, b, y_c, rows, cols, self.BLOCK)
        assert read_floats(y_t, N) == read_floats(y_c, N)
        for p in (x, b, y_t, y_c): cuda_free(p)

    def test_argmax_axis1(self):
        from gradtuity.kernels_triton.reduce_kernels import argmax_axis1_kernel as TK
        from gradtuity.kernels_cuda.reduce_kernels import argmax_axis1_kernel_launch as CK
        rows, cols = 16, 24
        x_vals = [(i * 13 + 7) % 97 / 100.0 for i in range(rows * cols)]
        x = alloc_floats(x_vals)
        out_t, out_c = cuda_malloc(rows * 4), cuda_malloc(rows * 4)
        TK[(rows,)](x, out_t, rows, cols, BLOCK_COLS=32)
        CK((rows,), x, out_c, rows, cols, 32)
        assert read_floats(out_t, rows) == read_floats(out_c, rows)
        for p in (x, out_t, out_c): cuda_free(p)


class TestMaskTranspose:
    BLOCK = 256

    def test_transpose2d(self):
        from gradtuity.kernels_triton.matmul_kernels import transpose2d_kernel as TK
        from gradtuity.kernels_cuda.matmul_kernels import transpose2d_kernel_launch as CK
        rows, cols = 17, 13  # deliberately non-square + non-aligned
        N = rows * cols
        x_vals = [float(i) for i in range(N)]
        x = alloc_floats(x_vals)
        d_t, d_c = cuda_malloc(N * 4), cuda_malloc(N * 4)
        TK[(triton.cdiv(N, self.BLOCK),)](x, d_t, rows, cols, BLOCK=self.BLOCK)
        CK((triton.cdiv(N, self.BLOCK),), x, d_c, rows, cols, self.BLOCK)
        assert read_floats(d_t, N) == read_floats(d_c, N)
        for p in (x, d_t, d_c): cuda_free(p)

    def test_transpose4d_12(self):
        from gradtuity.kernels_triton.mask_kernels import transpose4d_12_kernel as TK
        from gradtuity.kernels_cuda.mask_kernels import transpose4d_12_kernel_launch as CK
        B, A, C, D = 2, 3, 4, 5
        N = B * A * C * D
        x_vals = [float(i) for i in range(N)]
        x = alloc_floats(x_vals)
        d_t, d_c = cuda_malloc(N * 4), cuda_malloc(N * 4)
        TK[grid1d(N)](x, d_t, B, A, C, D, BLOCK=self.BLOCK)
        CK(grid1d(N), x, d_c, B, A, C, D, self.BLOCK)
        assert read_floats(d_t, N) == read_floats(d_c, N)
        for p in (x, d_t, d_c): cuda_free(p)

    def test_causal_mask_inplace(self):
        from gradtuity.kernels_triton.mask_kernels import causal_mask_inplace_kernel as TK
        from gradtuity.kernels_cuda.mask_kernels import causal_mask_inplace_kernel_launch as CK
        B, H, S = 2, 4, 16
        N = B * H * S * S
        x_vals = [0.1 * i for i in range(N)]
        x_t, x_c = alloc_floats(x_vals), alloc_floats(x_vals)
        BI = BJ = 8
        grid = (B * H, triton.cdiv(S, BI), triton.cdiv(S, BJ))
        TK[grid](x_t, B=B, H=H, S=S, NEG_INF=-1e9, BLOCK_I=BI, BLOCK_J=BJ)
        CK(grid, x_c, B, H, S, -1e9, BI, BJ)
        assert read_floats(x_t, N) == read_floats(x_c, N)
        for p in (x_t, x_c): cuda_free(p)

    def test_causal_mask_backward(self):
        from gradtuity.kernels_triton.mask_kernels import causal_mask_backward_kernel as TK
        from gradtuity.kernels_cuda.mask_kernels import causal_mask_backward_kernel_launch as CK
        B, H, S = 2, 4, 16
        N = B * H * S * S
        dout_vals = [0.01 * i for i in range(N)]
        dout = alloc_floats(dout_vals)
        d_t, d_c = cuda_malloc(N * 4), cuda_malloc(N * 4)
        TK[grid1d(N)](dout, d_t, B, H, S, BLOCK=self.BLOCK)
        CK(grid1d(N), dout, d_c, B, H, S, self.BLOCK)
        assert read_floats(d_t, N) == read_floats(d_c, N)
        for p in (dout, d_t, d_c): cuda_free(p)


# ---------------------------------------------------------------------------
# Gather / Conv / Loss
# ---------------------------------------------------------------------------
class TestGather:
    def test_embedding_gather(self):
        from gradtuity.kernels_triton.gather_kernels import embedding_gather_kernel as TK
        from gradtuity.kernels_cuda.gather_kernels import embedding_gather_kernel_launch as CK
        V, D, N = 32, 16, 24
        BLOCK_D = 16
        W_vals = [0.01 * i for i in range(V * D)]
        idx_vals = [float(i % V) for i in range(N)]
        W, idx = alloc_floats(W_vals), alloc_floats(idx_vals)
        out_t, out_c = cuda_malloc(N * D * 4), cuda_malloc(N * D * 4)
        grid = (N, triton.cdiv(D, BLOCK_D))
        TK[grid](W, idx, out_t, N, D, V, BLOCK_D=BLOCK_D)
        CK(grid, W, idx, out_c, N, D, V, BLOCK_D)
        assert read_floats(out_t, N * D) == read_floats(out_c, N * D)
        for p in (W, idx, out_t, out_c): cuda_free(p)

    def test_embedding_scatter_add(self):
        from gradtuity.kernels_triton.gather_kernels import embedding_scatter_add_kernel as TK
        from gradtuity.kernels_cuda.gather_kernels import embedding_scatter_add_kernel_launch as CK
        V, D, N = 32, 16, 24
        BLOCK_D = 16
        dOut_vals = [0.01 * (i + 1) for i in range(N * D)]
        # Use indices with collisions to exercise atomic_add.
        idx_vals = [float(i % 5) for i in range(N)]
        dOut, idx = alloc_floats(dOut_vals), alloc_floats(idx_vals)
        dW_t, dW_c = alloc_zeros(V * D), alloc_zeros(V * D)
        grid = (N, triton.cdiv(D, BLOCK_D))
        TK[grid](dOut, idx, dW_t, N, D, V, BLOCK_D=BLOCK_D)
        CK(grid, dOut, idx, dW_c, N, D, V, BLOCK_D)
        # Atomic order may differ — small tolerance.
        assert close_enough(read_floats(dW_t, V * D), read_floats(dW_c, V * D), 1e-5, 1e-6)
        for p in (dOut, idx, dW_t, dW_c): cuda_free(p)


class TestConv:
    def test_im2col_2d(self):
        from gradtuity.kernels_triton.conv_kernels import im2col_kernel_2d as TK
        from gradtuity.kernels_cuda.conv_kernels import im2col_kernel_2d_launch as CK
        N, C, H, W = 1, 2, 5, 5
        kH, kW = 3, 3
        sh, sw, ph, pw = 1, 1, 1, 1
        H_out, W_out = (H + 2 * ph - kH) // sh + 1, (W + 2 * pw - kW) // sw + 1
        num_cols = C * kH * kW
        num_rows = N * H_out * W_out
        BLOCK = 16
        x_vals = [float(i) for i in range(N * C * H * W)]
        x = alloc_floats(x_vals)
        col_t = cuda_malloc(num_rows * num_cols * 4)
        col_c = cuda_malloc(num_rows * num_cols * 4)
        grid = (num_rows, triton.cdiv(num_cols, BLOCK))
        TK[grid](
            x, col_t, N, C, H, W, kH, kW, sh, sw, ph, pw,
            H_out, W_out, num_cols, BLOCK=BLOCK,
        )
        CK(
            grid, x, col_c, N, C, H, W, kH, kW, sh, sw, ph, pw,
            H_out, W_out, num_cols, BLOCK,
        )
        assert read_floats(col_t, num_rows * num_cols) == read_floats(col_c, num_rows * num_cols)
        for p in (x, col_t, col_c): cuda_free(p)

    def test_col2im(self):
        from gradtuity.kernels_triton.conv_kernels import col2im_kernel as TK
        from gradtuity.kernels_cuda.conv_kernels import col2im_kernel_launch as CK
        N, C, H, W = 1, 2, 5, 5
        kH, kW = 3, 3
        sh, sw, ph, pw = 1, 1, 1, 1
        H_out, W_out = (H + 2 * ph - kH) // sh + 1, (W + 2 * pw - kW) // sw + 1
        num_cols = C * kH * kW
        num_rows = N * H_out * W_out
        BLOCK = 16
        col_vals = [0.01 * i for i in range(num_rows * num_cols)]
        col = alloc_floats(col_vals)
        x_t, x_c = alloc_zeros(N * C * H * W), alloc_zeros(N * C * H * W)
        grid = (num_rows, triton.cdiv(num_cols, BLOCK))
        TK[grid](
            col, x_t, N, C, H, W, kH, kW, sh, sw, ph, pw,
            H_out, W_out, num_cols, BLOCK=BLOCK,
        )
        CK(
            grid, col, x_c, N, C, H, W, kH, kW, sh, sw, ph, pw,
            H_out, W_out, num_cols, BLOCK,
        )
        assert close_enough(read_floats(x_t, N * C * H * W), read_floats(x_c, N * C * H * W), 1e-5, 1e-6)
        for p in (col, x_t, x_c): cuda_free(p)


class TestLoss:
    BLOCK = 256

    def test_mse_loss(self):
        from gradtuity.kernels_triton.loss_kernels import mse_loss_kernel as TK
        from gradtuity.kernels_cuda.loss_kernels import mse_loss_kernel_launch as CK
        N = 4096
        a_vals = [0.01 * (i % 23) for i in range(N)]
        b_vals = [0.01 * (i % 17 + 5) for i in range(N)]
        a, b = alloc_floats(a_vals), alloc_floats(b_vals)
        out_t, out_c = alloc_zeros(1), alloc_zeros(1)
        TK[grid1d(N)](a, b, out_t, N, BLOCK=self.BLOCK)
        CK(grid1d(N), a, b, out_c, N, self.BLOCK)
        rt, rc = read_floats(out_t, 1)[0], read_floats(out_c, 1)[0]
        assert abs(rt - rc) < 1e-3 + 1e-4 * abs(rt)
        for p in (a, b, out_t, out_c): cuda_free(p)

    def test_mse_loss_backward(self):
        from gradtuity.kernels_triton.loss_kernels import mse_loss_backward_kernel as TK
        from gradtuity.kernels_cuda.loss_kernels import mse_loss_backward_kernel_launch as CK
        N = 1024
        a_vals = [0.01 * (i % 23) for i in range(N)]
        b_vals = [0.01 * (i % 17 + 5) for i in range(N)]
        seed = [0.1 * i for i in range(N)]
        a, b = alloc_floats(a_vals), alloc_floats(b_vals)
        ga_t, ga_c = alloc_floats(seed), alloc_floats(seed)
        gb_t, gb_c = alloc_floats(seed), alloc_floats(seed)
        TK[grid1d(N)](
            ga_t, gb_t, a, b, 0.001, N, 1, 1, BLOCK=self.BLOCK,
        )
        CK(grid1d(N), ga_c, gb_c, a, b, 0.001, N, 1, 1, self.BLOCK)
        assert close_enough(read_floats(ga_t, N), read_floats(ga_c, N), 1e-6, 1e-7)
        assert close_enough(read_floats(gb_t, N), read_floats(gb_c, N), 1e-6, 1e-7)
        for p in (a, b, ga_t, ga_c, gb_t, gb_c): cuda_free(p)


# ---------------------------------------------------------------------------
# Dropout — different RNG; statistical + within-backend consistency only
# ---------------------------------------------------------------------------
class TestDropout:
    BLOCK = 256
    N = 1 << 14  # 16k for low-variance drop fraction

    def test_drop_fraction_within_tolerance(self):
        from gradtuity.kernels_cuda.dropout_kernels import dropout_forward_kernel_launch
        x_vals = [1.0] * self.N
        x = alloc_floats(x_vals)
        y = cuda_malloc(self.N * 4)
        p = 0.3
        dropout_forward_kernel_launch(grid1d(self.N), x, y, self.N, p, 7, 0, self.BLOCK)
        out = read_floats(y, self.N)
        kept = sum(1 for v in out if v != 0.0)
        kept_frac = kept / self.N
        assert abs(kept_frac - (1 - p)) < 0.05
        cuda_free(x); cuda_free(y)

    def test_forward_backward_use_same_mask(self):
        """Within the CUDA backend: dx[i] != 0 iff y[i] != 0 for the same (seed, offset)."""
        from gradtuity.kernels_cuda.dropout_kernels import (
            dropout_backward_kernel_launch,
            dropout_forward_kernel_launch,
        )
        x_vals = [1.0] * self.N
        x = alloc_floats(x_vals)
        y = cuda_malloc(self.N * 4)
        dx = alloc_zeros(self.N)
        dy = alloc_floats([1.0] * self.N)
        p = 0.4
        dropout_forward_kernel_launch(grid1d(self.N), x, y, self.N, p, 11, 0, self.BLOCK)
        dropout_backward_kernel_launch(grid1d(self.N), dx, dy, self.N, p, 11, 0, self.BLOCK)
        y_vals = read_floats(y, self.N)
        dx_vals = read_floats(dx, self.N)
        for yi, dxi in zip(y_vals, dx_vals):
            assert (yi != 0.0) == (dxi != 0.0)
        for p_ in (x, y, dx, dy): cuda_free(p_)
