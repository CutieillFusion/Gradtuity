"""
Phase 4 parity tests: 5 matmul variants + non-vectorized im2col.

Matmul tolerance is wider (~1e-3 rel) because:
  - Triton's tl.dot lowers to MMA which uses different summation order.
  - We accumulate in fp32 with naive K-loop ordering.

Both produce mathematically the same result; differences are last-bit ULP.
"""

import math
import struct

import pytest
import triton

from ._helpers import alloc_floats, alloc_zeros, close_enough, max_abs_diff, read_floats

from gradtuity.cuda_mem import cuda_free, cuda_malloc, cuda_memset

pytestmark = [pytest.mark.requires_cuda, pytest.mark.kernel_backend_agnostic]


# ---------------------------------------------------------------------------
# Matmul variants
# ---------------------------------------------------------------------------
class TestMatmul:
    def _ab(self, M, K, N):
        a_vals = [(((i * 13) % 41) - 20) / 100.0 for i in range(M * K)]
        b_vals = [(((i * 7) % 37) - 18) / 100.0 for i in range(K * N)]
        return alloc_floats(a_vals), alloc_floats(b_vals)

    @pytest.mark.parametrize("M,K,N", [(32, 32, 32), (64, 96, 48), (17, 33, 19)])
    def test_matmul(self, M, K, N):
        from gradtuity.kernels_triton.matmul_kernels import matmul_kernel as TK
        from gradtuity.kernels_cuda.matmul_kernels import matmul_kernel_launch as CK

        a, b = self._ab(M, K, N)
        c_t, c_c = cuda_malloc(M * N * 4), cuda_malloc(M * N * 4)
        BM = BN = BK = 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        TK[grid](
            a, b, c_t, M, N, K,
            K, 1, N, 1, N, 1,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        CK(grid, a, b, c_c, M, N, K, K, 1, N, 1, N, 1, BM, BN, BK)
        # Triton's tl.dot uses TF32 tensor cores on H100 (10-bit mantissa).
        # Our CUDA matmul uses straight fp32 — it's MORE accurate, not less.
        # TF32 quantization error grows with K; 1e-2 rel covers up to K~100.
        # 1e-3 atol covers values near zero where rel comparisons collapse.
        assert close_enough(read_floats(c_t, M * N), read_floats(c_c, M * N), 1e-2, 1e-3)
        for p in (a, b, c_t, c_c): cuda_free(p)

    @pytest.mark.parametrize("M,K,N", [(64, 96, 48), (17, 33, 19)])
    def test_matmul_bias(self, M, K, N):
        from gradtuity.kernels_triton.matmul_kernels import matmul_bias_kernel as TK
        from gradtuity.kernels_cuda.matmul_kernels import matmul_bias_kernel_launch as CK

        a, b = self._ab(M, K, N)
        bias = alloc_floats([0.05 * (i % 7) - 0.2 for i in range(N)])
        c_t, c_c = cuda_malloc(M * N * 4), cuda_malloc(M * N * 4)
        BM = BN = BK = 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        TK[grid](
            a, b, bias, c_t, M, N, K,
            K, 1, N, 1, N, 1,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        CK(grid, a, b, bias, c_c, M, N, K, K, 1, N, 1, N, 1, BM, BN, BK)
        # Triton's tl.dot uses TF32 tensor cores on H100 (10-bit mantissa).
        # Our CUDA matmul uses straight fp32 — it's MORE accurate, not less.
        # TF32 quantization error grows with K; 1e-2 rel covers up to K~100.
        # 1e-3 atol covers values near zero where rel comparisons collapse.
        assert close_enough(read_floats(c_t, M * N), read_floats(c_c, M * N), 1e-2, 1e-3)
        for p in (a, b, bias, c_t, c_c): cuda_free(p)

    @pytest.mark.parametrize("M,K,N", [(64, 96, 48), (17, 33, 19)])
    def test_matmul_bias_relu(self, M, K, N):
        from gradtuity.kernels_triton.matmul_kernels import matmul_bias_relu_kernel as TK
        from gradtuity.kernels_cuda.matmul_kernels import matmul_bias_relu_kernel_launch as CK

        a, b = self._ab(M, K, N)
        bias = alloc_floats([0.05 * (i % 7) - 0.2 for i in range(N)])
        c_t, c_c = cuda_malloc(M * N * 4), cuda_malloc(M * N * 4)
        BM = BN = BK = 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        TK[grid](
            a, b, bias, c_t, M, N, K,
            K, 1, N, 1, N, 1,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        CK(grid, a, b, bias, c_c, M, N, K, K, 1, N, 1, N, 1, BM, BN, BK)
        # After ReLU, values are bounded; same tolerance.
        # Triton's tl.dot uses TF32 tensor cores on H100 (10-bit mantissa).
        # Our CUDA matmul uses straight fp32 — it's MORE accurate, not less.
        # TF32 quantization error grows with K; 1e-2 rel covers up to K~100.
        # 1e-3 atol covers values near zero where rel comparisons collapse.
        assert close_enough(read_floats(c_t, M * N), read_floats(c_c, M * N), 1e-2, 1e-3)
        for p in (a, b, bias, c_t, c_c): cuda_free(p)

    @pytest.mark.parametrize("M,N,K_in", [(48, 32, 64), (19, 17, 33)])
    def test_matmul_nt_acc(self, M, N, K_in):
        """C += A @ B^T. A: (M, K_in), B: (N, K_in) read transposed -> C: (M, N)."""
        from gradtuity.kernels_triton.matmul_kernels import matmul_nt_acc_kernel as TK
        from gradtuity.kernels_cuda.matmul_kernels import matmul_nt_acc_kernel_launch as CK

        a_vals = [(((i * 13) % 41) - 20) / 100.0 for i in range(M * K_in)]
        b_vals = [(((i * 7) % 37) - 18) / 100.0 for i in range(N * K_in)]
        a = alloc_floats(a_vals)
        b = alloc_floats(b_vals)
        c_init = [0.001 * i for i in range(M * N)]
        c_t = alloc_floats(c_init)
        c_c = alloc_floats(c_init)
        BM = BN = BK = 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        # strides: A (M, K_in) row-major -> sam=K_in, sak=1
        #          B (N, K_in) row-major -> sbn=K_in, sbk=1
        #          C (M, N)              -> scm=N, scn=1
        TK[grid](
            a, b, c_t, M, N, K_in,
            K_in, 1, K_in, 1, N, 1,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        CK(
            grid, a, b, c_c, M, N, K_in,
            K_in, 1, K_in, 1, N, 1, BM, BN, BK,
        )
        # Triton's tl.dot uses TF32 tensor cores on H100 (10-bit mantissa).
        # Our CUDA matmul uses straight fp32 — it's MORE accurate, not less.
        # TF32 quantization error grows with K; 1e-2 rel covers up to K~100.
        # 1e-3 atol covers values near zero where rel comparisons collapse.
        assert close_enough(read_floats(c_t, M * N), read_floats(c_c, M * N), 1e-2, 1e-3)
        for p in (a, b, c_t, c_c): cuda_free(p)

    @pytest.mark.parametrize("M,N,K_in", [(48, 32, 64), (19, 17, 33)])
    def test_matmul_tn_acc(self, M, N, K_in):
        """C += A^T @ B. A: (K_in, M) read transposed, B: (K_in, N) -> C: (M, N)."""
        from gradtuity.kernels_triton.matmul_kernels import matmul_tn_acc_kernel as TK
        from gradtuity.kernels_cuda.matmul_kernels import matmul_tn_acc_kernel_launch as CK

        a_vals = [(((i * 13) % 41) - 20) / 100.0 for i in range(K_in * M)]
        b_vals = [(((i * 7) % 37) - 18) / 100.0 for i in range(K_in * N)]
        a = alloc_floats(a_vals)
        b = alloc_floats(b_vals)
        c_init = [0.001 * i for i in range(M * N)]
        c_t = alloc_floats(c_init)
        c_c = alloc_floats(c_init)
        BM = BN = BK = 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        # strides: A (K_in, M) row-major -> sak=M, sam=1
        #          B (K_in, N) row-major -> sbk=N, sbn=1
        #          C (M, N)               -> scm=N, scn=1
        TK[grid](
            a, b, c_t, M, N, K_in,
            M, 1, N, 1, N, 1,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        CK(
            grid, a, b, c_c, M, N, K_in,
            M, 1, N, 1, N, 1, BM, BN, BK,
        )
        # Triton's tl.dot uses TF32 tensor cores on H100 (10-bit mantissa).
        # Our CUDA matmul uses straight fp32 — it's MORE accurate, not less.
        # TF32 quantization error grows with K; 1e-2 rel covers up to K~100.
        # 1e-3 atol covers values near zero where rel comparisons collapse.
        assert close_enough(read_floats(c_t, M * N), read_floats(c_c, M * N), 1e-2, 1e-3)
        for p in (a, b, c_t, c_c): cuda_free(p)


# ---------------------------------------------------------------------------
# Non-vectorized im2col
#
# The Triton im2col_kernel is broken — it uses `continue` inside a Triton.jit
# function, which modern Triton rejects (UnsupportedLanguageConstruct). The
# vectorized im2col_kernel_2d is what the framework actually calls; im2col_kernel
# is dead code there. So we validate the CUDA version against im2col_kernel_2d
# instead, which is the meaningful correctness oracle.
# ---------------------------------------------------------------------------
class TestIm2colLegacy:
    def test_im2col_matches_im2col_2d(self):
        from gradtuity.kernels_triton.conv_kernels import im2col_kernel_2d as TK_2d
        from gradtuity.kernels_cuda.conv_kernels import im2col_kernel_launch as CK

        N, C, H, W = 1, 2, 5, 5
        kH, kW = 3, 3
        sh, sw, ph, pw = 1, 1, 1, 1
        H_out = (H + 2 * ph - kH) // sh + 1
        W_out = (W + 2 * pw - kW) // sw + 1
        num_cols = C * kH * kW
        num_rows = N * H_out * W_out
        BLOCK_ROW = 8
        BLOCK_COL = 8

        x_vals = [float(i) for i in range(N * C * H * W)]
        x = alloc_floats(x_vals)
        col_t = cuda_malloc(num_rows * num_cols * 4)
        col_c = cuda_malloc(num_rows * num_cols * 4)

        # Triton oracle: vectorized 2D im2col (the version the framework uses).
        BLOCK_2D = 32
        grid_2d = (num_rows, triton.cdiv(num_cols, BLOCK_2D))
        TK_2d[grid_2d](
            x, col_t, N, C, H, W, kH, kW, sh, sw, ph, pw,
            H_out, W_out, num_cols, BLOCK=BLOCK_2D,
        )

        # CUDA legacy im2col: tile (BLOCK_ROW, BLOCK_COL) per program.
        grid = (triton.cdiv(num_rows, BLOCK_ROW), triton.cdiv(num_cols, BLOCK_COL))
        CK(
            grid, x, col_c, N, C, H, W, kH, kW, sh, sw, ph, pw,
            H_out, W_out, BLOCK_ROW, BLOCK_COL,
        )
        assert read_floats(col_t, num_rows * num_cols) == read_floats(col_c, num_rows * num_cols)
        for p in (x, col_t, col_c): cuda_free(p)
