"""CUDA-C twins of gradtuity.kernels.matmul_kernels.

Tiled with 32x32 shared-memory blocks (BLOCK_M=BLOCK_N=BLOCK_K=32) — same
block sizes as the Triton version, so the comparison is apples-to-apples on
tile shape. We do NOT use WMMA / tensor cores; Triton's tl.dot lowers to MMA
on supported arches and will be the fast reference. Expect Triton matmul
2-5x faster on Ampere/Hopper for medium shapes — the educational comparison
is the point, not chasing cuBLAS-level perf.
"""

from ..cuda_driver import kernel_module, i32, launch, ptr

# All 5 variants use a 32x32 thread block; thread (ty, tx) computes one output
# cell. Strides are runtime args so call sites can route different transpose
# layouts through the same kernel without materializing transposes.
_SRC = r"""
#define BM 32
#define BN 32
#define BK 32

extern "C" __global__ void matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int sam, int sak, int sbk, int sbn, int scm, int scn
) {
    int by = blockIdx.x;
    int bx = blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = by * BM + ty;
    int col = bx * BN + tx;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int ak = k0 + tx;
        // Address clamping to ptr+0 on OOB — safer than `if` which the
        // compiler may still fuse into a predicated load. The conv2d call
        // site at tensor.py:1987 also passes strides that imply a logically
        // larger buffer than the physical allocation; clamping prevents the
        // resulting OOB pointer from hitting unmapped memory.
        int a_in = (row < M && ak < K) ? 1 : 0;
        int a_idx = a_in ? (row * sam + ak * sak) : 0;
        float av = a[a_idx] * (float)a_in;
        As[ty][tx] = av;
        int bk = k0 + ty;
        int b_in = (bk < K && col < N) ? 1 : 0;
        int b_idx = b_in ? (bk * sbk + col * sbn) : 0;
        float bv = b[b_idx] * (float)b_in;
        Bs[ty][tx] = bv;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) c[row * scm + col * scn] = acc;
}

extern "C" __global__ void matmul_bias_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ c,
    int M, int N, int K,
    int sam, int sak, int sbk, int sbn, int scm, int scn
) {
    int by = blockIdx.x;
    int bx = blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = by * BM + ty;
    int col = bx * BN + tx;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int ak = k0 + tx;
        int a_in = (row < M && ak < K) ? 1 : 0;
        int a_idx = a_in ? (row * sam + ak * sak) : 0;
        float av = a[a_idx] * (float)a_in;
        As[ty][tx] = av;
        int bk = k0 + ty;
        int b_in = (bk < K && col < N) ? 1 : 0;
        int b_idx = b_in ? (bk * sbk + col * sbn) : 0;
        float bv = b[b_idx] * (float)b_in;
        Bs[ty][tx] = bv;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) c[row * scm + col * scn] = acc + bias[col];
}

extern "C" __global__ void matmul_bias_relu_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ c,
    int M, int N, int K,
    int sam, int sak, int sbk, int sbn, int scm, int scn
) {
    int by = blockIdx.x;
    int bx = blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = by * BM + ty;
    int col = bx * BN + tx;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int ak = k0 + tx;
        int a_in = (row < M && ak < K) ? 1 : 0;
        int a_idx = a_in ? (row * sam + ak * sak) : 0;
        float av = a[a_idx] * (float)a_in;
        As[ty][tx] = av;
        int bk = k0 + ty;
        int b_in = (bk < K && col < N) ? 1 : 0;
        int b_idx = b_in ? (bk * sbk + col * sbn) : 0;
        float bv = b[b_idx] * (float)b_in;
        Bs[ty][tx] = bv;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        float v = acc + bias[col];
        c[row * scm + col * scn] = v > 0.0f ? v : 0.0f;
    }
}

// C += A @ B^T. Caller passes B as (N, K) with stride_bn = row stride,
// stride_bk = col stride. The kernel reads B[col, k] = b[col * sbn + k * sbk]
// and accumulates into c.
extern "C" __global__ void matmul_nt_acc_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int sam, int sak, int sbn, int sbk, int scm, int scn
) {
    int by = blockIdx.x;
    int bx = blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = by * BM + ty;
    int col = bx * BN + tx;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int ak = k0 + tx;
        int a_in = (row < M && ak < K) ? 1 : 0;
        int a_idx = a_in ? (row * sam + ak * sak) : 0;
        float av = a[a_idx] * (float)a_in;
        As[ty][tx] = av;
        // We want Bs[k, n] = B[n, k]. Thread (ty, tx) stages Bs[ty, tx] = B[col, k0+ty].
        int bk = k0 + ty;
        int b_in = (bk < K && col < N) ? 1 : 0;
        int b_idx = b_in ? (col * sbn + bk * sbk) : 0;
        float bv = b[b_idx] * (float)b_in;
        Bs[ty][tx] = bv;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        c[row * scm + col * scn] = c[row * scm + col * scn] + acc;
    }
}

// C += A^T @ B. Caller passes A as (K, M) with stride_ak = row stride,
// stride_am = col stride. Kernel reads A[k, row] = a[k * sak + row * sam].
extern "C" __global__ void matmul_tn_acc_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int sak, int sam, int sbk, int sbn, int scm, int scn
) {
    int by = blockIdx.x;
    int bx = blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = by * BM + ty;
    int col = bx * BN + tx;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Stage As[m, k] = A^T[m, k] = A[k, m]. Thread (ty, tx) stages
        // As[ty, tx] = A[k0+tx, row].
        int ak = k0 + tx;
        int a_in = (row < M && ak < K) ? 1 : 0;
        int a_idx = a_in ? (ak * sak + row * sam) : 0;
        float av = a[a_idx] * (float)a_in;
        As[ty][tx] = av;
        int bk = k0 + ty;
        int b_in = (bk < K && col < N) ? 1 : 0;
        int b_idx = b_in ? (bk * sbk + col * sbn) : 0;
        float bv = b[b_idx] * (float)b_in;
        Bs[ty][tx] = bv;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        c[row * scm + col * scn] = c[row * scm + col * scn] + acc;
    }
}

extern "C" __global__ void transpose2d_kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = rows * cols;
    if (idx >= numel) return;
    int i = idx / cols;
    int j = idx % cols;
    dst[j * rows + i] = src[i * cols + j];
}
"""

_h = kernel_module(_SRC)


def matmul_kernel_launch(
    grid, a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    launch(
        _h("matmul_kernel"), (grid[0], grid[1], 1), (32, 32, 1),
        [
            ptr(a_ptr), ptr(b_ptr), ptr(c_ptr),
            i32(M), i32(N), i32(K),
            i32(stride_am), i32(stride_ak),
            i32(stride_bk), i32(stride_bn),
            i32(stride_cm), i32(stride_cn),
        ],
    )


def matmul_bias_kernel_launch(
    grid, a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    launch(
        _h("matmul_bias_kernel"), (grid[0], grid[1], 1), (32, 32, 1),
        [
            ptr(a_ptr), ptr(b_ptr), ptr(bias_ptr), ptr(c_ptr),
            i32(M), i32(N), i32(K),
            i32(stride_am), i32(stride_ak),
            i32(stride_bk), i32(stride_bn),
            i32(stride_cm), i32(stride_cn),
        ],
    )


def matmul_bias_relu_kernel_launch(
    grid, a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    launch(
        _h("matmul_bias_relu_kernel"), (grid[0], grid[1], 1), (32, 32, 1),
        [
            ptr(a_ptr), ptr(b_ptr), ptr(bias_ptr), ptr(c_ptr),
            i32(M), i32(N), i32(K),
            i32(stride_am), i32(stride_ak),
            i32(stride_bk), i32(stride_bn),
            i32(stride_cm), i32(stride_cn),
        ],
    )


def matmul_nt_acc_kernel_launch(
    grid, a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    launch(
        _h("matmul_nt_acc_kernel"), (grid[0], grid[1], 1), (32, 32, 1),
        [
            ptr(a_ptr), ptr(b_ptr), ptr(c_ptr),
            i32(M), i32(N), i32(K),
            i32(stride_am), i32(stride_ak),
            i32(stride_bn), i32(stride_bk),
            i32(stride_cm), i32(stride_cn),
        ],
    )


def matmul_tn_acc_kernel_launch(
    grid, a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ak, stride_am, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    launch(
        _h("matmul_tn_acc_kernel"), (grid[0], grid[1], 1), (32, 32, 1),
        [
            ptr(a_ptr), ptr(b_ptr), ptr(c_ptr),
            i32(M), i32(N), i32(K),
            i32(stride_ak), i32(stride_am),
            i32(stride_bk), i32(stride_bn),
            i32(stride_cm), i32(stride_cn),
        ],
    )


def transpose2d_kernel_launch(grid, src_ptr, dst_ptr, rows, cols, BLOCK):
    launch(
        _h("transpose2d_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [ptr(src_ptr), ptr(dst_ptr), i32(rows), i32(cols)],
    )
