"""CUDA-C twins of gradtuity.kernels.gather_kernels."""

from ..cuda_driver import kernel_module, i32, launch, ptr

_SRC = r"""
extern "C" __global__ void embedding_gather_kernel(
    const float* __restrict__ W, const float* __restrict__ idx_f,
    float* __restrict__ out, int N, int D, int V
) {
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || d >= D) return;
    int row = (int)idx_f[n];
    out[n * D + d] = W[row * D + d];
}

extern "C" __global__ void embedding_scatter_add_kernel(
    const float* __restrict__ dOut, const float* __restrict__ idx_f,
    float* __restrict__ dW, int N, int D, int V
) {
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || d >= D) return;
    int row = (int)idx_f[n];
    if (row < 0 || row >= V) return;
    atomicAdd(dW + row * D + d, dOut[n * D + d]);
}
"""

_h = kernel_module(_SRC)


def embedding_gather_kernel_launch(grid, W_ptr, idx_ptr, out_ptr, N, D, V, BLOCK_D):
    # Triton grid: (N, cdiv(D, BLOCK_D)). CUDA same, BLOCK_D threads per d-block.
    launch(
        _h("embedding_gather_kernel"),
        (grid[0], grid[1], 1), (BLOCK_D, 1, 1),
        [ptr(W_ptr), ptr(idx_ptr), ptr(out_ptr), i32(N), i32(D), i32(V)],
    )


def embedding_scatter_add_kernel_launch(grid, dOut_ptr, idx_ptr, dW_ptr, N, D, V, BLOCK_D):
    launch(
        _h("embedding_scatter_add_kernel"),
        (grid[0], grid[1], 1), (BLOCK_D, 1, 1),
        [ptr(dOut_ptr), ptr(idx_ptr), ptr(dW_ptr), i32(N), i32(D), i32(V)],
    )
