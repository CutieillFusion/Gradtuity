"""CUDA-C twins of gradtuity.kernels.mask_kernels."""

from ..cuda_driver import kernel_module, f32, i32, launch, ptr

_SRC = r"""
extern "C" __global__ void transpose4d_12_kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int B, int A, int C, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = B * A * C * D;
    if (idx >= numel) return;
    int d = idx % D;
    int t = idx / D;
    int c = t % C;
    t = t / C;
    int a = t % A;
    int b = t / A;
    int src_idx = b * (A * C * D) + a * (C * D) + c * D + d;
    int dst_idx = b * (C * A * D) + c * (A * D) + a * D + d;
    dst[dst_idx] = src[src_idx];
}

extern "C" __global__ void causal_mask_inplace_kernel(
    float* __restrict__ scores, int B, int H, int S, float neg_inf
) {
    int bh = blockIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;
    if (bh >= B * H || i >= S || j >= S) return;
    int idx = bh * (S * S) + i * S + j;
    if (j > i) scores[idx] = neg_inf;
}

extern "C" __global__ void causal_mask_backward_kernel(
    const float* __restrict__ dout, float* __restrict__ dscores,
    int B, int H, int S
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = B * H * S * S;
    if (idx >= numel) return;
    int rest = idx % (S * S);
    int j = rest % S;
    int i = rest / S;
    dscores[idx] = (j <= i) ? dout[idx] : 0.0f;
}
"""

_h = kernel_module(_SRC)


def transpose4d_12_kernel_launch(grid, src_ptr, dst_ptr, B, A, C, D, BLOCK):
    launch(
        _h("transpose4d_12_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [ptr(src_ptr), ptr(dst_ptr), i32(B), i32(A), i32(C), i32(D)],
    )


def causal_mask_inplace_kernel_launch(
    grid, scores_ptr, B, H, S, NEG_INF, BLOCK_I, BLOCK_J
):
    # Triton: grid (B*H, cdiv(S, BLOCK_I), cdiv(S, BLOCK_J)), each program is a tile.
    # CUDA: same grid, but block dim covers the tile (BLOCK_J, BLOCK_I, 1) with
    # one thread per (i, j) cell.
    launch(
        _h("causal_mask_inplace_kernel"),
        (grid[0], grid[1], grid[2]),
        (BLOCK_J, BLOCK_I, 1),
        [ptr(scores_ptr), i32(B), i32(H), i32(S), f32(NEG_INF)],
    )


def causal_mask_backward_kernel_launch(grid, dout_ptr, dscores_ptr, B, H, S, BLOCK):
    launch(
        _h("causal_mask_backward_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [ptr(dout_ptr), ptr(dscores_ptr), i32(B), i32(H), i32(S)],
    )
