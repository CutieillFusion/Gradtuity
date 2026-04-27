"""CUDA-C twin of gradtuity.kernels.dropout_kernels.

Note: produces a *different* mask than Triton's tl.rand for the same (seed,
offset). Triton uses Philox; we use a Jenkins-style integer hash. Both are
deterministic, both are uniform over [0,1), and both satisfy the only
correctness invariant the rest of the framework relies on:
forward and backward, called with the same (seed, offset), produce the same
mask. Cross-backend bit-equality of dropout outputs is not expected.
"""

from ..cuda_driver import kernel_module, f32, i32, launch, ptr

_SRC = r"""
__device__ __forceinline__ unsigned int _wang_hash(unsigned int k) {
    k = (k ^ 61u) ^ (k >> 16);
    k = k + (k << 3);
    k = k ^ (k >> 4);
    k = k * 0x27d4eb2du;
    k = k ^ (k >> 15);
    return k;
}

__device__ __forceinline__ float _uniform_from_seed_offset(int seed, int offset) {
    unsigned int s = (unsigned int)seed;
    unsigned int o = (unsigned int)offset;
    // Combine seed and offset, then hash. Mantissa-only conversion to [0,1).
    unsigned int h = _wang_hash(s ^ _wang_hash(o));
    // 24-bit mantissa float in [0, 1).
    return (h >> 8) * (1.0f / 16777216.0f);
}

extern "C" __global__ void dropout_forward_kernel(
    const float* __restrict__ x, float* __restrict__ y,
    int n, float p, int seed, int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float r = _uniform_from_seed_offset(seed, offset + idx);
    float scale = 1.0f / (1.0f - p);
    y[idx] = (r >= p) ? (x[idx] * scale) : 0.0f;
}

extern "C" __global__ void dropout_backward_kernel(
    float* __restrict__ dx, const float* __restrict__ dy,
    int n, float p, int seed, int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float r = _uniform_from_seed_offset(seed, offset + idx);
    float scale = 1.0f / (1.0f - p);
    float dx_val = (r >= p) ? (dy[idx] * scale) : 0.0f;
    dx[idx] = dx[idx] + dx_val;  // accumulate into existing grad
}
"""

_h = kernel_module(_SRC)


def dropout_forward_kernel_launch(grid, x_ptr, y_ptr, n, p, seed, offset, BLOCK):
    launch(
        _h("dropout_forward_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [ptr(x_ptr), ptr(y_ptr), i32(n), f32(p), i32(seed), i32(offset)],
    )


def dropout_backward_kernel_launch(grid, dx_ptr, dy_ptr, n, p, seed, offset, BLOCK):
    launch(
        _h("dropout_backward_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [ptr(dx_ptr), ptr(dy_ptr), i32(n), f32(p), i32(seed), i32(offset)],
    )
