"""CUDA-C twins of gradtuity.kernels.reduce_kernels."""

from ..cuda_driver import kernel_module, i32, launch, ptr

# Reductions use shared memory + a single atomic per block. atomicAdd ordering
# differs from Triton's, so cross-backend results may differ in the last few
# ULPs for sum_all / sum_axis0; expected for any non-deterministic reduction.
_SRC = r"""
extern "C" __global__ void sum_all_kernel(
    const float* __restrict__ x, float* __restrict__ out, int numel
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (gid < numel) ? x[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

extern "C" __global__ void sum_axis0_kernel(
    const float* __restrict__ x, float* __restrict__ out,
    int rows, int cols
) {
    int col = blockIdx.x;
    if (col >= cols) return;
    float acc = 0.0f;
    for (int row = 0; row < rows; ++row) {
        acc += x[row * cols + col];
    }
    atomicAdd(out + col, acc);
}

extern "C" __global__ void add_scalar_inplace_kernel(
    float* __restrict__ x, const float* __restrict__ scalar_ptr, int numel
) {
    __shared__ float scalar;
    if (threadIdx.x == 0) scalar = *scalar_ptr;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    x[idx] = x[idx] + scalar;
}

extern "C" __global__ void add_bias_kernel(
    const float* __restrict__ x, const float* __restrict__ b,
    float* __restrict__ y, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = rows * cols;
    if (idx >= numel) return;
    int col = idx % cols;
    y[idx] = x[idx] + b[col];
}

extern "C" __global__ void argmax_axis1_kernel(
    const float* __restrict__ x, float* __restrict__ out,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xrow = x + row * cols;
    float mv = xrow[0];
    int mi = 0;
    for (int j = 1; j < cols; ++j) {
        float v = xrow[j];
        if (v > mv) { mv = v; mi = j; }
    }
    out[row] = (float)mi;
}
"""

_h = kernel_module(_SRC)


def sum_all_kernel_launch(grid, x_ptr, out_ptr, numel, BLOCK):
    # Shared mem = BLOCK floats
    launch(
        _h("sum_all_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [ptr(x_ptr), ptr(out_ptr), i32(numel)],
        shared_bytes=BLOCK * 4,
    )


def sum_axis0_kernel_launch(grid, x_ptr, out_ptr, rows, cols, BLOCK_ROWS):
    # Triton's grid is (cols,); per-column accumulation.
    launch(
        _h("sum_axis0_kernel"), (grid[0], 1, 1), (1, 1, 1),
        [ptr(x_ptr), ptr(out_ptr), i32(rows), i32(cols)],
    )


def add_scalar_inplace_kernel_launch(grid, x_ptr, scalar_ptr, numel, BLOCK):
    launch(
        _h("add_scalar_inplace_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [ptr(x_ptr), ptr(scalar_ptr), i32(numel)],
    )


def add_bias_kernel_launch(grid, x_ptr, b_ptr, y_ptr, rows, cols, BLOCK):
    launch(
        _h("add_bias_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [ptr(x_ptr), ptr(b_ptr), ptr(y_ptr), i32(rows), i32(cols)],
    )


def argmax_axis1_kernel_launch(grid, x_ptr, out_ptr, rows, cols, BLOCK_COLS):
    launch(
        _h("argmax_axis1_kernel"), (grid[0], 1, 1), (1, 1, 1),
        [ptr(x_ptr), ptr(out_ptr), i32(rows), i32(cols)],
    )
