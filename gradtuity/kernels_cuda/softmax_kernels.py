"""CUDA-C twins of gradtuity.kernels.softmax_kernels.

One CUDA block per row, BLOCK_COLS threads per block. Each thread strides
through the row, reducing in shared memory for max / sumexp.
"""

from ..cuda_driver import kernel_module, i32, launch, ptr

_SRC = r"""
// Block-wide reduction helpers. sdata must be at least blockDim.x floats.

__device__ __forceinline__ float _block_reduce_max(float v, float* sdata) {
    int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    return sdata[0];
}

__device__ __forceinline__ float _block_reduce_sum(float v, float* sdata) {
    int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

extern "C" __global__ void softmax_forward_kernel(
    const float* __restrict__ x, float* __restrict__ y,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int B = blockDim.x;

    // Pass 1: row max.
    float local_max = -__int_as_float(0x7f800000);  // -inf without <math.h>
    for (int j = tid; j < cols; j += B) {
        float v = x[row * cols + j];
        if (v > local_max) local_max = v;
    }
    float row_max = _block_reduce_max(local_max, sdata);

    // Pass 2: sumexp(x - row_max).
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += B) {
        local_sum += __expf(x[row * cols + j] - row_max);
    }
    float sumexp = _block_reduce_sum(local_sum, sdata);

    // Pass 3: y = exp(x - row_max) / sumexp.
    float inv = 1.0f / sumexp;
    for (int j = tid; j < cols; j += B) {
        y[row * cols + j] = __expf(x[row * cols + j] - row_max) * inv;
    }
}

extern "C" __global__ void softmax_backward_kernel(
    float* __restrict__ dx, const float* __restrict__ dy,
    const float* __restrict__ y, int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int B = blockDim.x;

    // dot = sum(dy * y) over the row.
    float local = 0.0f;
    for (int j = tid; j < cols; j += B) {
        local += dy[row * cols + j] * y[row * cols + j];
    }
    float dot = _block_reduce_sum(local, sdata);

    // dx_j = y_j * (dy_j - dot).
    for (int j = tid; j < cols; j += B) {
        float yi = y[row * cols + j];
        float dyi = dy[row * cols + j];
        dx[row * cols + j] = yi * (dyi - dot);
    }
}

extern "C" __global__ void softmax_with_causal_mask_forward_kernel(
    const float* __restrict__ x, float* __restrict__ y,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int B = blockDim.x;

    // i_in_block: position within the (S, S) block. Causal: only j <= i_in_block
    // contribute to max/sumexp; j > i_in_block written as 0.
    int i_in_block = row % cols;

    float local_max = -__int_as_float(0x7f800000);  // -inf without <math.h>
    for (int j = tid; j <= i_in_block; j += B) {
        float v = x[row * cols + j];
        if (v > local_max) local_max = v;
    }
    float row_max = _block_reduce_max(local_max, sdata);

    float local_sum = 0.0f;
    for (int j = tid; j <= i_in_block; j += B) {
        local_sum += __expf(x[row * cols + j] - row_max);
    }
    float sumexp = _block_reduce_sum(local_sum, sdata);

    float inv = 1.0f / sumexp;
    for (int j = tid; j < cols; j += B) {
        if (j <= i_in_block) {
            y[row * cols + j] = __expf(x[row * cols + j] - row_max) * inv;
        } else {
            y[row * cols + j] = 0.0f;
        }
    }
}
"""

_h = kernel_module(_SRC)


def _shared_bytes(BLOCK_COLS: int) -> int:
    return BLOCK_COLS * 4


def softmax_forward_kernel_launch(grid, x_ptr, y_ptr, rows, cols, BLOCK_COLS):
    launch(
        _h("softmax_forward_kernel"), (grid[0], 1, 1), (BLOCK_COLS, 1, 1),
        [ptr(x_ptr), ptr(y_ptr), i32(rows), i32(cols)],
        shared_bytes=_shared_bytes(BLOCK_COLS),
    )


def softmax_backward_kernel_launch(grid, dx_ptr, dy_ptr, y_ptr, rows, cols, BLOCK_COLS):
    launch(
        _h("softmax_backward_kernel"), (grid[0], 1, 1), (BLOCK_COLS, 1, 1),
        [ptr(dx_ptr), ptr(dy_ptr), ptr(y_ptr), i32(rows), i32(cols)],
        shared_bytes=_shared_bytes(BLOCK_COLS),
    )


def softmax_with_causal_mask_forward_kernel_launch(
    grid, x_ptr, y_ptr, rows, cols, BLOCK_COLS
):
    launch(
        _h("softmax_with_causal_mask_forward_kernel"),
        (grid[0], 1, 1), (BLOCK_COLS, 1, 1),
        [ptr(x_ptr), ptr(y_ptr), i32(rows), i32(cols)],
        shared_bytes=_shared_bytes(BLOCK_COLS),
    )
