"""CUDA-C twins of gradtuity.kernels.layernorm_kernels.

One CUDA block per row. Two-pass mean/variance (the Triton version uses
Welford to merge per-chunk stats; outputs agree to ~1e-5). Backward uses
atomicAdd for dgamma/dbeta exactly like the Triton version.
"""

from ..cuda_driver import kernel_module, f32, i32, launch, ptr

_SRC = r"""
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

extern "C" __global__ void layernorm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ xhat,
    float* __restrict__ rstd,
    int N, int H, float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int B = blockDim.x;

    // Pass 1: mean.
    float local_sum = 0.0f;
    for (int j = tid; j < H; j += B) local_sum += x[row * H + j];
    float mean = _block_reduce_sum(local_sum, sdata) / (float)H;

    // Pass 2: variance.
    float local_sq = 0.0f;
    for (int j = tid; j < H; j += B) {
        float d = x[row * H + j] - mean;
        local_sq += d * d;
    }
    float var = _block_reduce_sum(local_sq, sdata) / (float)H;
    float rstd_val = rsqrtf(var + eps);
    if (tid == 0) rstd[row] = rstd_val;

    // Pass 3: write xhat and y.
    for (int j = tid; j < H; j += B) {
        float xh = (x[row * H + j] - mean) * rstd_val;
        xhat[row * H + j] = xh;
        y[row * H + j] = xh * gamma[j] + beta[j];
    }
}

extern "C" __global__ void layernorm_bwd_kernel(
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    const float* __restrict__ dy,
    const float* __restrict__ xhat,
    const float* __restrict__ rstd,
    const float* __restrict__ gamma,
    int N, int H
) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int B = blockDim.x;
    float rstd_val = rstd[row];

    // sum1 = sum(dxhat), sum2 = sum(dxhat * xhat) over row, dxhat = dy * gamma.
    float local1 = 0.0f, local2 = 0.0f;
    for (int j = tid; j < H; j += B) {
        float dxhat = dy[row * H + j] * gamma[j];
        local1 += dxhat;
        local2 += dxhat * xhat[row * H + j];
    }
    float sum1 = _block_reduce_sum(local1, sdata);
    float sum2 = _block_reduce_sum(local2, sdata);

    float invH = 1.0f / (float)H;
    for (int j = tid; j < H; j += B) {
        float dxhat = dy[row * H + j] * gamma[j];
        float xh = xhat[row * H + j];
        dx[row * H + j] = rstd_val * invH * ((float)H * dxhat - sum1 - xh * sum2);
        // dgamma += dy * xhat; dbeta += dy. Atomic per-element matches Triton.
        atomicAdd(dgamma + j, dy[row * H + j] * xh);
        atomicAdd(dbeta + j, dy[row * H + j]);
    }
}
"""

_h = kernel_module(_SRC)


def layernorm_fwd_kernel_launch(
    grid, x_ptr, gamma_ptr, beta_ptr, y_ptr, xhat_ptr, rstd_ptr,
    N, H, eps, BLOCK_H,
):
    launch(
        _h("layernorm_fwd_kernel"), (grid[0], 1, 1), (BLOCK_H, 1, 1),
        [
            ptr(x_ptr), ptr(gamma_ptr), ptr(beta_ptr),
            ptr(y_ptr), ptr(xhat_ptr), ptr(rstd_ptr),
            i32(N), i32(H), f32(eps),
        ],
        shared_bytes=BLOCK_H * 4,
    )


def layernorm_bwd_kernel_launch(
    grid, dx_ptr, dgamma_ptr, dbeta_ptr, dy_ptr, xhat_ptr, rstd_ptr, gamma_ptr,
    N, H, BLOCK_H,
):
    launch(
        _h("layernorm_bwd_kernel"), (grid[0], 1, 1), (BLOCK_H, 1, 1),
        [
            ptr(dx_ptr), ptr(dgamma_ptr), ptr(dbeta_ptr),
            ptr(dy_ptr), ptr(xhat_ptr), ptr(rstd_ptr), ptr(gamma_ptr),
            i32(N), i32(H),
        ],
        shared_bytes=BLOCK_H * 4,
    )
