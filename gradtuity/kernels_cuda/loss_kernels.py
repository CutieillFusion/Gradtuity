"""CUDA-C twins of gradtuity.kernels.loss_kernels.

MSE forward/backward (Phase 2) and cross-entropy forward/backward (Phase 3)
share this module since their wrappers all live alongside each other in the
Triton version too. CE uses one block per row with shared-mem reductions for
row-max and sumexp; the math mirrors the Triton kernel exactly.
"""

from ..cuda_driver import kernel_module, f32, i32, launch, ptr

_SRC = r"""
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

extern "C" __global__ void mse_loss_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int numel
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float diff = (gid < numel) ? (a[gid] - b[gid]) : 0.0f;
    sdata[tid] = diff * diff;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

extern "C" __global__ void mse_loss_backward_kernel(
    float* __restrict__ grad_a, float* __restrict__ grad_b,
    const float* __restrict__ a, const float* __restrict__ b,
    float scale, int numel,
    int compute_grad_a, int compute_grad_b
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float gv = 2.0f * (a[idx] - b[idx]) * scale;
    if (compute_grad_a) grad_a[idx] = grad_a[idx] + gv;
    if (compute_grad_b) grad_b[idx] = grad_b[idx] - gv;
}

extern "C" __global__ void cross_entropy_forward_kernel(
    const float* __restrict__ logits, const float* __restrict__ targets,
    float* __restrict__ out, int B, int C
) {
    int row = blockIdx.x;
    if (row >= B) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int Bt = blockDim.x;

    float local_max = -__int_as_float(0x7f800000);  // -inf without <math.h>
    for (int j = tid; j < C; j += Bt) {
        float v = logits[row * C + j];
        if (v > local_max) local_max = v;
    }
    float row_max = _block_reduce_max(local_max, sdata);

    float local_sum = 0.0f;
    for (int j = tid; j < C; j += Bt) {
        local_sum += __expf(logits[row * C + j] - row_max);
    }
    float sumexp = _block_reduce_sum(local_sum, sdata);

    if (tid == 0) {
        float lse = __logf(sumexp) + row_max;
        int y = (int)targets[row];
        float logit_y = logits[row * C + y];
        atomicAdd(out, lse - logit_y);
    }
}

extern "C" __global__ void cross_entropy_backward_kernel(
    float* __restrict__ dlogits,
    const float* __restrict__ logits, const float* __restrict__ targets,
    float scale, int B, int C
) {
    int row = blockIdx.x;
    if (row >= B) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int Bt = blockDim.x;

    float local_max = -__int_as_float(0x7f800000);  // -inf without <math.h>
    for (int j = tid; j < C; j += Bt) {
        float v = logits[row * C + j];
        if (v > local_max) local_max = v;
    }
    float row_max = _block_reduce_max(local_max, sdata);

    float local_sum = 0.0f;
    for (int j = tid; j < C; j += Bt) {
        local_sum += __expf(logits[row * C + j] - row_max);
    }
    float sumexp = _block_reduce_sum(local_sum, sdata);
    float lse = __logf(sumexp) + row_max;
    int y = (int)targets[row];

    for (int j = tid; j < C; j += Bt) {
        float softmax_val = __expf(logits[row * C + j] - lse);
        float onehot = (j == y) ? 1.0f : 0.0f;
        float gv = softmax_val - onehot;
        dlogits[row * C + j] = dlogits[row * C + j] + scale * gv;
    }
}
"""

_h = kernel_module(_SRC)


def mse_loss_kernel_launch(grid, a_ptr, b_ptr, out_ptr, numel, BLOCK):
    launch(
        _h("mse_loss_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [ptr(a_ptr), ptr(b_ptr), ptr(out_ptr), i32(numel)],
        shared_bytes=BLOCK * 4,
    )


def mse_loss_backward_kernel_launch(
    grid, grad_a_ptr, grad_b_ptr, a_ptr, b_ptr,
    scale, numel, compute_grad_a, compute_grad_b, BLOCK,
):
    launch(
        _h("mse_loss_backward_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [
            ptr(grad_a_ptr), ptr(grad_b_ptr), ptr(a_ptr), ptr(b_ptr),
            f32(scale), i32(numel),
            i32(compute_grad_a), i32(compute_grad_b),
        ],
    )


def cross_entropy_forward_kernel_launch(
    grid, logits_ptr, targets_ptr, out_ptr, B, C, BLOCK_C
):
    launch(
        _h("cross_entropy_forward_kernel"), (grid[0], 1, 1), (BLOCK_C, 1, 1),
        [ptr(logits_ptr), ptr(targets_ptr), ptr(out_ptr), i32(B), i32(C)],
        shared_bytes=BLOCK_C * 4,
    )


def cross_entropy_backward_kernel_launch(
    grid, dlogits_ptr, logits_ptr, targets_ptr, scale, B, C, BLOCK_C
):
    launch(
        _h("cross_entropy_backward_kernel"), (grid[0], 1, 1), (BLOCK_C, 1, 1),
        [
            ptr(dlogits_ptr), ptr(logits_ptr), ptr(targets_ptr),
            f32(scale), i32(B), i32(C),
        ],
        shared_bytes=BLOCK_C * 4,
    )
