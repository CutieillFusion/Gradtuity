"""CUDA-C twins of gradtuity.kernels.optim_kernels."""

from ..cuda_driver import kernel_module, f32, i32, launch, ptr

_SRC = r"""
extern "C" __global__ void fill_kernel(
    float* __restrict__ dst, float value, int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = value;
}

extern "C" __global__ void sgd_update_kernel(
    float* __restrict__ p, const float* __restrict__ g,
    float lr, int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    p[idx] = p[idx] - lr * g[idx];
}

extern "C" __global__ void adamw_step_kernel(
    float* __restrict__ p, const float* __restrict__ g,
    float* __restrict__ m, float* __restrict__ v,
    int n,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float pi = p[idx];
    float gi = g[idx];
    float mi = beta1 * m[idx] + (1.0f - beta1) * gi;
    float vi = beta2 * v[idx] + (1.0f - beta2) * gi * gi;
    float m_hat = mi * bc1;
    float v_hat = vi * bc2;
    float upd = m_hat / (sqrtf(v_hat) + eps);
    pi = pi - lr * upd - lr * weight_decay * pi;
    m[idx] = mi;
    v[idx] = vi;
    p[idx] = pi;
}
"""

_h = kernel_module(_SRC)


def fill_kernel_launch(grid, dst_ptr, value, numel, BLOCK):
    launch(_h("fill_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
           [ptr(dst_ptr), f32(value), i32(numel)])


def sgd_update_kernel_launch(grid, param_ptr, grad_ptr, lr, numel, BLOCK):
    launch(_h("sgd_update_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
           [ptr(param_ptr), ptr(grad_ptr), f32(lr), i32(numel)])


def adamw_step_kernel_launch(
    grid, p_ptr, g_ptr, m_ptr, v_ptr, n_elements,
    lr, beta1, beta2, eps, weight_decay, bc1, bc2, BLOCK,
):
    launch(
        _h("adamw_step_kernel"), (grid[0], 1, 1), (BLOCK, 1, 1),
        [
            ptr(p_ptr), ptr(g_ptr), ptr(m_ptr), ptr(v_ptr),
            i32(n_elements),
            f32(lr), f32(beta1), f32(beta2), f32(eps),
            f32(weight_decay), f32(bc1), f32(bc2),
        ],
    )
