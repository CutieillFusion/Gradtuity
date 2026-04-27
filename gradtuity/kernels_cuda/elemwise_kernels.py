"""CUDA-C twins of gradtuity.kernels.elemwise_kernels."""

from ..cuda_driver import kernel_module, f32, i32, launch, ptr

# ---------------------------------------------------------------------------
# Sources — one extern "C" __global__ per kernel, behavior matched to Triton.
# Kept as a single source string so NVRTC compiles them all at once and the
# kernel objects share a CUmodule.
# ---------------------------------------------------------------------------
_SRC = r"""
#define IDX_GUARD(n) \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= (n)) return;

extern "C" __global__ void add_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int numel
) {
    IDX_GUARD(numel)
    c[idx] = a[idx] + b[idx];
}

extern "C" __global__ void mul_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int numel
) {
    IDX_GUARD(numel)
    c[idx] = a[idx] * b[idx];
}

extern "C" __global__ void mul_scalar_kernel(
    const float* __restrict__ a, float scalar,
    float* __restrict__ c, int numel
) {
    IDX_GUARD(numel)
    c[idx] = a[idx] * scalar;
}

extern "C" __global__ void mul_scalar_inplace_kernel(
    float* __restrict__ x, float scalar, int numel
) {
    IDX_GUARD(numel)
    x[idx] = x[idx] * scalar;
}

extern "C" __global__ void add_inplace_kernel(
    float* __restrict__ a, const float* __restrict__ b, int numel
) {
    IDX_GUARD(numel)
    a[idx] = a[idx] + b[idx];
}

extern "C" __global__ void mul_backward_kernel(
    float* __restrict__ grad,
    const float* __restrict__ out_grad,
    const float* __restrict__ other,
    int numel
) {
    IDX_GUARD(numel)
    grad[idx] = grad[idx] + out_grad[idx] * other[idx];
}

extern "C" __global__ void scale_backward_kernel(
    float* __restrict__ grad,
    const float* __restrict__ out_grad,
    float scalar, int numel
) {
    IDX_GUARD(numel)
    grad[idx] = grad[idx] + out_grad[idx] * scalar;
}

extern "C" __global__ void relu_kernel(
    const float* __restrict__ y, float* __restrict__ z, int numel
) {
    IDX_GUARD(numel)
    float v = y[idx];
    z[idx] = v > 0.0f ? v : 0.0f;
}

extern "C" __global__ void relu_backward_kernel(
    float* __restrict__ dy,
    const float* __restrict__ dz,
    const float* __restrict__ y,
    int numel
) {
    IDX_GUARD(numel)
    float m = y[idx] > 0.0f ? 1.0f : 0.0f;
    dy[idx] = dy[idx] + dz[idx] * m;
}

extern "C" __global__ void relu_mask_mul_kernel(
    float* __restrict__ c,
    const float* __restrict__ a,
    const float* __restrict__ y,
    int numel
) {
    IDX_GUARD(numel)
    float m = y[idx] > 0.0f ? 1.0f : 0.0f;
    c[idx] = a[idx] * m;
}

// GELU (tanh approximation). Matches Triton's branchy tanh form so numerics
// agree to within fast-math tolerance.
__device__ __forceinline__ float _tanh_branchy(float u) {
    if (u >= 0.0f) {
        return 2.0f / (1.0f + __expf(-2.0f * u)) - 1.0f;
    } else {
        float e = __expf(2.0f * u);
        return (e - 1.0f) / (e + 1.0f);
    }
}

extern "C" __global__ void gelu_kernel(
    const float* __restrict__ in, float* __restrict__ out, int numel
) {
    IDX_GUARD(numel)
    float x = in[idx];
    float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    float t = _tanh_branchy(u);
    out[idx] = 0.5f * x * (1.0f + t);
}

extern "C" __global__ void gelu_backward_kernel(
    float* __restrict__ dx,
    const float* __restrict__ dy,
    const float* __restrict__ x_in,
    int numel
) {
    IDX_GUARD(numel)
    float x = x_in[idx];
    float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    float t = _tanh_branchy(u);
    float sech2 = 1.0f - t * t;
    float du_dx = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x * x);
    float dgelu_dx = 0.5f * (1.0f + t) + 0.5f * x * sech2 * du_dx;
    dx[idx] = dy[idx] * dgelu_dx;
}
"""

_h = kernel_module(_SRC)


def _grid1d(grid: tuple[int]) -> tuple[int, int, int]:
    return (grid[0], 1, 1)


# ---------------------------------------------------------------------------
# Wrappers — signatures mirror gradtuity.kernels.elemwise_kernels exactly so
# call sites can swap one for the other through the dispatcher.
# ---------------------------------------------------------------------------
def add_kernel_launch(grid, a_ptr, b_ptr, c_ptr, numel, BLOCK):
    launch(_h("add_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(a_ptr), ptr(b_ptr), ptr(c_ptr), i32(numel)])


def mul_kernel_launch(grid, a_ptr, b_ptr, c_ptr, numel, BLOCK):
    launch(_h("mul_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(a_ptr), ptr(b_ptr), ptr(c_ptr), i32(numel)])


def mul_scalar_kernel_launch(grid, a_ptr, scalar, c_ptr, numel, BLOCK):
    launch(_h("mul_scalar_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(a_ptr), f32(scalar), ptr(c_ptr), i32(numel)])


def mul_scalar_inplace_kernel_launch(grid, x_ptr, scalar, numel, BLOCK):
    launch(_h("mul_scalar_inplace_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(x_ptr), f32(scalar), i32(numel)])


def add_inplace_kernel_launch(grid, a_ptr, b_ptr, numel, BLOCK):
    launch(_h("add_inplace_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(a_ptr), ptr(b_ptr), i32(numel)])


def mul_backward_kernel_launch(grid, grad_ptr, out_grad_ptr, other_ptr, numel, BLOCK):
    launch(_h("mul_backward_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(grad_ptr), ptr(out_grad_ptr), ptr(other_ptr), i32(numel)])


def scale_backward_kernel_launch(grid, grad_ptr, out_grad_ptr, scalar, numel, BLOCK):
    launch(_h("scale_backward_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(grad_ptr), ptr(out_grad_ptr), f32(scalar), i32(numel)])


def relu_kernel_launch(grid, y_ptr, z_ptr, numel, BLOCK):
    launch(_h("relu_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(y_ptr), ptr(z_ptr), i32(numel)])


def relu_backward_kernel_launch(grid, dy_ptr, dz_ptr, y_ptr, numel, BLOCK):
    launch(_h("relu_backward_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(dy_ptr), ptr(dz_ptr), ptr(y_ptr), i32(numel)])


def relu_mask_mul_kernel_launch(grid, c_ptr, a_ptr, y_ptr, numel, BLOCK):
    launch(_h("relu_mask_mul_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(c_ptr), ptr(a_ptr), ptr(y_ptr), i32(numel)])


def gelu_kernel_launch(grid, in_ptr, out_ptr, n_elements, BLOCK):
    launch(_h("gelu_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(in_ptr), ptr(out_ptr), i32(n_elements)])


def gelu_backward_kernel_launch(grid, dx_ptr, dy_ptr, x_ptr, n_elements, BLOCK):
    launch(_h("gelu_backward_kernel"), _grid1d(grid), (BLOCK, 1, 1),
           [ptr(dx_ptr), ptr(dy_ptr), ptr(x_ptr), i32(n_elements)])
