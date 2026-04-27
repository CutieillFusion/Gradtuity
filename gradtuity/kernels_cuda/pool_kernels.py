"""CUDA-C twins of gradtuity.kernels.pool_kernels."""

from ..cuda_driver import kernel_module, i32, launch, ptr

# Triton processes BLOCK_ELEMS output elements sequentially per program;
# CUDA does the same total work with one thread per output element.
_SRC = r"""
extern "C" __global__ void maxpool2d_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    float* __restrict__ idx_out,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int kH, int kW
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel_out = N * C * H_out * W_out;
    if (out_idx >= numel_out) return;

    int n = out_idx / (C * H_out * W_out);
    int rest = out_idx % (C * H_out * W_out);
    int c = rest / (H_out * W_out);
    rest = rest % (H_out * W_out);
    int h_out = rest / W_out;
    int w_out = rest % W_out;

    int h_start = h_out * stride_h;
    int w_start = w_out * stride_w;

    float max_val = -1e30f;
    int max_idx = 0;
    int idx = 0;
    for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
            int h_in = h_start + kh;
            int w_in = w_start + kw;
            if (h_in < H && w_in < W) {
                float v = x[n * (C * H * W) + c * (H * W) + h_in * W + w_in];
                if (v > max_val) {
                    max_val = v;
                    max_idx = idx;
                }
            }
            idx++;
        }
    }
    out[out_idx] = max_val;
    idx_out[out_idx] = (float)max_idx;
}

extern "C" __global__ void maxpool2d_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ idx_in,
    float* __restrict__ grad_in,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int kW
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel_out = N * C * H_out * W_out;
    if (out_idx >= numel_out) return;

    int n = out_idx / (C * H_out * W_out);
    int rest = out_idx % (C * H_out * W_out);
    int c = rest / (H_out * W_out);
    rest = rest % (H_out * W_out);
    int h_out = rest / W_out;
    int w_out = rest % W_out;

    int max_idx = (int)idx_in[out_idx];
    float g = grad_out[out_idx];
    int h_start = h_out * stride_h;
    int w_start = w_out * stride_w;
    int kh = max_idx / kW;
    int kw = max_idx % kW;
    int h_in = h_start + kh;
    int w_in = w_start + kw;

    int in_offset = n * (C * H * W) + c * (H * W) + h_in * W + w_in;
    atomicAdd(grad_in + in_offset, g);
}
"""

_h = kernel_module(_SRC)


def maxpool2d_forward_kernel_launch(
    grid, x_ptr, out_ptr, idx_ptr,
    N, C, H, W, H_out, W_out, stride_h, stride_w,
    BLOCK_KH, BLOCK_KW, BLOCK_ELEMS,
):
    launch(
        _h("maxpool2d_forward_kernel"), (grid[0], 1, 1), (BLOCK_ELEMS, 1, 1),
        [
            ptr(x_ptr), ptr(out_ptr), ptr(idx_ptr),
            i32(N), i32(C), i32(H), i32(W),
            i32(H_out), i32(W_out),
            i32(stride_h), i32(stride_w),
            i32(BLOCK_KH), i32(BLOCK_KW),
        ],
    )


def maxpool2d_backward_kernel_launch(
    grid, grad_out_ptr, idx_ptr, grad_in_ptr,
    N, C, H, W, H_out, W_out, stride_h, stride_w,
    BLOCK_KW, BLOCK_ELEMS,
):
    launch(
        _h("maxpool2d_backward_kernel"), (grid[0], 1, 1), (BLOCK_ELEMS, 1, 1),
        [
            ptr(grad_out_ptr), ptr(idx_ptr), ptr(grad_in_ptr),
            i32(N), i32(C), i32(H), i32(W),
            i32(H_out), i32(W_out),
            i32(stride_h), i32(stride_w),
            i32(BLOCK_KW),
        ],
    )
