"""CUDA-C twins of gradtuity.kernels.conv_kernels."""

from ..cuda_driver import kernel_module, i32, launch, ptr

# Note on `im2col_kernel`: the Triton version processes BLOCK_ROW x BLOCK_COL
# output cells per program in nested loops. The CUDA equivalent uses one thread
# per (row, col) pair — same total work, simpler indexing.
_SRC = r"""
extern "C" __global__ void im2col_kernel(
    const float* __restrict__ x, float* __restrict__ col,
    int N, int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out,
    int BLOCK_ROW, int BLOCK_COL
) {
    int num_rows = N * H_out * W_out;
    int num_cols = C * kH * kW;
    int row_idx = blockIdx.x * BLOCK_ROW + threadIdx.y;
    int col_idx = blockIdx.y * BLOCK_COL + threadIdx.x;
    if (row_idx >= num_rows || col_idx >= num_cols) return;

    int n = row_idx / (H_out * W_out);
    int rest = row_idx % (H_out * W_out);
    int h_out = rest / W_out;
    int w_out = rest % W_out;

    int c = col_idx / (kH * kW);
    int rest_c = col_idx % (kH * kW);
    int kh = rest_c / kW;
    int kw = rest_c % kW;

    int h_in = h_out * stride_h - pad_h + kh;
    int w_in = w_out * stride_w - pad_w + kw;

    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = x[n * (C * H * W) + c * (H * W) + h_in * W + w_in];
    }
    col[row_idx * num_cols + col_idx] = val;
}

extern "C" __global__ void im2col_kernel_2d(
    const float* __restrict__ x, float* __restrict__ col,
    int N, int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out, int num_cols
) {
    int row_idx = blockIdx.x;
    int col_off = blockIdx.y * blockDim.x + threadIdx.x;
    int num_rows = N * H_out * W_out;
    if (row_idx >= num_rows || col_off >= num_cols) return;

    int n = row_idx / (H_out * W_out);
    int rest = row_idx % (H_out * W_out);
    int h_out = rest / W_out;
    int w_out = rest % W_out;
    int h_start = h_out * stride_h - pad_h;
    int w_start = w_out * stride_w - pad_w;

    int c = col_off / (kH * kW);
    int rest_c = col_off % (kH * kW);
    int kh = rest_c / kW;
    int kw = rest_c % kW;
    int h_in = h_start + kh;
    int w_in = w_start + kw;

    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = x[n * (C * H * W) + c * (H * W) + h_in * W + w_in];
    }
    col[row_idx * num_cols + col_off] = val;
}

extern "C" __global__ void col2im_kernel(
    const float* __restrict__ col, float* __restrict__ x,
    int N, int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out, int num_cols
) {
    int row_idx = blockIdx.x;
    int col_off = blockIdx.y * blockDim.x + threadIdx.x;
    int num_rows = N * H_out * W_out;
    if (row_idx >= num_rows || col_off >= num_cols) return;

    int n = row_idx / (H_out * W_out);
    int rest = row_idx % (H_out * W_out);
    int h_out = rest / W_out;
    int w_out = rest % W_out;
    int h_start = h_out * stride_h - pad_h;
    int w_start = w_out * stride_w - pad_w;

    int c = col_off / (kH * kW);
    int rest_c = col_off % (kH * kW);
    int kh = rest_c / kW;
    int kw = rest_c % kW;
    int h_in = h_start + kh;
    int w_in = w_start + kw;

    float val = col[row_idx * num_cols + col_off];
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        atomicAdd(x + n * (C * H * W) + c * (H * W) + h_in * W + w_in, val);
    }
}
"""

_h = kernel_module(_SRC)


def im2col_kernel_launch(
    grid, x_ptr, col_ptr,
    N, C, H, W, kH, kW,
    stride_h, stride_w, pad_h, pad_w,
    H_out, W_out, BLOCK_ROW, BLOCK_COL,
):
    launch(
        _h("im2col_kernel"),
        (grid[0], grid[1], 1), (BLOCK_COL, BLOCK_ROW, 1),
        [
            ptr(x_ptr), ptr(col_ptr),
            i32(N), i32(C), i32(H), i32(W),
            i32(kH), i32(kW),
            i32(stride_h), i32(stride_w), i32(pad_h), i32(pad_w),
            i32(H_out), i32(W_out),
            i32(BLOCK_ROW), i32(BLOCK_COL),
        ],
    )


def im2col_kernel_2d_launch(
    grid, x_ptr, col_ptr,
    N, C, H, W, kH, kW,
    stride_h, stride_w, pad_h, pad_w,
    H_out, W_out, num_cols, BLOCK,
):
    launch(
        _h("im2col_kernel_2d"),
        (grid[0], grid[1], 1), (BLOCK, 1, 1),
        [
            ptr(x_ptr), ptr(col_ptr),
            i32(N), i32(C), i32(H), i32(W),
            i32(kH), i32(kW),
            i32(stride_h), i32(stride_w), i32(pad_h), i32(pad_w),
            i32(H_out), i32(W_out), i32(num_cols),
        ],
    )


def col2im_kernel_launch(
    grid, col_ptr, x_ptr,
    N, C, H, W, kH, kW,
    stride_h, stride_w, pad_h, pad_w,
    H_out, W_out, num_cols, BLOCK,
):
    launch(
        _h("col2im_kernel"),
        (grid[0], grid[1], 1), (BLOCK, 1, 1),
        [
            ptr(col_ptr), ptr(x_ptr),
            i32(N), i32(C), i32(H), i32(W),
            i32(kH), i32(kW),
            i32(stride_h), i32(stride_w), i32(pad_h), i32(pad_w),
            i32(H_out), i32(W_out), i32(num_cols),
        ],
    )
