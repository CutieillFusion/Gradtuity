"""CUDA-C twin of gradtuity.kernels.one_hot_kernels."""

from ..cuda_driver import kernel_module, f32, i32, launch, ptr

# Triton launches with grid (num_classes, B) and one program per output cell.
# We mirror that grid exactly (block = (1,1,1)) so behavior matches one-for-one.
_SRC = r"""
extern "C" __global__ void one_hot_kernel(
    const float* __restrict__ labels, float* __restrict__ out,
    int B, int num_classes, float on_value, float off_value
) {
    int col_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    if (row_idx >= B || col_idx >= num_classes) return;
    int label_int = (int)labels[row_idx];
    int out_idx = row_idx * num_classes + col_idx;
    out[out_idx] = (col_idx == label_int) ? on_value : off_value;
}
"""

_h = kernel_module(_SRC)


def one_hot_kernel_launch(grid, labels_ptr, out_ptr, B, num_classes, ON_VALUE, OFF_VALUE):
    # grid is (num_classes, B) per the Triton call site.
    launch(
        _h("one_hot_kernel"), (grid[0], grid[1], 1), (1, 1, 1),
        [
            ptr(labels_ptr), ptr(out_ptr),
            i32(B), i32(num_classes), f32(ON_VALUE), f32(OFF_VALUE),
        ],
    )
