"""
Triton kernel for one-hot encoding.

one_hot_kernel: Given labels (B,) float32 (class indices 0.0, 1.0, ...),
write output (B, num_classes) float32 with ON_VALUE at the label index and OFF_VALUE elsewhere.
"""

import triton
import triton.language as tl


# This kernel is actually slower than the Python implementation.
# TODO: Fix this or Remove it but its a preprocess step so it's not a big deal.
@triton.jit
def one_hot_kernel(
    labels_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    B: tl.int32,
    num_classes: tl.int32,
    ON_VALUE: tl.float32,
    OFF_VALUE: tl.float32,
):
    """
    One-hot encode labels: out[row, col] = ON_VALUE if col == labels[row] else OFF_VALUE.

    Grid: (num_classes, B). Each program handles one output element.

    Args:
        labels_ptr: Input labels GPU pointer (float32*), shape (B,).
        out_ptr: Output GPU pointer (float32*), shape (B, num_classes).
        B: Batch size.
        num_classes: Number of classes.
        ON_VALUE: Value at the correct class index (e.g. 1.0).
        OFF_VALUE: Value elsewhere (e.g. -1.0 for MSE).
    """
    col_idx = tl.program_id(0)
    row_idx = tl.program_id(1)

    if row_idx >= B or col_idx >= num_classes:
        return

    label_val = tl.load(labels_ptr + row_idx)
    label_int = tl.cast(label_val, tl.int32)
    out_idx = row_idx * num_classes + col_idx
    val = tl.where(col_idx == label_int, ON_VALUE, OFF_VALUE)
    tl.store(out_ptr + out_idx, val)
