"""
Triton kernels for reduction operations.

Includes:
- sum_all_kernel: Reduce all elements to a single scalar
- sum_axis0_kernel: Reduce along axis 0 (for bias gradient)
- add_scalar_inplace_kernel: Add a scalar to all elements (for sum backward)
- add_bias_kernel: Broadcast add bias to 2D tensor
- mse_loss_kernel: Fused MSE loss computation
- mse_loss_backward_kernel: Fused MSE loss backward
- argmax_axis1_kernel: Argmax along axis 1 (returns indices)
"""

import triton
import triton.language as tl


@triton.jit
def sum_all_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Reduce all elements to a single scalar: out = sum(x)

    Uses atomic_add for correctness when multiple blocks contribute.
    IMPORTANT: out_ptr MUST be zero-initialized before calling this kernel.

    Args:
        x_ptr: Input tensor GPU pointer (float32*).
        out_ptr: Output scalar GPU pointer (float32*), must be pre-zeroed.
        numel: Total number of elements in input.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    # Load elements (masked loads return 0 for out-of-bounds)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute partial sum within this block
    partial_sum = tl.sum(x, axis=0)

    # Atomically add to output (only one thread per block does this)
    tl.atomic_add(out_ptr, partial_sum)


@triton.jit
def sum_axis0_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    rows: tl.int32,
    cols: tl.int32,
    BLOCK_ROWS: tl.constexpr,
):
    """
    Reduce along axis 0: out[j] += sum_{i=0..rows-1} x[i, j]

    Used for bias gradient computation: db += sum over batch of dY.
    Uses atomic_add for correctness.
    IMPORTANT: out_ptr should be pre-zeroed (typically by zero_grad).

    Args:
        x_ptr: Input 2D tensor GPU pointer (float32*), shape (rows, cols).
        out_ptr: Output 1D tensor GPU pointer (float32*), shape (cols,).
        rows: Number of rows (batch size).
        cols: Number of columns (features).
        BLOCK_ROWS: Number of rows to process per block (compile-time constant).
    """
    # Each program handles one column
    col_idx = tl.program_id(0)

    if col_idx >= cols:
        return

    # Process rows in blocks
    partial_sum = 0.0
    for row_start in range(0, rows, BLOCK_ROWS):
        row_offsets = row_start + tl.arange(0, BLOCK_ROWS)
        mask = row_offsets < rows

        # Linear index: row * cols + col
        indices = row_offsets * cols + col_idx
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)
        partial_sum += tl.sum(x, axis=0)

    # Atomically add to output
    tl.atomic_add(out_ptr + col_idx, partial_sum)


@triton.jit
def add_scalar_inplace_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    scalar_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Add a scalar to all elements in-place: x += scalar

    Used for sum backward: dZ += dloss (broadcast scalar gradient).

    Args:
        x_ptr: Tensor GPU pointer (float32*), modified in-place.
        scalar_ptr: Pointer to scalar value (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    # Load the scalar (same value for all threads)
    scalar = tl.load(scalar_ptr)

    # Load, add, store
    x = tl.load(x_ptr + offsets, mask=mask)
    x = x + scalar
    tl.store(x_ptr + offsets, x, mask=mask)


@triton.jit
def add_bias_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    y_ptr: tl.pointer_type(tl.float32),
    rows: tl.int32,
    cols: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Add bias with broadcasting: Y[i, j] = X[i, j] + b[j]

    Args:
        x_ptr: Input 2D tensor GPU pointer (float32*), shape (rows, cols).
        b_ptr: Bias 1D tensor GPU pointer (float32*), shape (cols,).
        y_ptr: Output 2D tensor GPU pointer (float32*), shape (rows, cols).
        rows: Number of rows (batch size).
        cols: Number of columns (features).
        BLOCK: Block size for flattened iteration (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    numel = rows * cols
    mask = offsets < numel

    # Compute column indices for bias lookup
    col_indices = offsets % cols

    # Load input and bias
    x = tl.load(x_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + col_indices, mask=mask)

    # Compute output
    y = x + b

    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def mse_loss_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Fused MSE loss: out = sum((A - B)^2)

    Computes the sum of squared differences in a single pass.
    IMPORTANT: out_ptr MUST be zero-initialized before calling this kernel.

    The final loss value should be divided by numel on the CPU side
    to get the mean: loss = kernel_result / numel

    Args:
        a_ptr: First input tensor GPU pointer (float32*).
        b_ptr: Second input tensor GPU pointer (float32*).
        out_ptr: Output scalar GPU pointer (float32*), must be pre-zeroed.
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    # Load elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Compute (a - b)^2
    diff = a - b
    squared = diff * diff

    # Compute partial sum within this block
    partial_sum = tl.sum(squared, axis=0)

    # Atomically add to output
    tl.atomic_add(out_ptr, partial_sum)


@triton.jit
def mse_loss_backward_kernel(
    grad_a_ptr: tl.pointer_type(tl.float32),
    grad_b_ptr: tl.pointer_type(tl.float32),
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    scale: tl.float32,
    numel: tl.int32,
    compute_grad_a: tl.int32,
    compute_grad_b: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Fused MSE loss backward: accumulate gradients for both inputs.

    For MSE loss L = sum((A - B)^2) / N:
    - dL/dA = 2 * (A - B) / N
    - dL/dB = -2 * (A - B) / N = -dL/dA

    This kernel accumulates: grad_a += scale * 2 * (A - B)
                             grad_b += scale * (-2) * (A - B)
    where scale = out_grad / numel (typically 1/numel for mean).

    Args:
        grad_a_ptr: Gradient for A, accumulated in-place (float32*).
        grad_b_ptr: Gradient for B, accumulated in-place (float32*).
        a_ptr: First input tensor (float32*).
        b_ptr: Second input tensor (float32*).
        scale: Scaling factor (out_grad / numel for MSE).
        numel: Total number of elements.
        compute_grad_a: 1 if grad_a should be computed, 0 otherwise.
        compute_grad_b: 1 if grad_b should be computed, 0 otherwise.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    # Load input values
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Compute gradient contribution: 2 * (a - b) * scale
    diff = a - b
    grad_val = 2.0 * diff * scale

    # Accumulate into grad_a if needed
    if compute_grad_a == 1:
        grad_a = tl.load(grad_a_ptr + offsets, mask=mask, other=0.0)
        grad_a = grad_a + grad_val
        tl.store(grad_a_ptr + offsets, grad_a, mask=mask)

    # Accumulate into grad_b if needed (negative gradient)
    if compute_grad_b == 1:
        grad_b = tl.load(grad_b_ptr + offsets, mask=mask, other=0.0)
        grad_b = grad_b - grad_val
        tl.store(grad_b_ptr + offsets, grad_b, mask=mask)


@triton.jit
def argmax_axis1_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    rows: tl.int32,
    cols: tl.int32,
    BLOCK_COLS: tl.constexpr,
):
    """
    Argmax along axis 1: out[i] = argmax_j(x[i, j])

    Each program handles one row and finds the index of the maximum element.
    Output is stored as float32 (indices cast to float for consistency).

    Args:
        x_ptr: Input 2D tensor GPU pointer (float32*), shape (rows, cols).
        out_ptr: Output 1D tensor GPU pointer (float32*), shape (rows,).
        rows: Number of rows.
        cols: Number of columns.
        BLOCK_COLS: Block size for column iteration (compile-time constant).
    """
    row_idx = tl.program_id(0)

    if row_idx >= rows:
        return

    # Find max value and its index for this row
    max_val = -float("inf")
    max_idx = 0

    for col_start in range(0, cols, BLOCK_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < cols

        # Linear index: row * cols + col
        indices = row_idx * cols + col_offsets
        x = tl.load(x_ptr + indices, mask=mask, other=-float("inf"))

        # Find local max and its position
        local_max = tl.max(x, axis=0)

        if local_max > max_val:
            max_val = local_max
            # Find which element is the max
            is_max = x == local_max
            # Get the first index where is_max is true
            # Use argmax on the mask (converted to float)
            is_max_float = is_max.to(tl.float32)
            local_argmax = tl.argmax(is_max_float, axis=0)
            max_idx = col_start + local_argmax

    # Store the index (as float32)
    tl.store(out_ptr + row_idx, max_idx.to(tl.float32))
