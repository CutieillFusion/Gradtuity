"""
Triton kernels for loss operations.

Includes:
- mse_loss_kernel: Fused MSE loss computation
- mse_loss_backward_kernel: Fused MSE loss backward
- cross_entropy_forward_kernel: Fused cross-entropy forward (log-softmax + gather)
- cross_entropy_backward_kernel: Cross-entropy backward into logits
"""

import triton
import triton.language as tl


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
def cross_entropy_forward_kernel(
    logits_ptr: tl.pointer_type(tl.float32),
    targets_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    B: tl.int32,
    C: tl.int32,
    BLOCK_C: tl.constexpr,
):
    """
    Cross-entropy forward: out += sum_i (lse_i - logits[i, targets[i]]).

    One program per row. Computes numerically stable log-softmax (max-subtraction
    + logsumexp) per row, then gathers the target logit and accumulates loss.
    IMPORTANT: out_ptr MUST be zero-initialized before calling this kernel.

    Args:
        logits_ptr: Logits GPU pointer (float32*), shape (B, C).
        targets_ptr: Target class indices GPU pointer (float32*), shape (B,).
        out_ptr: Output scalar GPU pointer (float32*), must be pre-zeroed.
        B: Batch size.
        C: Number of classes.
        BLOCK_C: Block size for column dimension (compile-time constant).
    """
    row_idx = tl.program_id(0)

    if row_idx >= B:
        return

    # Row max (numerically stable)
    m = -float("inf")
    for col_start in range(0, C, BLOCK_C):
        col_offsets = col_start + tl.arange(0, BLOCK_C)
        mask = col_offsets < C
        indices = row_idx * C + col_offsets
        x = tl.load(logits_ptr + indices, mask=mask, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        m = tl.maximum(m, local_max)

    # sumexp = sum(exp(logits - m))
    sumexp = 0.0
    for col_start in range(0, C, BLOCK_C):
        col_offsets = col_start + tl.arange(0, BLOCK_C)
        mask = col_offsets < C
        indices = row_idx * C + col_offsets
        x = tl.load(logits_ptr + indices, mask=mask, other=0.0)
        exp_val = tl.where(mask, tl.exp(x - m), 0.0)
        sumexp += tl.sum(exp_val, axis=0)

    lse = tl.log(sumexp) + m

    # Load target index and logit at target
    y = tl.load(targets_ptr + row_idx).to(tl.int32)
    logit_y = tl.load(logits_ptr + row_idx * C + y)
    loss_i = lse - logit_y
    tl.atomic_add(out_ptr, loss_i)


@triton.jit
def cross_entropy_backward_kernel(
    dlogits_ptr: tl.pointer_type(tl.float32),
    logits_ptr: tl.pointer_type(tl.float32),
    targets_ptr: tl.pointer_type(tl.float32),
    scale: tl.float32,
    B: tl.int32,
    C: tl.int32,
    BLOCK_C: tl.constexpr,
):
    """
    Cross-entropy backward: dlogits += scale * (softmax(logits) - one_hot(targets)).

    One program per row. Recomputes softmax from logits, then accumulates
    gradient into dlogits. Caller must ensure dlogits_ptr is zero-initialized
    (e.g. via ensure_grad) before launch if accumulation is desired.

    Args:
        dlogits_ptr: Gradient for logits, accumulated in-place (float32*).
        logits_ptr: Logits GPU pointer (float32*), shape (B, C).
        targets_ptr: Target class indices GPU pointer (float32*), shape (B,).
        scale: Scaling factor (out_grad / B for mean, out_grad for sum).
        B: Batch size.
        C: Number of classes.
        BLOCK_C: Block size for column dimension (compile-time constant).
    """
    row_idx = tl.program_id(0)

    if row_idx >= B:
        return

    # Recompute row max and lse (same as forward)
    m = -float("inf")
    for col_start in range(0, C, BLOCK_C):
        col_offsets = col_start + tl.arange(0, BLOCK_C)
        mask = col_offsets < C
        indices = row_idx * C + col_offsets
        x = tl.load(logits_ptr + indices, mask=mask, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        m = tl.maximum(m, local_max)

    sumexp = 0.0
    for col_start in range(0, C, BLOCK_C):
        col_offsets = col_start + tl.arange(0, BLOCK_C)
        mask = col_offsets < C
        indices = row_idx * C + col_offsets
        x = tl.load(logits_ptr + indices, mask=mask, other=0.0)
        exp_val = tl.where(mask, tl.exp(x - m), 0.0)
        sumexp += tl.sum(exp_val, axis=0)

    lse = tl.log(sumexp) + m

    # Load target for this row
    y = tl.load(targets_ptr + row_idx).to(tl.int32)

    # For each column block: softmax - one_hot(y), then dlogits += scale * grad
    for col_start in range(0, C, BLOCK_C):
        col_offsets = col_start + tl.arange(0, BLOCK_C)
        mask = col_offsets < C
        indices = row_idx * C + col_offsets
        x = tl.load(logits_ptr + indices, mask=mask, other=0.0)
        softmax_val = tl.where(mask, tl.exp(x - lse), 0.0)
        one_hot = tl.where(col_offsets == y, 1.0, 0.0)
        grad_val = softmax_val - one_hot
        # Accumulate into dlogits
        dlogits_cur = tl.load(dlogits_ptr + indices, mask=mask, other=0.0)
        tl.store(dlogits_ptr + indices, dlogits_cur + scale * grad_val, mask=mask)
