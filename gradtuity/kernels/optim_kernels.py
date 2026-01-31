"""
Triton kernels for optimization and utility operations.

Includes:
- fill_kernel: Fill a tensor with a constant value
- sgd_update_kernel: In-place SGD parameter update
- adamw_step_kernel: Fused in-place AdamW update (m, v, p)
"""

import triton
import triton.language as tl


@triton.jit
def fill_kernel(
    dst_ptr: tl.pointer_type(tl.float32),
    value: tl.float32,
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Fill a GPU buffer with a constant value.

    Args:
        dst_ptr: Destination GPU pointer (float32*).
        value: The scalar value to fill with (float32).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    # Compute the block's starting position
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # Mask for out-of-bounds elements
    mask = offsets < numel

    # Store the value
    tl.store(dst_ptr + offsets, value, mask=mask)


@triton.jit
def sgd_update_kernel(
    param_ptr: tl.pointer_type(tl.float32),
    grad_ptr: tl.pointer_type(tl.float32),
    lr: tl.float32,
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    In-place SGD update: param -= lr * grad

    Args:
        param_ptr: Parameter GPU pointer (float32*).
        grad_ptr: Gradient GPU pointer (float32*).
        lr: Learning rate (float32).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    # Compute the block's starting position
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # Mask for out-of-bounds elements
    mask = offsets < numel

    # Load param and grad
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)

    # Update: param -= lr * grad
    param = param - lr * grad

    # Store back
    tl.store(param_ptr + offsets, param, mask=mask)


@triton.jit
def adamw_step_kernel(
    p_ptr: tl.pointer_type(tl.float32),
    g_ptr: tl.pointer_type(tl.float32),
    m_ptr: tl.pointer_type(tl.float32),
    v_ptr: tl.pointer_type(tl.float32),
    n_elements: tl.int32,
    lr: tl.float32,
    beta1: tl.float32,
    beta2: tl.float32,
    eps: tl.float32,
    weight_decay: tl.float32,
    bc1: tl.float32,
    bc2: tl.float32,
    BLOCK: tl.constexpr,
):
    """
    Fused in-place AdamW update: update moments m, v and parameter p.

    Args:
        p_ptr: Parameter GPU pointer (float32*).
        g_ptr: Gradient GPU pointer (float32*).
        m_ptr: First moment (exp_avg) GPU pointer (float32*).
        v_ptr: Second moment (exp_avg_sq) GPU pointer (float32*).
        n_elements: Total number of elements.
        lr: Learning rate.
        beta1: First beta.
        beta2: Second beta.
        eps: Epsilon for numerical stability.
        weight_decay: Decoupled weight decay.
        bc1: Bias correction for m: 1 / (1 - beta1^t).
        bc2: Bias correction for v: 1 / (1 - beta2^t).
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask)
    g = tl.load(g_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    # Update moments
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g

    # Bias-corrected estimates
    m_hat = m * bc1
    v_hat = v * bc2

    # Update: p -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)
    update = m_hat / (tl.sqrt(v_hat) + eps)
    p = p - lr * update - lr * weight_decay * p

    tl.store(m_ptr + offsets, m, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)
    tl.store(p_ptr + offsets, p, mask=mask)
