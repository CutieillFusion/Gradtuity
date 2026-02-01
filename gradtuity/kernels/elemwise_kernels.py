"""
Triton kernels for elementwise operations.

Includes:
- add_kernel: Elementwise addition (C = A + B)
- relu_kernel: ReLU forward (Z = max(Y, 0))
- relu_backward_kernel: ReLU backward with accumulation (dY += dZ * (Y > 0))
- relu_mask_mul_kernel: Multiply by ReLU mask (C = A * (Y > 0))
- add_inplace_kernel: In-place addition for gradient accumulation (A += B)
- mul_kernel: Elementwise multiplication (C = A * B)
- mul_scalar_kernel: Scalar multiplication (C = A * scalar)
- mul_backward_kernel: Elementwise multiply backward with accumulation
"""

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Elementwise addition: C = A + B

    Args:
        a_ptr: First input tensor GPU pointer (float32*).
        b_ptr: Second input tensor GPU pointer (float32*).
        c_ptr: Output tensor GPU pointer (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def relu_kernel(
    y_ptr: tl.pointer_type(tl.float32),
    z_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    ReLU forward: Z = max(Y, 0)

    Args:
        y_ptr: Input tensor GPU pointer (float32*).
        z_ptr: Output tensor GPU pointer (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.maximum(y, 0.0)

    tl.store(z_ptr + offsets, z, mask=mask)


@triton.jit
def relu_backward_kernel(
    dy_ptr: tl.pointer_type(tl.float32),
    dz_ptr: tl.pointer_type(tl.float32),
    y_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    ReLU backward with accumulation: dY += dZ * (Y > 0)

    Computes the gradient mask from the original input Y (not the output Z),
    since Z >= 0 always, so Z > 0 is not equivalent to Y > 0.

    Args:
        dy_ptr: Gradient w.r.t. input Y, accumulated in-place (float32*).
        dz_ptr: Gradient w.r.t. output Z (float32*).
        y_ptr: Original input Y for mask computation (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    # Load values
    dy = tl.load(dy_ptr + offsets, mask=mask)
    dz = tl.load(dz_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute relu mask: 1.0 where Y > 0, else 0.0
    relu_mask = (y > 0.0).to(tl.float32)

    # Accumulate: dY += dZ * mask
    dy = dy + dz * relu_mask

    tl.store(dy_ptr + offsets, dy, mask=mask)


@triton.jit
def relu_mask_mul_kernel(
    c_ptr: tl.pointer_type(tl.float32),
    a_ptr: tl.pointer_type(tl.float32),
    y_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Multiply by ReLU mask (non-accumulating): C = A * (Y > 0)

    Used for computing gradients through fused linear+relu operations.
    Unlike relu_backward_kernel, this writes directly (no accumulation).

    Args:
        c_ptr: Output tensor GPU pointer (float32*).
        a_ptr: Input tensor to be masked (float32*).
        y_ptr: ReLU output for mask computation (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    # Load values
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Compute relu mask: 1.0 where Y > 0, else 0.0
    relu_mask = (y > 0.0).to(tl.float32)

    # C = A * mask
    c = a * relu_mask

    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def add_inplace_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    In-place addition for gradient accumulation: A += B

    Args:
        a_ptr: Destination tensor GPU pointer (float32*), modified in-place.
        b_ptr: Source tensor GPU pointer (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    a = a + b

    tl.store(a_ptr + offsets, a, mask=mask)


@triton.jit
def mul_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Elementwise multiplication: C = A * B

    Args:
        a_ptr: First input tensor GPU pointer (float32*).
        b_ptr: Second input tensor GPU pointer (float32*).
        c_ptr: Output tensor GPU pointer (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a * b

    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def mul_scalar_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    scalar: tl.float32,
    c_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Scalar multiplication: C = A * scalar

    Args:
        a_ptr: Input tensor GPU pointer (float32*).
        scalar: Scalar value to multiply by.
        c_ptr: Output tensor GPU pointer (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    a = tl.load(a_ptr + offsets, mask=mask)
    c = a * scalar

    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def mul_backward_kernel(
    grad_ptr: tl.pointer_type(tl.float32),
    out_grad_ptr: tl.pointer_type(tl.float32),
    other_ptr: tl.pointer_type(tl.float32),
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Elementwise multiply backward with accumulation: grad += out_grad * other

    For C = A * B:
    - dA += dC * B (call with other_ptr = B)
    - dB += dC * A (call with other_ptr = A)

    Args:
        grad_ptr: Gradient to accumulate into (float32*), modified in-place.
        out_grad_ptr: Upstream gradient (float32*).
        other_ptr: The other operand for gradient computation (float32*).
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    grad = tl.load(grad_ptr + offsets, mask=mask)
    out_grad = tl.load(out_grad_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)

    grad = grad + out_grad * other

    tl.store(grad_ptr + offsets, grad, mask=mask)


@triton.jit
def scale_backward_kernel(
    grad_ptr: tl.pointer_type(tl.float32),
    out_grad_ptr: tl.pointer_type(tl.float32),
    scalar: tl.float32,
    numel: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Scalar multiply backward with accumulation: grad += out_grad * scalar

    For C = A * scalar:
    - dA += dC * scalar

    Args:
        grad_ptr: Gradient to accumulate into (float32*), modified in-place.
        out_grad_ptr: Upstream gradient (float32*).
        scalar: The scalar value used in forward pass.
        numel: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    grad = tl.load(grad_ptr + offsets, mask=mask)
    out_grad = tl.load(out_grad_ptr + offsets, mask=mask)

    grad = grad + out_grad * scalar

    tl.store(grad_ptr + offsets, grad, mask=mask)


@triton.jit
def gelu_kernel(
    in_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    n_elements: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    GELU forward (tanh approximation): out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))

    Args:
        in_ptr: Input tensor GPU pointer (float32*).
        out_ptr: Output tensor GPU pointer (float32*).
        n_elements: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # sqrt(2/pi) ~ 0.7978845608028654, GELU x^3 coef = 0.044715
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    # tanh(u): for u>=0 use 2/(1+exp(-2u))-1 (avoids overflow); for u<0 use (exp(2u)-1)/(exp(2u)+1)
    t = tl.where(
        u >= 0.0,
        2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0,
        (tl.exp(2.0 * u) - 1.0) / (tl.exp(2.0 * u) + 1.0),
    )
    out = 0.5 * x * (1.0 + t)

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def gelu_backward_kernel(
    dx_ptr: tl.pointer_type(tl.float32),
    dy_ptr: tl.pointer_type(tl.float32),
    x_ptr: tl.pointer_type(tl.float32),
    n_elements: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    GELU backward (tanh approximation): dx = dy * dgelu_dx, single write (no accumulation).

    Args:
        dx_ptr: Gradient w.r.t. input (float32*), written (not accumulated).
        dy_ptr: Upstream gradient (float32*).
        x_ptr: Original input x (float32*).
        n_elements: Total number of elements.
        BLOCK: Block size (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = tl.where(
        u >= 0.0,
        2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0,
        (tl.exp(2.0 * u) - 1.0) / (tl.exp(2.0 * u) + 1.0),
    )
    sech2 = 1.0 - t * t
    du_dx = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du_dx
    dx = dy * dgelu_dx

    tl.store(dx_ptr + offsets, dx, mask=mask)
