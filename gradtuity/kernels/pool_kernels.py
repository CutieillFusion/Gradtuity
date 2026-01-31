"""
Triton kernels for 2D max pooling.

Includes:
- maxpool2d_forward_kernel: Max over (kernel_size x kernel_size) windows, output + argmax indices
- maxpool2d_backward_kernel: Scatter gradient to positions that were max
"""

import triton
import triton.language as tl


@triton.jit
def maxpool2d_forward_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    idx_ptr: tl.pointer_type(tl.float32),
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    H_out: tl.int32,
    W_out: tl.int32,
    stride_h: tl.int32,
    stride_w: tl.int32,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    """
    MaxPool2d forward: (N, C, H, W) -> (N, C, H_out, W_out).
    Each program handles BLOCK_ELEMS output elements to keep grid size under CUDA limit.
    """
    pid = tl.program_id(0)
    numel_out = N * C * H_out * W_out

    for i in range(BLOCK_ELEMS):
        out_idx = pid * BLOCK_ELEMS + i
        if out_idx < numel_out:
            n = out_idx // (C * H_out * W_out)
            rest = out_idx % (C * H_out * W_out)
            c = rest // (H_out * W_out)
            rest = rest % (H_out * W_out)
            h_out = rest // W_out
            w_out = rest % W_out

            h_start = h_out * stride_h
            w_start = w_out * stride_w

            max_val = -1e30
            max_idx = 0
            idx = 0
            for kh in range(BLOCK_KH):
                for kw in range(BLOCK_KW):
                    h_in = h_start + kh
                    w_in = w_start + kw
                    if h_in < H and w_in < W:
                        x_offset = (
                            n * (C * H * W) + c * (H * W) + h_in * W + w_in
                        )
                        val = tl.load(x_ptr + x_offset)
                        if val > max_val:
                            max_val = val
                            max_idx = idx
                    idx += 1

            tl.store(out_ptr + out_idx, max_val)
            tl.store(idx_ptr + out_idx, max_idx.to(tl.float32))


@triton.jit
def maxpool2d_backward_kernel(
    grad_out_ptr: tl.pointer_type(tl.float32),
    idx_ptr: tl.pointer_type(tl.float32),
    grad_in_ptr: tl.pointer_type(tl.float32),
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    H_out: tl.int32,
    W_out: tl.int32,
    stride_h: tl.int32,
    stride_w: tl.int32,
    BLOCK_KW: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    """
    MaxPool2d backward: scatter grad_out to grad_in at argmax positions.
    Each program handles BLOCK_ELEMS output elements to keep grid size under CUDA limit.
    """
    pid = tl.program_id(0)
    numel_out = N * C * H_out * W_out

    for i in range(BLOCK_ELEMS):
        out_idx = pid * BLOCK_ELEMS + i
        if out_idx < numel_out:
            n = out_idx // (C * H_out * W_out)
            rest = out_idx % (C * H_out * W_out)
            c = rest // (H_out * W_out)
            rest = rest % (H_out * W_out)
            h_out = rest // W_out
            w_out = rest % W_out

            max_idx = tl.load(idx_ptr + out_idx).to(tl.int32)
            g = tl.load(grad_out_ptr + out_idx)

            h_start = h_out * stride_h
            w_start = w_out * stride_w
            kh = max_idx // BLOCK_KW
            kw = max_idx % BLOCK_KW
            h_in = h_start + kh
            w_in = w_start + kw

            in_offset = n * (C * H * W) + c * (H * W) + h_in * W + w_in
            tl.atomic_add(grad_in_ptr + in_offset, g)
