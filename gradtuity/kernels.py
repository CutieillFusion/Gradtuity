"""
Backend dispatcher: subscriptable proxies that route kernel[grid](...) to
either the Triton or the CUDA-C implementation, controlled by the
``GRADTUITY_KERNEL_BACKEND`` env var (default: ``triton``).

Both backends remain first-class. The Triton kernels in
``gradtuity.kernels_triton.*`` and the CUDA wrappers in
``gradtuity.kernels_cuda.*`` are still importable and callable directly —
useful for the benchmark harness, parity tests, and any caller that wants
to bypass the dispatch.

The proxy preserves Triton's launch syntax so call sites in tensor.py /
functional.py don't need to change beyond their imports::

    add_kernel[grid1d(n)](a_ptr, b_ptr, c_ptr, n, BLOCK=BLOCK)

This routes to ``triton.add_kernel[grid](...)`` or
``cuda.add_kernel_launch(grid, ...)`` depending on the active backend.
"""

import os

from .kernels_triton import (
    conv_kernels as _t_conv,
    dropout_kernels as _t_dropout,
    elemwise_kernels as _t_elemwise,
    gather_kernels as _t_gather,
    layernorm_kernels as _t_layernorm,
    loss_kernels as _t_loss,
    mask_kernels as _t_mask,
    matmul_kernels as _t_matmul,
    one_hot_kernels as _t_one_hot,
    optim_kernels as _t_optim,
    pool_kernels as _t_pool,
    reduce_kernels as _t_reduce,
    softmax_kernels as _t_softmax,
)

_BACKEND = os.environ.get("GRADTUITY_KERNEL_BACKEND", "triton").lower()
if _BACKEND not in ("triton", "cuda"):
    raise RuntimeError(
        f"GRADTUITY_KERNEL_BACKEND={_BACKEND!r} not recognized; "
        "expected 'triton' or 'cuda'."
    )


class _BoundCuda:
    """One-shot bound CUDA call: prepends grid to args, passes kwargs through.

    Triton allows the grid to be a callable that receives the meta dict of
    constexpr kwargs and returns the tuple grid. We mirror that here so call
    sites using ``kernel[lambda meta: ...](...)`` keep working.
    """

    __slots__ = ("_fn", "_grid")

    def __init__(self, fn, grid):
        self._fn = fn
        self._grid = grid

    def __call__(self, *args, **kwargs):
        grid = self._grid(kwargs) if callable(self._grid) else self._grid
        return self._fn(grid, *args, **kwargs)


class KernelDispatch:
    """Proxy: ``proxy[grid](...)`` routes to the active backend.

    Triton kernels are subscripted with a grid then called; CUDA wrappers take
    grid as their first positional arg. The proxy makes both look the same.
    """

    __slots__ = ("_triton", "_cuda_launch")

    def __init__(self, triton_kernel, cuda_launch):
        self._triton = triton_kernel
        self._cuda_launch = cuda_launch

    def __getitem__(self, grid):
        if _BACKEND == "cuda":
            return _BoundCuda(self._cuda_launch, grid)
        return self._triton[grid]


# ---------------------------------------------------------------------------
# Per-kernel dispatchers. Lazy import of CUDA wrappers so importing this
# module under GRADTUITY_KERNEL_BACKEND=triton doesn't pull in cuda_driver
# (which probes for libnvrtc / libcuda at import time).
# ---------------------------------------------------------------------------
def _load_cuda_wrappers():
    from .kernels_cuda import (
        conv_kernels as c_conv,
        dropout_kernels as c_dropout,
        elemwise_kernels as c_elemwise,
        gather_kernels as c_gather,
        layernorm_kernels as c_layernorm,
        loss_kernels as c_loss,
        mask_kernels as c_mask,
        matmul_kernels as c_matmul,
        one_hot_kernels as c_one_hot,
        optim_kernels as c_optim,
        pool_kernels as c_pool,
        reduce_kernels as c_reduce,
        softmax_kernels as c_softmax,
    )
    return {
        # Elementwise
        "add_kernel": c_elemwise.add_kernel_launch,
        "mul_kernel": c_elemwise.mul_kernel_launch,
        "mul_scalar_kernel": c_elemwise.mul_scalar_kernel_launch,
        "mul_scalar_inplace_kernel": c_elemwise.mul_scalar_inplace_kernel_launch,
        "add_inplace_kernel": c_elemwise.add_inplace_kernel_launch,
        "mul_backward_kernel": c_elemwise.mul_backward_kernel_launch,
        "scale_backward_kernel": c_elemwise.scale_backward_kernel_launch,
        "relu_kernel": c_elemwise.relu_kernel_launch,
        "relu_backward_kernel": c_elemwise.relu_backward_kernel_launch,
        "relu_mask_mul_kernel": c_elemwise.relu_mask_mul_kernel_launch,
        "gelu_kernel": c_elemwise.gelu_kernel_launch,
        "gelu_backward_kernel": c_elemwise.gelu_backward_kernel_launch,
        # Optim
        "fill_kernel": c_optim.fill_kernel_launch,
        "sgd_update_kernel": c_optim.sgd_update_kernel_launch,
        "adamw_step_kernel": c_optim.adamw_step_kernel_launch,
        # Dropout
        "dropout_forward_kernel": c_dropout.dropout_forward_kernel_launch,
        "dropout_backward_kernel": c_dropout.dropout_backward_kernel_launch,
        # One-hot
        "one_hot_kernel": c_one_hot.one_hot_kernel_launch,
        # Reduce
        "sum_all_kernel": c_reduce.sum_all_kernel_launch,
        "sum_axis0_kernel": c_reduce.sum_axis0_kernel_launch,
        "add_scalar_inplace_kernel": c_reduce.add_scalar_inplace_kernel_launch,
        "add_bias_kernel": c_reduce.add_bias_kernel_launch,
        "argmax_axis1_kernel": c_reduce.argmax_axis1_kernel_launch,
        # Mask / transpose
        "transpose4d_12_kernel": c_mask.transpose4d_12_kernel_launch,
        "causal_mask_inplace_kernel": c_mask.causal_mask_inplace_kernel_launch,
        "causal_mask_backward_kernel": c_mask.causal_mask_backward_kernel_launch,
        # Matmul
        "matmul_kernel": c_matmul.matmul_kernel_launch,
        "matmul_bias_kernel": c_matmul.matmul_bias_kernel_launch,
        "matmul_bias_relu_kernel": c_matmul.matmul_bias_relu_kernel_launch,
        "matmul_nt_acc_kernel": c_matmul.matmul_nt_acc_kernel_launch,
        "matmul_tn_acc_kernel": c_matmul.matmul_tn_acc_kernel_launch,
        "transpose2d_kernel": c_matmul.transpose2d_kernel_launch,
        # Conv
        "im2col_kernel": c_conv.im2col_kernel_launch,
        "im2col_kernel_2d": c_conv.im2col_kernel_2d_launch,
        "col2im_kernel": c_conv.col2im_kernel_launch,
        # Pool
        "maxpool2d_forward_kernel": c_pool.maxpool2d_forward_kernel_launch,
        "maxpool2d_backward_kernel": c_pool.maxpool2d_backward_kernel_launch,
        # Loss
        "mse_loss_kernel": c_loss.mse_loss_kernel_launch,
        "mse_loss_backward_kernel": c_loss.mse_loss_backward_kernel_launch,
        "cross_entropy_forward_kernel": c_loss.cross_entropy_forward_kernel_launch,
        "cross_entropy_backward_kernel": c_loss.cross_entropy_backward_kernel_launch,
        # Softmax
        "softmax_forward_kernel": c_softmax.softmax_forward_kernel_launch,
        "softmax_backward_kernel": c_softmax.softmax_backward_kernel_launch,
        "softmax_with_causal_mask_forward_kernel": c_softmax.softmax_with_causal_mask_forward_kernel_launch,
        # LayerNorm
        "layernorm_fwd_kernel": c_layernorm.layernorm_fwd_kernel_launch,
        "layernorm_bwd_kernel": c_layernorm.layernorm_bwd_kernel_launch,
        # Gather
        "embedding_gather_kernel": c_gather.embedding_gather_kernel_launch,
        "embedding_scatter_add_kernel": c_gather.embedding_scatter_add_kernel_launch,
    }


_TRITON_KERNELS = {
    # Elementwise
    "add_kernel": _t_elemwise.add_kernel,
    "mul_kernel": _t_elemwise.mul_kernel,
    "mul_scalar_kernel": _t_elemwise.mul_scalar_kernel,
    "mul_scalar_inplace_kernel": _t_elemwise.mul_scalar_inplace_kernel,
    "add_inplace_kernel": _t_elemwise.add_inplace_kernel,
    "mul_backward_kernel": _t_elemwise.mul_backward_kernel,
    "scale_backward_kernel": _t_elemwise.scale_backward_kernel,
    "relu_kernel": _t_elemwise.relu_kernel,
    "relu_backward_kernel": _t_elemwise.relu_backward_kernel,
    "relu_mask_mul_kernel": _t_elemwise.relu_mask_mul_kernel,
    "gelu_kernel": _t_elemwise.gelu_kernel,
    "gelu_backward_kernel": _t_elemwise.gelu_backward_kernel,
    # Optim
    "fill_kernel": _t_optim.fill_kernel,
    "sgd_update_kernel": _t_optim.sgd_update_kernel,
    "adamw_step_kernel": _t_optim.adamw_step_kernel,
    # Dropout
    "dropout_forward_kernel": _t_dropout.dropout_forward_kernel,
    "dropout_backward_kernel": _t_dropout.dropout_backward_kernel,
    # One-hot
    "one_hot_kernel": _t_one_hot.one_hot_kernel,
    # Reduce
    "sum_all_kernel": _t_reduce.sum_all_kernel,
    "sum_axis0_kernel": _t_reduce.sum_axis0_kernel,
    "add_scalar_inplace_kernel": _t_reduce.add_scalar_inplace_kernel,
    "add_bias_kernel": _t_reduce.add_bias_kernel,
    "argmax_axis1_kernel": _t_reduce.argmax_axis1_kernel,
    # Mask / transpose
    "transpose4d_12_kernel": _t_mask.transpose4d_12_kernel,
    "causal_mask_inplace_kernel": _t_mask.causal_mask_inplace_kernel,
    "causal_mask_backward_kernel": _t_mask.causal_mask_backward_kernel,
    # Matmul
    "matmul_kernel": _t_matmul.matmul_kernel,
    "matmul_bias_kernel": _t_matmul.matmul_bias_kernel,
    "matmul_bias_relu_kernel": _t_matmul.matmul_bias_relu_kernel,
    "matmul_nt_acc_kernel": _t_matmul.matmul_nt_acc_kernel,
    "matmul_tn_acc_kernel": _t_matmul.matmul_tn_acc_kernel,
    "transpose2d_kernel": _t_matmul.transpose2d_kernel,
    # Conv
    "im2col_kernel": _t_conv.im2col_kernel,
    "im2col_kernel_2d": _t_conv.im2col_kernel_2d,
    "col2im_kernel": _t_conv.col2im_kernel,
    # Pool
    "maxpool2d_forward_kernel": _t_pool.maxpool2d_forward_kernel,
    "maxpool2d_backward_kernel": _t_pool.maxpool2d_backward_kernel,
    # Loss
    "mse_loss_kernel": _t_loss.mse_loss_kernel,
    "mse_loss_backward_kernel": _t_loss.mse_loss_backward_kernel,
    "cross_entropy_forward_kernel": _t_loss.cross_entropy_forward_kernel,
    "cross_entropy_backward_kernel": _t_loss.cross_entropy_backward_kernel,
    # Softmax
    "softmax_forward_kernel": _t_softmax.softmax_forward_kernel,
    "softmax_backward_kernel": _t_softmax.softmax_backward_kernel,
    "softmax_with_causal_mask_forward_kernel": _t_softmax.softmax_with_causal_mask_forward_kernel,
    # LayerNorm
    "layernorm_fwd_kernel": _t_layernorm.layernorm_fwd_kernel,
    "layernorm_bwd_kernel": _t_layernorm.layernorm_bwd_kernel,
    # Gather
    "embedding_gather_kernel": _t_gather.embedding_gather_kernel,
    "embedding_scatter_add_kernel": _t_gather.embedding_scatter_add_kernel,
}


_cuda_kernels = _load_cuda_wrappers() if _BACKEND == "cuda" else None
_ALL_DISPATCHERS: dict[str, "KernelDispatch"] = {}


def _make_dispatch(name: str) -> KernelDispatch:
    triton_k = _TRITON_KERNELS[name]
    cuda_k = _cuda_kernels[name] if _cuda_kernels is not None else None
    d = KernelDispatch(triton_k, cuda_k)
    _ALL_DISPATCHERS[name] = d
    return d


def set_backend(name: str) -> None:
    """Switch the active backend at runtime. Used by tests to parametrize.

    Lazy-loads CUDA wrappers the first time ``cuda`` is selected — so a process
    that starts under ``triton`` and never switches doesn't pay for libnvrtc
    binding or cubin compilation.
    """
    global _BACKEND, _cuda_kernels
    name = name.lower()
    if name not in ("triton", "cuda"):
        raise ValueError(f"unknown backend {name!r}; expected 'triton' or 'cuda'")
    if name == "cuda" and _cuda_kernels is None:
        _cuda_kernels = _load_cuda_wrappers()
        # Backfill cuda_launch into dispatchers created before cuda was loaded.
        for kname, d in _ALL_DISPATCHERS.items():
            d._cuda_launch = _cuda_kernels[kname]
    _BACKEND = name


# Flatten into module-level names. This is the public API used by tensor.py
# and functional.py — every name matches a Triton kernel object so existing
# call-site syntax (kernel[grid](...)) continues to work.
add_kernel = _make_dispatch("add_kernel")
mul_kernel = _make_dispatch("mul_kernel")
mul_scalar_kernel = _make_dispatch("mul_scalar_kernel")
mul_scalar_inplace_kernel = _make_dispatch("mul_scalar_inplace_kernel")
add_inplace_kernel = _make_dispatch("add_inplace_kernel")
mul_backward_kernel = _make_dispatch("mul_backward_kernel")
scale_backward_kernel = _make_dispatch("scale_backward_kernel")
relu_kernel = _make_dispatch("relu_kernel")
relu_backward_kernel = _make_dispatch("relu_backward_kernel")
relu_mask_mul_kernel = _make_dispatch("relu_mask_mul_kernel")
gelu_kernel = _make_dispatch("gelu_kernel")
gelu_backward_kernel = _make_dispatch("gelu_backward_kernel")

fill_kernel = _make_dispatch("fill_kernel")
sgd_update_kernel = _make_dispatch("sgd_update_kernel")
adamw_step_kernel = _make_dispatch("adamw_step_kernel")

dropout_forward_kernel = _make_dispatch("dropout_forward_kernel")
dropout_backward_kernel = _make_dispatch("dropout_backward_kernel")

one_hot_kernel = _make_dispatch("one_hot_kernel")

sum_all_kernel = _make_dispatch("sum_all_kernel")
sum_axis0_kernel = _make_dispatch("sum_axis0_kernel")
add_scalar_inplace_kernel = _make_dispatch("add_scalar_inplace_kernel")
add_bias_kernel = _make_dispatch("add_bias_kernel")
argmax_axis1_kernel = _make_dispatch("argmax_axis1_kernel")

transpose4d_12_kernel = _make_dispatch("transpose4d_12_kernel")
causal_mask_inplace_kernel = _make_dispatch("causal_mask_inplace_kernel")
causal_mask_backward_kernel = _make_dispatch("causal_mask_backward_kernel")

matmul_kernel = _make_dispatch("matmul_kernel")
matmul_bias_kernel = _make_dispatch("matmul_bias_kernel")
matmul_bias_relu_kernel = _make_dispatch("matmul_bias_relu_kernel")
matmul_nt_acc_kernel = _make_dispatch("matmul_nt_acc_kernel")
matmul_tn_acc_kernel = _make_dispatch("matmul_tn_acc_kernel")
transpose2d_kernel = _make_dispatch("transpose2d_kernel")

im2col_kernel = _make_dispatch("im2col_kernel")
im2col_kernel_2d = _make_dispatch("im2col_kernel_2d")
col2im_kernel = _make_dispatch("col2im_kernel")

maxpool2d_forward_kernel = _make_dispatch("maxpool2d_forward_kernel")
maxpool2d_backward_kernel = _make_dispatch("maxpool2d_backward_kernel")

mse_loss_kernel = _make_dispatch("mse_loss_kernel")
mse_loss_backward_kernel = _make_dispatch("mse_loss_backward_kernel")
cross_entropy_forward_kernel = _make_dispatch("cross_entropy_forward_kernel")
cross_entropy_backward_kernel = _make_dispatch("cross_entropy_backward_kernel")

softmax_forward_kernel = _make_dispatch("softmax_forward_kernel")
softmax_backward_kernel = _make_dispatch("softmax_backward_kernel")
softmax_with_causal_mask_forward_kernel = _make_dispatch(
    "softmax_with_causal_mask_forward_kernel"
)

layernorm_fwd_kernel = _make_dispatch("layernorm_fwd_kernel")
layernorm_bwd_kernel = _make_dispatch("layernorm_bwd_kernel")

embedding_gather_kernel = _make_dispatch("embedding_gather_kernel")
embedding_scatter_add_kernel = _make_dispatch("embedding_scatter_add_kernel")


def active_backend() -> str:
    """Return the active backend name ('triton' or 'cuda')."""
    return _BACKEND
