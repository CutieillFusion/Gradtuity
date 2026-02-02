"""
Tensor class with from-scratch GPU storage and autograd support.

This module implements a minimal Tensor type that:
- Holds GPU storage as raw pointers managed via ctypes + CUDA runtime
- Tracks a computation graph and supports backward()
- Enforces invariants: CUDA only, float32 only, contiguous only, rank in {1, 2, 3, 4}
"""

from __future__ import annotations

import math
import os
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import triton

from .cuda_mem import (
    cuda_device_synchronize,
    cuda_free,
    cuda_malloc,
    cuda_memcpy_dtoh,
    cuda_memcpy_dtod,
    cuda_memcpy_htod,
    cuda_memset,
)
from .kernels.conv_kernels import col2im_kernel, im2col_kernel_2d
from .kernels.elemwise_kernels import (
    add_inplace_kernel,
    add_kernel,
    gelu_backward_kernel,
    gelu_kernel,
    mul_backward_kernel,
    mul_kernel,
    mul_scalar_inplace_kernel,
    mul_scalar_kernel,
    relu_backward_kernel,
    relu_kernel,
    relu_mask_mul_kernel,
    scale_backward_kernel,
)
from .kernels.matmul_kernels import (
    matmul_bias_kernel,
    matmul_bias_relu_kernel,
    matmul_kernel,
    matmul_nt_acc_kernel,
    matmul_tn_acc_kernel,
    transpose2d_kernel,
)
from .kernels.optim_kernels import fill_kernel
from .kernels.pool_kernels import (
    maxpool2d_backward_kernel,
    maxpool2d_forward_kernel,
)
from .kernels.loss_kernels import (
    cross_entropy_backward_kernel,
    cross_entropy_forward_kernel,
    mse_loss_backward_kernel,
    mse_loss_kernel,
)
from .kernels.reduce_kernels import (
    add_bias_kernel,
    add_scalar_inplace_kernel,
    argmax_axis1_kernel,
    sum_all_kernel,
    sum_axis0_kernel,
)
from .kernels.softmax_kernels import (
    softmax_backward_kernel,
    softmax_forward_kernel,
)
from .kernels.layernorm_kernels import (
    layernorm_bwd_kernel,
    layernorm_fwd_kernel,
)
from .kernels.gather_kernels import (
    embedding_gather_kernel,
    embedding_scatter_add_kernel,
)
from .kernels.mask_kernels import (
    causal_mask_backward_kernel,
    causal_mask_inplace_kernel,
    transpose4d_12_kernel,
)
# tensor_io imported lazily in save()/load() to avoid circular import with tensor_io -> tensor

# -------------------------------------------------------------------------
# Storage and helpers
# -------------------------------------------------------------------------

F32 = 4
BLOCK = 256


@dataclass
class Storage:
    """GPU memory buffer: ptr, nbytes, and ownership."""

    ptr: int
    nbytes: int
    owns: bool = True

    def free(self) -> None:
        if self.owns and self.ptr:
            cuda_free(self.ptr)
            self.ptr = 0

    def __del__(self) -> None:
        self.free()


def _validate_shape(shape: tuple[int, ...]) -> None:
    if len(shape) not in (1, 2, 3, 4):
        raise ValueError(
            f"Only rank 1, 2, 3, or 4 tensors supported, got rank {len(shape)}"
        )
    for i, dim in enumerate(shape):
        if dim <= 0:
            raise ValueError(f"Dimension {i} must be positive, got {dim}")


def prod(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def grid1d(n: int, block: int = BLOCK) -> tuple[int, ...]:
    return (triton.cdiv(n, block),)


def alloc_storage(nbytes: int, zero: bool = False) -> Storage:
    ptr = cuda_malloc(nbytes)
    if zero:
        cuda_memset(ptr, 0, nbytes)
    return Storage(ptr, nbytes, owns=True)


@contextmanager
def temp_storage(nbytes: int, zero: bool = False):
    st = alloc_storage(nbytes, zero=zero)
    try:
        yield st
    finally:
        st.free()


def empty_tensor(
    shape: tuple[int, ...],
    *,
    requires_grad: bool = False,
    zero: bool = False,
    name: str = "",
) -> "Tensor":
    """Allocate a tensor (optionally zero-filled). Returns Tensor._wrap(...)."""
    _validate_shape(shape)
    numel = prod(shape)
    st = alloc_storage(numel * F32, zero=zero)
    return Tensor._wrap(st, shape, requires_grad=requires_grad, name=name)


def ensure_grad(t: "Tensor") -> None:
    """Allocate zero grad tensor if t.grad is None."""
    if t.grad is None:
        t.grad = empty_tensor(t.shape, zero=True, requires_grad=False)


def accum_grad(t: "Tensor", g: "Tensor") -> None:
    """Accumulate g into t.grad (ensure_grad + add_inplace)."""
    ensure_grad(t)
    add_inplace_kernel[grid1d(t.numel)](t.grad.ptr, g.ptr, t.numel, BLOCK=BLOCK)


class Tensor:
    """
    A tensor that lives on GPU with autograd support.

    All data is stored on CUDA device as float32 in contiguous row-major layout.
    Ranks 1, 2, 3, and 4 are supported.
    """

    def __init__(
        self,
        data: list | tuple,
        shape: tuple[int, ...] | None = None,
        requires_grad: bool = False,
        name: str = "",
    ) -> None:
        """
        Create a Tensor from nested Python lists/tuples.

        Args:
            data: Nested list or tuple of floats.
            shape: Optional explicit shape. If None, inferred from data structure.
            requires_grad: Whether to track gradients for this tensor.
            name: Optional name for debugging.

        Raises:
            ValueError: If shape is invalid or data doesn't match shape.
        """
        # Infer shape and flatten data
        if shape is None:
            shape, flat_data = self._infer_shape_and_flatten(data)
        else:
            flat_data = self._flatten(data)
            expected_numel = 1
            for s in shape:
                expected_numel *= s
            if len(flat_data) != expected_numel:
                raise ValueError(
                    f"Data has {len(flat_data)} elements but shape {shape} "
                    f"requires {expected_numel} elements"
                )

        # Validate shape (rank must be 1, 2, 3, or 4)
        if len(shape) not in (1, 2, 3, 4):
            raise ValueError(
                f"Only rank 1, 2, 3, or 4 tensors supported, got rank {len(shape)}"
            )

        # Validate all dimensions are positive
        _validate_shape(shape)

        # Store shape and allocate GPU memory
        self._shape = tuple(shape)
        numel = len(flat_data)
        nbytes = numel * F32
        st = alloc_storage(nbytes, zero=False)
        host_bytes = struct.pack(f"{numel}f", *flat_data)
        cuda_memcpy_htod(st.ptr, host_bytes)
        self._st = st

        # Autograd fields
        self.requires_grad = requires_grad
        self.grad = None
        self.name = name
        self._parents = ()
        self._backward = None
        self._ctx = None

    @classmethod
    def _wrap(
        cls,
        st: Storage,
        shape: tuple[int, ...],
        requires_grad: bool = False,
        name: str = "",
    ) -> Tensor:
        """Create a Tensor from Storage (internal use)."""
        _validate_shape(shape)
        t = object.__new__(cls)
        t._st = st
        t._shape = tuple(shape)
        t.requires_grad = requires_grad
        t.grad = None
        t.name = name
        t._parents = ()
        t._backward = None
        t._ctx = None
        return t

    @classmethod
    def _from_ptr(
        cls,
        ptr: int,
        shape: tuple[int, ...],
        owns_memory: bool = True,
        requires_grad: bool = False,
        name: str = "",
    ) -> Tensor:
        """
        Create a Tensor from an existing GPU pointer (internal use).

        Args:
            ptr: Raw GPU pointer from cudaMalloc.
            shape: Shape of the tensor.
            owns_memory: If True, this tensor will free the memory on deletion.
            requires_grad: Whether to track gradients.
            name: Optional name for debugging.

        Returns:
            New Tensor wrapping the given pointer.
        """
        numel = prod(shape)
        nbytes = numel * F32
        st = Storage(ptr, nbytes, owns=owns_memory)
        return cls._wrap(st, shape, requires_grad=requires_grad, name=name)

    @classmethod
    def _zeros(
        cls,
        shape: tuple[int, ...],
        requires_grad: bool = False,
    ) -> Tensor:
        """Allocate a zero-initialized GPU tensor (internal use)."""
        return empty_tensor(shape, zero=True, requires_grad=requires_grad)

    @classmethod
    def _zeros_like(cls, t: Tensor, requires_grad: bool = False) -> Tensor:
        """Create a zero tensor with the same shape as t (internal use)."""
        return empty_tensor(t.shape, zero=True, requires_grad=requires_grad)

    @classmethod
    def _ones(
        cls,
        shape: tuple[int, ...],
        requires_grad: bool = False,
    ) -> Tensor:
        """Allocate a tensor filled with 1.0 (internal use)."""
        t = empty_tensor(shape, zero=True, requires_grad=requires_grad)
        fill_kernel[grid1d(t.numel)](t.ptr, 1.0, t.numel, BLOCK=BLOCK)
        return t

    @classmethod
    def _ones_like(cls, t: Tensor, requires_grad: bool = False) -> Tensor:
        """Create a tensor of ones with the same shape as t (internal use)."""
        return cls._ones(t.shape, requires_grad=requires_grad)

    def data_ptr(self) -> int:
        """
        Return raw GPU pointer for Triton kernels.

        Returns:
            GPU memory pointer as integer.
        """
        return self.ptr

    @property
    def ptr(self) -> int:
        """Raw GPU pointer for Triton kernels."""
        return self._st.ptr

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the tensor."""
        return self._shape

    @property
    def numel(self) -> int:
        """Return the total number of elements."""
        return prod(self._shape)

    @property
    def nbytes(self) -> int:
        """Return the total number of bytes."""
        return self._st.nbytes

    @property
    def ndim(self) -> int:
        """Return the number of dimensions (rank)."""
        return len(self._shape)

    @property
    def owns_memory(self) -> bool:
        """Return whether this tensor owns its storage (will free on deletion)."""
        storage = getattr(self, "_st", None)
        if storage is None:
            raise AttributeError("Tensor has no _st (storage)")
        return storage.owns

    # -------------------------------------------------------------------------
    # Autograd: backward() and graph construction
    # -------------------------------------------------------------------------

    def backward(self) -> None:
        """
        Perform backpropagation from this tensor.

        Requires this tensor to be a scalar (numel == 1).
        Seeds this tensor's grad with 1.0 and propagates gradients
        through the computation graph.

        Raises:
            ValueError: If tensor is not a scalar.
            RuntimeError: If requires_grad is False.
        """
        # Validate scalar
        if self.numel != 1:
            raise ValueError(
                f"backward() only works on scalar tensors (numel=1), "
                f"got numel={self.numel}"
            )

        if not self.requires_grad:
            raise RuntimeError(
                "backward() called on a tensor that doesn't require grad. "
                "Set requires_grad=True on leaf tensors."
            )

        # Seed grad with 1.0
        self.grad = Tensor._ones_like(self)

        # Build topological order using visited set keyed by id()
        # This is micrograd-style topo sort that handles shared subgraphs correctly
        visited: set[int] = set()
        topo: list[Tensor] = []

        def build_topo(v: Tensor) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for parent in v._parents:
                build_topo(parent)
            topo.append(v)

        build_topo(self)

        # Reverse traverse: from output to inputs
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

            # Optional: clear graph references to allow GC of intermediate tensors
            # (but NOT the grad tensors or leaf tensors)
            # We only clear non-leaf nodes (nodes with parents)
            if node._parents:
                node._parents = ()
                node._backward = None
                node._ctx = None

    def _set_graph(
        self,
        parents: tuple["Tensor", ...],
        backward_fn: Callable[["Tensor"], None],
        ctx: Optional[dict] = None,
    ) -> None:
        """
        Set up the computation graph for this tensor (conditional graph construction).

        This method implements conditional graph construction:
        - Sets requires_grad = any(p.requires_grad for p in parents)
        - Only attaches graph fields if requires_grad is True

        Args:
            parents: Tuple of parent tensors in the computation graph.
            backward_fn: The backward function to compute gradients.
            ctx: Optional context dictionary for backward (usually not needed
                 since backward_fn is a closure that captures context).

        Note:
            This method should be called by op implementations after creating
            the output tensor.
        """
        # Determine if we need gradients
        self.requires_grad = any(p.requires_grad for p in parents)

        if self.requires_grad:
            self._parents = parents
            self._backward = backward_fn
            self._ctx = ctx
        # If requires_grad is False, leave graph fields empty (already initialized to empty)

    def detach(self) -> Tensor:
        """
        Return a new Tensor sharing data but detached from the computation graph.

        The returned tensor has requires_grad=False and shares the same GPU memory.
        It will NOT free the memory when deleted (ownership stays with original).

        Returns:
            New Tensor with shared data pointer.
        """
        shared_st = Storage(self._st.ptr, self._st.nbytes, owns=False)
        return Tensor._wrap(
            shared_st,
            self._shape,
            requires_grad=False,
            name=f"{self.name}_detached" if self.name else "",
        )

    def view(self, new_shape: tuple[int | None, ...]) -> "Tensor":
        """
        Return a new tensor viewing the same storage with a different shape.

        Total number of elements must match. At most one dimension may be -1
        (or None), which is inferred from the remaining dimensions and numel.

        Args:
            new_shape: Target shape. Use -1 for one dimension to infer.

        Returns:
            New tensor sharing storage with self (no copy).

        Raises:
            ValueError: If product of new_shape does not match numel.
        """
        # Resolve -1 / None: replace with inferred size
        new_shape_list = list(new_shape)
        infer_idx = None
        product = 1
        for i, d in enumerate(new_shape_list):
            if d is None or d == -1:
                if infer_idx is not None:
                    raise ValueError("view() allows only one dimension to be -1")
                infer_idx = i
            else:
                if d <= 0:
                    raise ValueError(f"view() dimension {i} must be positive, got {d}")
                product *= d
        if infer_idx is not None:
            inferred = self.numel // product
            if inferred * product != self.numel:
                raise ValueError(
                    f"view() shape {new_shape} is incompatible with numel {self.numel}"
                )
            new_shape_list[infer_idx] = inferred
        else:
            if product != self.numel:
                raise ValueError(
                    f"view() shape {new_shape} has {product} elements, "
                    f"but tensor has {self.numel} elements"
                )
        resolved_shape = tuple(new_shape_list)

        out = Tensor._from_ptr(
            self.ptr,
            resolved_shape,
            owns_memory=False,
            requires_grad=False,
        )
        input_tensor = self

        def _backward(out_grad: Tensor) -> None:
            if input_tensor.requires_grad:
                grad_viewed = out_grad.view(input_tensor.shape)
                ensure_grad(input_tensor)
                input_tensor.grad = input_tensor.grad.add(grad_viewed)

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def matmul(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication: C = self @ other

        Computes C[M, N] = A[M, K] @ B[K, N]
        - self (A): shape (M, K) - e.g., (batch, in_features)
        - other (B): shape (K, N) - e.g., (in_features, out_features)
        - output (C): shape (M, N) - e.g., (batch, out_features)

        Args:
            other: Right-hand matrix, must be 2D with compatible dimensions.

        Returns:
            New tensor containing the matrix product.

        Raises:
            ValueError: If shapes are incompatible for matmul.
        """
        # Validate shapes
        if self.ndim != 2:
            raise ValueError(
                f"matmul requires 2D tensors, got self.shape={self._shape}"
            )
        if other.ndim != 2:
            raise ValueError(
                f"matmul requires 2D tensors, got other.shape={other._shape}"
            )
        if self._shape[1] != other._shape[0]:
            raise ValueError(
                f"matmul shape mismatch: {self._shape} @ {other._shape} "
                f"(inner dimensions {self._shape[1]} vs {other._shape[0]})"
            )

        M, K = self._shape
        K2, N = other._shape
        out_shape = (M, N)

        out = empty_tensor(out_shape)

        # Block sizes for matmul (tuned for typical small matrices)
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32

        # Launch matmul kernel
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_kernel[grid](
            self.ptr,
            other.ptr,
            out.ptr,
            M,
            N,
            K,
            # Strides for row-major layout
            K,
            1,  # A strides: stride_am=K, stride_ak=1
            N,
            1,  # B strides: stride_bk=N, stride_bn=1
            N,
            1,  # C strides: stride_cm=N, stride_cn=1
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        # Capture for backward
        a_tensor = self
        b_tensor = other

        def _backward(out_grad: Tensor) -> None:
            # dA += out_grad @ B^T  (M, N) @ (N, K) -> (M, K)
            # dB += A^T @ out_grad  (K, M) @ (M, N) -> (K, N)
            #
            # Using fused transposed matmul kernels that:
            # 1. Read matrices in transposed order (no materialized transpose)
            # 2. Accumulate directly into grad buffers (no temp + add_inplace)

            if a_tensor.requires_grad:
                ensure_grad(a_tensor)
                # dA += out_grad @ B^T using fused kernel
                # out_grad: (M, N), B: (K, N) read as transposed -> result: (M, K)
                # B is stored as (K, N) row-major: B[i,j] = b_ptr[i*N + j]
                # We want B^T[k, n] = B[n, k] where n is row index, k is col index
                # So stride_bn = N (row stride), stride_bk = 1 (col stride)
                grid_da = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
                matmul_nt_acc_kernel[grid_da](
                    out_grad.ptr,
                    b_tensor.ptr,
                    a_tensor.grad.ptr,
                    M,
                    K,
                    N,  # K_inner = N (shared dim)
                    N,
                    1,  # out_grad strides (M, N): stride_am=N, stride_ak=1
                    N,
                    1,  # B strides (K, N): stride_bn=N, stride_bk=1
                    K,
                    1,  # dA strides (M, K): stride_cm=K, stride_cn=1
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )

            if b_tensor.requires_grad:
                ensure_grad(b_tensor)
                # dB += A^T @ out_grad using fused kernel
                # A: (M, K) read as transposed, out_grad: (M, N) -> result: (K, N)
                grid_db = (triton.cdiv(K, BLOCK_M), triton.cdiv(N, BLOCK_N))
                matmul_tn_acc_kernel[grid_db](
                    a_tensor.ptr,
                    out_grad.ptr,
                    b_tensor.grad.ptr,
                    K,
                    N,
                    M,  # K_inner = M (shared dim)
                    K,
                    1,  # A strides (M, K): stride_ak=K, stride_am=1 (read transposed)
                    N,
                    1,  # out_grad strides (M, N): stride_bk=N, stride_bn=1
                    N,
                    1,  # dB strides (K, N): stride_cm=N, stride_cn=1
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )

        out._set_graph(parents=(self, other), backward_fn=_backward)
        return out

    def embedding(self, indices: list | tuple | "Tensor") -> "Tensor":
        """
        Row-gather (embedding lookup): out[n, d] = self[int(indices[n]), d].

        Weight must be 2D (V, D). Indices can be 1D (N,) or 2D (B, S); output is (N, D) or (B, S, D).
        Indices must be integer-valued and in range [0, V-1]. Gradients flow only to weight.

        Args:
            indices: Token IDs as Python list/tuple of ints or nested lists, or a Tensor of float32
                (integer-valued, e.g. 1.0, 2.0).

        Returns:
            Gathered rows, shape (N, D) or (B, S, D) matching indices layout.
        """
        if self.ndim != 2:
            raise ValueError(f"embedding requires 2D weight, got shape {self._shape}")
        V, D = self._shape[0], self._shape[1]

        # Convert indices to Tensor if needed; flatten to 1D for kernel
        if isinstance(indices, (list, tuple)):
            idx_tensor = Tensor(indices)
        else:
            idx_tensor = indices
        if idx_tensor.ndim not in (1, 2):
            raise ValueError(
                f"embedding indices must be 1D or 2D, got ndim={idx_tensor.ndim}"
            )
        if idx_tensor.ndim == 2:
            indices_shape = idx_tensor.shape
            idx_flat = idx_tensor.view((idx_tensor.numel,))
        else:
            indices_shape = None
            idx_flat = idx_tensor
        N = idx_flat.numel

        # Validate: integer-valued and in [0, V-1]
        flat_vals = Tensor._flatten(idx_flat.to_list())
        for v in flat_vals:
            if abs(v - round(v)) >= 1e-6:
                raise ValueError(
                    "embedding indices must be integer-valued (e.g. 1.0, 2.0)"
                )
            ri = int(round(v))
            if ri < 0 or ri >= V:
                raise ValueError(f"embedding index {ri} out of range [0, {V - 1}]")

        out_flat = empty_tensor((N, D))
        BLOCK_D = 128
        grid = (N, triton.cdiv(D, BLOCK_D))
        embedding_gather_kernel[grid](
            self.ptr,
            idx_flat.ptr,
            out_flat.ptr,
            N=N,
            D=D,
            V=V,
            BLOCK_D=BLOCK_D,
        )

        # For 2D indices, copy (N, D) into (B, S, D) so returned tensor owns its storage
        if indices_shape is not None:
            out = empty_tensor((*indices_shape, D))
            cuda_memcpy_dtod(out.ptr, out_flat.ptr, N * D * F32)
        else:
            out = out_flat

        weight_tensor = self
        idx_flat_captured = idx_flat
        idx_owner_captured = idx_tensor  # keep indices buffer alive until backward runs
        N_captured = N
        D_captured = D
        V_captured = V
        indices_shape_captured = indices_shape

        def _backward(out_grad: Tensor) -> None:
            _ = idx_owner_captured  # keep indices buffer alive until backward runs
            if weight_tensor.requires_grad:
                ensure_grad(weight_tensor)
                if indices_shape_captured is not None:
                    # Copy (B, S, D) to contiguous (N, D) so scatter kernel sees correct layout
                    dout_flat = empty_tensor((N_captured, D_captured))
                    cuda_memcpy_dtod(
                        dout_flat.ptr, out_grad.ptr, N_captured * D_captured * F32
                    )
                else:
                    dout_flat = out_grad
                grid_bw = (N_captured, triton.cdiv(D_captured, BLOCK_D))
                embedding_scatter_add_kernel[grid_bw](
                    dout_flat.ptr,
                    idx_flat_captured.ptr,
                    weight_tensor.grad.ptr,
                    N=N_captured,
                    D=D_captured,
                    V=V_captured,
                    BLOCK_D=BLOCK_D,
                )
                cuda_device_synchronize()

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def bmm(self, other: "Tensor") -> "Tensor":
        """
        Batched matrix multiplication: C = self @ other per batch slice.

        Supports rank-3 and rank-4:
        - Rank-3: a (B, M, K), b (B, K, N) -> out (B, M, N)
        - Rank-4: a (B, H, M, K), b (B, H, K, N) -> out (B, H, M, N) via flattening to (B*H, M, K) @ (B*H, K, N)

        Args:
            other: Right-hand batched matrix, same ndim as self with compatible dimensions.

        Returns:
            Batched matrix product, shape (B, M, N) or (B, H, M, N).
        """
        ndim = self.ndim
        if ndim not in (3, 4):
            raise ValueError(f"bmm requires ndim 3 or 4, got self.ndim={ndim}")
        if other.ndim != ndim:
            raise ValueError(
                f"bmm requires same ndim for both tensors, got {ndim} vs {other.ndim}"
            )

        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32

        if ndim == 3:
            Bflat, M, K = self.shape
            B2, K2, N = other.shape
            if B2 != Bflat:
                raise ValueError(
                    f"bmm batch dim mismatch: {self.shape} vs {other.shape}"
                )
            if K2 != K:
                raise ValueError(
                    f"bmm inner dim mismatch: a.shape[2]={K} vs b.shape[1]={K2}"
                )
            out_shape_flat = (Bflat, M, N)
            out = empty_tensor(out_shape_flat)
            is_rank4 = False
            B, H = None, None
        else:
            B, H, M, K = self.shape
            B2, H2, K2, N = other.shape
            if B2 != B or H2 != H:
                raise ValueError(
                    f"bmm batch/head dim mismatch: {self.shape} vs {other.shape}"
                )
            if K2 != K:
                raise ValueError(
                    f"bmm inner dim mismatch: a.shape[3]={K} vs b.shape[2]={K2}"
                )
            Bflat = B * H
            out = empty_tensor((B, H, M, N))
            is_rank4 = True

        a_tensor = self
        b_tensor = other

        # Forward: loop over batch slices (row-major slice i = b*H+h at offset i*M*N)
        for i in range(Bflat):
            a_ptr_i = self.ptr + i * (M * K) * F32
            b_ptr_i = other.ptr + i * (K * N) * F32
            c_ptr_i = out.ptr + i * (M * N) * F32
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_kernel[grid](
                a_ptr_i,
                b_ptr_i,
                c_ptr_i,
                M,
                N,
                K,
                K,
                1,
                N,
                1,
                N,
                1,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )

        def _backward(out_grad: Tensor) -> None:
            if is_rank4:
                out_grad_flat = out_grad.view((Bflat, M, N))
            else:
                out_grad_flat = out_grad
            for i in range(Bflat):
                dC_i = out_grad_flat.ptr + i * (M * N) * F32
                A_i = a_tensor.ptr + i * (M * K) * F32
                B_i = b_tensor.ptr + i * (K * N) * F32
                if a_tensor.requires_grad:
                    ensure_grad(a_tensor)
                    dA_i = a_tensor.grad.ptr + i * (M * K) * F32
                    grid_da = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
                    matmul_nt_acc_kernel[grid_da](
                        dC_i,
                        B_i,
                        dA_i,
                        M,
                        K,
                        N,
                        N,
                        1,
                        N,
                        1,
                        K,
                        1,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_K=BLOCK_K,
                    )
                if b_tensor.requires_grad:
                    ensure_grad(b_tensor)
                    dB_i = b_tensor.grad.ptr + i * (K * N) * F32
                    grid_db = (triton.cdiv(K, BLOCK_M), triton.cdiv(N, BLOCK_N))
                    matmul_tn_acc_kernel[grid_db](
                        A_i,
                        dC_i,
                        dB_i,
                        K,
                        N,
                        M,
                        K,
                        1,
                        N,
                        1,
                        N,
                        1,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_K=BLOCK_K,
                    )

        out._set_graph(parents=(self, other), backward_fn=_backward)
        return out

    def add(self, other: "Tensor") -> "Tensor":
        """
        Elementwise addition: C = self + other

        Both tensors must have the same shape.

        Args:
            other: Tensor to add, must have same shape as self.

        Returns:
            New tensor containing the elementwise sum.

        Raises:
            ValueError: If shapes don't match.
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for add: {self.shape} vs {other.shape}")

        out = empty_tensor(self.shape)
        add_kernel[grid1d(self.numel)](
            self.ptr, other.ptr, out.ptr, self.numel, BLOCK=BLOCK
        )

        a, b = self, other

        def _backward(g: Tensor) -> None:
            if a.requires_grad:
                accum_grad(a, g)
            if b.requires_grad:
                accum_grad(b, g)

        out._set_graph(parents=(self, other), backward_fn=_backward)
        return out

    def relu(self) -> "Tensor":
        """
        ReLU activation: Z = max(self, 0)

        Returns:
            New tensor with ReLU applied elementwise.
        """
        out = empty_tensor(self.shape)
        relu_kernel[grid1d(self.numel)](self.ptr, out.ptr, self.numel, BLOCK=BLOCK)

        x = self

        def _backward(g: Tensor) -> None:
            if x.requires_grad:
                ensure_grad(x)
                relu_backward_kernel[grid1d(x.numel)](
                    x.grad.ptr, g.ptr, x.ptr, x.numel, BLOCK=BLOCK
                )

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def gelu(self, approx: str = "tanh") -> "Tensor":
        """
        GELU activation (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))

        Args:
            approx: Only "tanh" supported in v1.

        Returns:
            New tensor with GELU applied elementwise.
        """
        if approx != "tanh":
            raise ValueError(f'gelu approx must be "tanh" in v1, got {approx!r}')
        out = empty_tensor(self.shape)
        gelu_kernel[grid1d(self.numel)](self.ptr, out.ptr, self.numel, BLOCK=BLOCK)

        x = self

        def _backward(g: Tensor) -> None:
            if x.requires_grad:
                ensure_grad(x)
                with temp_storage(x.numel * F32, zero=False) as st:
                    grad_tmp = Tensor._wrap(st, x.shape, requires_grad=False)
                    gelu_backward_kernel[grid1d(x.numel)](
                        grad_tmp.ptr, g.ptr, x.ptr, x.numel, BLOCK=BLOCK
                    )
                    add_inplace_kernel[grid1d(x.numel)](
                        x.grad.ptr, grad_tmp.ptr, x.numel, BLOCK=BLOCK
                    )

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def softmax(self, dim: int = -1) -> "Tensor":
        """
        Softmax over the last dimension (v1: only dim=-1 supported).

        Numerically stable: max-subtract then exp/sum per row.
        Treats input as (rows, cols) with cols = shape[-1], rows = numel // cols.

        Args:
            dim: Must be -1 in v1.

        Returns:
            Tensor of same shape; each row sums to 1.
        """
        if dim != -1:
            raise ValueError(f"softmax v1 only supports dim=-1, got dim={dim}")
        cols = self.shape[-1]
        rows = self.numel // cols
        out = empty_tensor(self.shape)
        BLOCK_COLS = 1024
        softmax_forward_kernel[(rows,)](
            self.ptr,
            out.ptr,
            rows=rows,
            cols=cols,
            BLOCK_COLS=BLOCK_COLS,
        )

        x = self
        out_saved = out

        def _backward(g: Tensor) -> None:
            if x.requires_grad:
                ensure_grad(x)
                softmax_backward_kernel[(rows,)](
                    x.grad.ptr,
                    g.ptr,
                    out_saved.ptr,
                    rows=rows,
                    cols=cols,
                    BLOCK_COLS=BLOCK_COLS,
                )

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def layer_norm(
        self, gamma: "Tensor", beta: "Tensor", eps: float = 1e-5
    ) -> "Tensor":
        """
        LayerNorm over the last dimension: y = (x - mean) * rstd * gamma + beta.

        v1: input ndim in (2, 3, 4); 3D/4D are flattened to (N, H) internally.
        gamma and beta must be 1D of shape (H,) where H = self.shape[-1].

        Args:
            gamma: Scale (H,), applied to normalized values.
            beta: Shift (H,).
            eps: Epsilon for variance stability.

        Returns:
            Tensor of same shape as self.
        """
        if self.ndim not in (2, 3, 4):
            raise ValueError(
                f"layer_norm v1 requires ndim in (2, 3, 4), got ndim={self.ndim}"
            )
        if gamma.ndim != 1 or beta.ndim != 1:
            raise ValueError(
                f"layer_norm requires 1D gamma and beta, got gamma.ndim={gamma.ndim}, beta.ndim={beta.ndim}"
            )
        H = self.shape[-1]
        if gamma.shape[0] != H or beta.shape[0] != H:
            raise ValueError(
                f"layer_norm gamma/beta must have shape (H,) with H={H}, got gamma={gamma.shape}, beta={beta.shape}"
            )

        original_shape = self.shape
        if self.ndim == 2:
            N, H = self.shape[0], self.shape[1]
            x = self
            out = empty_tensor((N, H))
        else:
            N = self.numel // H
            x = self.view((N, H))
            # Allocate output with original shape so kernel writes directly into returned tensor
            out = empty_tensor(original_shape)

        xhat = empty_tensor((N, H))
        rstd = empty_tensor((N,))

        BLOCK_H = 1024
        layernorm_fwd_kernel[(N,)](
            x.ptr,
            gamma.ptr,
            beta.ptr,
            out.ptr,
            xhat.ptr,
            rstd.ptr,
            N=N,
            H=H,
            eps=eps,
            BLOCK_H=BLOCK_H,
        )

        x_tensor = self
        gamma_tensor = gamma
        beta_tensor = beta
        xhat_saved = xhat
        rstd_saved = rstd

        def _backward(out_grad: Tensor) -> None:
            dy = out_grad.view((N, H)) if out_grad.shape != (N, H) else out_grad
            dx = empty_tensor((N, H), zero=True)
            with temp_storage(H * F32, zero=True) as st_dg:
                with temp_storage(H * F32, zero=True) as st_db:
                    if gamma_tensor.requires_grad:
                        ensure_grad(gamma_tensor)
                        dgamma_ptr = gamma_tensor.grad.ptr
                    else:
                        dgamma_ptr = st_dg.ptr
                    if beta_tensor.requires_grad:
                        ensure_grad(beta_tensor)
                        dbeta_ptr = beta_tensor.grad.ptr
                    else:
                        dbeta_ptr = st_db.ptr
                    layernorm_bwd_kernel[(N,)](
                        dx.ptr,
                        dgamma_ptr,
                        dbeta_ptr,
                        dy.ptr,
                        xhat_saved.ptr,
                        rstd_saved.ptr,
                        gamma_tensor.ptr,
                        N=N,
                        H=H,
                        BLOCK_H=BLOCK_H,
                    )
            if x_tensor.requires_grad:
                ensure_grad(x_tensor)
                add_inplace_kernel[grid1d(N * H)](
                    x_tensor.grad.ptr,
                    dx.ptr,
                    N * H,
                    BLOCK=BLOCK,
                )

        out._set_graph(parents=(self, gamma, beta), backward_fn=_backward)
        return out

    def transpose4d_last2(self) -> "Tensor":
        """
        Transpose last two dimensions of a 4D tensor: (B, H, S, D) -> (B, H, D, S).

        Rank must be 4. Allocates a new contiguous tensor. Backward is the same
        transpose (inverse permutation).

        Returns:
            Tensor of shape (B, H, D, S).
        """
        if self.ndim != 4:
            raise ValueError(f"transpose4d_last2 requires ndim=4, got ndim={self.ndim}")
        B, H, S, D = self.shape
        out_shape = (B, H, D, S)
        out = empty_tensor(out_shape)
        slice_elems = S * D
        for k in range(B * H):
            in_offset = k * slice_elems * F32
            out_offset = k * (D * S) * F32
            transpose2d_kernel[(triton.cdiv(S * D, BLOCK),)](
                self.ptr + in_offset,
                out.ptr + out_offset,
                rows=S,
                cols=D,
                BLOCK=BLOCK,
            )

        x = self

        def _backward(g: Tensor) -> None:
            if x.requires_grad:
                # Inverse of swap last two is same op: grad has shape (B,H,D,S), transpose -> (B,H,S,D)
                accum_grad(x, g.detach().transpose4d_last2())

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def transpose4d_12(self) -> "Tensor":
        """
        Transpose axes 1 and 2 of a 4D tensor: (B, A, C, D) -> (B, C, A, D).

        Rank must be 4. Allocates a new contiguous tensor. Backward is the same
        transpose (inverse is itself).

        Returns:
            Tensor of shape (B, C, A, D).
        """
        if self.ndim != 4:
            raise ValueError(
                f"transpose4d_12 requires ndim=4, got ndim={self.ndim}"
            )
        B, A, C, D = self.shape
        out_shape = (B, C, A, D)
        out = empty_tensor(out_shape)
        numel = B * A * C * D
        transpose4d_12_kernel[grid1d(numel)](
            self.ptr,
            out.ptr,
            B=B,
            A=A,
            C=C,
            D=D,
            BLOCK=BLOCK,
        )

        x = self

        def _backward(g: Tensor) -> None:
            if x.requires_grad:
                accum_grad(x, g.detach().transpose4d_12())

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def apply_causal_mask(self, neg_inf: float = -1e9) -> "Tensor":
        """
        Apply causal mask: copy tensor and set upper-triangular (j > i) to neg_inf.

        For 4D scores (B, H, S, S), position (i, j) is masked when j > i.
        Returns a new tensor; backward zeros gradients at masked positions.

        Args:
            neg_inf: Value to write for masked positions (default -1e9).

        Returns:
            New tensor of same shape with causal mask applied.
        """
        if self.ndim != 4:
            raise ValueError(
                f"apply_causal_mask requires ndim=4, got ndim={self.ndim}"
            )
        B, H, S, S_last = self.shape
        if S != S_last:
            raise ValueError(
                f"apply_causal_mask requires square last two dims, got shape {self.shape}"
            )
        out = empty_tensor(self.shape)
        cuda_memcpy_dtod(out.ptr, self.ptr, self.numel * F32)
        BLOCK_I = 32
        BLOCK_J = 32
        causal_mask_inplace_kernel[
            (B * H, triton.cdiv(S, BLOCK_I), triton.cdiv(S, BLOCK_J))
        ](
            out.ptr,
            B=B,
            H=H,
            S=S,
            NEG_INF=neg_inf,
            BLOCK_I=BLOCK_I,
            BLOCK_J=BLOCK_J,
        )

        input_tensor = self

        def _backward(g: Tensor) -> None:
            if input_tensor.requires_grad:
                grad_tmp = empty_tensor(self.shape)
                causal_mask_backward_kernel[grid1d(B * H * S * S)](
                    g.ptr,
                    grad_tmp.ptr,
                    B=B,
                    H=H,
                    S=S,
                    BLOCK=BLOCK,
                )
                accum_grad(input_tensor, grad_tmp)

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def sum(self) -> "Tensor":
        """
        Sum all elements to produce a scalar tensor.

        Returns:
            Scalar tensor with shape (1,) containing the sum of all elements.
        """
        out = empty_tensor((1,), zero=True)
        sum_all_kernel[grid1d(self.numel)](self.ptr, out.ptr, self.numel, BLOCK=BLOCK)

        input_tensor = self

        def _backward(g: Tensor) -> None:
            if input_tensor.requires_grad:
                ensure_grad(input_tensor)
                add_scalar_inplace_kernel[grid1d(input_tensor.numel)](
                    input_tensor.grad.ptr, g.ptr, input_tensor.numel, BLOCK=BLOCK
                )

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def mse_loss(self, target: "Tensor") -> "Tensor":
        """
        Compute mean squared error loss: L = mean((self - target)^2)

        This is a fused operation that computes the MSE in a single kernel,
        more efficient than separate subtract, square, and mean operations.

        Note:
            Backward performs one device→host read of the scalar gradient
            (typically 1.0). This sync is intentional for the current API.

        Args:
            target: Target tensor, must have same shape as self.

        Returns:
            Scalar tensor containing the MSE loss.

        Raises:
            ValueError: If shapes don't match.
        """
        if self.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for mse_loss: {self.shape} vs {target.shape}"
            )

        out = empty_tensor((1,), zero=True)
        mse_loss_kernel[grid1d(self.numel)](
            self.ptr, target.ptr, out.ptr, self.numel, BLOCK=BLOCK
        )
        scale = 1.0 / self.numel
        mul_scalar_kernel[grid1d(1)](out.ptr, scale, out.ptr, 1, BLOCK=BLOCK)

        pred_tensor = self
        target_tensor = target
        numel = self.numel

        def _backward(out_grad: Tensor) -> None:
            # MSE backward does one device→host read of scalar gradient (intentional).
            grad_bytes = cuda_memcpy_dtoh(out_grad.ptr, 4)
            grad_val = struct.unpack("f", grad_bytes)[0]
            scale = grad_val / numel

            if pred_tensor.requires_grad:
                ensure_grad(pred_tensor)
            if target_tensor.requires_grad:
                ensure_grad(target_tensor)

            mse_loss_backward_kernel[grid1d(numel)](
                pred_tensor.grad.ptr if pred_tensor.requires_grad else 0,
                target_tensor.grad.ptr if target_tensor.requires_grad else 0,
                pred_tensor.ptr,
                target_tensor.ptr,
                scale,
                numel,
                1 if pred_tensor.requires_grad else 0,
                1 if target_tensor.requires_grad else 0,
                BLOCK=BLOCK,
            )

        out._set_graph(parents=(self, target), backward_fn=_backward)
        return out

    def cross_entropy(self, targets: "Tensor", reduction: str = "mean") -> "Tensor":
        """
        Compute cross-entropy loss: L = -mean_i(log(softmax(logits_i)[targets_i])).

        Numerically stable via log-softmax (max-subtraction + logsumexp).
        Backward only into logits; targets are non-differentiable.

        Note:
            Backward performs one device→host read of the scalar gradient
            (typically 1.0). This sync is intentional for the current API.

        Args:
            targets: 1D tensor of class indices (float32 storage), shape (B,).
            reduction: "mean" (default) or "sum".

        Returns:
            Scalar tensor containing the cross-entropy loss.

        Raises:
            ValueError: If shapes or reduction are invalid.
        """
        if self.ndim != 2:
            raise ValueError(f"cross_entropy expects 2D logits, got ndim={self.ndim}")
        if targets.ndim != 1:
            raise ValueError(
                f"cross_entropy expects 1D targets, got ndim={targets.ndim}"
            )
        B, C = self.shape
        if targets.shape[0] != B:
            raise ValueError(
                f"cross_entropy batch size mismatch: logits {B} vs targets {targets.shape[0]}"
            )
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"cross_entropy reduction must be 'mean' or 'sum', got {reduction!r}"
            )
        if C > 1024:
            raise ValueError(f"cross_entropy supports at most 1024 classes, got C={C}")
        # BLOCK_C: next power of 2 >= C, capped at 1024
        block_c = min(1024, 1 << (C - 1).bit_length()) if C >= 1 else 1

        out = empty_tensor((1,), zero=True)
        cross_entropy_forward_kernel[(B,)](
            self.ptr,
            targets.ptr,
            out.ptr,
            B=B,
            C=C,
            BLOCK_C=block_c,
        )
        if reduction == "mean":
            scale = 1.0 / B
            mul_scalar_kernel[grid1d(1)](out.ptr, scale, out.ptr, 1, BLOCK=BLOCK)

        logits_tensor = self
        targets_tensor = targets

        def _backward(out_grad: Tensor) -> None:
            grad_bytes = cuda_memcpy_dtoh(out_grad.ptr, 4)
            grad_val = struct.unpack("f", grad_bytes)[0]
            scale = grad_val / B if reduction == "mean" else grad_val

            if logits_tensor.requires_grad:
                ensure_grad(logits_tensor)
                cross_entropy_backward_kernel[(B,)](
                    logits_tensor.grad.ptr,
                    logits_tensor.ptr,
                    targets_tensor.ptr,
                    scale=scale,
                    B=B,
                    C=C,
                    BLOCK_C=block_c,
                )

        out._set_graph(parents=(self, targets), backward_fn=_backward)
        return out

    def argmax(self, dim: int = 1) -> "Tensor":
        """
        Return indices of maximum values along a dimension.

        Currently only supports dim=1 (argmax along rows) for 2D tensors.
        This is a non-differentiable operation (no backward).

        Args:
            dim: Dimension to reduce (currently only dim=1 supported).

        Returns:
            1D tensor of indices (as float32 for GPU compatibility).

        Raises:
            ValueError: If tensor is not 2D or dim is not 1.
        """
        if self.ndim != 2:
            raise ValueError(f"argmax requires 2D tensor, got shape {self._shape}")
        if dim != 1:
            raise ValueError(f"argmax currently only supports dim=1, got dim={dim}")

        rows, cols = self._shape
        out = empty_tensor((rows,), requires_grad=False)

        # Launch kernel - one program per row
        # Use BLOCK_COLS that covers most column sizes efficiently
        BLOCK_COLS = 64 if cols <= 64 else 256
        argmax_axis1_kernel[(rows,)](
            self.ptr, out.ptr, rows, cols, BLOCK_COLS=BLOCK_COLS
        )

        # No backward for argmax (non-differentiable)
        return out

    def linear(self, weight: "Tensor", bias: "Tensor") -> "Tensor":
        """
        Fused linear layer: Y = X @ W + b

        This computes matrix multiplication and bias addition in a single kernel,
        more efficient than separate matmul and add_bias operations.

        Args:
            weight: Weight matrix of shape (in_features, out_features).
            bias: Bias vector of shape (out_features,).

        Returns:
            Output tensor of shape (batch, out_features).

        Raises:
            ValueError: If shapes are incompatible.
        """
        # Validate shapes
        if self.ndim != 2:
            raise ValueError(f"linear requires 2D input, got shape {self._shape}")
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got shape {weight._shape}")
        if bias.ndim != 1:
            raise ValueError(f"bias must be 1D, got shape {bias._shape}")
        if self._shape[1] != weight._shape[0]:
            raise ValueError(
                f"linear shape mismatch: input {self._shape} @ weight {weight._shape}"
            )
        if weight._shape[1] != bias._shape[0]:
            raise ValueError(
                f"bias size {bias._shape[0]} doesn't match weight out_features {weight._shape[1]}"
            )

        M, K = self._shape
        K2, N = weight._shape
        out_shape = (M, N)

        out = empty_tensor(out_shape)

        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_bias_kernel[grid](
            self.ptr,
            weight.ptr,
            bias.ptr,
            out.ptr,
            M,
            N,
            K,
            K,
            1,  # X strides (M, K)
            N,
            1,  # W strides (K, N)
            N,
            1,  # Y strides (M, N)
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        # Capture for backward
        x_tensor = self
        w_tensor = weight
        b_tensor = bias

        def _backward(out_grad: Tensor) -> None:
            if x_tensor.requires_grad:
                ensure_grad(x_tensor)
                grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
                matmul_nt_acc_kernel[grid_dx](
                    out_grad.ptr,
                    w_tensor.ptr,
                    x_tensor.grad.ptr,
                    M,
                    K,
                    N,
                    N,
                    1,  # out_grad strides
                    N,
                    1,  # W strides
                    K,
                    1,  # dX strides
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )

            if w_tensor.requires_grad:
                ensure_grad(w_tensor)
                grid_dw = (triton.cdiv(K, BLOCK_M), triton.cdiv(N, BLOCK_N))
                matmul_tn_acc_kernel[grid_dw](
                    x_tensor.ptr,
                    out_grad.ptr,
                    w_tensor.grad.ptr,
                    K,
                    N,
                    M,
                    K,
                    1,
                    N,
                    1,
                    N,
                    1,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )

            if b_tensor.requires_grad:
                ensure_grad(b_tensor)
                sum_axis0_kernel[(N,)](
                    out_grad.ptr,
                    b_tensor.grad.ptr,
                    M,
                    N,
                    BLOCK_ROWS=32,
                )

        out._set_graph(parents=(self, weight, bias), backward_fn=_backward)
        return out

    def linear_relu(self, weight: "Tensor", bias: "Tensor") -> "Tensor":
        """
        Fused linear layer with ReLU: Y = relu(X @ W + b)

        This computes matrix multiplication, bias addition, and ReLU in a single
        kernel - more efficient than separate linear and relu operations.

        Useful for hidden layers in neural networks.

        Args:
            weight: Weight matrix of shape (in_features, out_features).
            bias: Bias vector of shape (out_features,).

        Returns:
            Output tensor of shape (batch, out_features) with ReLU applied.

        Raises:
            ValueError: If shapes are incompatible.
        """
        # Validate shapes
        if self.ndim != 2:
            raise ValueError(f"linear_relu requires 2D input, got shape {self._shape}")
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got shape {weight._shape}")
        if bias.ndim != 1:
            raise ValueError(f"bias must be 1D, got shape {bias._shape}")
        if self._shape[1] != weight._shape[0]:
            raise ValueError(
                f"linear_relu shape mismatch: input {self._shape} @ weight {weight._shape}"
            )
        if weight._shape[1] != bias._shape[0]:
            raise ValueError(
                f"bias size {bias._shape[0]} doesn't match weight out_features {weight._shape[1]}"
            )

        M, K = self._shape
        K2, N = weight._shape
        out_shape = (M, N)
        out_numel = M * N

        out = empty_tensor(out_shape)

        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_bias_relu_kernel[grid](
            self.ptr,
            weight.ptr,
            bias.ptr,
            out.ptr,
            M,
            N,
            K,
            K,
            1,
            N,
            1,
            N,
            1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        x_tensor = self
        w_tensor = weight
        b_tensor = bias
        out_tensor = out

        def _backward(out_grad: Tensor) -> None:
            with temp_storage(out_numel * F32) as st:
                dz = Tensor._wrap(st, out_shape, requires_grad=False)
                relu_mask_mul_kernel[grid1d(out_numel)](
                    dz.ptr, out_grad.ptr, out_tensor.ptr, out_numel, BLOCK=BLOCK
                )
                if x_tensor.requires_grad:
                    ensure_grad(x_tensor)
                    grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
                    matmul_nt_acc_kernel[grid_dx](
                        dz.ptr,
                        w_tensor.ptr,
                        x_tensor.grad.ptr,
                        M,
                        K,
                        N,
                        N,
                        1,
                        N,
                        1,
                        K,
                        1,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_K=BLOCK_K,
                    )
                if w_tensor.requires_grad:
                    ensure_grad(w_tensor)
                    grid_dw = (triton.cdiv(K, BLOCK_M), triton.cdiv(N, BLOCK_N))
                    matmul_tn_acc_kernel[grid_dw](
                        x_tensor.ptr,
                        dz.ptr,
                        w_tensor.grad.ptr,
                        K,
                        N,
                        M,
                        K,
                        1,
                        N,
                        1,
                        N,
                        1,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_K=BLOCK_K,
                    )
                if b_tensor.requires_grad:
                    ensure_grad(b_tensor)
                    sum_axis0_kernel[(N,)](
                        dz.ptr, b_tensor.grad.ptr, M, N, BLOCK_ROWS=32
                    )

        out._set_graph(parents=(self, weight, bias), backward_fn=_backward)
        return out

    def conv2d(
        self,
        weight: "Tensor",
        bias: "Tensor",
        stride: int = 1,
        padding: int = 0,
    ) -> "Tensor":
        """
        2D convolution: input (N, C_in, H, W), weight (C_out, C_in, kH, kW), bias (C_out).

        Output (N, C_out, H_out, W_out) with H_out = (H + 2*padding - kH) // stride + 1.
        Implemented via im2col + matmul.
        """
        x = self
        if x.ndim != 4:
            raise ValueError(f"conv2d requires 4D input, got shape {x._shape}")
        if weight.ndim != 4:
            raise ValueError(f"conv2d weight must be 4D, got shape {weight._shape}")
        if bias.ndim != 1:
            raise ValueError(f"conv2d bias must be 1D, got shape {bias._shape}")
        N, C_in, H, W = x._shape
        C_out, C_in_w, kH, kW = weight._shape
        if C_in != C_in_w:
            raise ValueError(
                f"conv2d input channels {C_in} != weight channels {C_in_w}"
            )
        if bias._shape[0] != C_out:
            raise ValueError(
                f"conv2d bias size {bias._shape[0]} != out_channels {C_out}"
            )

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"conv2d output spatial size invalid: H_out={H_out}, W_out={W_out}"
            )

        num_rows = N * H_out * W_out
        num_cols = C_in * kH * kW
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32
        IM2COL_BLOCK = 64

        # im2col: x -> col (num_rows, num_cols)
        col_ptr = cuda_malloc(num_rows * num_cols * 4)
        cuda_memset(col_ptr, 0, num_rows * num_cols * 4)
        # CUDA requires grid dimensions >= 1; cdiv can be 0 if num_cols is 0
        grid_im2col = (
            max(1, num_rows),
            max(1, triton.cdiv(num_cols, IM2COL_BLOCK)),
        )
        im2col_kernel_2d[grid_im2col](
            x.ptr,
            col_ptr,
            N=N,
            C=C_in,
            H=H,
            W=W,
            kH=kH,
            kW=kW,
            stride_h=stride,
            stride_w=stride,
            pad_h=padding,
            pad_w=padding,
            H_out=H_out,
            W_out=W_out,
            num_cols=num_cols,
            BLOCK=IM2COL_BLOCK,
        )

        # col @ weight_flat.T -> (num_rows, C_out). weight is (C_out, C_in, kH, kW) -> view (C_out, num_cols)
        w_flat = weight.view((C_out, num_cols))
        out_flat_ptr = cuda_malloc(num_rows * C_out * 4)
        # owns_memory=False: we cuda_free(out_flat_ptr) below; avoid double-free in __del__
        out_flat = Tensor._from_ptr(out_flat_ptr, (num_rows, C_out), owns_memory=False)
        grid_mm = (
            max(1, triton.cdiv(num_rows, BLOCK_M)),
            max(1, triton.cdiv(C_out, BLOCK_N)),
        )
        matmul_kernel[grid_mm](
            col_ptr,
            w_flat.ptr,
            out_flat_ptr,
            M=num_rows,
            N=C_out,
            K=num_cols,
            stride_am=num_cols,
            stride_ak=1,
            stride_bk=num_cols,
            stride_bn=1,
            stride_cm=C_out,
            stride_cn=1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        # Save col for backward only when input needs grad; else free to avoid leak when backward is never called (e.g. eval).
        needs_col_saved = x.requires_grad
        col_ptr_saved = col_ptr if needs_col_saved else None
        if not needs_col_saved:
            cuda_free(col_ptr)

        # add bias: out_flat (num_rows, C_out) + bias (C_out)
        y_flat_ptr = cuda_malloc(num_rows * C_out * 4)
        y_flat = Tensor._from_ptr(y_flat_ptr, (num_rows, C_out), owns_memory=True)
        add_bias_kernel[(max(1, triton.cdiv(num_rows * C_out, 256)),)](
            out_flat.ptr,
            bias.ptr,
            y_flat_ptr,
            rows=num_rows,
            cols=C_out,
            BLOCK=256,
        )
        cuda_free(out_flat_ptr)

        # view to 4D
        y_4d = y_flat.view((N, C_out, H_out, W_out))

        # backward closure
        x_tensor = x
        w_tensor = weight
        b_tensor = bias
        # Keep y_flat alive: y_4d shares storage with it; if y_flat is GC'd, __del__ frees the ptr (use-after-free).
        y_flat_storage = y_flat

        def _backward(out_grad: Tensor) -> None:
            _ = y_flat_storage  # keep storage alive while any tensor using this ptr exists
            # out_grad is (N, C_out, H_out, W_out)
            d_y_flat = out_grad.view((num_rows, C_out))
            if b_tensor.requires_grad:
                ensure_grad(b_tensor)
                sum_axis0_kernel[(C_out,)](
                    d_y_flat.ptr,
                    b_tensor.grad.ptr,
                    num_rows,
                    C_out,
                    BLOCK_ROWS=32,
                )
            # Reuse col from forward when saved; else recompute im2col (eval path kept col_ptr_saved=None)
            if col_ptr_saved is not None:
                col_bw = Tensor._from_ptr(
                    col_ptr_saved, (num_rows, num_cols), owns_memory=False
                )
            else:
                col_ptr_bw = cuda_malloc(num_rows * num_cols * 4)
                cuda_memset(col_ptr_bw, 0, num_rows * num_cols * 4)
                im2col_kernel_2d[grid_im2col](
                    x_tensor.ptr,
                    col_ptr_bw,
                    N=N,
                    C=C_in,
                    H=H,
                    W=W,
                    kH=kH,
                    kW=kW,
                    stride_h=stride,
                    stride_w=stride,
                    pad_h=padding,
                    pad_w=padding,
                    H_out=H_out,
                    W_out=W_out,
                    num_cols=num_cols,
                    BLOCK=IM2COL_BLOCK,
                )
                col_bw = Tensor._from_ptr(
                    col_ptr_bw, (num_rows, num_cols), owns_memory=True
                )
            w_flat_bw = w_tensor.view((C_out, num_cols))
            if w_tensor.requires_grad:
                ensure_grad(w_tensor)
                with temp_storage(num_cols * C_out * F32, zero=True) as d_w_temp_st:
                    grid_dw = (
                        max(1, triton.cdiv(num_cols, BLOCK_M)),
                        max(1, triton.cdiv(C_out, BLOCK_N)),
                    )
                    matmul_tn_acc_kernel[grid_dw](
                        col_bw.ptr,
                        d_y_flat.ptr,
                        d_w_temp_st.ptr,
                        M=num_cols,
                        N=C_out,
                        K=num_rows,
                        stride_ak=num_cols,
                        stride_am=1,
                        stride_bk=C_out,
                        stride_bn=1,
                        stride_cm=C_out,
                        stride_cn=1,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_K=BLOCK_K,
                    )
                    with temp_storage(C_out * num_cols * F32) as d_w_T_st:
                        transpose2d_kernel[(triton.cdiv(num_cols * C_out, BLOCK),)](
                            d_w_temp_st.ptr,
                            d_w_T_st.ptr,
                            rows=num_cols,
                            cols=C_out,
                            BLOCK=BLOCK,
                        )
                        add_inplace_kernel[grid1d(C_out * num_cols)](
                            w_tensor.grad.ptr,
                            d_w_T_st.ptr,
                            C_out * num_cols,
                            BLOCK=BLOCK,
                        )
            with temp_storage(num_rows * num_cols * F32) as d_col_st:
                d_col = Tensor._wrap(
                    d_col_st, (num_rows, num_cols), requires_grad=False
                )
                grid_dcol = (
                    max(1, triton.cdiv(num_rows, BLOCK_M)),
                    max(1, triton.cdiv(num_cols, BLOCK_N)),
                )
                matmul_kernel[grid_dcol](
                    d_y_flat.ptr,
                    w_flat_bw.ptr,
                    d_col_st.ptr,
                    M=num_rows,
                    N=num_cols,
                    K=C_out,
                    stride_am=C_out,
                    stride_ak=1,
                    stride_bk=num_cols,
                    stride_bn=1,
                    stride_cm=num_cols,
                    stride_cn=1,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )
                if col_ptr_saved is not None:
                    cuda_free(col_ptr_saved)
                else:
                    cuda_free(col_ptr_bw)
                if x_tensor.requires_grad:
                    ensure_grad(x_tensor)
                    grid_col2im = (
                        max(1, num_rows),
                        max(1, triton.cdiv(num_cols, IM2COL_BLOCK)),
                    )
                    col2im_kernel[grid_col2im](
                        d_col_st.ptr,
                        x_tensor.grad.ptr,
                        N=N,
                        C=C_in,
                        H=H,
                        W=W,
                        kH=kH,
                        kW=kW,
                        stride_h=stride,
                        stride_w=stride,
                        pad_h=padding,
                        pad_w=padding,
                        H_out=H_out,
                        W_out=W_out,
                        num_cols=num_cols,
                        BLOCK=IM2COL_BLOCK,
                    )

        y_4d._set_graph(parents=(self, weight, bias), backward_fn=_backward)
        return y_4d

    def maxpool2d(
        self,
        kernel_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] | None = None,
    ) -> "Tensor":
        """
        2D max pooling: (N, C, H, W) -> (N, C, H_out, W_out).

        Default kernel_size=2, stride=2 (half spatial size).
        """
        x = self
        if x.ndim != 4:
            raise ValueError(f"maxpool2d requires 4D input, got shape {x._shape}")
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        if stride is None:
            stride_h = stride_w = kH  # default stride = kernel_size
        elif isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
        N, C, H, W = x._shape
        H_out = (H - kH) // stride_h + 1
        W_out = (W - kW) // stride_w + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"maxpool2d output size invalid: H_out={H_out}, W_out={W_out}"
            )
        out_shape = (N, C, H_out, W_out)
        numel_out = N * C * H_out * W_out
        out = empty_tensor(out_shape)
        idx_st = alloc_storage(numel_out * F32)

        BLOCK_ELEMS = 256
        grid_size = triton.cdiv(numel_out, BLOCK_ELEMS)
        maxpool2d_forward_kernel[(grid_size,)](
            x.ptr,
            out.ptr,
            idx_st.ptr,
            N=N,
            C=C,
            H=H,
            W=W,
            H_out=H_out,
            W_out=W_out,
            stride_h=stride_h,
            stride_w=stride_w,
            BLOCK_KH=kH,
            BLOCK_KW=kW,
            BLOCK_ELEMS=BLOCK_ELEMS,
        )

        x_tensor = x

        def _backward(out_grad: Tensor) -> None:
            if x_tensor.requires_grad:
                ensure_grad(x_tensor)
                grid_size_bw = triton.cdiv(numel_out, 256)
                maxpool2d_backward_kernel[(grid_size_bw,)](
                    out_grad.ptr,
                    idx_st.ptr,
                    x_tensor.grad.ptr,
                    N=N,
                    C=C,
                    H=H,
                    W=W,
                    H_out=H_out,
                    W_out=W_out,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    BLOCK_KW=kW,
                    BLOCK_ELEMS=256,
                )
            idx_st.free()

        out._set_graph(parents=(self,), backward_fn=_backward)
        if not out.requires_grad:
            idx_st.free()
        return out

    def add_bias(self, bias: "Tensor") -> "Tensor":
        """
        Add a 1D bias to a 2D tensor with broadcasting.

        Y[i, j] = self[i, j] + bias[j]

        Args:
            bias: 1D tensor of shape (H,) where self has shape (B, H).

        Returns:
            New tensor of same shape as self.

        Raises:
            ValueError: If shapes are incompatible.
        """
        # Validate shapes
        if self.ndim != 2:
            raise ValueError(f"add_bias requires 2D input, got shape {self._shape}")
        if bias.ndim != 1:
            raise ValueError(f"bias must be 1D, got shape {bias._shape}")
        if self._shape[1] != bias._shape[0]:
            raise ValueError(
                f"Bias size {bias._shape[0]} doesn't match input features {self._shape[1]}"
            )

        rows, cols = self._shape
        out = empty_tensor(self.shape)
        add_bias_kernel[grid1d(self.numel)](
            self.ptr, bias.ptr, out.ptr, rows, cols, BLOCK=BLOCK
        )

        input_tensor = self
        bias_tensor = bias

        def _backward(out_grad: Tensor) -> None:
            if input_tensor.requires_grad:
                accum_grad(input_tensor, out_grad)
            if bias_tensor.requires_grad:
                ensure_grad(bias_tensor)
                sum_axis0_kernel[(cols,)](
                    out_grad.ptr,
                    bias_tensor.grad.ptr,
                    rows,
                    cols,
                    BLOCK_ROWS=32,
                )

        out._set_graph(parents=(self, bias), backward_fn=_backward)
        return out

    def mul(self, other: "Tensor") -> "Tensor":
        """
        Elementwise multiplication: C = self * other

        Both tensors must have the same shape.

        Args:
            other: Tensor to multiply, must have same shape as self.

        Returns:
            New tensor containing the elementwise product.

        Raises:
            ValueError: If shapes don't match.
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for mul: {self.shape} vs {other.shape}")

        out = empty_tensor(self.shape)
        mul_kernel[grid1d(self.numel)](
            self.ptr, other.ptr, out.ptr, self.numel, BLOCK=BLOCK
        )

        a_tensor = self
        b_tensor = other

        def _backward(out_grad: Tensor) -> None:
            if a_tensor.requires_grad:
                ensure_grad(a_tensor)
                mul_backward_kernel[grid1d(a_tensor.numel)](
                    a_tensor.grad.ptr,
                    out_grad.ptr,
                    b_tensor.ptr,
                    a_tensor.numel,
                    BLOCK=BLOCK,
                )
            if b_tensor.requires_grad:
                ensure_grad(b_tensor)
                mul_backward_kernel[grid1d(b_tensor.numel)](
                    b_tensor.grad.ptr,
                    out_grad.ptr,
                    a_tensor.ptr,
                    b_tensor.numel,
                    BLOCK=BLOCK,
                )

        out._set_graph(parents=(self, other), backward_fn=_backward)
        return out

    def scale(self, scalar: float) -> "Tensor":
        """
        Multiply tensor by a scalar: C = self * scalar

        Args:
            scalar: The scalar value to multiply by.

        Returns:
            New tensor containing the scaled values.
        """
        out = empty_tensor(self.shape)
        mul_scalar_kernel[grid1d(self.numel)](
            self.ptr, scalar, out.ptr, self.numel, BLOCK=BLOCK
        )

        input_tensor = self
        scale_val = scalar

        def _backward(out_grad: Tensor) -> None:
            if input_tensor.requires_grad:
                ensure_grad(input_tensor)
                scale_backward_kernel[grid1d(input_tensor.numel)](
                    input_tensor.grad.ptr,
                    out_grad.ptr,
                    scale_val,
                    input_tensor.numel,
                    BLOCK=BLOCK,
                )

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def mul_scalar_inplace_(self, scalar: float) -> None:
        """
        In-place multiply by scalar: self *= scalar.

        No graph, no backward. Used for gradient clipping.

        Args:
            scalar: The scalar value to multiply by.
        """
        mul_scalar_inplace_kernel[grid1d(self.numel)](
            self.ptr, scalar, self.numel, BLOCK=BLOCK
        )

    # -------------------------------------------------------------------------
    # Operator overloads
    # -------------------------------------------------------------------------

    def __add__(self, other: "Tensor") -> "Tensor":
        """Addition: self + other"""
        return self.add(other)

    def __radd__(self, other: "Tensor") -> "Tensor":
        """Reverse addition: other + self"""
        return self.add(other)

    def __mul__(self, other):
        """Multiplication: self * other (tensor or scalar)"""
        if isinstance(other, Tensor):
            return self.mul(other)
        else:
            return self.scale(float(other))

    def __rmul__(self, other):
        """Reverse multiplication: other * self"""
        return self.__mul__(other)

    def __neg__(self) -> "Tensor":
        """Negation: -self"""
        return self.scale(-1.0)

    def __sub__(self, other: "Tensor") -> "Tensor":
        """Subtraction: self - other"""
        return self.add(other.scale(-1.0))

    def __rsub__(self, other: "Tensor") -> "Tensor":
        """Reverse subtraction: other - self"""
        return other.add(self.scale(-1.0))

    def to_list(self) -> list:
        """
        Copy data from GPU to CPU and return as nested Python list.

        Returns:
            Nested list matching the tensor's shape.
        """
        # Copy from GPU
        host_bytes = cuda_memcpy_dtoh(self.ptr, self.nbytes)

        # Unpack to floats
        flat_data = list(struct.unpack(f"{self.numel}f", host_bytes))

        # Reshape to nested list
        return self._unflatten(flat_data, self._shape)

    def item(self) -> float:
        """
        Return the scalar value for a single-element tensor.

        Returns:
            The scalar value as a Python float.

        Raises:
            ValueError: If tensor has more than one element.
        """
        if self.numel != 1:
            raise ValueError(
                f"item() only works for single-element tensors, got {self.numel} elements"
            )
        return self.to_list()[0]

    def save(
        self,
        path: str,
        name: str = "tensor",
        metadata: dict[str, str] | None = None,
    ) -> None:
        """
        Save this tensor to a SafeTensors-compatible file.

        Args:
            path: Output file path (.safetensors).
            name: Key name for this tensor in the file.
            metadata: Optional string->string metadata in __metadata__.
        """

        from .tensor_io import save_safetensors

        save_safetensors(path, {name: self}, metadata=metadata)

    @staticmethod
    def load(
        path: str,
        name: str = "tensor",
        *,
        requires_grad: bool = False,
    ) -> "Tensor":
        """
        Load a single tensor from a SafeTensors-compatible file.

        Args:
            path: Path to .safetensors file.
            name: Key name of the tensor to load.
            requires_grad: Applied to the returned tensor.

        Returns:
            The tensor with the given name.

        Raises:
            KeyError: If name is not in the file.
        """

        from .tensor_io import load_safetensors

        state = load_safetensors(path, requires_grad=requires_grad)
        if name not in state:
            raise KeyError(f"tensor {name!r} not found in {path}")
        return state[name]

    def __repr__(self) -> str:
        """Return string representation of the tensor."""
        name_str = f", name='{self.name}'" if self.name else ""
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self._shape}, ptr=0x{self.ptr:x}{grad_str}{name_str})"

    # -------------------------------------------------------------------------
    # Helper methods for data conversion
    # -------------------------------------------------------------------------

    @staticmethod
    def _infer_shape_and_flatten(
        data: list | tuple,
    ) -> tuple[tuple[int, ...], list[float]]:
        """
        Infer shape from nested structure and return flattened data.

        Args:
            data: Nested list or tuple of floats.

        Returns:
            Tuple of (shape, flat_data).

        Raises:
            ValueError: If structure is inconsistent (ragged array).
        """
        shape: list[int] = []
        current = data

        # Walk down to find shape at each level
        while isinstance(current, (list, tuple)):
            shape.append(len(current))
            if len(current) == 0:
                break
            current = current[0]

        # Flatten the data
        flat_data = Tensor._flatten(data)

        # Validate: product of shape must equal number of elements
        expected_numel = 1
        for s in shape:
            expected_numel *= s

        if len(flat_data) != expected_numel:
            raise ValueError(
                f"Inconsistent data structure (ragged array?). "
                f"Inferred shape {tuple(shape)} expects {expected_numel} elements, "
                f"but found {len(flat_data)}"
            )

        return tuple(shape), flat_data

    @staticmethod
    def _flatten(data: list | tuple | float | int) -> list[float]:
        """
        Recursively flatten nested lists/tuples to a flat list of floats.

        Args:
            data: Nested structure or scalar.

        Returns:
            Flat list of floats.
        """
        if isinstance(data, (int, float)):
            return [float(data)]

        result: list[float] = []
        for item in data:
            result.extend(Tensor._flatten(item))
        return result

    @staticmethod
    def _unflatten(flat_data: list[float], shape: tuple[int, ...]) -> list:
        """
        Reshape flat data into nested list matching shape.

        Args:
            flat_data: Flat list of floats.
            shape: Target shape.

        Returns:
            Nested list with given shape.
        """
        if len(shape) == 1:
            return flat_data[: shape[0]]

        # For 2D, chunk into rows
        if len(shape) == 2:
            rows, cols = shape
            result = []
            for i in range(rows):
                start = i * cols
                end = start + cols
                result.append(flat_data[start:end])
            return result

        # For 3D, chunk into 2D slices then rows
        if len(shape) == 3:
            d0, d1, d2 = shape
            result = []
            for i in range(d0):
                slice_result = []
                for j in range(d1):
                    start = (i * d1 + j) * d2
                    end = start + d2
                    slice_result.append(flat_data[start:end])
                result.append(slice_result)
            return result

        # For 4D, nest by batch -> channels -> height -> width
        if len(shape) == 4:
            n, c, h, w = shape
            result = []
            for i in range(n):
                batch_result = []
                for j in range(c):
                    chan_result = []
                    for k in range(h):
                        start = (i * c * h + j * h + k) * w
                        end = start + w
                        chan_result.append(flat_data[start:end])
                    batch_result.append(chan_result)
                result.append(batch_result)
            return result

        raise ValueError(f"Unsupported shape rank: {len(shape)}")
