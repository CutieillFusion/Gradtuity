"""
Tensor class with from-scratch GPU storage and autograd support.

This module implements a minimal Tensor type that:
- Holds GPU storage as raw pointers managed via ctypes + CUDA runtime
- Tracks a computation graph and supports backward()
- Enforces invariants: CUDA only, float32 only, contiguous only, rank in {1, 2, 3, 4}
"""

from __future__ import annotations

import os
import struct
from typing import Callable, Optional

from .cuda_mem import cuda_free, cuda_malloc, cuda_memcpy_dtoh, cuda_memcpy_htod


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
        for i, dim in enumerate(shape):
            if dim <= 0:
                raise ValueError(f"Dimension {i} must be positive, got {dim}")

        # Store shape info
        self._shape: tuple[int, ...] = tuple(shape)
        self._numel: int = len(flat_data)
        self._nbytes: int = self._numel * 4  # float32 = 4 bytes

        # Allocate GPU memory and copy data
        self._ptr: int = cuda_malloc(self._nbytes)
        self._owns_memory: bool = True  # Track ownership for safe cleanup

        # Pack floats to bytes and copy to GPU
        host_bytes = struct.pack(f"{self._numel}f", *flat_data)
        cuda_memcpy_htod(self._ptr, host_bytes)

        # Autograd fields
        self.requires_grad: bool = requires_grad
        self.grad: Optional[Tensor] = None
        self.name: str = name

        # Graph fields (set by ops when requires_grad is True)
        self._parents: tuple[Tensor, ...] = ()
        self._backward: Optional[Callable[[Tensor], None]] = None
        self._ctx: Optional[dict] = None

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
        # Validate shape
        if len(shape) not in (1, 2, 3, 4):
            raise ValueError(
                f"Only rank 1, 2, 3, or 4 tensors supported, got rank {len(shape)}"
            )

        # Create instance without calling __init__
        tensor = object.__new__(cls)

        # Compute numel
        numel = 1
        for s in shape:
            numel *= s

        # Set attributes directly
        tensor._ptr = ptr
        tensor._shape = tuple(shape)
        tensor._numel = numel
        tensor._nbytes = numel * 4
        tensor._owns_memory = owns_memory
        tensor.requires_grad = requires_grad
        tensor.grad = None
        tensor.name = name
        tensor._parents = ()
        tensor._backward = None
        tensor._ctx = None

        return tensor

    def __del__(self) -> None:
        """Free GPU memory when Tensor is garbage collected."""
        # Only free if we own the memory and pointer is valid
        if (
            getattr(self, "_owns_memory", False)
            and getattr(self, "_ptr", None) is not None
        ):
            cuda_free(self._ptr)
            self._ptr = None

    def data_ptr(self) -> int:
        """
        Return raw GPU pointer for Triton kernels.

        Returns:
            GPU memory pointer as integer.
        """
        return self._ptr

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the tensor."""
        return self._shape

    @property
    def numel(self) -> int:
        """Return the total number of elements."""
        return self._numel

    @property
    def nbytes(self) -> int:
        """Return the total number of bytes."""
        return self._nbytes

    @property
    def ndim(self) -> int:
        """Return the number of dimensions (rank)."""
        return len(self._shape)

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
        if self._numel != 1:
            raise ValueError(
                f"backward() only works on scalar tensors (numel=1), "
                f"got numel={self._numel}"
            )

        if not self.requires_grad:
            raise RuntimeError(
                "backward() called on a tensor that doesn't require grad. "
                "Set requires_grad=True on leaf tensors."
            )

        # Import here to avoid circular import
        from .functional import ones_like

        # Seed grad with 1.0
        self.grad = ones_like(self)

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
        return Tensor._from_ptr(
            ptr=self._ptr,
            shape=self._shape,
            owns_memory=False,  # Shared pointer, don't free
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
            inferred = self._numel // product
            if inferred * product != self._numel:
                raise ValueError(
                    f"view() shape {new_shape} is incompatible with numel {self._numel}"
                )
            new_shape_list[infer_idx] = inferred
        else:
            if product != self._numel:
                raise ValueError(
                    f"view() shape {new_shape} has {product} elements, "
                    f"but tensor has {self._numel} elements"
                )
        resolved_shape = tuple(new_shape_list)

        out = Tensor._from_ptr(
            self._ptr,
            resolved_shape,
            owns_memory=False,
            requires_grad=False,
        )
        input_tensor = self

        def _backward(out_grad: Tensor) -> None:
            if input_tensor.requires_grad:
                grad_viewed = out_grad.view(input_tensor._shape)
                if input_tensor.grad is None:
                    from .functional import zeros_like
                    input_tensor.grad = zeros_like(input_tensor)
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

        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc
        from .functional import zeros_like
        from .kernels.matmul_kernels import (
            matmul_kernel,
            matmul_nt_acc_kernel,
            matmul_tn_acc_kernel,
        )

        M, K = self._shape
        K2, N = other._shape
        out_shape = (M, N)
        out_numel = M * N

        # Allocate output
        out_ptr = cuda_malloc(out_numel * 4)
        out = Tensor._from_ptr(out_ptr, out_shape, owns_memory=True)

        # Block sizes for matmul (tuned for typical small matrices)
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32

        # Launch matmul kernel
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_kernel[grid](
            self._ptr,
            other._ptr,
            out._ptr,
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
                if a_tensor.grad is None:
                    a_tensor.grad = zeros_like(a_tensor)

                # dA += out_grad @ B^T using fused kernel
                # out_grad: (M, N), B: (K, N) read as transposed -> result: (M, K)
                # B is stored as (K, N) row-major: B[i,j] = b_ptr[i*N + j]
                # We want B^T[k, n] = B[n, k] where n is row index, k is col index
                # So stride_bn = N (row stride), stride_bk = 1 (col stride)
                grid_da = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
                matmul_nt_acc_kernel[grid_da](
                    out_grad._ptr,
                    b_tensor._ptr,
                    a_tensor.grad._ptr,
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
                if b_tensor.grad is None:
                    b_tensor.grad = zeros_like(b_tensor)

                # dB += A^T @ out_grad using fused kernel
                # A: (M, K) read as transposed, out_grad: (M, N) -> result: (K, N)
                grid_db = (triton.cdiv(K, BLOCK_M), triton.cdiv(N, BLOCK_N))
                matmul_tn_acc_kernel[grid_db](
                    a_tensor._ptr,
                    out_grad._ptr,
                    b_tensor.grad._ptr,
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
        # Validate shapes match
        if self._shape != other._shape:
            raise ValueError(f"Shape mismatch for add: {self._shape} vs {other._shape}")

        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc
        from .functional import zeros_like
        from .kernels.elemwise_kernels import add_inplace_kernel, add_kernel

        # Allocate output
        out_ptr = cuda_malloc(self._nbytes)
        out = Tensor._from_ptr(out_ptr, self._shape, owns_memory=True)

        # Launch kernel
        grid = lambda meta: (triton.cdiv(self._numel, meta["BLOCK"]),)
        add_kernel[grid](self._ptr, other._ptr, out._ptr, self._numel, BLOCK=256)

        # Set up backward
        def _backward(out_grad: Tensor) -> None:
            # dA += out_grad, dB += out_grad
            if self.requires_grad:
                if self.grad is None:
                    self.grad = zeros_like(self)
                grid = lambda meta: (triton.cdiv(self._numel, meta["BLOCK"]),)
                add_inplace_kernel[grid](
                    self.grad._ptr, out_grad._ptr, self._numel, BLOCK=256
                )

            if other.requires_grad:
                if other.grad is None:
                    other.grad = zeros_like(other)
                grid = lambda meta: (triton.cdiv(other._numel, meta["BLOCK"]),)
                add_inplace_kernel[grid](
                    other.grad._ptr, out_grad._ptr, other._numel, BLOCK=256
                )

        out._set_graph(parents=(self, other), backward_fn=_backward)
        return out

    def relu(self) -> "Tensor":
        """
        ReLU activation: Z = max(self, 0)

        Returns:
            New tensor with ReLU applied elementwise.
        """
        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc
        from .functional import zeros_like
        from .kernels.elemwise_kernels import relu_backward_kernel, relu_kernel

        # Allocate output
        out_ptr = cuda_malloc(self._nbytes)
        out = Tensor._from_ptr(out_ptr, self._shape, owns_memory=True)

        # Launch forward kernel
        grid = lambda meta: (triton.cdiv(self._numel, meta["BLOCK"]),)
        relu_kernel[grid](self._ptr, out._ptr, self._numel, BLOCK=256)

        # Set up backward - capture self (Y) for mask computation
        # Important: use self._ptr (original input), not out._ptr
        y_tensor = self  # Capture reference to input for backward

        def _backward(out_grad: Tensor) -> None:
            # dY += out_grad * (Y > 0)
            if y_tensor.requires_grad:
                if y_tensor.grad is None:
                    y_tensor.grad = zeros_like(y_tensor)
                grid = lambda meta: (triton.cdiv(y_tensor._numel, meta["BLOCK"]),)
                relu_backward_kernel[grid](
                    y_tensor.grad._ptr,
                    out_grad._ptr,
                    y_tensor._ptr,  # Original input for mask
                    y_tensor._numel,
                    BLOCK=256,
                )

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def sum(self) -> "Tensor":
        """
        Sum all elements to produce a scalar tensor.

        Returns:
            Scalar tensor with shape (1,) containing the sum of all elements.
        """
        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc, cuda_memset
        from .functional import zeros_like
        from .kernels.reduce_kernels import add_scalar_inplace_kernel, sum_all_kernel

        # Allocate zero-initialized output (MUST be zero for atomic adds)
        out_ptr = cuda_malloc(4)  # 1 float32 = 4 bytes
        cuda_memset(out_ptr, 0, 4)
        out = Tensor._from_ptr(out_ptr, (1,), owns_memory=True)

        # Launch forward kernel
        grid = lambda meta: (triton.cdiv(self._numel, meta["BLOCK"]),)
        sum_all_kernel[grid](self._ptr, out._ptr, self._numel, BLOCK=256)

        # Capture self for backward
        input_tensor = self

        def _backward(out_grad: Tensor) -> None:
            # dX += broadcast(out_grad) = add scalar to all elements
            if input_tensor.requires_grad:
                if input_tensor.grad is None:
                    input_tensor.grad = zeros_like(input_tensor)
                grid = lambda meta: (triton.cdiv(input_tensor._numel, meta["BLOCK"]),)
                add_scalar_inplace_kernel[grid](
                    input_tensor.grad._ptr,
                    out_grad._ptr,  # Scalar gradient
                    input_tensor._numel,
                    BLOCK=256,
                )

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

    def mse_loss(self, target: "Tensor") -> "Tensor":
        """
        Compute mean squared error loss: L = mean((self - target)^2)

        This is a fused operation that computes the MSE in a single kernel,
        more efficient than separate subtract, square, and mean operations.

        Args:
            target: Target tensor, must have same shape as self.

        Returns:
            Scalar tensor containing the MSE loss.

        Raises:
            ValueError: If shapes don't match.
        """
        # Validate shapes match
        if self._shape != target._shape:
            raise ValueError(
                f"Shape mismatch for mse_loss: {self._shape} vs {target._shape}"
            )

        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc, cuda_memset
        from .functional import zeros_like
        from .kernels.reduce_kernels import mse_loss_backward_kernel, mse_loss_kernel

        # Allocate zero-initialized output (MUST be zero for atomic adds)
        out_ptr = cuda_malloc(4)  # 1 float32 = 4 bytes
        cuda_memset(out_ptr, 0, 4)
        out = Tensor._from_ptr(out_ptr, (1,), owns_memory=True)

        # Launch forward kernel - computes sum((self - target)^2)
        grid = lambda meta: (triton.cdiv(self._numel, meta["BLOCK"]),)
        mse_loss_kernel[grid](self._ptr, target._ptr, out._ptr, self._numel, BLOCK=256)

        # Scale by 1/numel to get mean (done via a scale operation)
        # We'll do this in the backward by incorporating the scale factor
        # For forward, we need to divide the result
        from .kernels.elemwise_kernels import mul_scalar_kernel

        scale = 1.0 / self._numel
        mul_scalar_kernel[grid](out._ptr, scale, out._ptr, 1, BLOCK=256)

        # Capture for backward
        pred_tensor = self
        target_tensor = target
        numel = self._numel

        def _backward(out_grad: Tensor) -> None:
            # MSE gradient: d/dpred = 2 * (pred - target) / N
            #               d/dtarget = -2 * (pred - target) / N
            # out_grad is scalar (typically 1.0)

            # Get the scalar gradient value
            from .cuda_mem import cuda_memcpy_dtoh
            import struct

            grad_bytes = cuda_memcpy_dtoh(out_grad._ptr, 4)
            grad_val = struct.unpack("f", grad_bytes)[0]
            scale = grad_val / numel

            # Allocate gradients if needed
            if pred_tensor.requires_grad and pred_tensor.grad is None:
                pred_tensor.grad = zeros_like(pred_tensor)
            if target_tensor.requires_grad and target_tensor.grad is None:
                target_tensor.grad = zeros_like(target_tensor)

            # Use fused backward kernel
            grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)
            mse_loss_backward_kernel[grid](
                pred_tensor.grad._ptr if pred_tensor.requires_grad else 0,
                target_tensor.grad._ptr if target_tensor.requires_grad else 0,
                pred_tensor._ptr,
                target_tensor._ptr,
                scale,
                numel,
                1 if pred_tensor.requires_grad else 0,
                1 if target_tensor.requires_grad else 0,
                BLOCK=256,
            )

        out._set_graph(parents=(self, target), backward_fn=_backward)
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

        from .cuda_mem import cuda_malloc
        from .kernels.reduce_kernels import argmax_axis1_kernel

        rows, cols = self._shape

        # Allocate output: one index per row
        out_ptr = cuda_malloc(rows * 4)  # float32
        out = Tensor._from_ptr(out_ptr, (rows,), owns_memory=True, requires_grad=False)

        # Launch kernel - one program per row
        # Use BLOCK_COLS that covers most column sizes efficiently
        BLOCK_COLS = 64 if cols <= 64 else 256
        argmax_axis1_kernel[(rows,)](
            self._ptr, out._ptr, rows, cols, BLOCK_COLS=BLOCK_COLS
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

        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc
        from .functional import zeros_like
        from .kernels.matmul_kernels import (
            matmul_bias_kernel,
            matmul_nt_acc_kernel,
            matmul_tn_acc_kernel,
        )
        from .kernels.reduce_kernels import sum_axis0_kernel

        M, K = self._shape
        K2, N = weight._shape
        out_shape = (M, N)
        out_numel = M * N

        # Allocate output
        out_ptr = cuda_malloc(out_numel * 4)
        out = Tensor._from_ptr(out_ptr, out_shape, owns_memory=True)

        # Block sizes
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32

        # Launch fused matmul+bias kernel
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_bias_kernel[grid](
            self._ptr,
            weight._ptr,
            bias._ptr,
            out._ptr,
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
            # dX = out_grad @ W^T
            # dW = X^T @ out_grad
            # db = sum over axis 0 of out_grad

            if x_tensor.requires_grad:
                if x_tensor.grad is None:
                    x_tensor.grad = zeros_like(x_tensor)
                grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
                matmul_nt_acc_kernel[grid_dx](
                    out_grad._ptr,
                    w_tensor._ptr,
                    x_tensor.grad._ptr,
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
                if w_tensor.grad is None:
                    w_tensor.grad = zeros_like(w_tensor)
                grid_dw = (triton.cdiv(K, BLOCK_M), triton.cdiv(N, BLOCK_N))
                matmul_tn_acc_kernel[grid_dw](
                    x_tensor._ptr,
                    out_grad._ptr,
                    w_tensor.grad._ptr,
                    K,
                    N,
                    M,
                    K,
                    1,  # X strides
                    N,
                    1,  # out_grad strides
                    N,
                    1,  # dW strides
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )

            if b_tensor.requires_grad:
                if b_tensor.grad is None:
                    b_tensor.grad = zeros_like(b_tensor)
                # db = sum over rows of out_grad
                sum_axis0_kernel[(N,)](
                    out_grad._ptr,
                    b_tensor.grad._ptr,
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

        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc
        from .functional import zeros_like
        from .kernels.elemwise_kernels import relu_backward_kernel
        from .kernels.matmul_kernels import (
            matmul_bias_relu_kernel,
            matmul_nt_acc_kernel,
            matmul_tn_acc_kernel,
        )
        from .kernels.reduce_kernels import sum_axis0_kernel

        M, K = self._shape
        K2, N = weight._shape
        out_shape = (M, N)
        out_numel = M * N

        # Allocate output
        out_ptr = cuda_malloc(out_numel * 4)
        out = Tensor._from_ptr(out_ptr, out_shape, owns_memory=True)

        # Block sizes
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32

        # Launch fused matmul+bias+relu kernel
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_bias_relu_kernel[grid](
            self._ptr,
            weight._ptr,
            bias._ptr,
            out._ptr,
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
        out_tensor = out  # Need to keep reference for ReLU mask

        def _backward(out_grad: Tensor) -> None:
            # For Y = relu(X @ W + b):
            # Let Z = X @ W + b (pre-activation)
            # Y = relu(Z)
            # dZ = dY * (Z > 0) = dY * (Y > 0) since Y = relu(Z)
            # dX = dZ @ W^T
            # dW = X^T @ dZ
            # db = sum over axis 0 of dZ

            # First compute dZ = out_grad * relu_mask
            # We use Y > 0 as the mask since Y = relu(Z)
            from .kernels.elemwise_kernels import mul_kernel

            # Allocate dZ (gradient after ReLU mask)
            dz_ptr = cuda_malloc(out_numel * 4)
            dz = Tensor._from_ptr(dz_ptr, out_shape, owns_memory=True)

            # dZ = out_grad * (Y > 0)
            # We can use a kernel that does: dZ[i] = out_grad[i] if Y[i] > 0 else 0
            from .kernels.elemwise_kernels import relu_mask_mul_kernel

            grid_elem = lambda meta: (triton.cdiv(out_numel, meta["BLOCK"]),)
            relu_mask_mul_kernel[grid_elem](
                dz._ptr,
                out_grad._ptr,
                out_tensor._ptr,  # Y values for mask
                out_numel,
                BLOCK=256,
            )

            if x_tensor.requires_grad:
                if x_tensor.grad is None:
                    x_tensor.grad = zeros_like(x_tensor)
                grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
                matmul_nt_acc_kernel[grid_dx](
                    dz._ptr,
                    w_tensor._ptr,
                    x_tensor.grad._ptr,
                    M,
                    K,
                    N,
                    N,
                    1,  # dZ strides
                    N,
                    1,  # W strides
                    K,
                    1,  # dX strides
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )

            if w_tensor.requires_grad:
                if w_tensor.grad is None:
                    w_tensor.grad = zeros_like(w_tensor)
                grid_dw = (triton.cdiv(K, BLOCK_M), triton.cdiv(N, BLOCK_N))
                matmul_tn_acc_kernel[grid_dw](
                    x_tensor._ptr,
                    dz._ptr,
                    w_tensor.grad._ptr,
                    K,
                    N,
                    M,
                    K,
                    1,  # X strides
                    N,
                    1,  # dZ strides
                    N,
                    1,  # dW strides
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )

            if b_tensor.requires_grad:
                if b_tensor.grad is None:
                    b_tensor.grad = zeros_like(b_tensor)
                # db = sum over rows of dZ
                sum_axis0_kernel[(N,)](
                    dz._ptr,
                    b_tensor.grad._ptr,
                    M,
                    N,
                    BLOCK_ROWS=32,
                )

            # Free dZ
            from .cuda_mem import cuda_free

            cuda_free(dz_ptr)

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
        import triton

        from .cuda_mem import cuda_malloc, cuda_free, cuda_memset
        from .functional import zeros_like
        from .kernels.conv_kernels import col2im_kernel, im2col_kernel_2d
        from .kernels.matmul_kernels import (
            matmul_kernel,
            matmul_nt_acc_kernel,
            matmul_tn_acc_kernel,
        )
        from .kernels.reduce_kernels import add_bias_kernel, sum_axis0_kernel

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
            x._ptr,
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
        out_flat = Tensor._from_ptr(
            out_flat_ptr, (num_rows, C_out), owns_memory=False
        )
        grid_mm = (
            max(1, triton.cdiv(num_rows, BLOCK_M)),
            max(1, triton.cdiv(C_out, BLOCK_N)),
        )
        matmul_kernel[grid_mm](
            col_ptr,
            w_flat._ptr,
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
        y_flat = Tensor._from_ptr(
            y_flat_ptr, (num_rows, C_out), owns_memory=True
        )
        add_bias_kernel[
            (max(1, triton.cdiv(num_rows * C_out, 256)),)
        ](
            out_flat._ptr,
            bias._ptr,
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
                if b_tensor.grad is None:
                    b_tensor.grad = zeros_like(b_tensor)
                sum_axis0_kernel[(C_out,)](
                    d_y_flat._ptr,
                    b_tensor.grad._ptr,
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
                    x_tensor._ptr,
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
                if w_tensor.grad is None:
                    w_tensor.grad = zeros_like(w_tensor)
                # d_weight = col.T @ d_y_flat -> (num_cols, C_out); weight.grad is (C_out, num_cols)
                d_weight_temp_ptr = cuda_malloc(num_cols * C_out * 4)
                cuda_memset(d_weight_temp_ptr, 0, num_cols * C_out * 4)
                grid_dw = (
                    max(1, triton.cdiv(num_cols, BLOCK_M)),
                    max(1, triton.cdiv(C_out, BLOCK_N)),
                )
                matmul_tn_acc_kernel[grid_dw](
                    col_bw._ptr,
                    d_y_flat._ptr,
                    d_weight_temp_ptr,
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
                from .kernels.elemwise_kernels import add_inplace_kernel
                from .kernels.matmul_kernels import transpose2d_kernel
                d_weight_T_ptr = cuda_malloc(C_out * num_cols * 4)
                transpose2d_kernel[
                    (triton.cdiv(num_cols * C_out, 256),)
                ](
                    d_weight_temp_ptr,
                    d_weight_T_ptr,
                    rows=num_cols,
                    cols=C_out,
                    BLOCK=256,
                )
                cuda_free(d_weight_temp_ptr)
                add_inplace_kernel[
                    (triton.cdiv(C_out * num_cols, 256),)
                ](
                    w_tensor.grad._ptr,
                    d_weight_T_ptr,
                    C_out * num_cols,
                    BLOCK=256,
                )
                cuda_free(d_weight_T_ptr)
            # d_col = d_y_flat @ w_flat
            d_col_ptr = cuda_malloc(num_rows * num_cols * 4)
            d_col = Tensor._from_ptr(
                d_col_ptr, (num_rows, num_cols), owns_memory=True
            )
            grid_dcol = (
                max(1, triton.cdiv(num_rows, BLOCK_M)),
                max(1, triton.cdiv(num_cols, BLOCK_N)),
            )
            matmul_kernel[grid_dcol](
                d_y_flat._ptr,
                w_flat_bw._ptr,
                d_col_ptr,
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
                if x_tensor.grad is None:
                    x_tensor.grad = zeros_like(x_tensor)
                grid_col2im = (
                    max(1, num_rows),
                    max(1, triton.cdiv(num_cols, IM2COL_BLOCK)),
                )
                col2im_kernel[grid_col2im](
                    d_col_ptr,
                    x_tensor.grad._ptr,
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
            cuda_free(d_col_ptr)

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
        import triton

        from .cuda_mem import cuda_malloc, cuda_free, cuda_memset
        from .functional import zeros_like
        from .kernels.pool_kernels import (
            maxpool2d_backward_kernel,
            maxpool2d_forward_kernel,
        )

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
        out_ptr = cuda_malloc(numel_out * 4)
        idx_ptr = cuda_malloc(numel_out * 4)
        out = Tensor._from_ptr(out_ptr, out_shape, owns_memory=True)

        BLOCK_ELEMS = 256
        grid_size = triton.cdiv(numel_out, BLOCK_ELEMS)
        maxpool2d_forward_kernel[(grid_size,)](
            x._ptr,
            out_ptr,
            idx_ptr,
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
                if x_tensor.grad is None:
                    x_tensor.grad = zeros_like(x_tensor)
                BLOCK_ELEMS_BW = 256
                grid_size_bw = triton.cdiv(numel_out, BLOCK_ELEMS_BW)
                maxpool2d_backward_kernel[(grid_size_bw,)](
                    out_grad._ptr,
                    idx_ptr,
                    x_tensor.grad._ptr,
                    N=N,
                    C=C,
                    H=H,
                    W=W,
                    H_out=H_out,
                    W_out=W_out,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    BLOCK_KW=kW,
                    BLOCK_ELEMS=BLOCK_ELEMS_BW,
                )
            cuda_free(idx_ptr)

        out._set_graph(parents=(self,), backward_fn=_backward)
        if not out.requires_grad:
            cuda_free(idx_ptr)
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

        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc
        from .functional import zeros_like
        from .kernels.elemwise_kernels import add_inplace_kernel
        from .kernels.reduce_kernels import add_bias_kernel, sum_axis0_kernel

        rows, cols = self._shape

        # Allocate output
        out_ptr = cuda_malloc(self._nbytes)
        out = Tensor._from_ptr(out_ptr, self._shape, owns_memory=True)

        # Launch forward kernel
        grid = lambda meta: (triton.cdiv(self._numel, meta["BLOCK"]),)
        add_bias_kernel[grid](self._ptr, bias._ptr, out._ptr, rows, cols, BLOCK=256)

        # Capture for backward
        input_tensor = self
        bias_tensor = bias

        def _backward(out_grad: Tensor) -> None:
            # dX += out_grad (elementwise)
            if input_tensor.requires_grad:
                if input_tensor.grad is None:
                    input_tensor.grad = zeros_like(input_tensor)
                grid = lambda meta: (triton.cdiv(input_tensor._numel, meta["BLOCK"]),)
                add_inplace_kernel[grid](
                    input_tensor.grad._ptr,
                    out_grad._ptr,
                    input_tensor._numel,
                    BLOCK=256,
                )

            # db += sum over axis 0 of out_grad
            if bias_tensor.requires_grad:
                if bias_tensor.grad is None:
                    bias_tensor.grad = zeros_like(bias_tensor)
                # Each column gets summed independently
                # Grid size = number of columns
                sum_axis0_kernel[(cols,)](
                    out_grad._ptr,
                    bias_tensor.grad._ptr,
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
        # Validate shapes match
        if self._shape != other._shape:
            raise ValueError(f"Shape mismatch for mul: {self._shape} vs {other._shape}")

        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc
        from .functional import zeros_like
        from .kernels.elemwise_kernels import mul_backward_kernel, mul_kernel

        # Allocate output
        out_ptr = cuda_malloc(self._nbytes)
        out = Tensor._from_ptr(out_ptr, self._shape, owns_memory=True)

        # Launch kernel
        grid = lambda meta: (triton.cdiv(self._numel, meta["BLOCK"]),)
        mul_kernel[grid](self._ptr, other._ptr, out._ptr, self._numel, BLOCK=256)

        # Capture for backward
        a_tensor = self
        b_tensor = other

        def _backward(out_grad: Tensor) -> None:
            # dA += out_grad * B, dB += out_grad * A
            if a_tensor.requires_grad:
                if a_tensor.grad is None:
                    a_tensor.grad = zeros_like(a_tensor)
                grid = lambda meta: (triton.cdiv(a_tensor._numel, meta["BLOCK"]),)
                mul_backward_kernel[grid](
                    a_tensor.grad._ptr,
                    out_grad._ptr,
                    b_tensor._ptr,
                    a_tensor._numel,
                    BLOCK=256,
                )

            if b_tensor.requires_grad:
                if b_tensor.grad is None:
                    b_tensor.grad = zeros_like(b_tensor)
                grid = lambda meta: (triton.cdiv(b_tensor._numel, meta["BLOCK"]),)
                mul_backward_kernel[grid](
                    b_tensor.grad._ptr,
                    out_grad._ptr,
                    a_tensor._ptr,
                    b_tensor._numel,
                    BLOCK=256,
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
        # Import here to avoid circular imports
        import triton

        from .cuda_mem import cuda_malloc
        from .functional import zeros_like
        from .kernels.elemwise_kernels import mul_scalar_kernel, scale_backward_kernel

        # Allocate output
        out_ptr = cuda_malloc(self._nbytes)
        out = Tensor._from_ptr(out_ptr, self._shape, owns_memory=True)

        # Launch kernel
        grid = lambda meta: (triton.cdiv(self._numel, meta["BLOCK"]),)
        mul_scalar_kernel[grid](self._ptr, scalar, out._ptr, self._numel, BLOCK=256)

        # Capture for backward
        input_tensor = self
        scale_val = scalar

        def _backward(out_grad: Tensor) -> None:
            # dA += out_grad * scalar
            if input_tensor.requires_grad:
                if input_tensor.grad is None:
                    input_tensor.grad = zeros_like(input_tensor)
                grid = lambda meta: (triton.cdiv(input_tensor._numel, meta["BLOCK"]),)
                scale_backward_kernel[grid](
                    input_tensor.grad._ptr,
                    out_grad._ptr,
                    scale_val,
                    input_tensor._numel,
                    BLOCK=256,
                )

        out._set_graph(parents=(self,), backward_fn=_backward)
        return out

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
        host_bytes = cuda_memcpy_dtoh(self._ptr, self._nbytes)

        # Unpack to floats
        flat_data = list(struct.unpack(f"{self._numel}f", host_bytes))

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
        if self._numel != 1:
            raise ValueError(
                f"item() only works for single-element tensors, got {self._numel} elements"
            )
        return self.to_list()[0]

    def __repr__(self) -> str:
        """Return string representation of the tensor."""
        name_str = f", name='{self.name}'" if self.name else ""
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self._shape}, ptr=0x{self._ptr:x}{grad_str}{name_str})"

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
