"""
Tensor class with from-scratch GPU storage and autograd support.

This module implements a minimal Tensor type that:
- Holds GPU storage as raw pointers managed via ctypes + CUDA runtime
- Tracks a computation graph and supports backward()
- Enforces invariants: CUDA only, float32 only, contiguous only, rank in {1, 2}
"""

from __future__ import annotations

import struct
from typing import Callable, Optional

from .cuda_mem import cuda_malloc, cuda_free, cuda_memcpy_htod, cuda_memcpy_dtoh


class Tensor:
    """
    A tensor that lives on GPU with autograd support.

    All data is stored on CUDA device as float32 in contiguous row-major layout.
    Only ranks 1 and 2 are supported.
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

        # Validate shape (rank must be 1 or 2)
        if len(shape) not in (1, 2):
            raise ValueError(
                f"Only rank 1 or 2 tensors supported, got rank {len(shape)}"
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
        if len(shape) not in (1, 2):
            raise ValueError(
                f"Only rank 1 or 2 tensors supported, got rank {len(shape)}"
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

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

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
            raise ValueError(
                f"Shape mismatch for add: {self._shape} vs {other._shape}"
            )

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

        # Should not reach here given rank constraint, but handle gracefully
        raise ValueError(f"Unsupported shape rank: {len(shape)}")
