# Gradtuity: From-Scratch Tensor Autodiff Engine (Python + Triton)

## 0) Summary

Build a minimal, micrograd-style autodiff engine where each node is a Tensor (not a scalar), and all forward/backward math happens via Triton kernels on CUDA. The demo trains a tiny MLP:

```text
y = relu(XW + b),   L = sum(y)
```

**Constraints:** contiguous tensors only, ranks {1,2} only, dtype float32 only.

**From-scratch philosophy:**
- **No PyTorch, NumPy, or external tensor libraries.** The `Tensor` class is written entirely from scratch.
- GPU memory management uses `ctypes` to call CUDA runtime (`libcudart`) directly.
- All GPU compute (forward, backward, optimizer) happens via Triton kernels.
- Only Python standard library (`ctypes`, `struct`) + Triton are used.


## 1) Goals

Implement a Tensor type **from scratch** that:
- Holds GPU storage as raw pointers (integers) managed via `ctypes` + CUDA runtime
- Tracks a computation graph and supports `backward()`
- Supports a minimal op set sufficient to train the toy MLP:
  - matmul (2D)
  - add_bias (broadcast a 1D bias over batch)
  - relu
  - sum (reduce all elements to scalar shape (1,))
- Computes gradients via VJP rules implemented by us
- Uses Triton kernels for forward and backward paths (including reduction kernels)
- Runs a tiny SGD training loop

### Non-goals (toy demo)

- Strides/views, slicing, general broadcasting beyond bias-add
- Mixed precision (fp16/bf16), non-fp32
- Higher-order gradients
- Fancy losses (softmax/xent), layernorm, convs
- Performance parity with cuBLAS/PyTorch
- Using external tensor libraries (PyTorch, NumPy, CuPy)

## 2) Dependencies & Environment

**External dependencies:**
- Python 3.10+
- `triton` (kernel authoring + compilation)
- CUDA Toolkit (provides `libcudart.so` for memory management)
- CUDA GPU required

**Python standard library (no pip install needed):**
- `ctypes` — interface to CUDA runtime for GPU memory allocation
- `struct` — pack/unpack Python floats to bytes for GPU transfer
- `random` — generate random numbers for initialization

**NOT used:**
- No PyTorch
- No NumPy
- No CuPy or other tensor libraries

**Data type:** float32 only (4 bytes per element)


## 3) User-Facing API (MVP)

### Tensor

```python
Tensor(data: list | tuple, shape: tuple[int, ...] = None, requires_grad: bool = False, name: str = "")
```

**Construction from Python data:**
- Accepts nested lists/tuples of floats
- Flattens data, packs to bytes via `struct.pack`, copies to GPU via `cudaMemcpy`
- Shape is inferred from nested structure or explicitly provided

**Example:**
```python
# 2D tensor (2, 3)
x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

# 1D tensor (3,)
b = Tensor([0.1, 0.2, 0.3], requires_grad=True)
```

**Enforce invariants:**
- CUDA only (all data lives on GPU)
- float32 only (4 bytes per element)
- contiguous only (no strides/views)
- rank in {1, 2} only

### Properties (from-scratch storage model)

- **_ptr:** `int` — raw GPU pointer from `cudaMalloc`
- **_shape:** `tuple[int, ...]` — shape, e.g., `(32, 64)`
- **_numel:** `int` — total number of elements
- **_nbytes:** `int` — total bytes (`_numel * 4` for float32)
- **grad:** `Optional[Tensor]` — gradient tensor (same class), allocated lazily
- **requires_grad:** `bool`
- **graph fields:** `_parents`, `_backward`, `_ctx` (see Node model)

### Memory management

```python
def data_ptr(self) -> int:
    """Return raw GPU pointer for Triton kernels."""
    return self._ptr

def __del__(self):
    """Free GPU memory when Tensor is garbage collected."""
    if self._ptr is not None:
        cuda_free(self._ptr)
```

### Additional API (debugging / ergonomics)

**detach() / stop_grad():**
- Returns a new Tensor that shares the same data pointer but has `requires_grad=False` and no graph fields.
- Important: shared pointer means the new Tensor should NOT free memory in `__del__` (use reference counting or ownership flag).

**to_list():**
- Copies data from GPU to CPU and returns as nested Python list.
- Uses `cudaMemcpy` (device-to-host) + `struct.unpack`.

### Ops

- `C = A.matmul(W)` — (B,D) @ (D,H) -> (B,H)
- `Y = X.add_bias(b)` — (B,H) + (H,) -> (B,H)
- `C = A.add(B)` — elementwise add, same shape -> same shape
- `Z = Y.relu()` — (B,H)
- `loss = Z.sum()` — scalar Tensor with shape (1,)

### Conditional graph construction (prevents unnecessary retention)

Each op wrapper must set:
- `out.requires_grad = any(p.requires_grad for p in parents)`
- If `out.requires_grad == False`: do not set `_parents` / `_backward` / `_ctx` (leave empty/None). Still run the forward kernel and return the Tensor.

### Backward + grad management

**loss.backward():**
- Requires `loss._numel == 1` (scalar) and shape is `(1,)`
- Seeds `loss.grad` via Triton `fill_kernel` to set all elements to 1.0
- Performs topo sort and reverse traversal
- Calls each node's `_backward(out_grad)` and accumulates into parent grads via Triton kernels

Gradients accumulate by default, so training loops must call:
- `zero_grad(params)` — uses Triton `fill_kernel` to set `p.grad` to 0 (allocates if None)

### Optimizer (toy)

- `sgd_step(params, lr)`: updates `p._ptr` in-place: `p -= lr * p.grad`
- Implementation uses Triton `sgd_update_kernel` to remain consistent with "all math via Triton".


## 4) Data Model & Invariants

### Tensor storage (from scratch)

Each `Tensor` stores:
- `_ptr: int` — raw GPU pointer from `cudaMalloc`
- `_shape: tuple[int, ...]` — e.g., `(32, 64)`
- `_numel: int` — total elements (product of shape)
- `_nbytes: int` — total bytes (`_numel * 4`)

All tensors are:
- CUDA only (GPU memory)
- float32 only (4 bytes per element)
- contiguous only (row-major, no strides)

### Shapes allowed

- 2D: (B, D) and (D, H) for matmul
- 2D: (B, H) for intermediate activations
- 1D: (H,) for bias
- scalar loss: represented as shape (1,)

### Gradient storage

- `Tensor.grad` is `None` until first needed
- On first backward accumulation (and only if `requires_grad=True`):
  - If `Tensor.grad is None`: allocate via `zeros_like(Tensor)` (uses `cudaMalloc` + `cudaMemset`)
  - Accumulate via Triton `add_inplace_kernel`

### Backward allocation rule (important)

Inside each op's `_backward(out_grad)`:
- Only allocate/accumulate into a parent if `parent.requires_grad` is True
- This avoids allocating grads for constants or non-trainable leaves.

Explicit pattern for backward accumulation:

```python
# Inside each op's _backward(out_grad):
for parent in parents:
    if parent.requires_grad:
        if parent.grad is None:
            parent.grad = zeros_like(parent)  # cudaMalloc + cudaMemset
        # Accumulate via Triton kernel (not Python +=)
        add_inplace_kernel[grid](parent.grad._ptr, contribution._ptr, parent._numel, BLOCK=256)
```

### In-place ops

Not supported; every op produces a new output buffer (new `cudaMalloc` call).

### Memory lifecycle

- `cudaMalloc` when Tensor is created
- `cudaFree` when Tensor is garbage collected (`__del__`)
- Shared pointers (from `detach()`) must use ownership tracking to avoid double-free

## 5) Autograd Graph & Backward Engine

### Node model

Each op produces `out` with:
- **(out.requires_grad == True only):** `out._parents = (a, b, ...)`, `out._backward = fn(out_grad)` implemented by the op (must accumulate via Triton), `out._ctx` containing minimal info for backward
- **(out.requires_grad == False):** no graph fields retained

### _backward closure pattern

The `_backward` callable is a closure that captures the necessary context (parent tensors, shapes, or forward activations) at op creation time. The `_ctx` field is optional and may be used if storing context separately is preferred. For this toy implementation, prefer closures:

**Example (relu):**

```python
def make_relu_backward(Y, out):
    def _backward(out_grad):
        if Y.requires_grad:
            if Y.grad is None:
                Y.grad = zeros_like(Y)  # cudaMalloc + cudaMemset
            # mask = Y > 0, accumulate: Y.grad += out_grad * mask
            grid = lambda meta: (triton.cdiv(Y._numel, meta['BLOCK']),)
            relu_backward_kernel[grid](
                Y.grad._ptr, out_grad._ptr, Y._ptr, Y._numel, BLOCK=256
            )
    return _backward

out._backward = make_relu_backward(Y, out)
```

### Topological order (micrograd style)

**Must-do #3:** Visited-set topo sort (prevents double-counting grads on shared subgraphs). Topo construction must use a visited set keyed by object identity so reused nodes in a DAG are processed once.

Use `id(v)` rather than `v` in visited:

```python
visited = set()  # of ints
topo = []

def build(v):
    if id(v) in visited:
        return
    visited.add(id(v))
    for p in getattr(v, '_parents', ()):
        build(p)
    topo.append(v)
```

### Backward semantics (explicit)

**loss.backward():**
- Validate scalar: `loss._numel == 1` (and expected shape `(1,)`)
- Seed `loss.grad = ones_like(loss)` via Triton `fill_kernel` (sets to 1.0)
- `topo = topo_sort(loss)` using visited-set algorithm above
- Traverse nodes in reverse topo: let `out_grad = node.grad`; if `node._backward is not None`, call `node._backward(out_grad)` which must accumulate into parent grads via Triton kernels (leaf nodes have no `_backward`)
- No gradient graph; no higher-order grads

### Graph freeing (memory safety)

Optionally clear non-leaf graph references after processing each node:
- `node._parents = ()`
- `node._backward = None`
- `node._ctx = None`

Leaves (`requires_grad=True` with no parents) retain `_ptr` and `grad`.

**Important:** Graph freeing does NOT call `cudaFree` — that only happens when the Tensor object itself is garbage collected.


## 6) Kernel Backend Architecture

### High-level dataflow

Python op wrapper:
- validates inputs (shape rules, rank constraints)
- allocates output buffer via `cuda_malloc` + optionally `cuda_memset` for zero-init
- launches Triton kernel for forward (passes raw `_ptr` integers)
- stores minimal context for backward (parent Tensors, shapes)

### Backward write pattern (recommended)

Backward kernels should accumulate directly into destination grads: `kernel(..., dst_grad_ptr)` does `dst_grad += contribution`. Avoid allocating intermediate "contrib buffers" unless necessary for toy simplicity.

### Reduction correctness (important)

**Must-do #2:** Forward reductions must start from zero. Any kernel that reduces into a smaller output (or many writers to one output element) must:
- use `tl.atomic_add` (MVP), or
- use a 2-pass reduction

Additionally, if using atomics, the output buffer must be initialized to zero before the reduction:

```python
# Allocate zero-initialized buffer for reduction output
out_ptr = cuda_malloc(4)  # 4 bytes for one float32
cuda_memset(out_ptr, 0, 4)  # set to zero
```

This applies to forward reductions: **sum_all** producing (1,) — output buffer MUST be zero-initialized before the kernel runs.

**Must-do #2 clarification (forward vs backward):**
- **Forward reductions** (e.g., sum_all): output buffer MUST be zero-initialized before the kernel runs, since the kernel accumulates into it.
- **Backward reductions** (e.g., sum_axis0 for bias grad): the destination is a `.grad` tensor that should already be zeroed by `zero_grad()` at the start of each training iteration. The backward kernel simply accumulates.

**Design invariant:** Always call `zero_grad(params)` before backward (after forward, as shown in the demo). This ensures all `.grad` tensors are zeroed, and backward kernels can safely use `+=` accumulation without additional zero-init logic.

### Reduction determinism note

Atomic reductions are numerically correct but not bitwise deterministic due to accumulation order. Tests must use tolerances and may repeat runs to catch race-condition bugs.

### Kernel compilation + caching

- Use Triton's internal compilation caching
- Optional light Python cache keyed by (op_name, block_sizes, dtype=fp32, shape_class) if needed

### Training-loop support kernels (recommended)

To keep training loop "math" in Triton:
- `fill_kernel(dst_ptr, value, numel)` — for zero_grad and seeding loss.grad
- `sgd_update_kernel(param_ptr, grad_ptr, lr, numel)` — implementing `param -= lr * grad`


## 7) Ops: Forward + Backward Specifications

### 7.1 Matmul: C = A @ W

**Inputs:**
- A: (B, D) contiguous fp32 (where B is batch size)
- W: (D, H) contiguous fp32 (weight matrix)

**Output:** C: (B, H) contiguous fp32

**Forward kernel:** Blocked matmul using `tl.dot`. Accumulate in fp32.

**Backward:** Given dC (B, H): dA += dC @ W^T -> (B, D); dW += A^T @ dC -> (D, H).

**Backward implementation (toy-simplest):**

1. Compute transposed inputs (via transpose2d Triton kernel): `Wt = transpose2d(W)`, `At = transpose2d(A)`
2. Compute gradient contributions into temp buffers: `dA_contrib = matmul(dC, Wt)`, `dW_contrib = matmul(At, dC)`
3. Accumulate into parent grads (only if parent.requires_grad): `A.grad += dA_contrib`, `W.grad += dW_contrib` (via add_inplace_2d kernel)

**Notes:**
This allocates temporary buffers for dA_contrib and dB_contrib.
Acceptable for toy; a fused matmul_accumulate kernel would avoid this overhead.
Transpose copies are expensive but acceptable for toy demo.

### 7.2 Bias Add: Y = X + b

**Inputs:** X: (B, H) contiguous fp32; b: (H,) contiguous fp32  
**Output:** Y: (B, H) contiguous fp32

**Forward kernel:** `Y[i, j] = X[i, j] + b[j]`

**Backward:** Given dY (B, H): dX += dY; db += sum_{i=0..B-1} dY[i, j]

**Backward kernels:**
- add_inplace_2d: X.grad += dY (elementwise accumulation)
- reduce_sum_axis0: b.grad += sum over axis 0 of dY (uses `tl.atomic_add` for correctness; assumes b.grad pre-zeroed by zero_grad)

### 7.3 ReLU: Z = relu(Y)

**Inputs:** Y: (B, H)  
**Output:** Z: (B, H)

**Forward kernel:** Z = max(Y, 0)

**Backward:** Given dZ: dY += dZ * (Y > 0)

**Context:** Store pointer to Y (not Z) to recompute mask in backward. MVP: recompute (Y > 0) in backward (no stored bitmask). Note: Z cannot be used because Z >= 0 always, so `Z > 0` is not equivalent to `Y > 0`.

**Backward kernel:** elementwise multiply + accumulate into dY

### 7.4 Sum reduction: loss = sum(Z)

**Input:** Z: (B, H)  
**Output:** loss: (1,) fp32

**Forward kernel:** Reduce all elements of Z into one scalar. MVP correctness: use `tl.atomic_add(loss[0], partial)`. **Must-do #2:** output must start at 0:

```python
# Allocate zero-initialized scalar buffer
loss_ptr = cuda_malloc(4)      # 4 bytes for one float32
cuda_memset(loss_ptr, 0, 4)    # MUST be zero before atomic adds
```

**Backward:** Given dloss scalar (1,): dZ += dloss broadcasted to all elements.

**Backward kernel:** dZ += scalar (accumulating fill / add-scalar kernel)

### 7.5 Elementwise Add: C = A + B

**Inputs:** A: (*, H) contiguous fp32; B: (*, H) contiguous fp32 (same shape as A)  
**Output:** C: (*, H) contiguous fp32

**Forward kernel:** `C[i] = A[i] + B[i]` (elementwise)

**Backward:** Given dC: dA += dC; dB += dC

**Backward kernels:**
- add_inplace: A.grad += dC (elementwise accumulation)
- add_inplace: B.grad += dC (elementwise accumulation)

**Note:** This is a simple elementwise add with no broadcasting (both inputs must have identical shapes).


## 8) Dispatcher & Shape Handling

**MVP assumes:**
- all tensors are contiguous
- only ranks 1 and 2
- fixed semantics per op (no general broadcasting)
- only CUDA + fp32

**Validation checks in Python wrappers:** CUDA required; dtype fp32 required; contiguous required; shape compatibility required; scalar loss must have numel == 1 and represent as (1,).

## 9) Project Layout (suggested)

```text
gradtuity/
  cuda_mem.py        # ctypes CUDA memory interface (malloc, free, memcpy, memset)
  tensor.py          # Tensor class + topo backward + zero_grad + op wrappers
  functional.py      # zeros, ones, zeros_like, ones_like, randn helpers
  ops.py             # Python-level op implementations + ctx structs
  kernels/
    matmul.py        # triton matmul + transpose2d + matmul backward helpers
    elemwise.py      # add_bias, relu, relu_backward, add_inplace, add_scalar_inplace
    reduce.py        # sum_all (atomic/2-pass), sum_axis0 (atomic/2-pass)
    optim_kernels.py # fill_kernel, sgd_update_kernel
  optim.py           # sgd_step + zero_grad helpers (calls kernels)
  tests/
    test_ops.py      # forward/backward correctness vs manual calculations
    test_graph.py    # end-to-end MLP graph gradient test
  demo_train.py      # trains toy MLP
```

### cuda_mem.py (core from-scratch module)

```python
import ctypes

_libcudart = ctypes.CDLL("libcudart.so")

# cudaMemcpyKind enum
MEMCPY_H2D = 1  # Host to Device
MEMCPY_D2H = 2  # Device to Host
MEMCPY_D2D = 3  # Device to Device

def cuda_malloc(nbytes: int) -> int:
    """Allocate GPU memory, return pointer as int."""
    ptr = ctypes.c_void_p()
    status = _libcudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if status != 0:
        raise RuntimeError(f"cudaMalloc failed: {status}")
    return ptr.value

def cuda_free(ptr: int) -> None:
    """Free GPU memory."""
    _libcudart.cudaFree(ctypes.c_void_p(ptr))

def cuda_memset(ptr: int, value: int, nbytes: int) -> None:
    """Set nbytes at ptr to value (byte value, usually 0)."""
    _libcudart.cudaMemset(ctypes.c_void_p(ptr), ctypes.c_int(value), ctypes.c_size_t(nbytes))

def cuda_memcpy_htod(dst: int, src_bytes: bytes) -> None:
    """Copy bytes from host to device."""
    nbytes = len(src_bytes)
    src_buf = (ctypes.c_char * nbytes).from_buffer_copy(src_bytes)
    _libcudart.cudaMemcpy(ctypes.c_void_p(dst), src_buf, ctypes.c_size_t(nbytes), MEMCPY_H2D)

def cuda_memcpy_dtoh(src: int, nbytes: int) -> bytes:
    """Copy nbytes from device to host, return bytes."""
    dst_buf = (ctypes.c_char * nbytes)()
    _libcudart.cudaMemcpy(dst_buf, ctypes.c_void_p(src), ctypes.c_size_t(nbytes), MEMCPY_D2H)
    return bytes(dst_buf)

def cuda_memcpy_dtod(dst: int, src: int, nbytes: int) -> None:
    """Copy nbytes from device to device."""
    _libcudart.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(nbytes), MEMCPY_D2D)
```

### functional.py (tensor factory functions)

```python
import struct
import random
from .cuda_mem import cuda_malloc, cuda_memset, cuda_memcpy_htod
from .tensor import Tensor

def zeros(shape: tuple[int, ...]) -> Tensor:
    """Allocate zero-initialized GPU tensor."""
    numel = 1
    for s in shape:
        numel *= s
    nbytes = numel * 4
    ptr = cuda_malloc(nbytes)
    cuda_memset(ptr, 0, nbytes)
    return Tensor._from_ptr(ptr, shape, owns_memory=True)

def zeros_like(t: Tensor) -> Tensor:
    return zeros(t._shape)

def ones(shape: tuple[int, ...]) -> Tensor:
    """Allocate tensor filled with 1.0 via Triton fill kernel."""
    t = zeros(shape)
    from .kernels.optim_kernels import fill_kernel
    import triton
    grid = lambda meta: (triton.cdiv(t._numel, meta['BLOCK']),)
    fill_kernel[grid](t._ptr, 1.0, t._numel, BLOCK=256)
    return t

def ones_like(t: Tensor) -> Tensor:
    return ones(t._shape)

def randn(shape: tuple[int, ...], seed: int = None) -> Tensor:
    """Generate random normal tensor using Python's random module."""
    if seed is not None:
        random.seed(seed)
    numel = 1
    for s in shape:
        numel *= s
    data = [random.gauss(0, 1) for _ in range(numel)]
    host_bytes = struct.pack(f'{numel}f', *data)
    ptr = cuda_malloc(numel * 4)
    cuda_memcpy_htod(ptr, host_bytes)
    return Tensor._from_ptr(ptr, shape, owns_memory=True)
```

## 10) Correctness Strategy

### Golden reference

**Primary approach:** Hand-calculate expected values for small inputs. For example:
- matmul: manually compute 2x2 @ 2x2 results
- relu: verify positive values pass through, negatives become 0
- sum: verify total matches Python `sum()` of input list

**Optional (tests only):** If desired, tests may import PyTorch or NumPy **only in test files** as a reference for comparison. The main `gradtuity/` code must remain dependency-free.

### Recommended tests per op

- **Forward vs manual calculation:** max_abs_err < 1e-4 (fp32)
- **Backward vs manual gradients:** derive gradients by hand for small inputs, compare

### Reduction-specific tests

Reductions are most error-prone; test multiple sizes and ensure stability; repeat the same reduction multiple times to catch race-condition bugs; allow small tolerance due to atomic nondeterminism.

### End-to-end graph test (important)

Test composed pipeline: `loss = relu(X@W + b).sum()`. Compute expected dW, db by hand (or via optional torch reference in test file only) for multiple shapes, including non-multiples of tile sizes.

### High-value graph gotcha tests

1. **Shared subgraph reuse:** `y = x.add(x); loss = y.sum()` — expect `x.grad == 2 * ones_like(x)`
2. **Branching then merge:** `a = x.relu(); b = x.relu(); loss = a.add(b).sum()` — expect `x.grad == 2 * (x > 0)`

## 11) Demo Program (acceptance)

`demo_train.py` should:
- Initialize random X, W, b using `randn()` (Python `random` module + GPU transfer)
- **Forward:** `Z = relu(X.matmul(W).add_bias(b))`, `loss = Z.sum()`
- **Backward:** `zero_grad([W, b])`, `loss.backward()`
- **Update:** `sgd_step([W, b], lr)` via Triton `sgd_update_kernel`
- Print loss decreasing over iterations (trend, not strictly monotonic)

**Example:**

```python
from gradtuity import Tensor, randn, zero_grad, sgd_step

# Initialize (all from scratch, no external libraries)
X = randn((32, 64))                    # input: batch=32, features=64
W = randn((64, 16), requires_grad=True)  # weights
b = randn((16,), requires_grad=True)     # bias

for i in range(100):
    # Forward
    Z = X.matmul(W).add_bias(b).relu()
    loss = Z.sum()
    
    # Backward
    zero_grad([W, b])
    loss.backward()
    
    # Update
    sgd_step([W, b], lr=0.001)
    
    if i % 10 == 0:
        print(f"iter {i}: loss = {loss.to_list()[0]:.4f}")
```

## 12) Performance Expectations (toy)

- Correctness first; performance "reasonable"
- Transpose copies in matmul backward are expensive but acceptable
- Block sizes can be hardcoded initially (e.g., 16×16) and tuned later
- Atomics in reductions may be slower but acceptable for toy correctness

## 13) Risks & Mitigations

**Risk: silent gradient bugs (common)**  
Mitigation: enforce scalar-only backward; consistent accumulation via Triton kernels everywhere; visited-set topo sort to avoid double backward on shared subgraphs; end-to-end gradient tests; shared-subgraph and branching-merge gotcha tests.

**Risk: reduction race conditions (most common!)**  
Mitigation: reductions must use atomics or 2-pass; forward reduction outputs must start at 0 (via `cuda_memset`); dedicated tests for sum and bias grad; repeat reduction tests; tolerate small fp32 differences due to atomic ordering.

**Risk: GPU memory leaks**  
Mitigation: implement `__del__` on Tensor to call `cuda_free`; use ownership flag (`_owns_memory`) for shared pointers from `detach()`; consider explicit `del` or context managers for long-running code; test with small inputs to catch leaks early.

**Risk: double-free on shared pointers**  
Mitigation: when `detach()` creates a Tensor sharing the same `_ptr`, mark it with `_owns_memory=False` so its `__del__` does not call `cuda_free`. Only the original owner frees memory.

**Risk: memory blowup from graph retention**  
Mitigation: conditional graph construction (only attach parents/backward/ctx when out.requires_grad is True); optional graph freeing after backward traversal.

**Risk: ctypes/CUDA errors**  
Mitigation: check return status of all CUDA calls; wrap in try/except; provide clear error messages with status codes.

## 14) Implementation Milestones

1. **cuda_mem.py** — ctypes interface to CUDA runtime (malloc, free, memcpy, memset)
2. **Tensor class** — from-scratch storage (`_ptr`, `_shape`, `_numel`), `__del__` for memory cleanup, `data_ptr()` method
3. **functional.py** — `zeros`, `zeros_like`, `ones`, `ones_like`, `randn` using cuda_mem + Triton fill
4. **Topo sort backward** — visited set keyed by `id()`, scalar backward check
5. **Conditional graph construction** + `detach()` with ownership tracking
6. **Elementwise kernels** (add, relu, relu_backward) with accumulating grads via Triton
7. **Reduction kernels** (sum_all, sum_axis0) with atomic correctness (zero-init via `cuda_memset`) + repeat-tests
8. **Matmul forward** — blocked Triton matmul
9. **Transpose2d + matmul backward** — reuse matmul for gradient computation
10. **fill_kernel + sgd_update_kernel** — keep all training-loop math in Triton
11. **Full MLP demo** + per-op + end-to-end tests
