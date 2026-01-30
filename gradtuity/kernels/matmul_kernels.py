"""
Triton kernels for matrix multiplication operations.

Includes:
- matmul_kernel: Blocked matrix multiplication C = A @ B
- matmul_bias_kernel: Fused matmul with bias C = A @ B + bias
- matmul_bias_relu_kernel: Fused matmul with bias and ReLU C = relu(A @ B + bias)
- matmul_nt_acc_kernel: C += A @ B^T (B transposed, accumulating)
- matmul_tn_acc_kernel: C += A^T @ B (A transposed, accumulating)
- transpose2d_kernel: Transpose a 2D matrix (legacy, for reference)
"""

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32,
    stride_ak: tl.int32,
    stride_bk: tl.int32,
    stride_bn: tl.int32,
    stride_cm: tl.int32,
    stride_cn: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Blocked matrix multiplication: C = A @ B

    Computes C[M, N] = A[M, K] @ B[K, N]

    Args:
        a_ptr: Input matrix A GPU pointer (float32*), shape (M, K).
        b_ptr: Input matrix B GPU pointer (float32*), shape (K, N).
        c_ptr: Output matrix C GPU pointer (float32*), shape (M, N).
        M: Number of rows in A and C.
        N: Number of columns in B and C.
        K: Shared dimension (columns of A, rows of B).
        stride_am: Stride for A's M dimension (typically K for row-major).
        stride_ak: Stride for A's K dimension (typically 1 for row-major).
        stride_bk: Stride for B's K dimension (typically N for row-major).
        stride_bn: Stride for B's N dimension (typically 1 for row-major).
        stride_cm: Stride for C's M dimension (typically N for row-major).
        stride_cn: Stride for C's N dimension (typically 1 for row-major).
        BLOCK_M: Block size for M dimension (compile-time constant).
        BLOCK_N: Block size for N dimension (compile-time constant).
        BLOCK_K: Block size for K dimension (compile-time constant).
    """
    # Program ID determines which block of C we're computing
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting row and column for this block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator for this block of C
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        # Load block of A: shape (BLOCK_M, BLOCK_K)
        # A[rm, rk] at indices rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_offs = rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)

        # Load block of B: shape (BLOCK_K, BLOCK_N)
        # B[rk, rn] at indices rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_offs = rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        # Accumulate: acc += A @ B for this K-block
        acc += tl.dot(a, b)

    # Store the result block to C
    c_offs = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + c_offs, acc, mask=c_mask)


@triton.jit
def matmul_bias_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    bias_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32,
    stride_ak: tl.int32,
    stride_bk: tl.int32,
    stride_bn: tl.int32,
    stride_cm: tl.int32,
    stride_cn: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused matrix multiplication with bias: C = A @ B + bias

    Computes C[M, N] = A[M, K] @ B[K, N] + bias[N]
    where bias is broadcast across rows.

    This is more efficient than separate matmul + add_bias operations.

    Args:
        a_ptr: Input matrix A GPU pointer (float32*), shape (M, K).
        b_ptr: Input matrix B GPU pointer (float32*), shape (K, N).
        bias_ptr: Bias vector GPU pointer (float32*), shape (N,).
        c_ptr: Output matrix C GPU pointer (float32*), shape (M, N).
        M: Number of rows in A and C.
        N: Number of columns in B and C (and size of bias).
        K: Shared dimension (columns of A, rows of B).
        stride_am: Stride for A's M dimension.
        stride_ak: Stride for A's K dimension.
        stride_bk: Stride for B's K dimension.
        stride_bn: Stride for B's N dimension.
        stride_cm: Stride for C's M dimension.
        stride_cn: Stride for C's N dimension.
        BLOCK_M: Block size for M dimension (compile-time constant).
        BLOCK_N: Block size for N dimension (compile-time constant).
        BLOCK_K: Block size for K dimension (compile-time constant).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        a_offs = rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)

        b_offs = rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Load bias and broadcast add
    bias_mask = rn < N
    bias = tl.load(bias_ptr + rn, mask=bias_mask, other=0.0)
    # Broadcast bias across rows: acc[m, n] += bias[n]
    acc = acc + bias[None, :]

    # Store the result block to C
    c_offs = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + c_offs, acc, mask=c_mask)


@triton.jit
def matmul_bias_relu_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    bias_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32,
    stride_ak: tl.int32,
    stride_bk: tl.int32,
    stride_bn: tl.int32,
    stride_cm: tl.int32,
    stride_cn: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused matrix multiplication with bias and ReLU: C = relu(A @ B + bias)

    Computes C[M, N] = relu(A[M, K] @ B[K, N] + bias[N])
    This combines linear layer and ReLU activation in a single kernel.

    Args:
        a_ptr: Input matrix A GPU pointer (float32*), shape (M, K).
        b_ptr: Input matrix B GPU pointer (float32*), shape (K, N).
        bias_ptr: Bias vector GPU pointer (float32*), shape (N,).
        c_ptr: Output matrix C GPU pointer (float32*), shape (M, N).
        M: Number of rows in A and C.
        N: Number of columns in B and C (and size of bias).
        K: Shared dimension (columns of A, rows of B).
        stride_am: Stride for A's M dimension.
        stride_ak: Stride for A's K dimension.
        stride_bk: Stride for B's K dimension.
        stride_bn: Stride for B's N dimension.
        stride_cm: Stride for C's M dimension.
        stride_cn: Stride for C's N dimension.
        BLOCK_M: Block size for M dimension (compile-time constant).
        BLOCK_N: Block size for N dimension (compile-time constant).
        BLOCK_K: Block size for K dimension (compile-time constant).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        a_offs = rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)

        b_offs = rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Load bias and broadcast add
    bias_mask = rn < N
    bias = tl.load(bias_ptr + rn, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU: max(0, x)
    acc = tl.maximum(acc, 0.0)

    # Store the result block to C
    c_offs = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + c_offs, acc, mask=c_mask)


@triton.jit
def matmul_nt_acc_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32,
    stride_ak: tl.int32,
    stride_bn: tl.int32,
    stride_bk: tl.int32,
    stride_cm: tl.int32,
    stride_cn: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused transposed matmul with accumulation: C += A @ B^T

    Computes C[M, N] += A[M, K] @ B^T[K, N] where B is stored as (N, K).
    Reads B in transposed order without materializing the transpose.
    Accumulates directly into C (no temporary buffer needed).

    Used for matmul backward: dA += out_grad @ B^T

    Args:
        a_ptr: Input matrix A GPU pointer (float32*), shape (M, K).
        b_ptr: Input matrix B GPU pointer (float32*), shape (N, K) - read transposed.
        c_ptr: Output matrix C GPU pointer (float32*), shape (M, N) - accumulated.
        M: Number of rows in A and C.
        N: Number of columns in C (and rows in B).
        K: Shared dimension (columns of A, columns of B).
        stride_am: Stride for A's M dimension.
        stride_ak: Stride for A's K dimension.
        stride_bn: Stride for B's N dimension (row stride in physical layout).
        stride_bk: Stride for B's K dimension (col stride in physical layout).
        stride_cm: Stride for C's M dimension.
        stride_cn: Stride for C's N dimension.
        BLOCK_M: Block size for M dimension (compile-time constant).
        BLOCK_N: Block size for N dimension (compile-time constant).
        BLOCK_K: Block size for K dimension (compile-time constant).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        # Load block of A: shape (BLOCK_M, BLOCK_K)
        a_offs = rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)

        # Load block of B^T: we want B^T[k, n] = B[n, k]
        # B is stored as (N, K), so B[n, k] = b_ptr[n * stride_bn + k * stride_bk]
        # For B^T[k, n], we index with k on rows, n on cols
        # b_offs[k, n] = rn[n] * stride_bn + rk[k] * stride_bk
        b_offs = rn[None, :] * stride_bn + rk[:, None] * stride_bk
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Load existing C values and accumulate
    c_offs = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    c_old = tl.load(c_ptr + c_offs, mask=c_mask, other=0.0)
    tl.store(c_ptr + c_offs, c_old + acc, mask=c_mask)


@triton.jit
def matmul_tn_acc_kernel(
    a_ptr: tl.pointer_type(tl.float32),
    b_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_ak: tl.int32,
    stride_am: tl.int32,
    stride_bk: tl.int32,
    stride_bn: tl.int32,
    stride_cm: tl.int32,
    stride_cn: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused transposed matmul with accumulation: C += A^T @ B

    Computes C[M, N] += A^T[M, K] @ B[K, N] where A is stored as (K, M).
    Reads A in transposed order without materializing the transpose.
    Accumulates directly into C (no temporary buffer needed).

    Used for matmul backward: dB += A^T @ out_grad

    Args:
        a_ptr: Input matrix A GPU pointer (float32*), shape (K, M) - read transposed.
        b_ptr: Input matrix B GPU pointer (float32*), shape (K, N).
        c_ptr: Output matrix C GPU pointer (float32*), shape (M, N) - accumulated.
        M: Number of rows in C (and columns in A).
        N: Number of columns in B and C.
        K: Shared dimension (rows of A, rows of B).
        stride_ak: Stride for A's K dimension (row stride in physical layout).
        stride_am: Stride for A's M dimension (col stride in physical layout).
        stride_bk: Stride for B's K dimension.
        stride_bn: Stride for B's N dimension.
        stride_cm: Stride for C's M dimension.
        stride_cn: Stride for C's N dimension.
        BLOCK_M: Block size for M dimension (compile-time constant).
        BLOCK_N: Block size for N dimension (compile-time constant).
        BLOCK_K: Block size for K dimension (compile-time constant).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        # Load block of A^T: we want A^T[m, k] = A[k, m]
        # A is stored as (K, M), so A[k, m] = a_ptr[k * stride_ak + m * stride_am]
        # For A^T[m, k], we index with m on rows, k on cols
        # a_offs[m, k] = rk[k] * stride_ak + rm[m] * stride_am
        a_offs = rk[None, :] * stride_ak + rm[:, None] * stride_am
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)

        # Load block of B: shape (BLOCK_K, BLOCK_N)
        b_offs = rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Load existing C values and accumulate
    c_offs = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    c_old = tl.load(c_ptr + c_offs, mask=c_mask, other=0.0)
    tl.store(c_ptr + c_offs, c_old + acc, mask=c_mask)


@triton.jit
def transpose2d_kernel(
    src_ptr: tl.pointer_type(tl.float32),
    dst_ptr: tl.pointer_type(tl.float32),
    rows: tl.int32,
    cols: tl.int32,
    BLOCK: tl.constexpr,
):
    """
    Transpose a 2D matrix: dst[j, i] = src[i, j]

    Input shape: (rows, cols)
    Output shape: (cols, rows)

    Args:
        src_ptr: Input matrix GPU pointer (float32*), shape (rows, cols).
        dst_ptr: Output matrix GPU pointer (float32*), shape (cols, rows).
        rows: Number of rows in source.
        cols: Number of columns in source.
        BLOCK: Block size for flattened iteration (compile-time constant).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    numel = rows * cols
    mask = offsets < numel

    # Compute source (i, j) from flat index
    i = offsets // cols  # row in source
    j = offsets % cols  # col in source

    # Load from source[i, j]
    src_idx = i * cols + j
    val = tl.load(src_ptr + src_idx, mask=mask)

    # Store to dest[j, i] = dest at row j, col i
    # dest has shape (cols, rows), so stride is rows
    dst_idx = j * rows + i
    tl.store(dst_ptr + dst_idx, val, mask=mask)
