"""
Benchmark specs — one Spec per (kernel, shape) pair.

Covers a representative slice of each kernel category. Add more specs by
appending to ALL_SPECS. Shapes are chosen to match real GPT-2 small training
on this codebase (vocab=50257, d_model=768, ctx=1024, batch=8) where
applicable; otherwise medium sizes that exercise the kernel meaningfully.
"""

from __future__ import annotations

import triton

from ..cuda_mem import cuda_free, cuda_malloc, cuda_memset
from ..tensor import grid1d
from .runner import Spec, alloc_floats, alloc_zeros, deterministic_floats


# ---------------------------------------------------------------------------
# Elementwise — add (memory-bound canonical)
# ---------------------------------------------------------------------------
def _add_spec(numel: int, BLOCK: int = 256) -> Spec:
    from ..kernels_triton.elemwise_kernels import add_kernel as TK
    from ..kernels_cuda.elemwise_kernels import add_kernel_launch as CK
    bytes_moved = 3 * 4 * numel
    return Spec(
        name="add_kernel",
        shape_label=f"n={numel}",
        bytes_moved=bytes_moved,
        flops=numel,
        output_size=numel,
        setup=lambda: {
            "a": alloc_floats(deterministic_floats(numel)),
            "b": alloc_floats(deterministic_floats(numel, scale=0.07)),
            "out": cuda_malloc(numel * 4),
        },
        run_triton=lambda s: TK[grid1d(numel)](s["a"], s["b"], s["out"], numel, BLOCK=BLOCK),
        run_cuda=lambda s: CK(grid1d(numel), s["a"], s["b"], s["out"], numel, BLOCK),
        teardown=lambda s: [cuda_free(s["a"]), cuda_free(s["b"]), cuda_free(s["out"])],
    )


# ---------------------------------------------------------------------------
# Elementwise — GELU (transcendental: exp/tanh)
# ---------------------------------------------------------------------------
def _gelu_spec(numel: int, BLOCK: int = 256) -> Spec:
    from ..kernels_triton.elemwise_kernels import gelu_kernel as TK
    from ..kernels_cuda.elemwise_kernels import gelu_kernel_launch as CK
    return Spec(
        name="gelu_kernel",
        shape_label=f"n={numel}",
        bytes_moved=2 * 4 * numel,
        flops=10 * numel,  # ~10 ops per element (mul, add, exp, etc.)
        output_size=numel,
        setup=lambda: {
            "x": alloc_floats(deterministic_floats(numel)),
            "out": cuda_malloc(numel * 4),
        },
        run_triton=lambda s: TK[grid1d(numel)](s["x"], s["out"], numel, BLOCK=BLOCK),
        run_cuda=lambda s: CK(grid1d(numel), s["x"], s["out"], numel, BLOCK),
        teardown=lambda s: [cuda_free(s["x"]), cuda_free(s["out"])],
    )


# ---------------------------------------------------------------------------
# Reduction — sum_all (atomicAdd)
# ---------------------------------------------------------------------------
def _sum_all_spec(numel: int, BLOCK: int = 256) -> Spec:
    from ..kernels_triton.reduce_kernels import sum_all_kernel as TK
    from ..kernels_cuda.reduce_kernels import sum_all_kernel_launch as CK

    def setup():
        x = alloc_floats(deterministic_floats(numel, scale=0.001))
        out = alloc_zeros(1)
        return {"x": x, "out": out}

    def run_triton(s):
        cuda_memset(s["out"], 0, 4)
        TK[grid1d(numel)](s["x"], s["out"], numel, BLOCK=BLOCK)

    def run_cuda(s):
        cuda_memset(s["out"], 0, 4)
        CK(grid1d(numel), s["x"], s["out"], numel, BLOCK)

    return Spec(
        name="sum_all_kernel",
        shape_label=f"n={numel}",
        bytes_moved=4 * numel,
        flops=numel,
        output_size=1,
        setup=setup,
        run_triton=run_triton,
        run_cuda=run_cuda,
        teardown=lambda s: [cuda_free(s["x"]), cuda_free(s["out"])],
    )


# ---------------------------------------------------------------------------
# Softmax forward (row-wise, common shape for attention scores / logits)
# ---------------------------------------------------------------------------
def _softmax_spec(rows: int, cols: int, BLOCK_COLS: int = 128) -> Spec:
    from ..kernels_triton.softmax_kernels import softmax_forward_kernel as TK
    from ..kernels_cuda.softmax_kernels import softmax_forward_kernel_launch as CK
    n = rows * cols
    return Spec(
        name="softmax_forward_kernel",
        shape_label=f"{rows}x{cols}",
        bytes_moved=2 * 4 * n,
        flops=5 * n,
        output_size=n,
        setup=lambda: {
            "x": alloc_floats(deterministic_floats(n)),
            "out": cuda_malloc(n * 4),
        },
        run_triton=lambda s: TK[(rows,)](s["x"], s["out"], rows, cols, BLOCK_COLS=BLOCK_COLS),
        run_cuda=lambda s: CK((rows,), s["x"], s["out"], rows, cols, BLOCK_COLS),
        teardown=lambda s: [cuda_free(s["x"]), cuda_free(s["out"])],
    )


# ---------------------------------------------------------------------------
# LayerNorm forward
# ---------------------------------------------------------------------------
def _layernorm_spec(N: int, H: int, BLOCK_H: int = 128) -> Spec:
    from ..kernels_triton.layernorm_kernels import layernorm_fwd_kernel as TK
    from ..kernels_cuda.layernorm_kernels import layernorm_fwd_kernel_launch as CK
    n = N * H

    def setup():
        x = alloc_floats(deterministic_floats(n))
        gamma = alloc_floats([1.0 + 0.01 * (i % 5) for i in range(H)])
        beta = alloc_floats([0.5 - 0.02 * (i % 5) for i in range(H)])
        return {
            "x": x, "gamma": gamma, "beta": beta,
            "out": cuda_malloc(n * 4),
            "xhat": cuda_malloc(n * 4),
            "rstd": cuda_malloc(N * 4),
        }

    def run_triton(s):
        TK[(N,)](
            s["x"], s["gamma"], s["beta"], s["out"], s["xhat"], s["rstd"],
            N, H, 1e-5, BLOCK_H=BLOCK_H,
        )

    def run_cuda(s):
        CK(
            (N,), s["x"], s["gamma"], s["beta"], s["out"], s["xhat"], s["rstd"],
            N, H, 1e-5, BLOCK_H,
        )

    return Spec(
        name="layernorm_fwd_kernel",
        shape_label=f"{N}x{H}",
        bytes_moved=4 * (4 * n + 2 * H + N),  # x, out, xhat (3n) + gamma+beta (2H) + rstd (N)
        flops=10 * n,
        output_size=n,
        setup=setup,
        run_triton=run_triton,
        run_cuda=run_cuda,
        teardown=lambda s: [cuda_free(s[k]) for k in ("x", "gamma", "beta", "out", "xhat", "rstd")],
    )


# ---------------------------------------------------------------------------
# Matmul (compute-bound; expected to show Triton MMA advantage)
# ---------------------------------------------------------------------------
def _matmul_spec(M: int, K: int, N: int) -> Spec:
    from ..kernels_triton.matmul_kernels import matmul_kernel as TK
    from ..kernels_cuda.matmul_kernels import matmul_kernel_launch as CK
    BLOCK_M = BLOCK_N = BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    def setup():
        return {
            "a": alloc_floats(deterministic_floats(M * K)),
            "b": alloc_floats(deterministic_floats(K * N, scale=0.07)),
            "out": cuda_malloc(M * N * 4),
        }

    def run_triton(s):
        TK[grid](
            s["a"], s["b"], s["out"], M, N, K,
            K, 1, N, 1, N, 1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

    def run_cuda(s):
        CK(grid, s["a"], s["b"], s["out"], M, N, K, K, 1, N, 1, N, 1, BLOCK_M, BLOCK_N, BLOCK_K)

    return Spec(
        name="matmul_kernel",
        shape_label=f"{M}x{K}x{N}",
        bytes_moved=4 * (M * K + K * N + M * N),
        flops=2 * M * N * K,
        output_size=M * N,
        setup=setup,
        run_triton=run_triton,
        run_cuda=run_cuda,
        teardown=lambda s: [cuda_free(s[k]) for k in ("a", "b", "out")],
    )


# ---------------------------------------------------------------------------
# Cross-entropy forward (per-row reduction with atomic — typical LM-head shape)
# ---------------------------------------------------------------------------
def _ce_spec(B: int, C: int, BLOCK_C: int = 128) -> Spec:
    from ..kernels_triton.loss_kernels import cross_entropy_forward_kernel as TK
    from ..kernels_cuda.loss_kernels import cross_entropy_forward_kernel_launch as CK

    def setup():
        return {
            "logits": alloc_floats(deterministic_floats(B * C)),
            "targets": alloc_floats([float(i % C) for i in range(B)]),
            "out": alloc_zeros(1),
        }

    def run_triton(s):
        cuda_memset(s["out"], 0, 4)
        TK[(B,)](s["logits"], s["targets"], s["out"], B, C, BLOCK_C=BLOCK_C)

    def run_cuda(s):
        cuda_memset(s["out"], 0, 4)
        CK((B,), s["logits"], s["targets"], s["out"], B, C, BLOCK_C)

    return Spec(
        name="cross_entropy_forward_kernel",
        shape_label=f"B={B} C={C}",
        bytes_moved=4 * B * C,
        flops=4 * B * C,
        output_size=1,
        setup=setup,
        run_triton=run_triton,
        run_cuda=run_cuda,
        teardown=lambda s: [cuda_free(s[k]) for k in ("logits", "targets", "out")],
    )


# ---------------------------------------------------------------------------
# Embedding gather (token embedding lookup — common LM bottleneck)
# ---------------------------------------------------------------------------
def _embedding_gather_spec(V: int, D: int, N: int, BLOCK_D: int = 128) -> Spec:
    from ..kernels_triton.gather_kernels import embedding_gather_kernel as TK
    from ..kernels_cuda.gather_kernels import embedding_gather_kernel_launch as CK
    grid = (N, triton.cdiv(D, BLOCK_D))

    def setup():
        return {
            "W": alloc_floats(deterministic_floats(V * D)),
            "idx": alloc_floats([float(i % V) for i in range(N)]),
            "out": cuda_malloc(N * D * 4),
        }

    def run_triton(s):
        TK[grid](s["W"], s["idx"], s["out"], N, D, V, BLOCK_D=BLOCK_D)

    def run_cuda(s):
        CK(grid, s["W"], s["idx"], s["out"], N, D, V, BLOCK_D)

    return Spec(
        name="embedding_gather_kernel",
        shape_label=f"V={V} D={D} N={N}",
        bytes_moved=4 * (N * D + N),  # gather reads N*D from W + idx
        flops=0,  # pure data movement
        output_size=N * D,
        setup=setup,
        run_triton=run_triton,
        run_cuda=run_cuda,
        teardown=lambda s: [cuda_free(s[k]) for k in ("W", "idx", "out")],
    )


# ---------------------------------------------------------------------------
# Causal mask inplace (attention mask)
# ---------------------------------------------------------------------------
def _causal_mask_spec(B: int, H: int, S: int, BLOCK_I: int = 32, BLOCK_J: int = 32) -> Spec:
    from ..kernels_triton.mask_kernels import causal_mask_inplace_kernel as TK
    from ..kernels_cuda.mask_kernels import causal_mask_inplace_kernel_launch as CK
    grid = (B * H, triton.cdiv(S, BLOCK_I), triton.cdiv(S, BLOCK_J))
    n = B * H * S * S

    def setup():
        # Re-allocate per call so the timing loop sees a fresh state — but
        # since causal_mask_inplace is idempotent (re-applying mask gives same
        # result), we can reuse the same buffer across iterations.
        return {"out": alloc_floats(deterministic_floats(n))}

    def run_triton(s):
        TK[grid](s["out"], B=B, H=H, S=S, NEG_INF=-1e9, BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J)

    def run_cuda(s):
        CK(grid, s["out"], B, H, S, -1e9, BLOCK_I, BLOCK_J)

    return Spec(
        name="causal_mask_inplace_kernel",
        shape_label=f"B={B} H={H} S={S}",
        bytes_moved=4 * 2 * n,  # read + write
        flops=n,
        output_size=n,
        setup=setup,
        run_triton=run_triton,
        run_cuda=run_cuda,
        teardown=lambda s: [cuda_free(s["out"])],
    )


# ---------------------------------------------------------------------------
# AdamW step (optimizer step over a typical parameter shape)
# ---------------------------------------------------------------------------
def _adamw_spec(numel: int, BLOCK: int = 256) -> Spec:
    from ..kernels_triton.optim_kernels import adamw_step_kernel as TK
    from ..kernels_cuda.optim_kernels import adamw_step_kernel_launch as CK

    def setup():
        return {
            "p": alloc_floats(deterministic_floats(numel, scale=0.05)),
            "g": alloc_floats(deterministic_floats(numel, scale=0.01)),
            "m": alloc_zeros(numel),
            "v": alloc_zeros(numel),
            "out": 0,  # set below
        }

    def setup_with_out():
        s = setup()
        s["out"] = s["p"]  # reuse p as the read-back point
        return s

    def run_triton(s):
        TK[grid1d(numel)](
            s["p"], s["g"], s["m"], s["v"], numel,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 10.0, 1000.0,
            BLOCK=BLOCK,
        )

    def run_cuda(s):
        CK(
            grid1d(numel), s["p"], s["g"], s["m"], s["v"], numel,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 10.0, 1000.0,
            BLOCK,
        )

    return Spec(
        name="adamw_step_kernel",
        shape_label=f"n={numel}",
        bytes_moved=4 * 4 * numel,  # read+write p, g, m, v approximately
        flops=8 * numel,
        output_size=numel,
        output_key="out",
        setup=setup_with_out,
        run_triton=run_triton,
        run_cuda=run_cuda,
        teardown=lambda s: [cuda_free(s[k]) for k in ("p", "g", "m", "v")],
    )


# ---------------------------------------------------------------------------
# All specs — small sizes so the suite finishes in a few seconds.
# ---------------------------------------------------------------------------
def all_specs() -> list[Spec]:
    return [
        _add_spec(1 << 20),                          # 1M
        _add_spec(1 << 24),                          # 16M
        _gelu_spec(1 << 20),
        _sum_all_spec(1 << 20),
        _softmax_spec(rows=1024, cols=512),
        _softmax_spec(rows=128, cols=50257),         # LM-head softmax
        _layernorm_spec(N=1024, H=768),              # GPT-2 small d_model
        _matmul_spec(256, 256, 256),
        _matmul_spec(1024, 768, 768),                # GPT-2 small attn proj
        _ce_spec(B=128, C=50257),                    # LM cross-entropy
        _embedding_gather_spec(V=50257, D=768, N=1024),
        _causal_mask_spec(B=4, H=12, S=512),
        _adamw_spec(1 << 20),
    ]
