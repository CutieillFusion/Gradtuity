"""
Benchmark runner. Each Spec defines: a name, shape grid, byte/flop accounting,
input setup, and one callable per backend that launches the kernel into a
caller-supplied output buffer.

The runner times each launch with EventTimer (cuEventRecord pairs around the
launch, cuEventSynchronize to wait, cuEventElapsedTime for the ms result), so
we measure kernel-side time, not Python overhead.
"""

from __future__ import annotations

import statistics
import struct
from dataclasses import dataclass
from typing import Callable, Iterable

import triton

from ..cuda_driver import EventTimer
from ..cuda_mem import (
    cuda_free,
    cuda_malloc,
    cuda_memcpy_dtoh,
    cuda_memcpy_htod,
    cuda_memset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def alloc_floats(values: list[float]) -> int:
    n = len(values)
    p = cuda_malloc(n * 4)
    cuda_memcpy_htod(p, struct.pack(f"{n}f", *values))
    return p


def alloc_zeros(n: int) -> int:
    p = cuda_malloc(n * 4)
    cuda_memset(p, 0, n * 4)
    return p


def read_floats(p: int, n: int) -> list[float]:
    return list(struct.unpack(f"{n}f", cuda_memcpy_dtoh(p, n * 4)))


def deterministic_floats(n: int, scale: float = 0.1) -> list[float]:
    """Repeatable, no-numpy-dependency input generator."""
    return [(((i * 13) % 41) - 20) * scale / 20.0 for i in range(n)]


# ---------------------------------------------------------------------------
# Spec & result types
# ---------------------------------------------------------------------------
@dataclass
class Spec:
    """One benchmarkable kernel-shape combination.

    ``run_triton`` and ``run_cuda`` accept the dict returned by ``setup`` and
    must perform a single in-place launch. They return the device pointer of
    the output buffer for parity comparison (or None if the kernel writes
    to an existing buffer in setup).
    """

    name: str
    shape_label: str
    bytes_moved: int
    flops: int
    setup: Callable[[], dict]
    run_triton: Callable[[dict], int | None]
    run_cuda: Callable[[dict], int | None]
    teardown: Callable[[dict], None]
    output_size: int  # number of floats to read for parity check
    output_key: str = "out"


@dataclass
class Result:
    name: str
    shape_label: str
    triton_ms: float
    cuda_ms: float
    bytes_moved: int
    flops: int
    max_abs_diff: float


def _gbps(bytes_moved: int, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return bytes_moved / (ms * 1e-3) / 1e9


def _tflops(flops: int, ms: float) -> float:
    if ms <= 0 or flops == 0:
        return float("nan")
    return flops / (ms * 1e-3) / 1e12


def _max_abs(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def time_one(launch_fn: Callable[[dict], int | None], state: dict, warmup=5, iters=50) -> float:
    for _ in range(warmup):
        launch_fn(state)
    samples = []
    for _ in range(iters):
        with EventTimer() as t:
            launch_fn(state)
        samples.append(t.elapsed_ms)
    return statistics.median(samples)


def run_spec(spec: Spec, warmup=5, iters=50) -> Result:
    # Parity: fresh state per backend, single launch each, then diff.
    # This handles stateful kernels (e.g. adamw) where many iterations would
    # accumulate divergence; the actual correctness suite does the heavy
    # parity testing — we just sanity-check here.
    parity_t_state = spec.setup()
    try:
        spec.run_triton(parity_t_state)
        out_triton = read_floats(parity_t_state[spec.output_key], spec.output_size)
    finally:
        spec.teardown(parity_t_state)
    parity_c_state = spec.setup()
    try:
        spec.run_cuda(parity_c_state)
        out_cuda = read_floats(parity_c_state[spec.output_key], spec.output_size)
    finally:
        spec.teardown(parity_c_state)
    diff = _max_abs(out_triton, out_cuda)

    # Timing: each backend gets its own fresh state so iteration N doesn't
    # depend on iteration N-1 of the other backend.
    t_state = spec.setup()
    try:
        triton_ms = time_one(spec.run_triton, t_state, warmup, iters)
    finally:
        spec.teardown(t_state)
    c_state = spec.setup()
    try:
        cuda_ms = time_one(spec.run_cuda, c_state, warmup, iters)
    finally:
        spec.teardown(c_state)

    return Result(
        name=spec.name,
        shape_label=spec.shape_label,
        triton_ms=triton_ms,
        cuda_ms=cuda_ms,
        bytes_moved=spec.bytes_moved,
        flops=spec.flops,
        max_abs_diff=diff,
    )


def _ratio(r: Result) -> float:
    """Triton time / CUDA time. >1 means CUDA is faster, <1 means Triton wins."""
    return r.triton_ms / r.cuda_ms if r.cuda_ms > 0 else float("nan")


def format_markdown(results: Iterable[Result]) -> str:
    rows = [
        "| Kernel | Shape | Triton ms | CUDA ms | CUDA / Triton | Triton GB/s | CUDA GB/s | Max \\|Δ\\| |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        rows.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.2f}× | {:.1f} | {:.1f} | {:.2e} |".format(
                r.name, r.shape_label, r.triton_ms, r.cuda_ms, _ratio(r),
                _gbps(r.bytes_moved, r.triton_ms), _gbps(r.bytes_moved, r.cuda_ms),
                r.max_abs_diff,
            )
        )
    return "\n".join(rows)


def format_csv(results: Iterable[Result]) -> str:
    out = ["kernel,shape,triton_ms,cuda_ms,cuda_speedup_over_triton,bytes,flops,triton_gbps,cuda_gbps,triton_tflops,cuda_tflops,max_abs_diff"]
    for r in results:
        out.append(
            "{},{},{:.6f},{:.6f},{:.4f},{},{},{:.3f},{:.3f},{:.4f},{:.4f},{:.4e}".format(
                r.name, r.shape_label, r.triton_ms, r.cuda_ms, _ratio(r),
                r.bytes_moved, r.flops,
                _gbps(r.bytes_moved, r.triton_ms),
                _gbps(r.bytes_moved, r.cuda_ms),
                _tflops(r.flops, r.triton_ms),
                _tflops(r.flops, r.cuda_ms),
                r.max_abs_diff,
            )
        )
    return "\n".join(out)
