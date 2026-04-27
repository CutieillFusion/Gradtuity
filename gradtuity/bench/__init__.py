"""Benchmark harness comparing Triton vs hand-written CUDA-C kernels.

CLI: ``python -m gradtuity.bench [pattern]``

Times each kernel pair (Triton + CUDA-C) on the same inputs and the same
default CUDA stream, using cuEvent for kernel-side timing. Reports median
of N iterations after warmup, achieved GB/s, and the max-abs-diff between
the two backends' outputs.
"""
