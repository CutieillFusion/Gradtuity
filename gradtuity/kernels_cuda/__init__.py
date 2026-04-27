"""Hand-written CUDA-C kernels (NVRTC-compiled) — peer of gradtuity.kernels.

Each module here mirrors a file in gradtuity.kernels. Kernels are stored as
CUDA-C source strings, compiled lazily on first launch via cuda_driver, and
launched through cuLaunchKernel. Wrappers expose the same calling shape as
the matching Triton kernel so the two backends can be swapped at the dispatch
layer (gradtuity.kernels.__init__).
"""

from .elemwise_kernels import add_kernel_launch

__all__ = ["add_kernel_launch"]
