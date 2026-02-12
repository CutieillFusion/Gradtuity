"""
Pytest configuration for gradtuity tests.

Provides capability checks (CUDA, Triton, NCCL, multi-GPU) and skips tests
marked requires_cuda, requires_triton, requires_nccl, or requires_multigpu
when the corresponding capability is not available.
"""

import ctypes
import os

import pytest


def _check_cuda_available() -> bool:
    """Check if CUDA runtime is available (for ctypes-based operations)."""
    lib_name = os.environ.get("GRADTUITY_LIBCUDART_LIBRARY", "libcudart.so")
    try:
        libcudart = ctypes.CDLL(lib_name)
        device_count = ctypes.c_int()
        status = libcudart.cudaGetDeviceCount(ctypes.byref(device_count))
        return status == 0 and device_count.value > 0
    except Exception:
        return False


def _check_nccl_available() -> bool:
    """Check if NCCL library is available."""
    lib_name = os.environ.get("GRADTUITY_NCCL_LIBRARY", "libnccl.so.2")
    try:
        ctypes.CDLL(lib_name)
        return True
    except OSError:
        return False


def _check_multigpu_available() -> bool:
    """Check if at least 2 GPUs are available."""
    if not _check_cuda_available():
        return False
    lib_name = os.environ.get("GRADTUITY_LIBCUDART_LIBRARY", "libcudart.so")
    try:
        libcudart = ctypes.CDLL(lib_name)
        device_count = ctypes.c_int()
        status = libcudart.cudaGetDeviceCount(ctypes.byref(device_count))
        return status == 0 and device_count.value >= 2
    except Exception:
        return False


# Check availability at module load time
CUDA_AVAILABLE = _check_cuda_available()
TRITON_AVAILABLE = CUDA_AVAILABLE
NCCL_AVAILABLE = _check_nccl_available()
MULTIGPU_AVAILABLE = _check_multigpu_available()


def pytest_collection_modifyitems(config, items):
    """Skip tests based on hardware/library availability."""
    skip_triton = pytest.mark.skip(reason="Triton not available (no active CUDA GPU)")
    skip_cuda = pytest.mark.skip(reason="CUDA runtime not available")
    skip_nccl = pytest.mark.skip(reason="NCCL not available")
    skip_multigpu = pytest.mark.skip(reason="Fewer than 2 GPUs available")

    for item in items:
        if "requires_triton" in item.keywords and not TRITON_AVAILABLE:
            item.add_marker(skip_triton)
        if "requires_cuda" in item.keywords and not CUDA_AVAILABLE:
            item.add_marker(skip_cuda)
        if "requires_nccl" in item.keywords and not NCCL_AVAILABLE:
            item.add_marker(skip_nccl)
        if "requires_multigpu" in item.keywords and not MULTIGPU_AVAILABLE:
            item.add_marker(skip_multigpu)
