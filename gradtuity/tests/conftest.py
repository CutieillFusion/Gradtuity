"""
Pytest configuration and fixtures for gradtuity tests.
"""

import pytest


def _check_cuda_available() -> bool:
    """Check if CUDA runtime is available (for ctypes-based operations)."""
    try:
        import ctypes

        libcudart = ctypes.CDLL("libcudart.so")
        # Try a simple CUDA call
        device_count = ctypes.c_int()
        status = libcudart.cudaGetDeviceCount(ctypes.byref(device_count))
        return status == 0 and device_count.value > 0
    except Exception:
        return False


# Check availability at module load time
CUDA_AVAILABLE = _check_cuda_available()
# Triton requires CUDA toolkit + GPU, assume available if CUDA works
TRITON_AVAILABLE = CUDA_AVAILABLE


# Define markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_triton: mark test as requiring Triton (active CUDA GPU)"
    )
    config.addinivalue_line(
        "markers", "requires_cuda: mark test as requiring CUDA runtime"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on hardware availability."""
    skip_triton = pytest.mark.skip(reason="Triton not available (no active CUDA GPU)")
    skip_cuda = pytest.mark.skip(reason="CUDA runtime not available")

    for item in items:
        if "requires_triton" in item.keywords and not TRITON_AVAILABLE:
            item.add_marker(skip_triton)
        if "requires_cuda" in item.keywords and not CUDA_AVAILABLE:
            item.add_marker(skip_cuda)
