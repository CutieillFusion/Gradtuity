"""
Pytest configuration for gradtuity tests.

Provides capability checks (CUDA, Triton, NCCL, multi-GPU) and skips tests
marked requires_cuda, requires_triton, requires_nccl, or requires_multigpu
when the corresponding capability is not available.

Also auto-parametrizes every test over both kernel backends (Triton + CUDA-C)
in a single pytest process, by flipping ``gradtuity.kernels.dispatch`` state
between tests. Mark a test ``kernel_backend_agnostic`` to opt out of the CUDA
repetition (used by the parity tests that exercise both backends explicitly).
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


# ---------------------------------------------------------------------------
# Backend parametrization
#
# ``gradtuity.kernels.dispatch.set_backend`` flips which backend the higher-
# level Tensor / nn / training tests route through, so a single pytest process
# exercises both code paths for every test. Mark a test
# ``kernel_backend_agnostic`` to opt out of the CUDA repetition — used by the
# kernel-pair parity tests that already exercise both backends within a
# single test body.
# ---------------------------------------------------------------------------
_BACKEND_OPT_OUT = os.environ.get("GRADTUITY_TEST_BACKENDS")


@pytest.fixture(autouse=True, params=["triton", "cuda"])
def _kernel_backend(request):
    if _BACKEND_OPT_OUT and request.param not in _BACKEND_OPT_OUT.split(","):
        # Caller restricted the backends explicitly (e.g., for fast local
        # iteration: ``GRADTUITY_TEST_BACKENDS=triton pytest``).
        pytest.skip(f"backend {request.param!r} not in GRADTUITY_TEST_BACKENDS")
    if request.param == "cuda":
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA backend selected but CUDA runtime not available")
        if request.node.get_closest_marker("kernel_backend_agnostic"):
            pytest.skip("backend-agnostic test; one backend is enough")
    from gradtuity import kernels as dispatch
    old = dispatch.active_backend()
    dispatch.set_backend(request.param)
    try:
        yield request.param
    finally:
        dispatch.set_backend(old)
