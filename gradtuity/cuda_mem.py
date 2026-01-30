"""
ctypes interface to CUDA runtime for GPU memory management.

This module provides low-level GPU memory operations by calling CUDA runtime
functions (libcudart.so) directly via ctypes. No PyTorch, NumPy, or other
tensor libraries are used.
"""

import ctypes

# Load CUDA runtime library
_libcudart = ctypes.CDLL("libcudart.so")

# cudaMemcpyKind enum values
MEMCPY_H2D = 1  # Host to Device
MEMCPY_D2H = 2  # Device to Host
MEMCPY_D2D = 3  # Device to Device


def cuda_malloc(nbytes: int) -> int:
    """
    Allocate GPU memory.

    Args:
        nbytes: Number of bytes to allocate.

    Returns:
        Raw GPU pointer as an integer.

    Raises:
        RuntimeError: If cudaMalloc fails.
    """
    ptr = ctypes.c_void_p()
    status = _libcudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if status != 0:
        raise RuntimeError(f"cudaMalloc failed with status: {status}")
    return ptr.value


def cuda_free(ptr: int) -> None:
    """
    Free GPU memory.

    Args:
        ptr: Raw GPU pointer (as integer) to free.
    """
    _libcudart.cudaFree(ctypes.c_void_p(ptr))


def cuda_memset(ptr: int, value: int, nbytes: int) -> None:
    """
    Set nbytes at ptr to a byte value.

    Args:
        ptr: Raw GPU pointer (as integer).
        value: Byte value to set (usually 0).
        nbytes: Number of bytes to set.
    """
    _libcudart.cudaMemset(
        ctypes.c_void_p(ptr), ctypes.c_int(value), ctypes.c_size_t(nbytes)
    )


def cuda_memcpy_htod(dst: int, src_bytes: bytes) -> None:
    """
    Copy bytes from host (CPU) to device (GPU).

    Args:
        dst: Destination GPU pointer (as integer).
        src_bytes: Source bytes from host memory.
    """
    nbytes = len(src_bytes)
    src_buf = (ctypes.c_char * nbytes).from_buffer_copy(src_bytes)
    _libcudart.cudaMemcpy(
        ctypes.c_void_p(dst), src_buf, ctypes.c_size_t(nbytes), MEMCPY_H2D
    )


def cuda_memcpy_dtoh(src: int, nbytes: int) -> bytes:
    """
    Copy nbytes from device (GPU) to host (CPU).

    Args:
        src: Source GPU pointer (as integer).
        nbytes: Number of bytes to copy.

    Returns:
        Bytes copied from GPU memory.
    """
    dst_buf = (ctypes.c_char * nbytes)()
    _libcudart.cudaMemcpy(
        dst_buf, ctypes.c_void_p(src), ctypes.c_size_t(nbytes), MEMCPY_D2H
    )
    return bytes(dst_buf)


def cuda_memcpy_dtod(dst: int, src: int, nbytes: int) -> None:
    """
    Copy nbytes from device to device.

    Args:
        dst: Destination GPU pointer (as integer).
        src: Source GPU pointer (as integer).
        nbytes: Number of bytes to copy.
    """
    _libcudart.cudaMemcpy(
        ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(nbytes), MEMCPY_D2D
    )
