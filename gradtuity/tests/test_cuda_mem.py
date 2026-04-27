"""
Tests for cuda_mem.py - CUDA memory management functions.

These tests require a CUDA-enabled GPU to run.
"""

import struct

import pytest

from gradtuity.cuda_mem import (
    cuda_free,
    cuda_malloc,
    cuda_memcpy_dtod,
    cuda_memcpy_dtoh,
    cuda_memcpy_htod,
    cuda_memset,
)

# Mark all tests in this module as requiring CUDA
pytestmark = pytest.mark.requires_cuda


class TestCudaMalloc:
    """Tests for cuda_malloc function."""

    def test_allocate_small_buffer(self):
        """Test allocating a small buffer (1KB)."""
        ptr = cuda_malloc(1024)
        assert ptr is not None
        assert isinstance(ptr, int)
        assert ptr > 0
        cuda_free(ptr)

    def test_allocate_single_float(self):
        """Test allocating space for a single float32 (4 bytes)."""
        ptr = cuda_malloc(4)
        assert ptr is not None
        assert ptr > 0
        cuda_free(ptr)

    def test_allocate_large_buffer(self):
        """Test allocating a larger buffer (1MB)."""
        ptr = cuda_malloc(1024 * 1024)
        assert ptr is not None
        assert ptr > 0
        cuda_free(ptr)

    def test_allocate_zero_bytes(self):
        """0-byte allocation may succeed or raise; we only verify no crash."""
        try:
            ptr = cuda_malloc(0)
            if ptr is not None:
                cuda_free(ptr)
        except RuntimeError:
            pass


class TestCudaFree:
    """Tests for cuda_free function."""

    def test_free_allocated_memory(self):
        """Test freeing allocated memory doesn't raise."""
        ptr = cuda_malloc(1024)
        # Should not raise
        cuda_free(ptr)

    def test_double_free_null(self):
        """Test that freeing NULL (0) doesn't crash (CUDA allows this)."""
        # cudaFree(NULL) is safe and returns cudaSuccess
        cuda_free(0)


class TestCudaMemset:
    """Tests for cuda_memset function."""

    def test_memset_zeros(self):
        """Test setting memory to zero."""
        nbytes = 16  # 4 floats
        ptr = cuda_malloc(nbytes)
        cuda_memset(ptr, 0, nbytes)

        # Read back and verify all zeros
        result = cuda_memcpy_dtoh(ptr, nbytes)
        assert result == b"\x00" * nbytes

        cuda_free(ptr)

    def test_memset_nonzero_byte(self):
        """Test setting memory to a non-zero byte value."""
        nbytes = 8
        ptr = cuda_malloc(nbytes)
        cuda_memset(ptr, 0xFF, nbytes)

        # Read back and verify
        result = cuda_memcpy_dtoh(ptr, nbytes)
        assert result == b"\xff" * nbytes

        cuda_free(ptr)

    def test_memset_partial(self):
        """Test setting only part of allocated memory."""
        nbytes = 16
        ptr = cuda_malloc(nbytes)

        # Zero entire buffer first
        cuda_memset(ptr, 0, nbytes)

        # Set only first 8 bytes to 0xAA
        cuda_memset(ptr, 0xAA, 8)

        # Read back and verify
        result = cuda_memcpy_dtoh(ptr, nbytes)
        assert result[:8] == b"\xaa" * 8
        assert result[8:] == b"\x00" * 8

        cuda_free(ptr)


class TestCudaMemcpyHtoD:
    """Tests for cuda_memcpy_htod function (host to device)."""

    def test_copy_bytes_to_device(self):
        """Test copying raw bytes to device."""
        data = b"hello world!"
        nbytes = len(data)

        ptr = cuda_malloc(nbytes)
        cuda_memcpy_htod(ptr, data)

        # Read back and verify
        result = cuda_memcpy_dtoh(ptr, nbytes)
        assert result == data

        cuda_free(ptr)

    def test_copy_float32_values(self):
        """Test copying float32 values to device."""
        values = [1.0, 2.5, -3.14, 0.0]
        data = struct.pack(f"{len(values)}f", *values)
        nbytes = len(data)

        ptr = cuda_malloc(nbytes)
        cuda_memcpy_htod(ptr, data)

        # Read back and verify
        result = cuda_memcpy_dtoh(ptr, nbytes)
        result_values = struct.unpack(f"{len(values)}f", result)

        for expected, actual in zip(values, result_values):
            assert abs(expected - actual) < 1e-6

        cuda_free(ptr)

    def test_copy_empty_bytes(self):
        """Test copying empty bytes (edge case)."""
        ptr = cuda_malloc(4)  # Allocate some space
        cuda_memcpy_htod(ptr, b"")  # Copy nothing
        cuda_free(ptr)


class TestCudaMemcpyDtoH:
    """Tests for cuda_memcpy_dtoh function (device to host)."""

    def test_read_zeroed_memory(self):
        """Test reading zero-initialized memory."""
        nbytes = 16
        ptr = cuda_malloc(nbytes)
        cuda_memset(ptr, 0, nbytes)

        result = cuda_memcpy_dtoh(ptr, nbytes)
        assert result == b"\x00" * nbytes

        cuda_free(ptr)

    def test_read_float32_values(self):
        """Test reading float32 values from device."""
        values = [42.0, -1.5, 3.14159, 0.001]
        data = struct.pack(f"{len(values)}f", *values)
        nbytes = len(data)

        ptr = cuda_malloc(nbytes)
        cuda_memcpy_htod(ptr, data)

        # Read back
        result = cuda_memcpy_dtoh(ptr, nbytes)
        result_values = struct.unpack(f"{len(values)}f", result)

        for expected, actual in zip(values, result_values):
            assert abs(expected - actual) < 1e-6

        cuda_free(ptr)

    def test_read_partial(self):
        """Test reading only part of the allocated memory."""
        full_data = b"0123456789ABCDEF"
        ptr = cuda_malloc(len(full_data))
        cuda_memcpy_htod(ptr, full_data)

        # Read only first 8 bytes
        result = cuda_memcpy_dtoh(ptr, 8)
        assert result == b"01234567"

        cuda_free(ptr)


class TestCudaMemcpyDtoD:
    """Tests for cuda_memcpy_dtod function (device to device)."""

    def test_copy_between_buffers(self):
        """Test copying data between two GPU buffers."""
        data = b"test data for dtod copy"
        nbytes = len(data)

        # Allocate source and destination
        src_ptr = cuda_malloc(nbytes)
        dst_ptr = cuda_malloc(nbytes)

        # Initialize destination to zeros
        cuda_memset(dst_ptr, 0, nbytes)

        # Copy data to source
        cuda_memcpy_htod(src_ptr, data)

        # Copy from source to destination on device
        cuda_memcpy_dtod(dst_ptr, src_ptr, nbytes)

        # Read destination and verify
        result = cuda_memcpy_dtoh(dst_ptr, nbytes)
        assert result == data

        cuda_free(src_ptr)
        cuda_free(dst_ptr)

    def test_copy_float32_array(self):
        """Test copying float32 array between GPU buffers."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        data = struct.pack(f"{len(values)}f", *values)
        nbytes = len(data)

        src_ptr = cuda_malloc(nbytes)
        dst_ptr = cuda_malloc(nbytes)

        cuda_memcpy_htod(src_ptr, data)
        cuda_memcpy_dtod(dst_ptr, src_ptr, nbytes)

        result = cuda_memcpy_dtoh(dst_ptr, nbytes)
        result_values = struct.unpack(f"{len(values)}f", result)

        for expected, actual in zip(values, result_values):
            assert abs(expected - actual) < 1e-6

        cuda_free(src_ptr)
        cuda_free(dst_ptr)

    def test_copy_partial(self):
        """Test copying only part of the data between buffers."""
        full_data = b"AAAABBBBCCCCDDDD"
        nbytes = len(full_data)

        src_ptr = cuda_malloc(nbytes)
        dst_ptr = cuda_malloc(nbytes)

        # Initialize both
        cuda_memcpy_htod(src_ptr, full_data)
        cuda_memset(dst_ptr, 0, nbytes)

        # Copy only first 8 bytes
        cuda_memcpy_dtod(dst_ptr, src_ptr, 8)

        result = cuda_memcpy_dtoh(dst_ptr, nbytes)
        assert result[:8] == b"AAAABBBB"
        assert result[8:] == b"\x00" * 8

        cuda_free(src_ptr)
        cuda_free(dst_ptr)


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_roundtrip_float_tensor(self):
        """Test full roundtrip: allocate, write floats, read back."""
        # Simulate a small 2x3 tensor of float32
        shape = (2, 3)
        numel = shape[0] * shape[1]
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        nbytes = numel * 4

        # Pack to bytes
        host_bytes = struct.pack(f"{numel}f", *values)

        # Allocate and copy to GPU
        ptr = cuda_malloc(nbytes)
        cuda_memcpy_htod(ptr, host_bytes)

        # Read back
        result_bytes = cuda_memcpy_dtoh(ptr, nbytes)
        result_values = struct.unpack(f"{numel}f", result_bytes)

        # Verify
        assert len(result_values) == numel
        for expected, actual in zip(values, result_values):
            assert abs(expected - actual) < 1e-6

        cuda_free(ptr)

    def test_zero_initialize_then_write(self):
        """Test zero-init followed by partial write."""
        nbytes = 32  # 8 floats

        ptr = cuda_malloc(nbytes)
        cuda_memset(ptr, 0, nbytes)

        # Write 4 floats (16 bytes) starting at the beginning
        values = [1.0, 2.0, 3.0, 4.0]
        data = struct.pack("4f", *values)
        cuda_memcpy_htod(ptr, data)

        # Read all 8 floats
        result = cuda_memcpy_dtoh(ptr, nbytes)
        all_values = struct.unpack("8f", result)

        # First 4 should be our values
        assert all_values[:4] == tuple(values)
        # Last 4 should still be zero
        assert all_values[4:] == (0.0, 0.0, 0.0, 0.0)

        cuda_free(ptr)

    def test_multiple_allocations(self):
        """Test multiple simultaneous allocations."""
        ptrs = []
        num_allocs = 10
        size = 1024

        # Allocate multiple buffers
        for i in range(num_allocs):
            ptr = cuda_malloc(size)
            cuda_memset(ptr, i, size)  # Set each to different byte value
            ptrs.append(ptr)

        # Verify each buffer independently
        for i, ptr in enumerate(ptrs):
            result = cuda_memcpy_dtoh(ptr, size)
            expected = bytes([i] * size)
            assert result == expected

        # Free all
        for ptr in ptrs:
            cuda_free(ptr)
