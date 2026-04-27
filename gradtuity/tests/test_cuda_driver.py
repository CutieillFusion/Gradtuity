"""
Tests for cuda_driver.py — NVRTC compile + driver-API launch + event timing.

Phase 1 exit criterion: a CUDA add_kernel compiles, launches, and produces
results bit-equal to the Triton add_kernel; EventTimer returns a sane number.
"""

import struct

import pytest

pytestmark = [pytest.mark.requires_cuda, pytest.mark.kernel_backend_agnostic]


def _make_buf(values: list[float]) -> tuple[int, int]:
    """Allocate device buffer and copy host floats into it. Returns (ptr, n)."""
    from gradtuity.cuda_mem import cuda_malloc, cuda_memcpy_htod

    n = len(values)
    nbytes = n * 4
    ptr = cuda_malloc(nbytes)
    cuda_memcpy_htod(ptr, struct.pack(f"{n}f", *values))
    return ptr, n


def _read_floats(ptr: int, n: int) -> list[float]:
    from gradtuity.cuda_mem import cuda_memcpy_dtoh

    return list(struct.unpack(f"{n}f", cuda_memcpy_dtoh(ptr, n * 4)))


class TestInit:
    def test_compute_capability(self):
        from gradtuity.cuda_driver import compute_capability

        major, minor = compute_capability()
        assert major >= 5  # Anything Maxwell or newer
        assert 0 <= minor <= 9


class TestCompile:
    def test_compile_returns_handle(self):
        from gradtuity.cuda_driver import compile_kernel

        src = r"""
        extern "C" __global__ void noop() {}
        """
        h = compile_kernel(src, "noop")
        assert h.func is not None
        assert h.name == "noop"

    def test_compile_caches_in_process(self):
        from gradtuity.cuda_driver import compile_kernel

        src = r"""
        extern "C" __global__ void noop_cached() {}
        """
        h1 = compile_kernel(src, "noop_cached")
        h2 = compile_kernel(src, "noop_cached")
        assert h1 is h2  # cache hit returns the same object

    def test_compile_error_surfaces_log(self):
        from gradtuity.cuda_driver import compile_kernel

        bad_src = r"""
        extern "C" __global__ void broken(int x) {
            this_function_does_not_exist(x);
        }
        """
        with pytest.raises(RuntimeError) as exc:
            compile_kernel(bad_src, "broken")
        # NVRTC log should mention the unknown identifier
        msg = str(exc.value).lower()
        assert "nvrtc" in msg
        assert "this_function_does_not_exist" in msg or "identifier" in msg


class TestLaunch:
    def test_add_kernel_matches_triton(self):
        """End-to-end: CUDA add_kernel result == Triton add_kernel result."""
        import triton

        from gradtuity.cuda_mem import cuda_free, cuda_malloc
        from gradtuity.kernels_triton.elemwise_kernels import add_kernel as triton_add
        from gradtuity.kernels_cuda.elemwise_kernels import add_kernel_launch

        n = 1024
        a_vals = [float(i) for i in range(n)]
        b_vals = [float(2 * i + 1) for i in range(n)]

        a_ptr, _ = _make_buf(a_vals)
        b_ptr, _ = _make_buf(b_vals)
        c_cuda = cuda_malloc(n * 4)
        c_triton = cuda_malloc(n * 4)

        BLOCK = 256
        grid = (triton.cdiv(n, BLOCK),)

        # Triton path
        triton_add[grid](a_ptr, b_ptr, c_triton, n, BLOCK=BLOCK)
        # CUDA path
        add_kernel_launch(grid, a_ptr, b_ptr, c_cuda, n, BLOCK)

        out_cuda = _read_floats(c_cuda, n)
        out_triton = _read_floats(c_triton, n)
        expected = [a + b for a, b in zip(a_vals, b_vals)]

        assert out_cuda == expected
        assert out_cuda == out_triton  # bit-exact for elementwise add

        cuda_free(a_ptr)
        cuda_free(b_ptr)
        cuda_free(c_cuda)
        cuda_free(c_triton)

    def test_add_kernel_handles_unaligned_size(self):
        """The bounds-check guard works when numel isn't a multiple of BLOCK."""
        from gradtuity.cuda_mem import cuda_free, cuda_malloc
        from gradtuity.kernels_cuda.elemwise_kernels import add_kernel_launch

        n = 1000  # deliberately not a multiple of 256
        a_vals = [1.0] * n
        b_vals = [2.0] * n

        a_ptr, _ = _make_buf(a_vals)
        b_ptr, _ = _make_buf(b_vals)
        c_ptr = cuda_malloc(n * 4)

        BLOCK = 256
        grid = ((n + BLOCK - 1) // BLOCK,)
        add_kernel_launch(grid, a_ptr, b_ptr, c_ptr, n, BLOCK)

        out = _read_floats(c_ptr, n)
        assert out == [3.0] * n

        cuda_free(a_ptr)
        cuda_free(b_ptr)
        cuda_free(c_ptr)


class TestEventTimer:
    def test_records_positive_elapsed_ms(self):
        from gradtuity.cuda_driver import EventTimer
        from gradtuity.cuda_mem import cuda_free, cuda_malloc
        from gradtuity.kernels_cuda.elemwise_kernels import add_kernel_launch

        n = 1 << 20  # 1M elements — must take measurable time
        a_ptr = cuda_malloc(n * 4)
        b_ptr = cuda_malloc(n * 4)
        c_ptr = cuda_malloc(n * 4)
        BLOCK = 256
        grid = ((n + BLOCK - 1) // BLOCK,)

        # Warmup so the timed launch isn't paying compile cost.
        add_kernel_launch(grid, a_ptr, b_ptr, c_ptr, n, BLOCK)

        with EventTimer() as t:
            add_kernel_launch(grid, a_ptr, b_ptr, c_ptr, n, BLOCK)
        assert t.elapsed_ms is not None
        assert t.elapsed_ms >= 0.0
        assert t.elapsed_ms < 1000.0  # 1 second is way more than enough

        cuda_free(a_ptr)
        cuda_free(b_ptr)
        cuda_free(c_ptr)
