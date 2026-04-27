"""
ctypes interface to NVRTC + CUDA driver API for runtime kernel compilation.

Mirrors cuda_mem.py style: env-var-driven library loading, raw ctypes,
no third-party deps. Allows compiling CUDA C source at runtime via NVRTC,
loading the resulting PTX as a CUDA module, and launching kernels via the
driver API. Designed to coexist with cuda_mem.py (runtime API) on the same
primary context.
"""

import ctypes
import hashlib
import os
import threading
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Library loading (env-var driven, matches cuda_mem.py convention)
# ---------------------------------------------------------------------------


def _load_lib(env_var: str, hint: str) -> ctypes.CDLL:
    name = os.environ.get(env_var)
    if not name:
        raise RuntimeError(
            f"{env_var} is not set. Set it to the {hint} library "
            f"(e.g. {hint}.so or {hint}.so.12)."
        )
    return ctypes.CDLL(name)


_libnvrtc = _load_lib("GRADTUITY_LIBNVRTC_LIBRARY", "libnvrtc")
_libcuda = _load_lib("GRADTUITY_LIBCUDA_LIBRARY", "libcuda")


# ---------------------------------------------------------------------------
# NVRTC bindings
# ---------------------------------------------------------------------------
_nvrtcProgram = ctypes.c_void_p

_libnvrtc.nvrtcCreateProgram.restype = ctypes.c_int
_libnvrtc.nvrtcCreateProgram.argtypes = [
    ctypes.POINTER(_nvrtcProgram),
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]

_libnvrtc.nvrtcDestroyProgram.restype = ctypes.c_int
_libnvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(_nvrtcProgram)]

_libnvrtc.nvrtcCompileProgram.restype = ctypes.c_int
_libnvrtc.nvrtcCompileProgram.argtypes = [
    _nvrtcProgram,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
]

# Cubin (SASS) output — emitted instead of PTX so the driver doesn't have to
# JIT, which sidesteps NVRTC-vs-driver PTX ISA version skew.
_libnvrtc.nvrtcGetCUBINSize.restype = ctypes.c_int
_libnvrtc.nvrtcGetCUBINSize.argtypes = [
    _nvrtcProgram,
    ctypes.POINTER(ctypes.c_size_t),
]

_libnvrtc.nvrtcGetCUBIN.restype = ctypes.c_int
_libnvrtc.nvrtcGetCUBIN.argtypes = [_nvrtcProgram, ctypes.c_char_p]

_libnvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
_libnvrtc.nvrtcGetProgramLogSize.argtypes = [
    _nvrtcProgram,
    ctypes.POINTER(ctypes.c_size_t),
]

_libnvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
_libnvrtc.nvrtcGetProgramLog.argtypes = [_nvrtcProgram, ctypes.c_char_p]

_libnvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
_libnvrtc.nvrtcGetErrorString.argtypes = [ctypes.c_int]


def _read_program_log(prog: _nvrtcProgram) -> str:
    log_size = ctypes.c_size_t(0)
    rc = _libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    if rc != 0 or log_size.value == 0:
        return ""
    buf = ctypes.create_string_buffer(log_size.value)
    rc = _libnvrtc.nvrtcGetProgramLog(prog, buf)
    if rc != 0:
        return ""
    return buf.value.decode(errors="replace")


def _check_nvrtc(status: int, prog: _nvrtcProgram | None = None) -> None:
    if status == 0:
        return
    err_bytes = _libnvrtc.nvrtcGetErrorString(status)
    err = err_bytes.decode() if err_bytes else f"unknown ({status})"
    log = _read_program_log(prog) if prog is not None else ""
    raise RuntimeError(f"NVRTC error ({status}): {err}\n{log}")


# ---------------------------------------------------------------------------
# Driver API bindings
# ---------------------------------------------------------------------------
_CUdevice = ctypes.c_int
_CUcontext = ctypes.c_void_p
_CUmodule = ctypes.c_void_p
_CUfunction = ctypes.c_void_p
_CUstream = ctypes.c_void_p
_CUevent = ctypes.c_void_p

# CUdevice_attribute enum
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

_libcuda.cuInit.restype = ctypes.c_int
_libcuda.cuInit.argtypes = [ctypes.c_uint]

_libcuda.cuDeviceGet.restype = ctypes.c_int
_libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(_CUdevice), ctypes.c_int]

_libcuda.cuDeviceGetAttribute.restype = ctypes.c_int
_libcuda.cuDeviceGetAttribute.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    _CUdevice,
]

_libcuda.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
_libcuda.cuDevicePrimaryCtxRetain.argtypes = [
    ctypes.POINTER(_CUcontext),
    _CUdevice,
]

_libcuda.cuCtxSetCurrent.restype = ctypes.c_int
_libcuda.cuCtxSetCurrent.argtypes = [_CUcontext]

_libcuda.cuCtxGetCurrent.restype = ctypes.c_int
_libcuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(_CUcontext)]

_libcuda.cuModuleLoadData.restype = ctypes.c_int
_libcuda.cuModuleLoadData.argtypes = [
    ctypes.POINTER(_CUmodule),
    ctypes.c_void_p,
]

_libcuda.cuModuleGetFunction.restype = ctypes.c_int
_libcuda.cuModuleGetFunction.argtypes = [
    ctypes.POINTER(_CUfunction),
    _CUmodule,
    ctypes.c_char_p,
]

_libcuda.cuLaunchKernel.restype = ctypes.c_int
_libcuda.cuLaunchKernel.argtypes = [
    _CUfunction,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    _CUstream,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
]

_libcuda.cuEventCreate.restype = ctypes.c_int
_libcuda.cuEventCreate.argtypes = [ctypes.POINTER(_CUevent), ctypes.c_uint]

_libcuda.cuEventRecord.restype = ctypes.c_int
_libcuda.cuEventRecord.argtypes = [_CUevent, _CUstream]

_libcuda.cuEventSynchronize.restype = ctypes.c_int
_libcuda.cuEventSynchronize.argtypes = [_CUevent]

_libcuda.cuEventElapsedTime.restype = ctypes.c_int
_libcuda.cuEventElapsedTime.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    _CUevent,
    _CUevent,
]

_libcuda.cuEventDestroy_v2.restype = ctypes.c_int
_libcuda.cuEventDestroy_v2.argtypes = [_CUevent]

_libcuda.cuGetErrorString.restype = ctypes.c_int
_libcuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]


def _check_cu(status: int) -> None:
    if status == 0:
        return
    err = ctypes.c_char_p()
    _libcuda.cuGetErrorString(status, ctypes.byref(err))
    msg = err.value.decode() if err.value else f"unknown ({status})"
    raise RuntimeError(f"CUDA driver error ({status}): {msg}")


# ---------------------------------------------------------------------------
# Lazy primary-context attach (called on first compile / launch / timer)
# ---------------------------------------------------------------------------
_init_lock = threading.Lock()
_initialized = False
_compute_capability: tuple[int, int] | None = None


def _ensure_initialized() -> None:
    global _initialized, _compute_capability
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        _check_cu(_libcuda.cuInit(0))
        device_id = int(os.environ.get("GRADTUITY_DEVICE", "0"))
        device = _CUdevice()
        _check_cu(_libcuda.cuDeviceGet(ctypes.byref(device), device_id))
        # Primary context — same one cudaMalloc implicitly creates. Retaining
        # auto-initializes if needed; setting current binds this thread.
        ctx = _CUcontext()
        _check_cu(_libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device))
        _check_cu(_libcuda.cuCtxSetCurrent(ctx))
        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        _check_cu(
            _libcuda.cuDeviceGetAttribute(
                ctypes.byref(major),
                _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            )
        )
        _check_cu(
            _libcuda.cuDeviceGetAttribute(
                ctypes.byref(minor),
                _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            )
        )
        _compute_capability = (major.value, minor.value)
        _initialized = True


def compute_capability() -> tuple[int, int]:
    """Return (major, minor) compute capability of the active device."""
    _ensure_initialized()
    return _compute_capability  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Compile + launch
# ---------------------------------------------------------------------------
class KernelHandle:
    """Compiled+loaded kernel: holds the CUfunction passed to cuLaunchKernel."""

    __slots__ = ("func", "name")

    def __init__(self, func: int, name: str):
        self.func = func
        self.name = name


_CUBIN_CACHE_DIR = Path(
    os.environ.get(
        "GRADTUITY_CUBIN_CACHE_DIR",
        str(Path.home() / ".cache" / "gradtuity" / "cubin"),
    )
)

_kernel_cache: dict[tuple[str, str, str], KernelHandle] = {}
_compile_lock = threading.Lock()


def _arch_flag(arch: str | None) -> str:
    if arch is None:
        major, minor = compute_capability()
        return f"sm_{major}{minor}"
    return arch


def _compile_to_cubin(src: str, name: str, arch: str) -> bytes:
    """Compile CUDA source -> cubin (SASS) via NVRTC. Cached on disk by hash.

    We emit cubin instead of PTX so the driver doesn't have to JIT — that
    sidesteps version skew between NVRTC and the installed driver.
    """
    key = hashlib.sha256(f"{arch}\0{src}".encode()).hexdigest()
    _CUBIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CUBIN_CACHE_DIR / f"{key}.cubin"
    if cache_path.exists():
        return cache_path.read_bytes()

    prog = _nvrtcProgram()
    src_b = src.encode()
    name_b = f"{name}.cu".encode()
    _check_nvrtc(
        _libnvrtc.nvrtcCreateProgram(
            ctypes.byref(prog), src_b, name_b, 0, None, None
        )
    )
    try:
        opt_strs = [
            f"--gpu-architecture={arch}".encode(),
            b"--use_fast_math",
            b"-default-device",
        ]
        opts = (ctypes.c_char_p * len(opt_strs))(*opt_strs)
        _check_nvrtc(
            _libnvrtc.nvrtcCompileProgram(prog, len(opt_strs), opts),
            prog=prog,
        )
        cubin_size = ctypes.c_size_t(0)
        _check_nvrtc(
            _libnvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_size))
        )
        cubin_buf = ctypes.create_string_buffer(cubin_size.value)
        _check_nvrtc(_libnvrtc.nvrtcGetCUBIN(prog, cubin_buf))
        cubin = cubin_buf.raw[: cubin_size.value]
    finally:
        _libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    cache_path.write_bytes(cubin)
    return cubin


def compile_kernel(src: str, name: str, arch: str | None = None) -> KernelHandle:
    """
    Compile CUDA source via NVRTC, load as a module, return the function handle.

    `name` must match an `extern "C" __global__` symbol in `src`.
    Cached in-process and on disk by (arch, src) hash, so repeat calls are cheap.
    """
    _ensure_initialized()
    arch = _arch_flag(arch)
    cache_key = (src, name, arch)
    cached = _kernel_cache.get(cache_key)
    if cached is not None:
        return cached
    with _compile_lock:
        cached = _kernel_cache.get(cache_key)
        if cached is not None:
            return cached
        cubin = _compile_to_cubin(src, name, arch)
        module = _CUmodule()
        _check_cu(_libcuda.cuModuleLoadData(ctypes.byref(module), cubin))
        func = _CUfunction()
        _check_cu(
            _libcuda.cuModuleGetFunction(
                ctypes.byref(func), module, name.encode()
            )
        )
        handle = KernelHandle(func.value, name)
        _kernel_cache[cache_key] = handle
        return handle


def kernel_module(src: str):
    """Returns a getter that lazy-compiles named kernels from a shared source.

    Each ``gradtuity.kernels_cuda.*`` module instantiates one of these so the
    13 modules don't have to repeat the same lookup-and-cache boilerplate.
    """
    def get(name: str) -> KernelHandle:
        return compile_kernel(src, name)
    return get


# ---------------------------------------------------------------------------
# Argument typing helpers — Python ints and floats are ambiguous in C, so
# kernel call sites use these to declare the exact ABI type per arg.
# ---------------------------------------------------------------------------
class KernelArg:
    __slots__ = ("box",)

    def __init__(self, box):
        self.box = box


def ptr(p: int) -> KernelArg:
    """Device pointer (int from cuda_malloc) -> void*."""
    return KernelArg(ctypes.c_void_p(p))


def i32(x: int) -> KernelArg:
    return KernelArg(ctypes.c_int(x))


def i64(x: int) -> KernelArg:
    return KernelArg(ctypes.c_longlong(x))


def u32(x: int) -> KernelArg:
    return KernelArg(ctypes.c_uint(x))


def f32(x: float) -> KernelArg:
    return KernelArg(ctypes.c_float(x))


def _pack_args(args: Sequence) -> tuple[ctypes.Array, list]:
    keepalive = []
    addrs = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        if isinstance(a, KernelArg):
            box = a.box
        elif isinstance(a, bool):
            box = ctypes.c_int(int(a))
        elif isinstance(a, int):
            box = ctypes.c_int(a)
        elif isinstance(a, float):
            box = ctypes.c_float(a)
        else:
            raise TypeError(
                f"Unsupported kernel arg at index {i}: {type(a).__name__}. "
                "Use ptr(), i32(), i64(), f32(), etc. to disambiguate."
            )
        keepalive.append(box)
        addrs[i] = ctypes.cast(ctypes.pointer(box), ctypes.c_void_p)
    return addrs, keepalive


_DEBUG_SYNC = os.environ.get("GRADTUITY_CUDA_LAUNCH_DEBUG") == "1"


def launch(
    handle: KernelHandle,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: Sequence,
    shared_bytes: int = 0,
    stream: int = 0,
) -> None:
    """Launch a compiled kernel via cuLaunchKernel.

    If ``GRADTUITY_CUDA_LAUNCH_DEBUG=1`` is set, synchronously fault-check
    after each launch — useful for localizing async kernel errors.
    """
    _ensure_initialized()
    addrs, _keep = _pack_args(args)
    _check_cu(
        _libcuda.cuLaunchKernel(
            handle.func,
            ctypes.c_uint(grid[0]),
            ctypes.c_uint(grid[1]),
            ctypes.c_uint(grid[2]),
            ctypes.c_uint(block[0]),
            ctypes.c_uint(block[1]),
            ctypes.c_uint(block[2]),
            ctypes.c_uint(shared_bytes),
            ctypes.c_void_p(stream),
            addrs,
            None,
        )
    )
    if _DEBUG_SYNC:
        from .cuda_mem import cuda_device_synchronize
        try:
            cuda_device_synchronize()
        except Exception as e:
            raise RuntimeError(f"CUDA kernel {handle.name!r} failed: {e}") from None


# ---------------------------------------------------------------------------
# Event timing
# ---------------------------------------------------------------------------
class EventTimer:
    """
    Records a CUDA-event pair. After `__exit__`, `.elapsed_ms` holds the
    measured kernel-side duration (millisecond resolution from cuEvents).

    Usage:
        with EventTimer() as t:
            launch(...)
        print(t.elapsed_ms)
    """

    def __init__(self, stream: int = 0):
        self.stream = stream
        self._start = _CUevent()
        self._stop = _CUevent()
        self.elapsed_ms: float | None = None

    def __enter__(self) -> "EventTimer":
        _ensure_initialized()
        _check_cu(_libcuda.cuEventCreate(ctypes.byref(self._start), 0))
        _check_cu(_libcuda.cuEventCreate(ctypes.byref(self._stop), 0))
        _check_cu(
            _libcuda.cuEventRecord(self._start, ctypes.c_void_p(self.stream))
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _check_cu(
            _libcuda.cuEventRecord(self._stop, ctypes.c_void_p(self.stream))
        )
        _check_cu(_libcuda.cuEventSynchronize(self._stop))
        ms = ctypes.c_float(0.0)
        _check_cu(
            _libcuda.cuEventElapsedTime(
                ctypes.byref(ms), self._start, self._stop
            )
        )
        self.elapsed_ms = ms.value
        _libcuda.cuEventDestroy_v2(self._start)
        _libcuda.cuEventDestroy_v2(self._stop)
