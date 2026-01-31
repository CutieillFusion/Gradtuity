"""
SafeTensors-compatible save/load for GPU Tensor (stdlib-only, no safetensors dep).

Saves/loads CUDA-backed float32 tensors to .safetensors files that are readable
by ecosystem tooling. Uses only struct, json, io, os and cuda_mem.
"""

from __future__ import annotations

import json
import math
import os
import struct
from .cuda_mem import cuda_memcpy_dtoh, cuda_memcpy_htod
from .tensor import Tensor, alloc_storage

# -------------------------------------------------------------------------
# Constants and helpers
# -------------------------------------------------------------------------

_MAX_HEADER_BYTES = 100 * 1024 * 1024  # 100 MiB DoS guard
F32_BYTES = 4

_DEFAULT_METADATA = {
    "format": "gradtuity-safetensors-compat",
    "tensor_version": "1",
    "endianness": "little",
    "layout": "C",
    "note": "float32 contiguous cuda tensors",
}


def _prod(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _validate_tensor_for_save(t: Tensor, name: str) -> None:
    if t.ndim not in (1, 2, 3, 4):
        raise ValueError(
            f"rank must be 1..4; dims must be > 0 (tensor {name!r} has ndim={t.ndim})"
        )
    for i, d in enumerate(t.shape):
        if d <= 0:
            raise ValueError(
                f"rank must be 1..4; dims must be > 0 (tensor {name!r} dim {i}={d})"
            )
    if t.numel * F32_BYTES != t.nbytes:
        raise ValueError(
            f"tensor byte size does not match shape*dtype (tensor {name!r}: "
            f"numel*4={t.numel * F32_BYTES} != nbytes={t.nbytes})"
        )


def _validate_key_utf8(name: str) -> None:
    name.encode("utf-8")


# -------------------------------------------------------------------------
# Writer
# -------------------------------------------------------------------------


def save_safetensors(
    path: str,
    tensors: dict[str, Tensor],
    metadata: dict[str, str] | None = None,
    include_grads: bool = False,
) -> None:
    """
    Save tensors to a SafeTensors-compatible file.

    Args:
        path: Output file path (.safetensors).
        tensors: Name -> Tensor mapping. All tensors must be float32, contiguous, rank 1-4.
        metadata: Optional string->string metadata merged into __metadata__.
        include_grads: If True, for each tensor with t.grad set, save an extra key "{name}.grad".

    Raises:
        ValueError: If a tensor or key is invalid.
    """
    if not tensors:
        raise ValueError("tensors dict must not be empty")

    # Validate keys: must be strings (and UTF-8 encodable)
    for name in tensors:
        if not isinstance(name, str):
            raise ValueError(
                f"tensor keys must be strings, got {type(name).__name__} for key {name!r}"
            )
        _validate_key_utf8(name)
    # Grad key collision: include_grads would add "k.grad"; reject if already present
    if include_grads:
        for k in tensors:
            grad_key = f"{k}.grad"
            if grad_key in tensors:
                raise ValueError(
                    f"include_grads=True: key {grad_key!r} would collide with grad of {k!r}; "
                    "do not pass .grad keys explicitly when include_grads=True"
                )

    # Validate tensors
    for name in tensors:
        _validate_tensor_for_save(tensors[name], name)
        if include_grads and tensors[name].grad is not None:
            g = tensors[name].grad
            _validate_tensor_for_save(g, f"{name}.grad")

    # Build ordered list of (name, tensor) for deterministic layout
    entries: list[tuple[str, Tensor]] = []
    for k in sorted(tensors.keys()):
        entries.append((k, tensors[k]))
        if include_grads and tensors[k].grad is not None:
            entries.append((f"{k}.grad", tensors[k].grad))

    # Compute offsets (no holes)
    running_offset = 0
    header_tensors: dict[str, dict] = {}
    for name, t in entries:
        nbytes = t.nbytes
        begin = running_offset
        end = begin + nbytes
        running_offset = end
        header_tensors[name] = {
            "dtype": "F32",
            "shape": list(t.shape),
            "data_offsets": [begin, end],
        }

    # Build metadata
    meta = dict(_DEFAULT_METADATA)
    if metadata:
        for k, v in metadata.items():
            if not isinstance(v, str):
                raise ValueError(f"metadata values must be strings, got {type(v).__name__} for key {k!r}")
            meta[k] = v

    header = {"__metadata__": meta, **header_tensors}

    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError(f"header size {len(header_bytes)} exceeds max {_MAX_HEADER_BYTES}")

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for _name, t in entries:
            host_bytes = cuda_memcpy_dtoh(t.ptr, t.nbytes)
            f.write(host_bytes)


# -------------------------------------------------------------------------
# Reader
# -------------------------------------------------------------------------


def load_safetensors(
    path: str,
    *,
    device: str = "cuda",
    requires_grad: bool = False,
) -> dict[str, Tensor]:
    """
    Load tensors from a SafeTensors-compatible file into GPU memory.

    Args:
        path: Path to .safetensors file.
        device: Must be "cuda" (only supported device in v1).
        requires_grad: Applied to all returned tensors.

    Returns:
        Dict mapping tensor names to Tensor (on GPU).

    Raises:
        ValueError: If file is corrupt, unsupported, or device is not "cuda".
    """
    if device != "cuda":
        raise ValueError("Only device='cuda' supported")

    file_size = os.stat(path).st_size
    if file_size < 8:
        raise ValueError("corrupt or unsupported safetensors file: file too small for header length")

    with open(path, "rb") as f:
        n_bytes = f.read(8)
        if len(n_bytes) != 8:
            raise ValueError("corrupt or unsupported safetensors file: truncated header length")
        N = struct.unpack("<Q", n_bytes)[0]

        if N == 0:
            raise ValueError("corrupt or unsupported safetensors file: header length is 0")
        if N > _MAX_HEADER_BYTES:
            raise ValueError(
                f"corrupt or unsupported safetensors file: header length {N} exceeds max {_MAX_HEADER_BYTES}"
            )

        header_bytes = f.read(N)
        if len(header_bytes) != N:
            raise ValueError("corrupt or unsupported safetensors file: truncated header")

    if not header_bytes.lstrip().startswith(b"{"):
        raise ValueError("corrupt or unsupported safetensors file: header must start with {")

    try:
        header_str = header_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(
            "corrupt or unsupported safetensors file: invalid UTF-8 in header"
        ) from e
    try:
        header = json.loads(header_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"corrupt or unsupported safetensors file: invalid JSON ({e})") from e

    if not isinstance(header, dict):
        raise ValueError("corrupt or unsupported safetensors file: header must be a JSON object")

    # Validate __metadata__
    if "__metadata__" in header:
        meta = header["__metadata__"]
        if not isinstance(meta, dict):
            raise ValueError("corrupt or unsupported safetensors file: __metadata__ must be an object")
        for k, v in meta.items():
            if not isinstance(v, str):
                raise ValueError(
                    f"corrupt or unsupported safetensors file: __metadata__ values must be strings (key {k!r})"
                )

    data_start = 8 + N
    data_size = file_size - data_start
    if data_size < 0:
        raise ValueError("corrupt or unsupported safetensors file: negative data size")

    # Collect tensor entries (exclude __metadata__)
    tensor_entries: list[tuple[str, dict]] = []
    for key, val in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(val, dict):
            raise ValueError(f"corrupt or unsupported safetensors file: tensor entry {key!r} must be an object")
        if "dtype" not in val or "shape" not in val or "data_offsets" not in val:
            raise ValueError(
                f"corrupt or unsupported safetensors file: tensor entry {key!r} missing dtype/shape/data_offsets"
            )
        dtype = val["dtype"]
        shape = val["shape"]
        data_offsets = val["data_offsets"]
        if dtype != "F32":
            raise ValueError(f"Only F32 supported, found {dtype!r}")
        if not isinstance(shape, list) or len(shape) not in (1, 2, 3, 4):
            raise ValueError("rank must be 1..4; dims must be > 0")
        for i, d in enumerate(shape):
            if not isinstance(d, int) or d <= 0:
                raise ValueError("rank must be 1..4; dims must be > 0")
        if not isinstance(data_offsets, (list, tuple)) or len(data_offsets) != 2:
            raise ValueError(f"corrupt or unsupported safetensors file: data_offsets must be [BEGIN, END] for {key!r}")
        begin, end = int(data_offsets[0]), int(data_offsets[1])
        if begin < 0 or end < begin:
            raise ValueError("corrupt or unsupported safetensors file: offset outside data buffer")
        expected_bytes = _prod(tuple(shape)) * F32_BYTES
        if end - begin != expected_bytes:
            raise ValueError("tensor byte size does not match shape*dtype")
        if begin >= data_size or end > data_size:
            raise ValueError("corrupt or unsupported safetensors file: offset outside data buffer")
        tensor_entries.append((key, {"shape": tuple(shape), "begin": begin, "end": end}))

    # Sort by begin and check no holes / no overlap
    tensor_entries.sort(key=lambda x: x[1]["begin"])
    prev_end = 0
    for key, info in tensor_entries:
        begin, end = info["begin"], info["end"]
        if begin != prev_end:
            if begin < prev_end:
                raise ValueError(
                    "corrupt or unsupported safetensors file: offsets overlap"
                )
            raise ValueError(
                "corrupt or unsupported safetensors file: offsets contain holes"
            )
        prev_end = end
    if prev_end != data_size:
        raise ValueError(
            "corrupt or unsupported safetensors file: offsets contain holes"
        )

    # Load each tensor into GPU
    result: dict[str, Tensor] = {}
    with open(path, "rb") as f:
        for key, info in tensor_entries:
            begin, end = info["begin"], info["end"]
            shape = info["shape"]
            expected_bytes = end - begin
            f.seek(data_start + begin)
            host_bytes = f.read(expected_bytes)
            if len(host_bytes) != expected_bytes:
                raise ValueError("corrupt or unsupported safetensors file: truncated data")
            st = alloc_storage(expected_bytes, zero=False)
            cuda_memcpy_htod(st.ptr, host_bytes)
            t = Tensor._wrap(st, shape, requires_grad=requires_grad, name=key)
            result[key] = t

    return result
