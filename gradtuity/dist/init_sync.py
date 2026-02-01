"""
One-time synchronization of initial model parameters (and buffers) from a source rank.

Broadcasts from rank src (default 0) so all ranks start with identical parameters,
avoiding RNG desync during model creation. Call after model construction and before
optimizer creation.
"""

from __future__ import annotations

import os
import struct

from . import comm
from . import env as dist_env
from .. import cuda_mem

# Signature is 2 x int64 = 16 bytes = 4 float32 elements for NCCL broadcast
_SIGNATURE_NUMEL = 4
_SIGNATURE_NBYTES = 16


def _extract_params_and_buffers(model_or_params, sync_buffers: bool) -> tuple[list, list]:
    """Return (params, buffers) from a model or a list/tuple of tensors."""
    if isinstance(model_or_params, (list, tuple)):
        params = list(model_or_params)
        buffers: list = []
        return params, buffers
    params = list(model_or_params.parameters())
    buffers = (
        list(model_or_params.buffers())
        if sync_buffers and hasattr(model_or_params, "buffers")
        else []
    )
    return params, buffers


def init_sync(
    model_or_params,
    *,
    src: int = 0,
    sync_buffers: bool = True,
    bucket_mb: float = 25.0,  # reserved for future bucketing; v1 ignored
    strict: bool = True,
) -> None:
    """
    One-time synchronization of initial parameters (and buffers) from src rank.

    Broadcasts from rank src to all ranks so initial model state is bitwise identical.
    Call after init(), after model construction, and before optimizer creation.

    Args:
        model_or_params: A model (with parameters() and optionally buffers()) or a
            list/tuple of Tensors (parameters only).
        src: Rank that sends data (default 0).
        sync_buffers: If True and model has buffers(), broadcast buffers too.
        bucket_mb: Reserved for future bucketing; ignored in v1.
        strict: If True, run preflight check that all ranks have same num_tensors
            and total_numel; fail fast with clear error on mismatch.
    """
    _ = bucket_mb  # reserved
    world_size = dist_env.get_world_size()
    if world_size == 1:
        return

    rank = dist_env.get_rank()
    if not (0 <= src < world_size):
        raise ValueError(
            f"init_sync src must be in [0, {world_size}), got {src}"
        )

    params, buffers = _extract_params_and_buffers(model_or_params, sync_buffers)
    all_tensors = params + buffers

    num_tensors = len(all_tensors)
    total_numel = sum(t.numel for t in all_tensors)

    if os.environ.get("GRADTUITY_DIST_DEBUG_INIT_SYNC") == "1" and rank == 0:
        print(f"init_sync: broadcasting {num_tensors} params (total numel {total_numel})")

    if strict:
        signature_bytes = struct.pack("qq", num_tensors, total_numel)
        assert len(signature_bytes) == _SIGNATURE_NBYTES
        sig_ptr = cuda_mem.cuda_malloc(_SIGNATURE_NBYTES)
        try:
            if rank == src:
                cuda_mem.cuda_memcpy_htod(sig_ptr, signature_bytes)
            comm.broadcast(sig_ptr, _SIGNATURE_NUMEL, src=src)
            received = cuda_mem.cuda_memcpy_dtoh(sig_ptr, _SIGNATURE_NBYTES)
            root_num_tensors, root_total_numel = struct.unpack("qq", received)
            if num_tensors != root_num_tensors or total_numel != root_total_numel:
                raise RuntimeError(
                    f"init_sync strict check failed: rank {rank} has num_tensors={num_tensors} "
                    f"total_numel={total_numel}, root has num_tensors={root_num_tensors} "
                    f"total_numel={root_total_numel}. Ensure same model architecture and "
                    "parameter order on all ranks."
                )
            if os.environ.get("GRADTUITY_DIST_DEBUG_INIT_SYNC") == "1":
                print(f"init_sync [rank {rank}] signature ok: {num_tensors} tensors, {total_numel} numel")
        finally:
            cuda_mem.cuda_free(sig_ptr)

    for t in all_tensors:
        comm.broadcast(t.ptr, t.numel, src=src)
