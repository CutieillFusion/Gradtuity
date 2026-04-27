"""
Gradient synchronization for data parallel: bucketing + AllReduce.

sync_grads(params, bucket_mb): ensure every param has grad, pack grads into
buckets (up to bucket_mb MB), AllReduce each bucket (sum), scale by 1/world_size,
scatter back to param.grad. All ranks must pass the same params in the same order.
"""

from __future__ import annotations

import os

from .. import cuda_mem
from ..tensor import ensure_grad
from ..tensor import BLOCK, grid1d
from ..kernels import mul_scalar_kernel
from . import comm
from . import env as dist_env

# Number of bytes per float32
F32 = 4


def sync_grads(params: list, bucket_mb: float = 25) -> None:
    """
    Synchronize gradients across ranks: AllReduce (sum) then scale by 1/world_size.

    Every parameter must have a grad tensor (zeros if missing). Params must be
    in the same order on all ranks (e.g. model.parameters() with deterministic
    order). Uses bucketing to reduce AllReduce launch overhead.

    Tiny bucket mode: set bucket_mb very small (e.g. 0.001) or set env
    GRADTUITY_BUCKET_MB (e.g. 0.001) to force one param per AllReduce and
    isolate param-order/packing issues when debugging explosions.

    Args:
        params: List of Tensor (model parameters) in fixed order.
        bucket_mb: Max bucket size in MB (e.g. 25). Overridden by GRADTUITY_BUCKET_MB if set.
    """
    world_size = dist_env.get_world_size()
    if world_size == 1:
        return

    env_bucket_mb = os.environ.get("GRADTUITY_BUCKET_MB")
    if env_bucket_mb is not None:
        bucket_mb = float(env_bucket_mb)

    for p in params:
        ensure_grad(p)

    bucket_max_bytes = int(bucket_mb * 1024 * 1024)
    # Build list of (grad_ptr, nbytes) in param order
    entries: list[tuple[int, int]] = []
    for p in params:
        nbytes = p.grad.nbytes
        assert nbytes % 4 == 0, f"grad nbytes must be multiple of 4 (float32), got {nbytes}"
        entries.append((p.grad.ptr, nbytes))

    # Bucketize: group consecutive entries until we'd exceed bucket_max_bytes
    buckets: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    current_bytes = 0
    for (ptr, nbytes) in entries:
        if current_bytes + nbytes > bucket_max_bytes and current:
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append((ptr, nbytes))
        current_bytes += nbytes
    if current:
        buckets.append(current)

    cuda_mem.cuda_device_synchronize()

    for bucket_entries in buckets:
        bucket_bytes = sum(n for _, n in bucket_entries)
        assert bucket_bytes % 4 == 0, f"bucket_bytes must be multiple of 4, got {bucket_bytes}"
        bucket_numel = bucket_bytes // F32
        assert bucket_numel * F32 == bucket_bytes, (
            f"bucket truncation: bucket_numel*F32={bucket_numel * F32} != bucket_bytes={bucket_bytes}"
        )
        if bucket_numel == 0:
            continue
        buf_ptr = cuda_mem.cuda_malloc(bucket_bytes)
        try:
            # Pack: copy each grad slice into buffer
            offset = 0
            for (grad_ptr, nbytes) in bucket_entries:
                cuda_mem.cuda_memcpy_dtod(buf_ptr + offset, grad_ptr, nbytes)
                offset += nbytes
            comm.allreduce_inplace(buf_ptr, bucket_numel)
            cuda_mem.cuda_device_synchronize()
            # Scale by 1/world_size
            scale = 1.0 / world_size
            mul_scalar_kernel[grid1d(bucket_numel)](
                buf_ptr, scale, buf_ptr, bucket_numel, BLOCK=BLOCK
            )
            # Unpack: copy buffer back to grads
            offset = 0
            for (grad_ptr, nbytes) in bucket_entries:
                cuda_mem.cuda_memcpy_dtod(grad_ptr, buf_ptr + offset, nbytes)
                offset += nbytes
        finally:
            cuda_mem.cuda_free(buf_ptr)

    cuda_mem.cuda_device_synchronize()
