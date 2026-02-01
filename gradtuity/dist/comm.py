"""
Process group and NCCL communicator: init, destroy, allreduce.

Single-node: rank 0 creates UniqueId and writes to a file; other ranks
poll until file exists then read. All ranks call ncclCommInitRank.
Optional barrier (allreduce 1 float) after init for robustness.
"""

from __future__ import annotations

import os
import time

from . import env as dist_env
from . import nccl
from .. import cuda_mem


# Module-level communicator (set by init_process_group, cleared by destroy)
_comm: int | None = None


def _unique_id_file() -> str:
    """Path for file-based UniqueId exchange (single-node)."""
    # Tests can set GRADTUITY_NCCL_UID_FILE to a shared path for multi-process tests
    path = os.environ.get("GRADTUITY_NCCL_UID_FILE")
    if path is not None:
        return path
    # SLURM_JOB_ID is set by Slurm, so use it for a shared path.
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        return f"/tmp/gradtuity_nccl_{job_id}.bin"
    # Otherwise use a unique path based on PID and timestamp.
    return f"/tmp/gradtuity_nccl_{os.getpid()}_{int(time.time())}.bin"


def init_process_group() -> None:
    """
    Initialize distributed process group: set CUDA device from local rank,
    exchange NCCL UniqueId (file-based), create NCCL communicator.

    Call once at startup. Reads rank/size from env (RANK/WORLD_SIZE or
    SLURM_*). Single-node only: rank 0 writes UniqueId to a file, others
    read it.
    """
    global _comm
    if _comm is not None:
        return

    rank = dist_env.get_rank()
    world_size = dist_env.get_world_size()
    local_rank = dist_env.get_local_rank()

    cuda_mem.cuda_set_device(local_rank)

    if world_size == 1:
        _comm = None
        return

    uid_file = _unique_id_file()

    if rank == 0:
        comm_id = nccl.nccl_get_unique_id()
        if len(comm_id) != nccl.NCCL_UNIQUE_ID_BYTES:
            raise RuntimeError(
                f"nccl_get_unique_id returned {len(comm_id)} bytes, expected {nccl.NCCL_UNIQUE_ID_BYTES}"
            )
        with open(uid_file, "wb") as f:
            f.write(comm_id)
            f.flush()
            os.fsync(f.fileno())
    else:
        # Wait until rank 0 has written the full UniqueId (file exists and has correct size)
        wait_start = time.monotonic()
        wait_timeout = 60.0
        while not os.path.isfile(uid_file) or os.path.getsize(uid_file) != nccl.NCCL_UNIQUE_ID_BYTES:
            if time.monotonic() - wait_start > wait_timeout:
                raise RuntimeError(
                    f"Rank {rank}: UniqueId file {uid_file!r} did not reach {nccl.NCCL_UNIQUE_ID_BYTES} bytes "
                    f"within {wait_timeout}s (file exists={os.path.isfile(uid_file)}, "
                    f"size={os.path.getsize(uid_file) if os.path.isfile(uid_file) else 0}). "
                    "Ensure rank 0 is running and GRADTUITY_NCCL_UID_FILE is the same on all ranks."
                )
            time.sleep(0.01)
        with open(uid_file, "rb") as f:
            comm_id = f.read()
        if len(comm_id) != nccl.NCCL_UNIQUE_ID_BYTES:
            raise RuntimeError(
                f"Rank {rank}: read {len(comm_id)} bytes from {uid_file!r}, expected {nccl.NCCL_UNIQUE_ID_BYTES}. "
                "File may have been truncated or not yet fully written."
            )
    if len(comm_id) != nccl.NCCL_UNIQUE_ID_BYTES:
        raise RuntimeError(
            f"UniqueId must be {nccl.NCCL_UNIQUE_ID_BYTES} bytes, got {len(comm_id)}"
        )

    _comm = nccl.nccl_comm_init_rank(world_size, comm_id, rank)

    if rank == 0:
        try:
            os.remove(uid_file)
        except OSError:
            pass

    # Optional barrier: allreduce 1 float so everyone is in sync
    barrier()


def destroy_process_group() -> None:
    """Destroy NCCL communicator and clear module state."""
    global _comm
    if _comm is not None:
        nccl.nccl_comm_destroy(_comm)
        _comm = None


def allreduce_inplace(ptr: int, numel: int) -> None:
    """
    In-place AllReduce (sum) of numel float32 elements at ptr.

    All ranks must call with same ptr layout. Sync device before calling
    so all prior kernels are done; uses default stream.
    """
    if _comm is None:
        world_size = dist_env.get_world_size()
        if world_size == 1:
            return
        raise RuntimeError("Process group not initialized; call init() first")
    cuda_mem.cuda_device_synchronize()
    nccl._nccl_all_reduce_inplace(_comm, ptr, numel, None)
    cuda_mem.cuda_device_synchronize()


def broadcast(ptr: int, numel: int, src: int = 0) -> None:
    """
    In-place Broadcast of numel float32 elements at ptr from src rank to all ranks.

    All ranks must call with same numel. Sync device before calling; uses default stream.
    """
    world_size = dist_env.get_world_size()
    if world_size == 1:
        return
    if _comm is None:
        raise RuntimeError("Process group not initialized; call init() first")
    cuda_mem.cuda_device_synchronize()
    nccl._nccl_broadcast_inplace(_comm, ptr, numel, src, None)
    cuda_mem.cuda_device_synchronize()


def get_comm() -> int | None:
    """Return current NCCL communicator handle or None."""
    return _comm


def barrier() -> None:
    """Block until all ranks reach this point (allreduce on 1 float)."""
    world_size = dist_env.get_world_size()
    if world_size == 1:
        return
    if _comm is None:
        raise RuntimeError("Process group not initialized; call init() first")
    # Use a tiny device buffer: allocate 4 bytes, fill with 0, allreduce sum, free
    ptr = cuda_mem.cuda_malloc(4)
    try:
        cuda_mem.cuda_memset(ptr, 0, 4)
        cuda_mem.cuda_device_synchronize()
        nccl._nccl_all_reduce_inplace(_comm, ptr, 1, None)
        cuda_mem.cuda_device_synchronize()
    finally:
        cuda_mem.cuda_free(ptr)
