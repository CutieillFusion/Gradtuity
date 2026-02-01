"""
Distributed training environment: rank, world size, and master address.

Reads from explicit env vars (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR,
MASTER_PORT) or from Slurm (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID,
SLURM_JOB_ID). Used by init_process_group for single-node multi-GPU.
"""

from __future__ import annotations

import os


def get_rank() -> int:
    """
    Return global rank (0 .. world_size - 1).

    Prefers RANK; falls back to SLURM_PROCID.
    Raises RuntimeError if neither is set.
    """
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank)
    rank = os.environ.get("SLURM_PROCID")
    if rank is not None:
        return int(rank)
    raise RuntimeError(
        "Distributed rank not set. Set RANK or SLURM_PROCID."
    )


def get_world_size() -> int:
    """
    Return number of processes in the job.

    Prefers WORLD_SIZE; falls back to SLURM_NTASKS.
    Raises RuntimeError if neither is set.
    """
    size = os.environ.get("WORLD_SIZE")
    if size is not None:
        return int(size)
    size = os.environ.get("SLURM_NTASKS")
    if size is not None:
        return int(size)
    raise RuntimeError(
        "World size not set. Set WORLD_SIZE or SLURM_NTASKS."
    )


def get_local_rank() -> int:
    """
    Return rank within the node (0 .. local_size - 1).

    Prefers LOCAL_RANK; falls back to SLURM_LOCALID.
    Raises RuntimeError if neither is set.
    """
    local = os.environ.get("LOCAL_RANK")
    if local is not None:
        return int(local)
    local = os.environ.get("SLURM_LOCALID")
    if local is not None:
        return int(local)
    raise RuntimeError(
        "Local rank not set. Set LOCAL_RANK or SLURM_LOCALID."
    )


def get_master_addr() -> str:
    """
    Return master address for rendezvous (e.g. UniqueId exchange).

    Requires MASTER_ADDR to be set.
    """
    addr = os.environ.get("MASTER_ADDR")
    if addr is not None:
        return addr
    raise RuntimeError(
        "Master address not set. Set MASTER_ADDR."
    )


def get_master_port() -> str:
    """
    Return master port for rendezvous.

    Requires MASTER_PORT to be set.
    """
    port = os.environ.get("MASTER_PORT")
    if port is not None:
        return port
    raise RuntimeError(
        "Master port not set. Set MASTER_PORT."
    )


def get_env_ranks() -> tuple[int, int, int, str, str]:
    """
    Return (rank, world_size, local_rank, master_addr, master_port) from env.

    Uses get_rank(), get_world_size(), get_local_rank(), get_master_addr(),
    get_master_port(). Raises if any required env var is unset (e.g. when not
    launched by gradtuity.launch or Slurm).
    """
    return (
        get_rank(),
        get_world_size(),
        get_local_rank(),
        get_master_addr(),
        get_master_port(),
    )
