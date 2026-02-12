"""
Distributed training: single-node data parallel (NCCL DDP).

Usage:
    from gradtuity.dist import init, sync_grads, get_rank, get_world_size

    init()
    for step in range(steps):
        loss = model(x).cross_entropy(y)
        loss.backward()
        sync_grads(model.parameters(), bucket_mb=25)
        optimizer.step()
        optimizer.zero_grad()
"""

from __future__ import annotations

from . import env as _env
from . import comm
from . import ddp
from . import init_sync as _init_sync
from . import sampler

# Env
get_rank = _env.get_rank
get_world_size = _env.get_world_size
get_local_rank = _env.get_local_rank
get_master_addr = _env.get_master_addr
get_master_port = _env.get_master_port
get_env_ranks = _env.get_env_ranks

# Process group
def init() -> None:
    """Initialize process group (device + NCCL). Call once at startup."""
    comm.init_process_group()


def destroy_process_group() -> None:
    """Destroy NCCL communicator."""
    comm.destroy_process_group()


def barrier() -> None:
    """Block until all ranks reach this point."""
    comm.barrier()


def sync_grads(params: list, bucket_mb: float = 25) -> None:
    """AllReduce gradients and scale by 1/world_size. Params must be in same order on all ranks."""
    ddp.sync_grads(params, bucket_mb=bucket_mb)


def init_sync(model_or_params, *, src: int = 0, sync_buffers: bool = True, bucket_mb: float = 25.0, strict: bool = True) -> None:
    """One-time sync of initial parameters (and buffers) from src rank. Call after model build, before optimizer."""
    _init_sync.init_sync(model_or_params, src=src, sync_buffers=sync_buffers, bucket_mb=bucket_mb, strict=strict)


# Sampler
distributed_indices = sampler.distributed_indices
shard_size = sampler.shard_size

__all__ = [
    "init",
    "destroy_process_group",
    "sync_grads",
    "init_sync",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "get_master_addr",
    "get_master_port",
    "get_env_ranks",
    "barrier",
    "distributed_indices",
    "shard_size",
]
