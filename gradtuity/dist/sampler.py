"""
Distributed data sharding: indices for this rank's shard.

Simple offset/stride or index range so each rank sees different data
in single-node data parallel training.
"""

from __future__ import annotations

from . import env as dist_env


def distributed_indices(num_samples: int) -> range:
    """
    Return the range of sample indices for this rank (shard).

    Rank i gets indices i, i + world_size, i + 2*world_size, ...
    So each rank has roughly num_samples / world_size indices.

    Args:
        num_samples: Total number of samples in the dataset.

    Returns:
        range(rank, num_samples, world_size)
    """
    rank = dist_env.get_rank()
    world_size = dist_env.get_world_size()
    return range(rank, num_samples, world_size)


def shard_size(num_samples: int) -> int:
    """
    Return the number of samples this rank's shard has.

    Args:
        num_samples: Total number of samples.

    Returns:
        len(distributed_indices(num_samples))
    """
    return len(distributed_indices(num_samples))
