"""
Worker script for multi-process dist tests.

Run as: python -m gradtuity.tests.dist_worker MODE [--outfile PATH]

Modes:
  allreduce: init, 1 float set to rank+1, AllReduce (sum), write result.
  init_sync: RNG desync (rank 1 extra rand), same model, init_sync(model), write first param sample.
  init_sync_strict_fail: rank 0 Linear(4,2), rank 1 Linear(4,1), init_sync(strict=True) -> rank 1 raises.

Reads RANK, WORLD_SIZE, LOCAL_RANK from env (or GRADTUITY_NCCL_UID_FILE for UniqueId path).
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

# Add project root so gradtuity is importable when run as __main__
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from gradtuity import cuda_mem
from gradtuity.dist import comm, destroy_process_group, init, init_sync


def main_allreduce(outfile: str | None) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        val = 1.0
    else:
        init()
        ptr = cuda_mem.cuda_malloc(4)
        try:
            val = float(rank + 1)
            cuda_mem.cuda_memcpy_htod(ptr, struct.pack("f", val))
            comm.allreduce_inplace(ptr, 1)
            buf = cuda_mem.cuda_memcpy_dtoh(ptr, 4)
            val = struct.unpack("f", buf)[0]
        finally:
            cuda_mem.cuda_free(ptr)
        destroy_process_group()
    path = outfile or f"/tmp/gradtuity_dist_result_{rank}.txt"
    with open(path, "w") as f:
        f.write(str(val))
    print(path, val)


def main_init_sync(outfile: str | None) -> None:
    """RNG desync: rank 1 consumes extra RNG; both build same model, init_sync, write first param sample."""
    import numpy as np

    from gradtuity.nn import Linear

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        model = Linear(4, 2)
        first_param_sample = model.parameters()[0].to_list()
        path = outfile or f"/tmp/gradtuity_init_sync_{rank}.txt"
        with open(path, "w") as f:
            f.write(repr(first_param_sample[:8]))
        print(path)
        return

    init()
    try:
        if rank == 1:
            np.random.rand()  # desync RNG so rank 1 init differs from rank 0
        model = Linear(4, 2)
        init_sync(model)
        first_param = model.parameters()[0]
        first_param_sample = first_param.to_list()
        path = outfile or f"/tmp/gradtuity_init_sync_{rank}.txt"
        with open(path, "w") as f:
            f.write(repr(first_param_sample[:8]))
        print(path)
        # Sanity: barrier to confirm comm still healthy
        comm.barrier()
    except Exception as e:
        path = outfile or f"/tmp/gradtuity_init_sync_{rank}.txt"
        with open(path, "w") as f:
            f.write(f"ERROR:{e!r}")
        raise
    finally:
        destroy_process_group()


def main_init_sync_strict_fail(outfile: str | None) -> None:
    """Rank 0 Linear(4,2), rank 1 Linear(4,1); init_sync(strict=True) -> rank 1 raises."""
    from gradtuity.nn import Linear

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        path = outfile or f"/tmp/gradtuity_init_sync_strict_{rank}.txt"
        with open(path, "w") as f:
            f.write("SKIP:world_size=1")
        print(path)
        return

    init()
    try:
        if rank == 0:
            model = Linear(4, 2)
        else:
            model = Linear(4, 1)
        init_sync(model, strict=True)
        path = outfile or f"/tmp/gradtuity_init_sync_strict_{rank}.txt"
        with open(path, "w") as f:
            f.write("ok")
        print(path)
    except RuntimeError as e:
        path = outfile or f"/tmp/gradtuity_init_sync_strict_{rank}.txt"
        with open(path, "w") as f:
            f.write(f"ERROR:{e!r}")
        print(path)
        raise
    finally:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["allreduce", "init_sync", "init_sync_strict_fail"],
        help="Test mode",
    )
    parser.add_argument("--outfile", default=None, help="Output file path")
    args = parser.parse_args()
    if args.mode == "allreduce":
        main_allreduce(args.outfile)
    elif args.mode == "init_sync":
        main_init_sync(args.outfile)
    elif args.mode == "init_sync_strict_fail":
        main_init_sync_strict_fail(args.outfile)
    else:
        raise SystemExit(1)
