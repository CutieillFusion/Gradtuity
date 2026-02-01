"""
Local multi-GPU launcher: spawn N worker processes with distributed env vars.

Usage:
    python -m gradtuity.launch --nproc 8 [--master-addr 127.0.0.1] [--master-port 29600] train.py --arg1 ...
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import subprocess
import sys
import time


def _pick_port(user_port: int | None) -> int:
    if user_port is not None:
        return user_port
    return 29500 + random.randint(0, 1000)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m gradtuity.launch")
    ap.add_argument(
        "--nproc", type=int, required=True, help="Number of worker processes"
    )
    ap.add_argument("--master-addr", default="127.0.0.1", help="Rendezvous address")
    ap.add_argument(
        "--master-port", type=int, default=None, help="Rendezvous port (default: auto)"
    )
    ap.add_argument("script", help="Training script path")
    ap.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="Arguments passed to script"
    )
    args = ap.parse_args()

    nproc = args.nproc
    if nproc <= 0:
        raise SystemExit("nproc must be > 0")

    master_port = _pick_port(args.master_port)

    procs: list[subprocess.Popen[bytes]] = []
    alive = True

    def shutdown() -> None:
        nonlocal alive
        if not alive:
            return
        alive = False
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(1.0)
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

    def handle_sig(signum: int, frame: object) -> None:
        shutdown()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    base_env = os.environ.copy()
    base_env["MASTER_ADDR"] = args.master_addr
    base_env["MASTER_PORT"] = str(master_port)
    base_env["WORLD_SIZE"] = str(nproc)
    base_env["GRADTUITY_LAUNCHER"] = "1"
    base_env["GRADTUITY_NCCL_UID_FILE"] = (
        f"/tmp/gradtuity_nccl_launch_{os.getpid()}.bin"
    )

    for local_rank in range(nproc):
        env = base_env.copy()
        env["RANK"] = str(local_rank)
        env["LOCAL_RANK"] = str(local_rank)

        cmd = [sys.executable, "-u", args.script] + args.script_args
        procs.append(subprocess.Popen(cmd, env=env))

    rc = 0
    try:
        for p in procs:
            r = p.wait()
            if r != 0:
                rc = r
                shutdown()
                break
    finally:
        shutdown()

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
