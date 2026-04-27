"""
Tests for gradtuity.dist: env, init, sync_grads, bucket, NCCL glue, parity, deadlock.

- Env/sampler: imported from submodules so collection works without libnccl.so.
- Tests that need init/sync_grads import gradtuity.dist inside the test (loads libnccl).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from subprocess import TimeoutExpired

import pytest

# Import env and sampler only (no comm -> no nccl). Env/sampler tests run without NCCL.
from gradtuity.dist import env as dist_env
from gradtuity.dist.sampler import distributed_indices, shard_size


class TestEnv:
    """Environment variable parsing (no CUDA required)."""

    def test_get_rank_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("SLURM_PROCID", raising=False)
        with pytest.raises(RuntimeError, match="rank not set|RANK|SLURM_PROCID"):
            dist_env.get_rank()

    def test_get_rank_from_rank(self, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        assert dist_env.get_rank() == 3

    def test_get_rank_from_slurm(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("SLURM_PROCID", "2")
        assert dist_env.get_rank() == 2

    def test_get_world_size_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("SLURM_NTASKS", raising=False)
        with pytest.raises(
            RuntimeError, match="World size not set|WORLD_SIZE|SLURM_NTASKS"
        ):
            dist_env.get_world_size()

    def test_get_world_size_from_env(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert dist_env.get_world_size() == 8

    def test_get_local_rank_from_env(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert dist_env.get_local_rank() == 1

    def test_get_local_rank_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.delenv("SLURM_LOCALID", raising=False)
        with pytest.raises(
            RuntimeError, match="Local rank not set|LOCAL_RANK|SLURM_LOCALID"
        ):
            dist_env.get_local_rank()

    def test_get_master_addr_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        with pytest.raises(RuntimeError, match="Master address not set|MASTER_ADDR"):
            dist_env.get_master_addr()

    def test_get_master_port_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("MASTER_PORT", raising=False)
        with pytest.raises(RuntimeError, match="Master port not set|MASTER_PORT"):
            dist_env.get_master_port()


class TestDistributedIndices:
    """Distributed sampler indices (no CUDA required)."""

    def test_single_rank(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        r = distributed_indices(10)
        assert list(r) == list(range(10))

    def test_two_ranks(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        r0 = distributed_indices(10)
        assert list(r0) == [0, 2, 4, 6, 8]
        monkeypatch.setenv("RANK", "1")
        r1 = distributed_indices(10)
        assert list(r1) == [1, 3, 5, 7, 9]

    def test_shard_size(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert shard_size(10) == 3
        assert shard_size(12) == 3


class TestCudaRuntimeHelpers:
    """cuda_set_device, cuda_get_device (requires CUDA)."""

    @pytest.mark.requires_cuda
    @pytest.mark.requires_triton
    def test_set_and_get_device(self):
        from gradtuity import cuda_mem

        cuda_mem.cuda_set_device(0)
        assert cuda_mem.cuda_get_device() == 0

    @pytest.mark.requires_cuda
    @pytest.mark.requires_triton
    def test_device_synchronize(self):
        from gradtuity import cuda_mem

        cuda_mem.cuda_set_device(0)
        cuda_mem.cuda_device_synchronize()


class TestDistSingleProcess:
    """Single process (WORLD_SIZE=1): init and sync_grads are no-ops."""

    @pytest.mark.requires_triton
    def test_init_world_size_one(self, monkeypatch):
        from gradtuity.dist import destroy_process_group, init

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        init()
        destroy_process_group()

    @pytest.mark.requires_triton
    def test_sync_grads_world_size_one_no_op(self, monkeypatch):
        from gradtuity import Tensor
        from gradtuity.dist import destroy_process_group, init, sync_grads
        from gradtuity.nn import Linear

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        init()
        try:
            layer = Linear(4, 2)
            params = layer.parameters()
            x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
            y = layer(x)
            loss = y.sum()
            loss.backward()
            # With world_size=1, sync_grads returns without changing grads
            sync_grads(params)
            for p in params:
                assert p.grad is not None
        finally:
            destroy_process_group()


class TestSyncGradsEnsuresGrad:
    """sync_grads calls ensure_grad so every param has a grad (no hang)."""

    @pytest.mark.requires_triton
    def test_missing_grad_gets_zeros(self, monkeypatch):
        from gradtuity.dist import destroy_process_group, init, sync_grads
        from gradtuity.nn import Linear

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        init()
        try:
            layer = Linear(3, 2)
            params = layer.parameters()
            layer.zero_grad()
            # No backward: all grads are None. sync_grads should ensure_grad (allocate zeros)
            sync_grads(params)
            for p in params:
                assert p.grad is not None
                assert p.grad.shape == p.shape
        finally:
            destroy_process_group()


class TestBucketPackUnpackSingleProcess:
    """With world_size=1, sync_grads leaves grad values unchanged."""

    @pytest.mark.requires_triton
    def test_sync_grads_preserves_grads_when_world_size_one(self, monkeypatch):
        from gradtuity import Tensor
        from gradtuity.dist import destroy_process_group, init, sync_grads
        from gradtuity.nn import Linear

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        init()
        try:
            layer = Linear(4, 2)
            params = layer.parameters()
            x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
            y = layer(x)
            loss = y.sum()
            loss.backward()
            # Snapshot grad values (copy to list)
            grad_snapshots = [p.grad.to_list() for p in params]
            sync_grads(params, bucket_mb=0.001)
            for p, snap in zip(params, grad_snapshots):
                assert p.grad is not None
                assert p.grad.to_list() == snap
        finally:
            destroy_process_group()


class TestNCCLAllReduceTwoProcesses:
    """Multi-process: 2 ranks, each sets buffer to rank+1, AllReduce sum => 3.0."""

    @pytest.mark.requires_nccl
    @pytest.mark.requires_multigpu
    def test_allreduce_sum_two_ranks(self, monkeypatch):
        # Use a closed temp file so subprocesses have exclusive access
        fd, uid_file = tempfile.mkstemp(suffix=".bin", prefix="gradtuity_nccl_uid_")
        os.close(fd)
        try:
            # Both processes see GPUs 0,1; LOCAL_RANK selects device (torchrun/DDP style)
            env0 = {
                **os.environ,
                "RANK": "0",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "0",
                "GRADTUITY_NCCL_UID_FILE": uid_file,
                "CUDA_VISIBLE_DEVICES": "0,1",
            }
            env1 = {
                **os.environ,
                "RANK": "1",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "1",
                "GRADTUITY_NCCL_UID_FILE": uid_file,
                "CUDA_VISIBLE_DEVICES": "0,1",
            }
            out0 = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            out1 = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            out0.close()
            out1.close()
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "gradtuity.tests.dist_worker",
                    "allreduce",
                    "--outfile",
                    out0.name,
                ]
                proc0 = subprocess.Popen(
                    cmd,
                    env=env0,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                cmd1 = [
                    sys.executable,
                    "-m",
                    "gradtuity.tests.dist_worker",
                    "allreduce",
                    "--outfile",
                    out1.name,
                ]
                proc1 = subprocess.Popen(
                    cmd1,
                    env=env1,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    ret0 = proc0.wait(timeout=60)
                    ret1 = proc1.wait(timeout=60)
                except TimeoutExpired:
                    out0_text = (
                        proc0.stdout.read().decode(errors="replace")
                        if proc0.stdout
                        else ""
                    )
                    err0_text = (
                        proc0.stderr.read().decode(errors="replace")
                        if proc0.stderr
                        else ""
                    )
                    out1_text = (
                        proc1.stdout.read().decode(errors="replace")
                        if proc1.stdout
                        else ""
                    )
                    err1_text = (
                        proc1.stderr.read().decode(errors="replace")
                        if proc1.stderr
                        else ""
                    )
                    print("--- rank 0 stdout ---\n", out0_text)
                    print("--- rank 0 stderr ---\n", err0_text)
                    print("--- rank 1 stdout ---\n", out1_text)
                    print("--- rank 1 stderr ---\n", err1_text)
                    raise
                err0 = proc0.stderr.read().decode() if proc0.stderr else ""
                err1 = proc1.stderr.read().decode() if proc1.stderr else ""
                assert ret0 == 0, f"rank 0 failed: {err0}"
                assert ret1 == 0, f"rank 1 failed: {err1}"
                with open(out0.name) as f:
                    val0 = float(f.read().strip())
                with open(out1.name) as f:
                    val1 = float(f.read().strip())
                # AllReduce(sum): 1 + 2 = 3.0 on both ranks
                assert abs(val0 - 3.0) < 1e-5
                assert abs(val1 - 3.0) < 1e-5
            finally:
                try:
                    os.unlink(out0.name)
                except OSError:
                    pass
                try:
                    os.unlink(out1.name)
                except OSError:
                    pass
        finally:
            try:
                os.unlink(uid_file)
            except OSError:
                pass


class TestInitSyncSingleProcess:
    """Single process (WORLD_SIZE=1): init_sync is a no-op."""

    @pytest.mark.requires_triton
    def test_init_sync_world_size_one_no_op(self, monkeypatch):
        from gradtuity.dist import destroy_process_group, init, init_sync
        from gradtuity.nn import Linear

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        init()
        try:
            model = Linear(4, 2)
            params_before = [p.to_list() for p in model.parameters()]
            init_sync(model)
            params_after = [p.to_list() for p in model.parameters()]
            assert params_before == params_after
        finally:
            destroy_process_group()


class TestInitSyncTwoProcesses:
    """Two processes: RNG desync before model build, init_sync makes params identical."""

    @pytest.mark.requires_nccl
    @pytest.mark.requires_multigpu
    def test_init_sync_params_identical_after_broadcast(self, monkeypatch):
        fd, uid_file = tempfile.mkstemp(suffix=".bin", prefix="gradtuity_nccl_uid_")
        os.close(fd)
        try:
            env0 = {
                **os.environ,
                "RANK": "0",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "0",
                "GRADTUITY_NCCL_UID_FILE": uid_file,
                "CUDA_VISIBLE_DEVICES": "0,1",
            }
            env1 = {
                **os.environ,
                "RANK": "1",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "1",
                "GRADTUITY_NCCL_UID_FILE": uid_file,
                "CUDA_VISIBLE_DEVICES": "0,1",
            }
            out0 = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            out1 = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            out0.close()
            out1.close()
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "gradtuity.tests.dist_worker",
                    "init_sync",
                    "--outfile",
                    out0.name,
                ]
                proc0 = subprocess.Popen(
                    cmd,
                    env=env0,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                cmd1 = [
                    sys.executable,
                    "-m",
                    "gradtuity.tests.dist_worker",
                    "init_sync",
                    "--outfile",
                    out1.name,
                ]
                proc1 = subprocess.Popen(
                    cmd1,
                    env=env1,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    ret0 = proc0.wait(timeout=60)
                    ret1 = proc1.wait(timeout=60)
                except TimeoutExpired:
                    proc0.kill()
                    proc1.kill()
                    proc0.wait()
                    proc1.wait()
                    raise
                err0 = proc0.stderr.read().decode() if proc0.stderr else ""
                err1 = proc1.stderr.read().decode() if proc1.stderr else ""
                assert ret0 == 0, f"rank 0 failed: {err0}"
                assert ret1 == 0, f"rank 1 failed: {err1}"
                with open(out0.name) as f:
                    sample0 = f.read().strip()
                with open(out1.name) as f:
                    sample1 = f.read().strip()
                assert sample0 == sample1, (
                    f"init_sync should make params identical; rank0 sample={sample0!r}, rank1 sample={sample1!r}"
                )
            finally:
                try:
                    os.unlink(out0.name)
                except OSError:
                    pass
                try:
                    os.unlink(out1.name)
                except OSError:
                    pass
        finally:
            try:
                os.unlink(uid_file)
            except OSError:
                pass


class TestInitSyncStrictMismatch:
    """Two processes: rank 1 has different model (fewer params); strict=True fails fast."""

    @pytest.mark.requires_nccl
    @pytest.mark.requires_multigpu
    def test_init_sync_strict_fails_on_param_mismatch(self, monkeypatch):
        fd, uid_file = tempfile.mkstemp(suffix=".bin", prefix="gradtuity_nccl_uid_")
        os.close(fd)
        try:
            env0 = {
                **os.environ,
                "RANK": "0",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "0",
                "GRADTUITY_NCCL_UID_FILE": uid_file,
                "CUDA_VISIBLE_DEVICES": "0,1",
            }
            env1 = {
                **os.environ,
                "RANK": "1",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "1",
                "GRADTUITY_NCCL_UID_FILE": uid_file,
                "CUDA_VISIBLE_DEVICES": "0,1",
            }
            out0 = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            out1 = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            out0.close()
            out1.close()
            try:
                cmd0 = [
                    sys.executable,
                    "-m",
                    "gradtuity.tests.dist_worker",
                    "init_sync_strict_fail",
                    "--outfile",
                    out0.name,
                ]
                cmd1 = [
                    sys.executable,
                    "-m",
                    "gradtuity.tests.dist_worker",
                    "init_sync_strict_fail",
                    "--outfile",
                    out1.name,
                ]
                proc0 = subprocess.Popen(
                    cmd0,
                    env=env0,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                proc1 = subprocess.Popen(
                    cmd1,
                    env=env1,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                ret0 = proc0.wait(timeout=30)
                ret1 = proc1.wait(timeout=30)
                err0 = proc0.stderr.read().decode() if proc0.stderr else ""
                err1 = proc1.stderr.read().decode() if proc1.stderr else ""
                with open(out0.name) as f:
                    content0 = f.read()
                with open(out1.name) as f:
                    content1 = f.read()
                combined_stderr = err0 + err1
                combined_out = content0 + content1
                assert (
                    "strict check failed" in combined_stderr
                    or "strict check failed" in combined_out
                ), (
                    f"Expected 'strict check failed' in stderr or outfile; ret0={ret0} ret1={ret1} "
                    f"stderr0={err0!r} stderr1={err1!r} out0={content0!r} out1={content1!r}"
                )
            finally:
                try:
                    os.unlink(out0.name)
                except OSError:
                    pass
                try:
                    os.unlink(out1.name)
                except OSError:
                    pass
        finally:
            try:
                os.unlink(uid_file)
            except OSError:
                pass


class TestDeadlockMissingGrad:
    """Ensure sync_grads calls ensure_grad so no rank has missing grad (collective order matches)."""

    @pytest.mark.requires_triton
    def test_all_params_have_grad_after_sync(self, monkeypatch):
        from gradtuity.dist import destroy_process_group, init, sync_grads
        from gradtuity.nn import MLP

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        init()
        try:
            model = MLP(4, [8, 4])
            params = model.parameters()
            model.zero_grad()
            sync_grads(params)
            for p in params:
                assert p.grad is not None
        finally:
            destroy_process_group()


class TestLaunch:
    """Smoke tests for gradtuity.launch (no CUDA/NCCL required)."""

    def test_launcher_spawns_workers_with_env(self):
        """Launcher sets RANK, WORLD_SIZE, LOCAL_RANK per worker; both ranks run."""
        outdir = tempfile.mkdtemp(prefix="gradtuity_launch_test_")
        script_path = os.path.join(outdir, "write_env.py")
        with open(script_path, "w") as f:
            f.write(
                "import os\n"
                "d = os.environ.get('GRADTUITY_LAUNCH_TEST_DIR')\n"
                "if d:\n"
                "    r = os.environ.get('RANK', '')\n"
                "    w = os.environ.get('WORLD_SIZE', '')\n"
                "    l = os.environ.get('LOCAL_RANK', '')\n"
                "    with open(os.path.join(d, r + '.txt'), 'w') as out:\n"
                "        out.write(r + ' ' + w + ' ' + l)\n"
            )
        try:
            env = os.environ.copy()
            env["GRADTUITY_LAUNCH_TEST_DIR"] = outdir
            cmd = [
                sys.executable,
                "-m",
                "gradtuity.launch",
                "--nproc",
                "2",
                script_path,
            ]
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            result = subprocess.run(
                cmd, env=env, cwd=repo_root, capture_output=True, text=True, timeout=30
            )
            assert result.returncode == 0, (result.stdout, result.stderr)
            with open(os.path.join(outdir, "0.txt")) as f:
                line0 = f.read().strip()
            with open(os.path.join(outdir, "1.txt")) as f:
                line1 = f.read().strip()
            assert line0 == "0 2 0"
            assert line1 == "1 2 1"
        finally:
            for f in ("0.txt", "1.txt", "write_env.py"):
                p = os.path.join(outdir, f)
                if os.path.exists(p):
                    os.unlink(p)
            os.rmdir(outdir)

    def test_launcher_propagates_failure(self):
        """When a worker exits nonzero, launcher exits with that code."""
        outdir = tempfile.mkdtemp(prefix="gradtuity_launch_test_")
        script_path = os.path.join(outdir, "exit_two.py")
        with open(script_path, "w") as f:
            f.write("import sys\nsys.exit(2)\n")
        try:
            cmd = [
                sys.executable,
                "-m",
                "gradtuity.launch",
                "--nproc",
                "2",
                script_path,
            ]
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            result = subprocess.run(
                cmd, cwd=repo_root, capture_output=True, text=True, timeout=30
            )
            assert result.returncode == 2
        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)
            os.rmdir(outdir)
