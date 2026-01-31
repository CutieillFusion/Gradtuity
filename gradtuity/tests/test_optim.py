"""
Tests for optimizers (gradtuity/optim.py): Optimizer base, AdamW, SGD.

Requires CUDA and Triton.
"""

import math
import pytest

from gradtuity import AdamW, Optimizer, SGD, Tensor

pytestmark = [pytest.mark.requires_cuda, pytest.mark.requires_triton]


def _adamw_step_reference(
    p: list[float],
    g: list[float],
    m: list[float],
    v: list[float],
    t: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> tuple[list[float], list[float], list[float]]:
    """Pure Python float32-ish reference for one AdamW step."""
    n = len(p)
    m_new = [0.0] * n
    v_new = [0.0] * n
    p_new = [0.0] * n
    bc1 = 1.0 / (1.0 - beta1**t)
    bc2 = 1.0 / (1.0 - beta2**t)
    for i in range(n):
        m_new[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
        v_new[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]
        m_hat = m_new[i] * bc1
        v_hat = v_new[i] * bc2
        update = m_hat / (math.sqrt(v_hat) + eps)
        p_new[i] = p[i] - lr * update - lr * weight_decay * p[i]
    return (m_new, v_new, p_new)


class TestAdamWCoreCorrectness:
    """Core correctness: single step, bias correction, weight decay decoupling."""

    def test_adamw_single_step_matches_reference(self):
        """Single step matches pure-Python reference (float32)."""
        p = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        p.grad = Tensor([1.0, 1.0, 1.0, 1.0])
        lr, beta1, beta2, eps, wd = 0.1, 0.9, 0.999, 1e-8, 0.01
        opt = AdamW([p], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)
        opt.step()
        expected_m, expected_v, expected_p = _adamw_step_reference(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            t=1,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=wd,
        )
        assert p.to_list() == pytest.approx(expected_p, rel=1e-5, abs=1e-5)

    def test_adamw_bias_correction_two_steps(self):
        """Two steps: step counter and params match reference."""
        p = Tensor([2.0, 4.0], requires_grad=True)
        p.grad = Tensor([0.5, -0.5])
        lr, beta1, beta2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.0
        opt = AdamW([p], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)
        m_ref, v_ref = [0.0, 0.0], [0.0, 0.0]
        p_ref = [2.0, 4.0]
        g_ref = [0.5, -0.5]
        for t in range(1, 3):
            opt.step()
            m_ref, v_ref, p_ref = _adamw_step_reference(
                p_ref, g_ref, m_ref, v_ref, t, lr, beta1, beta2, eps, wd
            )
            assert opt.step_count == t
            assert p.to_list() == pytest.approx(p_ref, rel=1e-5, abs=1e-5)

    def test_adamw_weight_decay_decoupled_zero_grad(self):
        """With zero gradient, weight decay still shrinks param (decoupled)."""
        p = Tensor([10.0, 20.0], requires_grad=True)
        p.grad = Tensor([0.0, 0.0])  # zero grad so update term is 0
        lr, wd = 0.1, 0.01
        opt = AdamW([p], lr=lr, weight_decay=wd)
        opt.step()
        # p_new = p - lr*0 - lr*wd*p = p*(1 - lr*wd)
        expected = [10.0 * (1 - lr * wd), 20.0 * (1 - lr * wd)]
        assert p.to_list() == pytest.approx(expected, rel=1e-5, abs=1e-5)


class TestAdamWStateAndBehavior:
    """Lazy state init and zero_grad modes."""

    def test_adamw_lazy_state_init(self):
        """State m/v allocated only for params that had grad on step()."""
        p1 = Tensor([1.0, 2.0], requires_grad=True)
        p1.grad = Tensor([0.1, 0.2])
        p2 = Tensor([3.0], requires_grad=True)
        p2.grad = None  # no grad
        opt = AdamW([p1, p2], lr=0.01)
        opt.step()
        assert opt.m[0] is not None
        assert opt.v[0] is not None
        assert opt.m[1] is None
        assert opt.v[1] is None

    def test_adamw_zero_grad_set_to_none_false(self):
        """zero_grad(set_to_none=False) zeros the gradient buffer."""
        p = Tensor([1.0, 2.0], requires_grad=True)
        p.grad = Tensor([1.0, 2.0])
        opt = AdamW([p])
        opt.zero_grad(set_to_none=False)
        assert p.grad is not None
        assert p.grad.to_list() == pytest.approx([0.0, 0.0])

    def test_adamw_zero_grad_set_to_none_true(self):
        """zero_grad(set_to_none=True) sets grad to None."""
        p = Tensor([1.0, 2.0], requires_grad=True)
        p.grad = Tensor([1.0, 2.0])
        opt = AdamW([p])
        opt.zero_grad(set_to_none=True)
        assert p.grad is None


class TestAdamWSaveLoad:
    """Save/load continuity."""

    def test_adamw_save_load_continuity(self, tmp_path):
        """Load then continue training matches uninterrupted run (same final params)."""
        # Uninterrupted: N+M steps
        p_a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        opt_a = AdamW([p_a], lr=0.01, weight_decay=0.001)
        for _ in range(5):
            p_a.grad = Tensor([0.1, -0.1, 0.05])
            opt_a.step()
        final_uninterrupted = p_a.to_list()

        # Resumed: N steps, save, load, M steps
        p_b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        opt_b = AdamW([p_b], lr=0.01, weight_decay=0.001)
        for _ in range(3):
            p_b.grad = Tensor([0.1, -0.1, 0.05])
            opt_b.step()
        prefix = str(tmp_path / "optim")
        opt_b.save(prefix)
        opt_b_loaded = AdamW.load(prefix, [p_b])
        for _ in range(2):
            p_b.grad = Tensor([0.1, -0.1, 0.05])
            opt_b_loaded.step()
        final_resumed = p_b.to_list()

        assert final_resumed == pytest.approx(final_uninterrupted, rel=1e-5, abs=1e-5)

    def test_adamw_save_load_round_trip_with_tmp_path(self, tmp_path):
        """Save and load round-trip with tmp_path; optimizer state (step, hyperparams, m, v) restored."""
        p = Tensor([5.0, 6.0], requires_grad=True)
        p.grad = Tensor([0.2, -0.2])
        opt = AdamW([p], lr=0.02, weight_decay=0.01)
        opt.step()
        opt.step()
        step_before = opt.step_count
        prefix = str(tmp_path / "optim")
        opt.save(prefix)
        p2 = Tensor([5.0, 6.0], requires_grad=True)
        opt2 = AdamW.load(prefix, [p2])
        assert opt2.step_count == step_before
        assert opt2.lr == pytest.approx(0.02)
        assert opt2.weight_decay == pytest.approx(0.01)
        # Optimizer save/load does not save parameter values; m/v state was restored
        assert opt2.m[0] is not None
        assert opt2.v[0] is not None


class TestAdamWEdgeCases:
    """Edge cases and validation."""

    def test_adamw_step_raises_on_grad_shape_mismatch(self):
        """step() raises ValueError when grad shape != param shape."""
        p = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        p.grad = Tensor([1.0, 2.0])  # wrong length
        opt = AdamW([p])
        with pytest.raises(ValueError, match="grad shape.*param shape"):
            opt.step()

    def test_adamw_step_increments_even_with_no_grads(self):
        """Global step increments on step() even when no param has grad."""
        p = Tensor([1.0], requires_grad=True)
        p.grad = None
        opt = AdamW([p])
        assert opt.step_count == 0
        opt.step()
        assert opt.step_count == 1
        opt.step()
        assert opt.step_count == 2

    def test_adamw_validate_hyperparams_in_step(self):
        """Invalid hyperparameters in step() raise ValueError."""
        p = Tensor([1.0], requires_grad=True)
        p.grad = Tensor([0.1])
        opt = AdamW([p], lr=0.01)
        opt.lr = -0.1
        with pytest.raises(ValueError, match="lr"):
            opt.step()

    def test_adamw_load_raises_on_param_count_mismatch(self, tmp_path):
        """load() raises when param_count != len(params)."""
        p = Tensor([1.0, 2.0], requires_grad=True)
        opt = AdamW([p])
        opt.step()
        prefix = str(tmp_path / "optim")
        opt.save(prefix)
        params_two = [Tensor([1.0]), Tensor([2.0])]
        with pytest.raises(ValueError, match="param_count"):
            AdamW.load(prefix, params_two)

    def test_adamw_load_raises_on_param_shape_mismatch(self, tmp_path):
        """load() raises when saved param_shapes != current param shapes."""
        p = Tensor([1.0, 2.0], requires_grad=True)
        p.grad = Tensor([0.1, 0.1])
        opt = AdamW([p])
        opt.step()
        prefix = str(tmp_path / "optim")
        opt.save(prefix)
        p_wrong_shape = Tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="shape"):
            AdamW.load(prefix, [p_wrong_shape])


class TestSGD:
    """Tests for SGD optimizer class."""

    def test_sgd_step_matches_reference(self):
        """SGD step matches expected param -= lr * grad."""
        p = Tensor([10.0, 20.0], requires_grad=True)
        p.grad = Tensor([1.0, 2.0])
        opt = SGD([p], lr=0.5)
        opt.step()
        assert p.to_list() == pytest.approx([9.5, 19.0])

    def test_sgd_zero_grad_set_to_none_false(self):
        """zero_grad(set_to_none=False) zeros the gradient buffer."""
        p = Tensor([1.0, 2.0], requires_grad=True)
        p.grad = Tensor([1.0, 2.0])
        opt = SGD([p])
        opt.zero_grad(set_to_none=False)
        assert p.grad is not None
        assert p.grad.to_list() == pytest.approx([0.0, 0.0])

    def test_sgd_zero_grad_set_to_none_true(self):
        """zero_grad(set_to_none=True) sets grad to None."""
        p = Tensor([1.0, 2.0], requires_grad=True)
        p.grad = Tensor([1.0, 2.0])
        opt = SGD([p])
        opt.zero_grad(set_to_none=True)
        assert p.grad is None

    def test_sgd_save_load_round_trip(self, tmp_path):
        """Save and load SGD; lr and param_count restored; continued steps match uninterrupted."""
        p_a = Tensor([1.0, 2.0], requires_grad=True)
        opt_a = SGD([p_a], lr=0.01)
        p_a.grad = Tensor([0.1, -0.1])
        opt_a.step()
        opt_a.step()
        final_uninterrupted = p_a.to_list()

        p_b = Tensor([1.0, 2.0], requires_grad=True)
        opt_b = SGD([p_b], lr=0.01)
        p_b.grad = Tensor([0.1, -0.1])
        opt_b.step()
        prefix = str(tmp_path / "sgd")
        opt_b.save(prefix)
        opt_b_loaded = SGD.load(prefix, [p_b])
        assert opt_b_loaded.lr == pytest.approx(0.01)
        p_b.grad = Tensor([0.1, -0.1])
        opt_b_loaded.step()
        final_resumed = p_b.to_list()
        assert final_resumed == pytest.approx(final_uninterrupted, rel=1e-5, abs=1e-5)


class TestOptimizerLoadDispatch:
    """Optimizer.load() dispatches to correct subclass."""

    def test_optimizer_load_returns_adamw(self, tmp_path):
        """Save AdamW, load via Optimizer.load -> isinstance AdamW."""
        p = Tensor([1.0, 2.0], requires_grad=True)
        p.grad = Tensor([0.1, 0.1])
        opt = AdamW([p], lr=0.01)
        opt.step()
        prefix = str(tmp_path / "opt")
        opt.save(prefix)
        p2 = Tensor([1.0, 2.0], requires_grad=True)
        loaded = Optimizer.load(prefix, [p2])
        assert isinstance(loaded, AdamW)
        assert loaded.lr == pytest.approx(0.01)

    def test_optimizer_load_returns_sgd(self, tmp_path):
        """Save SGD, load via Optimizer.load -> isinstance SGD."""
        p = Tensor([1.0, 2.0], requires_grad=True)
        p.grad = Tensor([0.1, 0.1])
        opt = SGD([p], lr=0.02)
        opt.step()
        prefix = str(tmp_path / "opt")
        opt.save(prefix)
        p2 = Tensor([1.0, 2.0], requires_grad=True)
        loaded = Optimizer.load(prefix, [p2])
        assert isinstance(loaded, SGD)
        assert loaded.lr == pytest.approx(0.02)
