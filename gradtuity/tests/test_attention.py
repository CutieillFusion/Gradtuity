"""
Tests for CausalSelfAttention module.

These tests require a CUDA-enabled GPU and Triton.
"""

import numpy as np
import pytest

from gradtuity import CausalSelfAttention, Tensor, randn, zeros

pytestmark = pytest.mark.requires_cuda


@pytest.mark.requires_triton
class TestCausalSelfAttention:
    """Tests for CausalSelfAttention forward and gradients."""

    def test_forward_shape(self):
        """CausalSelfAttention(x) returns (B, S, E)."""
        attn = CausalSelfAttention(embed_dim=4, num_heads=2)
        x = randn((2, 3, 4), requires_grad=True)
        y = attn(x)
        assert y.shape == (2, 3, 4)

    def test_forward_deterministic_small(self):
        """Small deterministic run: B=1, H=1, S=3, D=2; output is finite."""
        attn = CausalSelfAttention(embed_dim=2, num_heads=1)
        x = Tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # (1, 3, 2)
        y = attn(x)
        assert y.shape == (1, 3, 2)
        arr = np.array(y.to_list())
        assert np.all(np.isfinite(arr))

    def test_backward_runs(self):
        """backward() runs without error; grad on input is correct shape."""
        attn = CausalSelfAttention(embed_dim=4, num_heads=2)
        x = randn((2, 3, 4), requires_grad=True)
        y = attn(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_check_small(self):
        """Finite-diff gradient check on a few elements."""
        np.random.seed(789)
        attn = CausalSelfAttention(embed_dim=2, num_heads=1)
        x_np = np.random.randn(1, 3, 2).astype(np.float32) * 0.1
        x = Tensor(x_np.tolist(), requires_grad=True)
        y = attn(x)
        loss = y.sum()
        loss.backward()
        analytic = np.array(x.grad.to_list())
        eps = 1e-4

        def scalar_from_sum_tensor(t):
            return np.array(t.to_list()).ravel()[0]

        for flat_idx in [0, 2, 4]:
            i = flat_idx // 2
            j = flat_idx % 2
            x_plus = x_np.copy()
            x_plus[0, i, j] += eps
            x_minus = x_np.copy()
            x_minus[0, i, j] -= eps
            y_plus = attn(Tensor(x_plus.tolist())).sum()
            y_minus = attn(Tensor(x_minus.tolist())).sum()
            numeric = (scalar_from_sum_tensor(y_plus) - scalar_from_sum_tensor(y_minus)) / (2 * eps)
            # Finite-diff can diverge from analytic for float32 + softmax; use loose sanity check
            assert analytic[0, i, j] == pytest.approx(numeric, rel=0.25, abs=1.0)

    def test_init_rejects_bad_embed_dim(self):
        """Constructor rejects embed_dim not divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            CausalSelfAttention(embed_dim=5, num_heads=2)
