"""
Tests for causal mask (apply_causal_mask) and transpose4d_12.

These tests require a CUDA-enabled GPU and Triton.
"""

import numpy as np
import pytest

from gradtuity import Tensor, randn

pytestmark = pytest.mark.requires_cuda


@pytest.mark.requires_triton
class TestTranspose4D12:
    """Tests for transpose4d_12: (B, A, C, D) -> (B, C, A, D)."""

    def test_transpose4d_12_round_trip(self):
        """Round-trip: transpose4d_12().transpose4d_12() equals identity."""
        np.random.seed(42)
        x_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        x = Tensor(x_np.tolist())
        y = x.transpose4d_12()
        assert y.shape == (2, 4, 3, 5)
        z = y.transpose4d_12()
        assert z.shape == (2, 3, 4, 5)
        flat_x = np.array(x.to_list()).ravel()
        flat_z = np.array(z.to_list()).ravel()
        assert flat_z == pytest.approx(flat_x, rel=1e-5, abs=1e-6)

    def test_transpose4d_12_rejects_non_4d(self):
        """transpose4d_12 rejects non-4D tensors."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="ndim=4"):
            x.transpose4d_12()


@pytest.mark.requires_triton
class TestApplyCausalMask:
    """Tests for apply_causal_mask on 4D scores (B, H, S, S)."""

    def test_causal_mask_upper_triangle_set(self):
        """Upper-triangular positions (j > i) become NEG_INF; lower unchanged."""
        # (B=1, H=1, S=4, S=4)
        np.random.seed(123)
        scores_np = np.random.randn(1, 1, 4, 4).astype(np.float32)
        scores = Tensor(scores_np.tolist())
        masked = scores.apply_causal_mask(neg_inf=-1e9)
        arr = np.array(masked.to_list())
        assert arr.shape == (1, 1, 4, 4)
        mat = arr[0, 0]
        for i in range(4):
            for j in range(4):
                if j > i:
                    assert mat[i, j] == pytest.approx(-1e9, rel=0, abs=1e3)
                else:
                    assert mat[i, j] == pytest.approx(
                        scores_np[0, 0, i, j], rel=1e-5, abs=1e-6
                    )

    def test_causal_mask_softmax_masked_near_zero(self):
        """After apply_causal_mask, softmax gives ~0 probability in masked positions."""
        np.random.seed(456)
        scores_np = np.random.randn(1, 1, 4, 4).astype(np.float32)
        scores = Tensor(scores_np.tolist())
        masked = scores.apply_causal_mask(neg_inf=-1e9)
        probs = masked.softmax(dim=-1)
        arr = np.array(probs.to_list())
        mat = arr[0, 0]
        for i in range(4):
            row_sum = mat[i, :].sum()
            assert row_sum == pytest.approx(1.0, rel=1e-5, abs=1e-5)
            for j in range(4):
                if j > i:
                    assert mat[i, j] == pytest.approx(0.0, abs=1e-6)

    def test_apply_causal_mask_rejects_non_4d(self):
        """apply_causal_mask rejects non-4D tensors."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="ndim=4"):
            x.apply_causal_mask()

    def test_apply_causal_mask_requires_square_last_two(self):
        """apply_causal_mask requires shape[-2] == shape[-1]."""
        x = Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])  # (1,1,2,3)
        with pytest.raises(ValueError, match="square last two"):
            x.apply_causal_mask()
