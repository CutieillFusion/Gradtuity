"""
Tests for embedding (row-gather) and nn.Embedding.

These tests require a CUDA-enabled GPU and Triton.
"""

import pytest

from gradtuity import Embedding
from gradtuity.tensor import Tensor

pytestmark = pytest.mark.requires_cuda


@pytest.mark.requires_triton
class TestEmbeddingForward:
    """Tests for weight.embedding(indices) forward."""

    def test_embedding_forward_deterministic(self):
        """Small deterministic: W (3,2), indices [2,0,2] -> [[4,5],[0,1],[4,5]]."""
        # W: row 0 = [0,1], row 1 = [2,3], row 2 = [4,5]
        W = Tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        indices = [2, 0, 2]
        out = W.embedding(indices)
        assert out.shape == (3, 2)
        assert out.to_list()[0] == pytest.approx([4.0, 5.0])
        assert out.to_list()[1] == pytest.approx([0.0, 1.0])
        assert out.to_list()[2] == pytest.approx([4.0, 5.0])

    def test_embedding_forward_2d_indices(self):
        """2D indices [[1,2],[0,1]] -> output (2, 2, 2)."""
        W = Tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])  # (3, 2)
        indices = [[1, 2], [0, 1]]
        out = W.embedding(indices)
        assert out.shape == (2, 2, 2)
        # out[0,0] = W[1], out[0,1] = W[2], out[1,0] = W[0], out[1,1] = W[1]
        assert out.to_list()[0][0] == pytest.approx([2.0, 3.0])
        assert out.to_list()[0][1] == pytest.approx([4.0, 5.0])
        assert out.to_list()[1][0] == pytest.approx([0.0, 1.0])
        assert out.to_list()[1][1] == pytest.approx([2.0, 3.0])

    def test_embedding_forward_indices_as_tensor_float(self):
        """Indices as Tensor of integer-valued floats 1.0, 2.0, 0.0 -> same as ints."""
        W = Tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        indices = Tensor([1.0, 2.0, 0.0])
        out = W.embedding(indices)
        assert out.shape == (3, 2)
        assert out.to_list()[0] == pytest.approx([2.0, 3.0])
        assert out.to_list()[1] == pytest.approx([4.0, 5.0])
        assert out.to_list()[2] == pytest.approx([0.0, 1.0])

    def test_embedding_bounds_index_negative_raises(self):
        """Index -1 should raise."""
        W = Tensor([[0.0, 1.0], [2.0, 3.0]])
        with pytest.raises(ValueError, match="out of range"):
            W.embedding([0, -1, 1])

    def test_embedding_bounds_index_equals_v_raises(self):
        """Index V (e.g. 3 for V=3) should raise."""
        W = Tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])  # V=3
        with pytest.raises(ValueError, match="out of range"):
            W.embedding([0, 3, 1])

    def test_embedding_rejects_non_integer_valued_float(self):
        """Indices with non-integer float (e.g. 1.5) should raise."""
        W = Tensor([[0.0, 1.0], [2.0, 3.0]])
        with pytest.raises(ValueError, match="integer-valued"):
            W.embedding([0.0, 1.5, 1.0])

    def test_embedding_rejects_ndim_3(self):
        """Indices must be 1D or 2D."""
        W = Tensor([[0.0, 1.0], [2.0, 3.0]])
        idx_3d = Tensor([[[0.0, 1.0]]])  # (1, 1, 2)
        with pytest.raises(ValueError, match="1D or 2D"):
            W.embedding(idx_3d)

    def test_embedding_rejects_weight_not_2d(self):
        """Weight must be 2D."""
        W = Tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D"):
            W.embedding([0, 1])


@pytest.mark.requires_triton
class TestEmbeddingBackward:
    """Tests for embedding backward and gradient accumulation."""

    def test_embedding_backward_exact_sum_loss(self):
        """L = out.sum(); weight.grad should equal count of each index per row (ones)."""
        W = Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
        )  # (3, 3)
        indices = [0, 1, 2, 1]  # N=4; row 0 once, row 1 twice, row 2 once
        out = W.embedding(indices)
        loss = out.sum()
        loss.backward()
        assert W.grad is not None
        assert W.grad.shape == (3, 3)
        # d(sum(out))/d(W) = for each (i,j), dW[i,j] = sum over n where idx[n]==i of dOut[n,j]; for sum loss dOut = 1
        # So dW[0] += 1 (once), dW[1] += 1 (twice), dW[2] += 1 (once)
        assert W.grad.to_list()[0] == pytest.approx([1.0, 1.0, 1.0])
        assert W.grad.to_list()[1] == pytest.approx([2.0, 2.0, 2.0])
        assert W.grad.to_list()[2] == pytest.approx([1.0, 1.0, 1.0])

    def test_embedding_backward_repeated_indices(self):
        """indices [1,1,1], L = out.sum() -> row 1 grad = [3,3,3] (or 3 ones)."""
        W = Tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], requires_grad=True)
        indices = [1, 1, 1]
        out = W.embedding(indices)
        loss = out.sum()
        loss.backward()
        assert W.grad is not None
        assert W.grad.to_list()[0] == pytest.approx([0.0, 0.0])
        assert W.grad.to_list()[1] == pytest.approx([3.0, 3.0])
        assert W.grad.to_list()[2] == pytest.approx([0.0, 0.0])

    def test_embedding_backward_weight_requires_grad_false(self):
        """When weight.requires_grad=False, output should not require_grad (no graph)."""
        W = Tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=False)
        out = W.embedding([0, 1])
        assert out.requires_grad is False
        assert out._parents == ()
        assert out._backward is None

    def test_embedding_backward_2d_indices(self):
        """Backward with 2D indices (B, S) -> out (B, S, D); grad flows correctly."""
        W = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        indices = [[0, 1], [1, 2]]  # (2, 2)
        out = W.embedding(indices)
        assert out.shape == (2, 2, 2)
        # out[0,0]=W[0], out[0,1]=W[1], out[1,0]=W[1], out[1,1]=W[2]
        assert out.to_list()[0][0] == pytest.approx([1.0, 2.0])
        assert out.to_list()[0][1] == pytest.approx([3.0, 4.0])
        assert out.to_list()[1][0] == pytest.approx([3.0, 4.0])
        assert out.to_list()[1][1] == pytest.approx([5.0, 6.0])
        loss = out.sum()
        loss.backward()
        assert W.grad is not None
        # Each of 4 positions contributes 1 to one row: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->2
        # So row 0: 1, row 1: 2, row 2: 1
        assert W.grad.to_list()[0] == pytest.approx([1.0, 1.0])
        assert W.grad.to_list()[1] == pytest.approx([2.0, 2.0])
        assert W.grad.to_list()[2] == pytest.approx([1.0, 1.0])


@pytest.mark.requires_triton
class TestEmbeddingNoGrad:
    """Tests for no-grad and graph behavior."""

    def test_embedding_indices_never_require_grad(self):
        """Indices are not differentiable; only weight gets grad."""
        W = Tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
        indices = Tensor([0.0, 1.0])  # indices.requires_grad could be False by default
        out = W.embedding(indices)
        # Only weight is a parent (indices are not in the graph); check before backward() clears graph
        assert out._parents == (W,)
        loss = out.sum()
        loss.backward()
        assert W.grad is not None


@pytest.mark.requires_triton
class TestNNEmbedding:
    """Tests for nn.Embedding module."""

    def test_nn_embedding_forward_list_indices(self):
        """nn.Embedding forward with list indices; output shape and values match weight.embedding."""
        emb = Embedding(3, 2)
        # Use known weight so we can check exact values
        known_weight = Tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        emb.load_state_dict({"weight": known_weight}, strict=True)
        indices = [2, 0, 2]
        out = emb(indices)
        expected = known_weight.embedding(indices)
        assert out.shape == (3, 2)
        assert out.shape == expected.shape
        assert out.to_list()[0] == pytest.approx(expected.to_list()[0])
        assert out.to_list()[1] == pytest.approx(expected.to_list()[1])
        assert out.to_list()[2] == pytest.approx(expected.to_list()[2])

    def test_nn_embedding_forward_tensor_indices(self):
        """nn.Embedding forward with Tensor indices."""
        emb = Embedding(3, 2)
        known_weight = Tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        emb.load_state_dict({"weight": known_weight}, strict=True)
        indices = Tensor([1.0, 2.0, 0.0])
        out = emb(indices)
        expected = known_weight.embedding(indices)
        assert out.shape == (3, 2)
        assert out.to_list()[0] == pytest.approx(expected.to_list()[0])
        assert out.to_list()[1] == pytest.approx(expected.to_list()[1])
        assert out.to_list()[2] == pytest.approx(expected.to_list()[2])

    def test_nn_embedding_forward_2d_indices(self):
        """nn.Embedding forward with 2D indices -> (B, S, D)."""
        emb = Embedding(3, 2)
        known_weight = Tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        emb.load_state_dict({"weight": known_weight}, strict=True)
        indices = [[1, 2], [0, 1]]
        out = emb(indices)
        assert out.shape == (2, 2, 2)
        expected = known_weight.embedding(indices)
        assert out.to_list()[0][0] == pytest.approx(expected.to_list()[0][0])
        assert out.to_list()[0][1] == pytest.approx(expected.to_list()[0][1])
        assert out.to_list()[1][0] == pytest.approx(expected.to_list()[1][0])
        assert out.to_list()[1][1] == pytest.approx(expected.to_list()[1][1])

    def test_nn_embedding_state_dict(self):
        """nn.Embedding state_dict has single key 'weight'."""
        emb = Embedding(10, 4)
        state = emb.state_dict()
        assert set(state.keys()) == {"weight"}
        assert state["weight"] is emb.weight
        assert state["weight"].shape == (10, 4)

    def test_nn_embedding_load_state_dict_round_trip(self):
        """load_state_dict(state_dict()) preserves module; forward still runs."""
        emb = Embedding(5, 3)
        state = emb.state_dict()
        emb.load_state_dict(state)
        out = emb([0, 1, 2])
        assert out.shape == (3, 3)

    def test_nn_embedding_backward(self):
        """nn.Embedding backward: loss = out.sum() gives correct weight.grad."""
        emb = Embedding(3, 3)
        known_weight = Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            requires_grad=True,
        )
        emb.load_state_dict({"weight": known_weight}, strict=True)
        indices = [0, 1, 2, 1]
        out = emb(indices)
        loss = out.sum()
        loss.backward()
        assert emb.weight.grad is not None
        assert emb.weight.grad.shape == (3, 3)
        assert emb.weight.grad.to_list()[0] == pytest.approx([1.0, 1.0, 1.0])
        assert emb.weight.grad.to_list()[1] == pytest.approx([2.0, 2.0, 2.0])
        assert emb.weight.grad.to_list()[2] == pytest.approx([1.0, 1.0, 1.0])
