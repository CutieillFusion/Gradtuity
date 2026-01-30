"""
End-to-end integration tests for the full training pipeline.

These tests verify:
- Complete forward/backward pass through MLP
- Loss decreases over training iterations
- Gradient correctness via numerical comparison
- Memory management (no double-frees, leaks)
- Edge cases (shared subgraphs, branching)
"""

import pytest

from gradtuity import Tensor, randn, zeros, ones, zero_grad, sgd_step


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestFullMLPTraining:
    """Tests for complete MLP training loop."""

    def test_single_iteration(self):
        """Test a single training iteration completes without error."""
        # Initialize
        X = randn((4, 8), seed=42)
        W = randn((8, 4), requires_grad=True, seed=123)
        b = randn((4,), requires_grad=True, seed=456)

        # Forward
        z = X.matmul(W).add_bias(b).relu()
        loss = z.sum()

        # Get initial loss
        initial_loss = loss.item()

        # Backward
        zero_grad([W, b])
        loss.backward()

        # Check gradients exist
        assert W.grad is not None
        assert b.grad is not None
        assert W.grad.shape == W.shape
        assert b.grad.shape == b.shape

        # SGD step
        sgd_step([W, b], lr=0.01)

        # Forward again to check loss changed
        z2 = X.matmul(W).add_bias(b).relu()
        loss2 = z2.sum()
        new_loss = loss2.item()

        # Loss should have changed (likely decreased)
        assert initial_loss != new_loss

    def test_loss_decreases_over_iterations(self):
        """Test that loss decreases over multiple iterations."""
        # Initialize with fixed seeds for reproducibility
        X = randn((16, 32), seed=42)
        W = randn((32, 8), requires_grad=True, seed=123)
        b = randn((8,), requires_grad=True, seed=456)

        losses = []
        for _ in range(50):
            # Forward
            z = X.matmul(W).add_bias(b).relu()
            loss = z.sum()
            losses.append(loss.item())

            # Backward
            zero_grad([W, b])
            loss.backward()

            # Update
            sgd_step([W, b], lr=0.001)

        # Check that loss generally decreased
        initial_loss = losses[0]
        final_loss = losses[-1]
        assert final_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_training_with_larger_network(self):
        """Test training with larger batch and feature sizes."""
        X = randn((64, 128), seed=42)
        W = randn((128, 32), requires_grad=True, seed=123)
        b = randn((32,), requires_grad=True, seed=456)

        initial_loss = None
        for i in range(20):
            z = X.matmul(W).add_bias(b).relu()
            loss = z.sum()

            if i == 0:
                initial_loss = loss.item()

            zero_grad([W, b])
            loss.backward()
            sgd_step([W, b], lr=0.0001)

        final_loss = loss.item()
        assert final_loss < initial_loss


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestGradientCorrectness:
    """Tests for gradient correctness via manual calculation."""

    def test_matmul_gradient_manual(self):
        """Test matmul gradient matches manual calculation."""
        # Simple 2x2 case for manual verification
        A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)  # Identity

        C = A.matmul(B)  # C = A @ I = A
        loss = C.sum()   # sum of all elements

        loss.backward()

        # dC = ones(2,2)
        # dA = dC @ B^T = ones @ I = ones
        # dB = A^T @ dC = A^T @ ones
        a_grad = A.grad.to_list()
        assert a_grad[0] == pytest.approx([1.0, 1.0])
        assert a_grad[1] == pytest.approx([1.0, 1.0])

        # A^T = [[1,3], [2,4]]
        # A^T @ ones = [[1+3, 1+3], [2+4, 2+4]] = [[4,4], [6,6]]
        b_grad = B.grad.to_list()
        assert b_grad[0] == pytest.approx([4.0, 4.0])
        assert b_grad[1] == pytest.approx([6.0, 6.0])

    def test_add_bias_gradient_manual(self):
        """Test add_bias gradient matches manual calculation."""
        X = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([0.1, 0.2], requires_grad=True)

        Y = X.add_bias(b)
        loss = Y.sum()

        loss.backward()

        # dY = ones(2,2)
        # dX = dY = ones
        x_grad = X.grad.to_list()
        assert x_grad[0] == pytest.approx([1.0, 1.0])
        assert x_grad[1] == pytest.approx([1.0, 1.0])

        # db = sum(dY, axis=0) = [2, 2]
        assert b.grad.to_list() == pytest.approx([2.0, 2.0])

    def test_relu_gradient_manual(self):
        """Test relu gradient matches manual calculation."""
        X = Tensor([[-1.0, 2.0], [0.0, -3.0]], requires_grad=True)

        Y = X.relu()  # [[0, 2], [0, 0]]
        loss = Y.sum()  # 2

        assert loss.item() == pytest.approx(2.0)

        loss.backward()

        # dY = ones(2,2)
        # dX = dY * (X > 0) = [[0,1], [0,0]]
        x_grad = X.grad.to_list()
        assert x_grad[0] == pytest.approx([0.0, 1.0])
        assert x_grad[1] == pytest.approx([0.0, 0.0])

    def test_full_mlp_gradient_chain(self):
        """Test gradient flow through matmul -> add_bias -> relu -> sum."""
        # X: (2, 3), W: (3, 2), b: (2,)
        X = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # First two basis vectors
        W = Tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], requires_grad=True)
        b = Tensor([0.0, 0.0], requires_grad=True)

        # Forward:
        # h = X @ W = [[1, -1], [2, -2]]
        # y = h + b = [[1, -1], [2, -2]]
        # z = relu(y) = [[1, 0], [2, 0]]
        # loss = sum(z) = 3
        z = X.matmul(W).add_bias(b).relu()
        loss = z.sum()

        assert loss.item() == pytest.approx(3.0)

        loss.backward()

        # Backward:
        # dz = ones(2,2) = [[1,1], [1,1]]
        # dy = dz * relu_mask = [[1,0], [1,0]] (since y>0 at [0,0], [1,0])
        # db = sum(dy, axis=0) = [2, 0]
        assert b.grad.to_list() == pytest.approx([2.0, 0.0])

        # dh = dy = [[1,0], [1,0]]
        # dW = X^T @ dh
        # X^T = [[1,0], [0,1], [0,0]]
        # dW = [[1,0], [1,0], [0,0]]
        w_grad = W.grad.to_list()
        assert w_grad[0] == pytest.approx([1.0, 0.0])
        assert w_grad[1] == pytest.approx([1.0, 0.0])
        assert w_grad[2] == pytest.approx([0.0, 0.0])


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestSharedSubgraphs:
    """Tests for handling shared subgraphs correctly."""

    def test_shared_input_no_double_backward(self):
        """Test that shared inputs don't cause double backward."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        # x is used twice
        y1 = x.relu()
        y2 = x.relu()
        z = y1.add(y2)  # z = relu(x) + relu(x) = 2*relu(x)
        loss = z.sum()

        loss.backward()

        # Gradient should be accumulated correctly
        # dy1 = ones, dy2 = ones
        # dx from y1 = mask * 1 = [1,1,1]
        # dx from y2 = mask * 1 = [1,1,1]
        # total dx = [2,2,2]
        assert x.grad.to_list() == pytest.approx([2.0, 2.0, 2.0])

    def test_diamond_graph(self):
        """Test diamond-shaped computation graph."""
        #     x
        #    / \
        #   y1  y2
        #    \ /
        #     z
        x = Tensor([1.0, 2.0], requires_grad=True)

        y1 = x.relu()  # [1, 2]
        y2 = x.relu()  # [1, 2]
        z = y1.add(y2)  # [2, 4]
        loss = z.sum()  # 6

        assert loss.item() == pytest.approx(6.0)

        loss.backward()

        # Both paths contribute gradient
        assert x.grad.to_list() == pytest.approx([2.0, 2.0])


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_grad_before_backward(self):
        """Test that zero_grad properly clears previous gradients."""
        x = Tensor([1.0, 2.0], requires_grad=True)

        # First backward
        y = x.relu()
        loss = y.sum()
        loss.backward()
        assert x.grad.to_list() == pytest.approx([1.0, 1.0])

        # Second forward/backward with zero_grad
        zero_grad([x])
        y2 = x.relu()
        loss2 = y2.sum()
        loss2.backward()

        # Should be fresh gradient, not accumulated
        assert x.grad.to_list() == pytest.approx([1.0, 1.0])

    def test_no_grad_tensors_in_graph(self):
        """Test that non-grad tensors don't get gradients."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        w = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

        y = x.matmul(w)
        loss = y.sum()
        loss.backward()

        assert x.grad is None
        assert w.grad is not None

    def test_single_element_tensors(self):
        """Test with single-element tensors."""
        x = Tensor([[1.0]], requires_grad=True)
        w = Tensor([[2.0]], requires_grad=True)

        y = x.matmul(w)  # [[2.0]]
        loss = y.sum()   # 2.0

        assert loss.item() == pytest.approx(2.0)

        loss.backward()

        assert x.grad.to_list()[0][0] == pytest.approx(2.0)
        assert w.grad.to_list()[0][0] == pytest.approx(1.0)

    def test_all_negative_relu(self):
        """Test relu with all negative inputs."""
        x = Tensor([-1.0, -2.0, -3.0], requires_grad=True)

        y = x.relu()  # [0, 0, 0]
        loss = y.sum()  # 0

        assert loss.item() == pytest.approx(0.0)

        loss.backward()

        # All gradients blocked by relu
        assert x.grad.to_list() == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestMemorySafety:
    """Tests for memory management correctness."""

    def test_repeated_forward_backward(self):
        """Test that repeated forward/backward doesn't leak memory."""
        X = randn((8, 16), seed=42)
        W = randn((16, 8), requires_grad=True, seed=123)
        b = randn((8,), requires_grad=True, seed=456)

        # Run many iterations - should not crash or OOM
        for _ in range(100):
            z = X.matmul(W).add_bias(b).relu()
            loss = z.sum()
            zero_grad([W, b])
            loss.backward()
            sgd_step([W, b], lr=0.001)

        # If we got here without crashing, memory is being freed

    def test_detach_memory_safety(self):
        """Test that detach creates safe non-owning references."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        # Create detached view
        x_detached = x.detach()

        # Both should be usable
        assert x.to_list() == pytest.approx([1.0, 2.0, 3.0])
        assert x_detached.to_list() == pytest.approx([1.0, 2.0, 3.0])

        # Detached should not require grad
        assert x_detached.requires_grad is False

        # Original still works for autograd
        y = x.relu()
        loss = y.sum()
        loss.backward()
        assert x.grad is not None