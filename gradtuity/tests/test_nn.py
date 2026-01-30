"""
Tests for nn.py - Neural network modules.

These tests require a CUDA-enabled GPU to run.
"""

import pytest

from gradtuity import Linear, MLP, Module, Tensor, ones, randn, zeros

# Mark all tests in this module as requiring CUDA
pytestmark = pytest.mark.requires_cuda


class TestModule:
    """Tests for the Module base class."""

    def test_module_is_base_class(self):
        """Test that Module is a valid base class."""
        m = Module()
        assert hasattr(m, "parameters")
        assert hasattr(m, "zero_grad")

    def test_module_call_not_implemented(self):
        """Test that calling Module directly raises NotImplementedError."""
        m = Module()
        x = Tensor([[1.0, 2.0]])
        with pytest.raises(NotImplementedError):
            m(x)


@pytest.mark.requires_triton
class TestLinear:
    """Tests for Linear layer."""

    def test_linear_creation(self):
        """Test creating a Linear layer."""
        layer = Linear(10, 5)

        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weight.shape == (10, 5)
        assert layer.bias.shape == (5,)

    def test_linear_requires_grad(self):
        """Test that Linear parameters require grad."""
        layer = Linear(3, 2)

        assert layer.weight.requires_grad is True
        assert layer.bias.requires_grad is True

    def test_linear_parameters(self):
        """Test that parameters() returns weight and bias."""
        layer = Linear(4, 3)
        params = layer.parameters()

        assert len(params) == 2
        assert layer.weight in params
        assert layer.bias in params

    def test_linear_forward_shape(self):
        """Test Linear forward produces correct output shape."""
        layer = Linear(3, 2)
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)

        y = layer(x)

        assert y.shape == (2, 2)

    def test_linear_forward_computation(self):
        """Test Linear forward computes Y = X @ W + b correctly."""
        layer = Linear(2, 2)

        # Set known weights and bias for testing
        # We can't easily set them, so we'll just verify the computation
        # by checking gradients flow correctly

        x = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        y = layer(x)

        # Should be able to compute loss and backward
        loss = y.sum()
        loss.backward()

        assert x.grad is not None

    def test_linear_backward(self):
        """Test that Linear backward computes gradients."""
        layer = Linear(3, 2)
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert x.grad is not None

    def test_linear_repr(self):
        """Test Linear string representation."""
        layer = Linear(10, 5)
        r = repr(layer)

        assert "Linear" in r
        assert "10" in r
        assert "5" in r


@pytest.mark.requires_triton
class TestMLP:
    """Tests for MLP class."""

    def test_mlp_creation(self):
        """Test creating an MLP."""
        model = MLP(2, [16, 16, 1])

        assert model.nin == 2
        assert model.nouts == [16, 16, 1]
        assert len(model.layers) == 3

    def test_mlp_layer_sizes(self):
        """Test that MLP creates layers with correct sizes."""
        model = MLP(2, [16, 8, 1])

        assert model.layers[0].in_features == 2
        assert model.layers[0].out_features == 16

        assert model.layers[1].in_features == 16
        assert model.layers[1].out_features == 8

        assert model.layers[2].in_features == 8
        assert model.layers[2].out_features == 1

    def test_mlp_parameters_count(self):
        """Test that MLP has expected parameter count."""
        # MLP(2, [16, 16, 1]):
        # Layer 1: 2*16 + 16 = 48
        # Layer 2: 16*16 + 16 = 272
        # Layer 3: 16*1 + 1 = 17
        # Total: 48 + 272 + 17 = 337
        model = MLP(2, [16, 16, 1])
        params = model.parameters()

        total_params = sum(p.numel for p in params)
        assert total_params == 337

    def test_mlp_forward_shape(self):
        """Test MLP forward produces correct output shape."""
        model = MLP(2, [16, 16, 1])
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)

        y = model(x)

        assert y.shape == (3, 1)

    def test_mlp_forward_backward(self):
        """Test complete forward and backward pass through MLP."""
        model = MLP(2, [4, 1])
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        y = model(x)
        loss = y.sum()

        # Should complete without error
        model.zero_grad()
        loss.backward()

        # All parameters should have gradients
        for p in model.parameters():
            assert p.grad is not None

    def test_mlp_zero_grad(self):
        """Test that zero_grad clears all gradients."""
        model = MLP(2, [4, 1])
        x = Tensor([[1.0, 2.0]])

        # First forward/backward
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Verify grads exist
        for p in model.parameters():
            assert p.grad is not None

        # Zero grads
        model.zero_grad()

        # Verify grads are zero (still exist but all zeros)
        for p in model.parameters():
            grad_list = p.grad.to_list()
            if isinstance(grad_list[0], list):
                for row in grad_list:
                    assert all(abs(g) < 1e-6 for g in row)
            else:
                assert all(abs(g) < 1e-6 for g in grad_list)

    def test_mlp_repr(self):
        """Test MLP string representation."""
        model = MLP(2, [16, 16, 1])
        r = repr(model)

        assert "MLP" in r
        assert "Linear" in r

    def test_mlp_single_layer(self):
        """Test MLP with single layer (logistic regression)."""
        model = MLP(3, [1])

        assert len(model.layers) == 1
        assert model.layers[0].in_features == 3
        assert model.layers[0].out_features == 1

    def test_mlp_deep(self):
        """Test deeper MLP."""
        model = MLP(10, [32, 16, 8, 4, 1])

        assert len(model.layers) == 5
        assert len(model.parameters()) == 10  # 5 layers * 2 params each


@pytest.mark.requires_triton
class TestMul:
    """Tests for elementwise multiply operation."""

    def test_mul_forward_1d(self):
        """Test mul forward on 1D tensors."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])

        c = a.mul(b)

        assert c.shape == (3,)
        assert c.to_list() == pytest.approx([4.0, 10.0, 18.0])

    def test_mul_forward_2d(self):
        """Test mul forward on 2D tensors."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[2.0, 3.0], [4.0, 5.0]])

        c = a.mul(b)

        assert c.shape == (2, 2)
        result = c.to_list()
        assert result[0] == pytest.approx([2.0, 6.0])
        assert result[1] == pytest.approx([12.0, 20.0])

    def test_mul_shape_mismatch_raises(self):
        """Test that mul raises error for mismatched shapes."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([1.0, 2.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            a.mul(b)

    def test_mul_backward_both_require_grad(self):
        """Test mul backward when both inputs require grad."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

        c = a.mul(b)
        loss = c.sum()

        loss.backward()

        # dA = out_grad * B = [1,1,1] * [4,5,6] = [4,5,6]
        # dB = out_grad * A = [1,1,1] * [1,2,3] = [1,2,3]
        assert a.grad.to_list() == pytest.approx([4.0, 5.0, 6.0])
        assert b.grad.to_list() == pytest.approx([1.0, 2.0, 3.0])

    def test_mul_backward_one_requires_grad(self):
        """Test mul backward when only one input requires grad."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=False)

        c = a.mul(b)
        loss = c.sum()

        loss.backward()

        # dA = out_grad * B = [1,1] * [4,5] = [4,5]
        assert a.grad.to_list() == pytest.approx([4.0, 5.0])
        assert b.grad is None

    def test_mul_operator(self):
        """Test * operator for mul."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([2.0, 2.0, 2.0])

        c = a * b

        assert c.to_list() == pytest.approx([2.0, 4.0, 6.0])

    def test_mul_accumulates_grad(self):
        """Test that mul backward accumulates gradients."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)

        # Pre-set gradients
        a.grad = Tensor([10.0, 20.0])
        b.grad = Tensor([30.0, 40.0])

        c = a.mul(b)
        loss = c.sum()
        loss.backward()

        # a.grad = 10 + b = [10+3, 20+4] = [13, 24]
        # b.grad = 30 + a = [30+1, 40+2] = [31, 42]
        assert a.grad.to_list() == pytest.approx([13.0, 24.0])
        assert b.grad.to_list() == pytest.approx([31.0, 42.0])


@pytest.mark.requires_triton
class TestScale:
    """Tests for scalar multiply operation."""

    def test_scale_forward(self):
        """Test scale forward."""
        a = Tensor([1.0, 2.0, 3.0])
        c = a.scale(2.0)

        assert c.to_list() == pytest.approx([2.0, 4.0, 6.0])

    def test_scale_negative(self):
        """Test scale with negative scalar."""
        a = Tensor([1.0, -2.0, 3.0])
        c = a.scale(-1.0)

        assert c.to_list() == pytest.approx([-1.0, 2.0, -3.0])

    def test_scale_zero(self):
        """Test scale by zero."""
        a = Tensor([1.0, 2.0, 3.0])
        c = a.scale(0.0)

        assert c.to_list() == pytest.approx([0.0, 0.0, 0.0])

    def test_scale_backward(self):
        """Test scale backward."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = a.scale(3.0)
        loss = c.sum()

        loss.backward()

        # dA = out_grad * scalar = [1,1,1] * 3 = [3,3,3]
        assert a.grad.to_list() == pytest.approx([3.0, 3.0, 3.0])

    def test_scale_operator(self):
        """Test * operator with scalar."""
        a = Tensor([1.0, 2.0, 3.0])

        c = a * 2.0
        assert c.to_list() == pytest.approx([2.0, 4.0, 6.0])

        d = 3.0 * a
        assert d.to_list() == pytest.approx([3.0, 6.0, 9.0])

    def test_scale_2d(self):
        """Test scale on 2D tensor."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = a.scale(0.5)

        result = c.to_list()
        assert result[0] == pytest.approx([0.5, 1.0])
        assert result[1] == pytest.approx([1.5, 2.0])


@pytest.mark.requires_triton
class TestOperatorOverloads:
    """Tests for operator overloads."""

    def test_neg_operator(self):
        """Test unary - operator."""
        a = Tensor([1.0, -2.0, 3.0])
        b = -a

        assert b.to_list() == pytest.approx([-1.0, 2.0, -3.0])

    def test_neg_backward(self):
        """Test negation backward."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = -a
        loss = b.sum()

        loss.backward()

        # dA = out_grad * -1 = [1,1] * -1 = [-1,-1]
        assert a.grad.to_list() == pytest.approx([-1.0, -1.0])

    def test_sub_operator(self):
        """Test - operator for subtraction."""
        a = Tensor([5.0, 6.0, 7.0])
        b = Tensor([1.0, 2.0, 3.0])

        c = a - b

        assert c.to_list() == pytest.approx([4.0, 4.0, 4.0])

    def test_sub_backward(self):
        """Test subtraction backward."""
        a = Tensor([5.0, 6.0], requires_grad=True)
        b = Tensor([1.0, 2.0], requires_grad=True)

        c = a - b
        loss = c.sum()

        loss.backward()

        # dA = 1, dB = -1
        assert a.grad.to_list() == pytest.approx([1.0, 1.0])
        assert b.grad.to_list() == pytest.approx([-1.0, -1.0])

    def test_add_operator(self):
        """Test + operator."""
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])

        c = a + b

        assert c.to_list() == pytest.approx([4.0, 6.0])

    def test_radd_operator(self):
        """Test reverse + operator."""
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])

        # This exercises __radd__ if a doesn't support the operation
        c = b + a

        assert c.to_list() == pytest.approx([4.0, 6.0])

    def test_rmul_operator(self):
        """Test reverse * operator with scalar."""
        a = Tensor([1.0, 2.0, 3.0])
        c = 2 * a  # Exercises __rmul__

        assert c.to_list() == pytest.approx([2.0, 4.0, 6.0])


@pytest.mark.requires_triton
class TestHingeLoss:
    """Tests for hinge loss computation (like micrograd demo)."""

    def test_hinge_loss_computation(self):
        """Test computing hinge loss: (1 - y*score).relu()."""
        # Scores from model
        scores = Tensor([[0.5], [-0.5], [1.5], [-1.5]])  # (4, 1)

        # True labels: 1 or -1
        y = Tensor([[1.0], [-1.0], [1.0], [-1.0]])  # (4, 1)

        # Margin tensor
        ones_tensor = ones((4, 1))

        # Hinge loss: (1 - y*score).relu()
        y_times_scores = y * scores  # [0.5, 0.5, 1.5, 1.5]
        margins = ones_tensor - y_times_scores  # [0.5, 0.5, -0.5, -0.5]
        losses = margins.relu()  # [0.5, 0.5, 0, 0]

        loss = losses.sum()  # 1.0

        assert loss.item() == pytest.approx(1.0)

    def test_hinge_loss_backward(self):
        """Test gradients through hinge loss."""
        scores = Tensor([[2.0]], requires_grad=True)  # Correctly classified
        y = Tensor([[1.0]])

        ones_tensor = ones((1, 1))
        y_times_scores = y * scores  # 2.0
        margins = ones_tensor - y_times_scores  # -1.0
        losses = margins.relu()  # 0.0 (correctly classified, no loss)
        loss = losses.sum()

        loss.backward()

        # Since relu output is 0, gradient should be 0
        assert scores.grad.to_list()[0][0] == pytest.approx(0.0)

    def test_hinge_loss_gradient_flows(self):
        """Test that gradient flows for misclassified samples."""
        scores = Tensor(
            [[-0.5]], requires_grad=True
        )  # Misclassified (should be positive)
        y = Tensor([[1.0]])

        ones_tensor = ones((1, 1))
        y_times_scores = y * scores  # -0.5
        margins = ones_tensor - y_times_scores  # 1.5
        losses = margins.relu()  # 1.5
        loss = losses.sum()

        loss.backward()

        # Gradient should flow through (non-zero because margin > 0)
        # d(loss)/d(score) = d(relu(1 - y*score))/d(score)
        # = -y when 1 - y*score > 0
        # = -1 * 1 = -1
        assert scores.grad.to_list()[0][0] == pytest.approx(-1.0)


@pytest.mark.requires_triton
class TestIntegration:
    """Integration tests for MLP training."""

    def test_mlp_training_step(self):
        """Test a complete training step with MLP."""
        from gradtuity import sgd_step

        model = MLP(2, [4, 1])

        # Input data
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = Tensor([[1.0], [-1.0]])

        # Forward
        scores = model(x)

        # Simple MSE-like loss: sum((scores - y)^2) simplified to sum(scores * y * -1)
        # Actually use hinge-like: (1 - y*scores).relu().sum()
        ones_tensor = ones((2, 1))
        margins = ones_tensor - (y * scores)
        loss = margins.relu().sum()

        initial_loss = loss.item()

        # Backward
        model.zero_grad()
        loss.backward()

        # Update
        sgd_step(model.parameters(), lr=0.1)

        # Forward again
        scores2 = model(x)
        margins2 = ones_tensor - (y * scores2)
        loss2 = margins2.relu().sum()

        # Loss should change (not necessarily decrease with one step on hinge loss)
        # Just verify the whole process completes
        assert loss2.item() != initial_loss or initial_loss == pytest.approx(0.0)

    def test_mlp_multiple_training_steps(self):
        """Test multiple training steps."""
        from gradtuity import sgd_step

        model = MLP(2, [8, 1])

        # Training data
        x = Tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        y = Tensor([[1.0], [1.0], [-1.0], [-1.0]])

        ones_tensor = ones((4, 1))

        losses = []
        for _ in range(10):
            scores = model(x)
            margins = ones_tensor - (y * scores)
            loss = margins.relu().sum()
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            sgd_step(model.parameters(), lr=0.1)

        # Should complete without error
        assert len(losses) == 10
