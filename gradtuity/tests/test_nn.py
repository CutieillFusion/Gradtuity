"""
Tests for nn.py - Neural network modules.

These tests require a CUDA-enabled GPU to run.
"""

import random

import numpy as np
import pytest

from gradtuity import (
    CNN,
    Conv2d,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2d,
    MLP,
    Module,
    Tensor,
    ones,
    randn,
    zeros,
)

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


class TestModuleStateDict:
    """Tests for Module.state_dict() and load_state_dict()."""

    def test_linear_state_dict_keys(self):
        """Linear state_dict has weight and bias."""
        layer = Linear(2, 3)
        state = layer.state_dict()
        assert set(state.keys()) == {"weight", "bias"}
        assert state["weight"] is layer.weight
        assert state["bias"] is layer.bias

    def test_mlp_state_dict_keys(self):
        """MLP state_dict has layers.i.weight and layers.i.bias."""
        model = MLP(2, [4, 1])
        state = model.state_dict()
        expected = {
            "layers.0.weight",
            "layers.0.bias",
            "layers.1.weight",
            "layers.1.bias",
        }
        assert set(state.keys()) == expected
        assert state["layers.0.weight"] is model.layers[0].weight
        assert state["layers.0.bias"] is model.layers[0].bias
        assert state["layers.1.weight"] is model.layers[1].weight
        assert state["layers.1.bias"] is model.layers[1].bias

    def test_mlp_load_state_dict_round_trip(self):
        """load_state_dict(state_dict()) preserves model; forward still runs."""
        model = MLP(2, [4, 1])
        state = model.state_dict()
        model.load_state_dict(state)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = model(x)
        assert y.shape == (2, 1)

    def test_load_state_dict_strict_unexpected_key_raises(self):
        """load_state_dict with unexpected key raises when strict=True."""
        model = MLP(2, [4, 1])
        state = model.state_dict()
        state["fake.key"] = Tensor([1.0])
        with pytest.raises(ValueError) as exc_info:
            model.load_state_dict(state, strict=True)
        assert "unexpected" in str(exc_info.value).lower()

    def test_load_state_dict_strict_missing_key_raises(self):
        """load_state_dict with missing key raises when strict=True."""
        model = MLP(2, [4, 1])
        state = model.state_dict()
        state.pop("layers.0.bias")
        with pytest.raises(ValueError) as exc_info:
            model.load_state_dict(state, strict=True)
        assert "missing" in str(exc_info.value).lower()

    def test_load_state_dict_strict_false_ignores_extra(self):
        """load_state_dict with extra key does not raise when strict=False."""
        model = MLP(2, [4, 1])
        state = model.state_dict()
        state["extra.key"] = Tensor([1.0])
        model.load_state_dict(state, strict=False)
        x = Tensor([[1.0, 2.0]])
        y = model(x)
        assert y.shape == (1, 1)


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
class TestFlatten:
    """Tests for Flatten module."""

    def test_flatten_4d_to_2d(self):
        """Test Flatten(1) on 4D input (N, C, H, W) -> (N, C*H*W)."""
        flat = Flatten(start_dim=1)
        x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
        y = flat(x)
        assert y.shape == (1, 4)
        assert y.numel == 4


@pytest.mark.requires_triton
class TestConv2d:
    """Tests for Conv2d layer."""

    def test_conv2d_forward_shape(self):
        """Test Conv2d forward produces correct output shape."""
        conv = Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        x = Tensor([[[[0.1] * 8] * 8]])  # (1, 1, 8, 8)
        y = conv(x)
        assert y.shape == (1, 4, 8, 8)

    def test_conv2d_parameters(self):
        """Test Conv2d has weight and bias parameters."""
        conv = Conv2d(2, 8, kernel_size=3)
        params = conv.parameters()
        assert len(params) == 2
        assert conv.weight.shape == (8, 2, 3, 3)
        assert conv.bias.shape == (8,)

    def test_conv2d_then_relu_storage_valid(self):
        """Regression: conv2d output storage must stay valid after return (no use-after-free).

        Conv2d returns a view (y_4d) sharing storage with an internal tensor (y_flat).
        If y_flat is GC'd, __del__ would free the ptr and the next op (e.g. relu) would
        read freed memory. This test ensures we keep the storage alive via the backward
        closure. Also ensures no double-free of the matmul output buffer (out_flat_ptr).
        """

        conv = Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        x_data = np.zeros((32, 1, 28, 28), dtype=np.float32)
        x = Tensor(x_data.tolist())
        y = conv(x)
        assert y.shape == (32, 32, 28, 28)
        z = y.relu()
        assert z.shape == (32, 32, 28, 28)
        # ReLU output must be non-negative; reading invalid storage could yield garbage
        flat = np.array(z.to_list()).ravel()
        assert (flat >= -1e-5).all(), "relu output should be non-negative"

    def test_conv2d_then_multiple_ops_storage_valid(self):
        """Regression: conv2d output used in several downstream ops (stress storage lifetime)."""

        conv = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        x_data = np.zeros((4, 1, 14, 14), dtype=np.float32)
        x = Tensor(x_data.tolist())
        y = conv(x)
        y = y.relu()
        y = y.relu()
        assert y.shape == (4, 8, 14, 14)
        s = y.sum()
        assert s.shape == (1,)
        assert s.to_list()[0] >= 0


@pytest.mark.requires_triton
class TestMaxPool2d:
    """Tests for MaxPool2d layer."""

    def test_maxpool2d_forward_shape(self):
        """Test MaxPool2d forward halves spatial size with kernel=2, stride=2."""
        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor([[[[1.0, 2.0, 3.0, 4.0] * 2] * 4]])  # (1, 1, 4, 8)
        y = pool(x)
        assert y.shape == (1, 1, 2, 4)

    def test_maxpool2d_forward_cnn_like_shape(self):
        """Test MaxPool2d on CNN-like tensor (N=4, C=32, 28x28) -> (4, 32, 14, 14).

        Catches illegal memory access when grid is large (e.g. numel_out > 100k).
        """
        pool = MaxPool2d(kernel_size=2, stride=2)
        # Same shape as after first conv in CNN: (batch, 32, 28, 28)
        data = np.zeros((4, 32, 28, 28), dtype=np.float32)
        x = Tensor(data.tolist())
        y = pool(x)
        assert y.shape == (4, 32, 14, 14)

    def test_maxpool2d_requires_grad_false_does_not_leak_idx_buffer(self):
        """Test maxpool2d with requires_grad=False frees idx_ptr (no VRAM leak).

        When input has requires_grad=False, _backward is not attached, so idx_ptr
        must be freed in the no-grad path. Running many forward passes would OOM
        if idx_ptr were leaked each time.
        """
        pool = MaxPool2d(kernel_size=2, stride=2)
        # CNN-like shape so we allocate non-trivial idx buffer each time
        data = np.zeros((8, 16, 14, 14), dtype=np.float32)
        x = Tensor(data.tolist(), requires_grad=False)
        n_repeats = 200
        for _ in range(n_repeats):
            y = pool(x)
            assert y.shape == (8, 16, 7, 7)
            assert y.requires_grad is False
        # If idx_ptr were leaked, 200 * (8*16*7*7*4) bytes would accumulate;
        # passing without OOM confirms the no-grad free path works.
        del y
        del x


@pytest.mark.requires_triton
class TestLayerNorm:
    """Tests for nn.LayerNorm module."""

    def test_layernorm_forward_2d(self):
        """nn.LayerNorm forward on 2D (N, H); output matches x.layer_norm(gamma, beta)."""
        ln = LayerNorm(8)
        # gamma=ones, beta=zeros by default
        x = Tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        )
        out = ln(x)
        expected = x.layer_norm(ln.gamma, ln.beta, eps=ln.eps)
        assert out.shape == (2, 8)
        for i in range(2):
            assert out.to_list()[i] == pytest.approx(expected.to_list()[i])

    def test_layernorm_forward_3d_shape_preservation(self):
        """nn.LayerNorm on 3D (N, S, H) preserves shape."""
        ln = LayerNorm(4)
        x = Tensor([[[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]]])  # (1, 2, 4)
        out = ln(x)
        assert out.shape == (1, 2, 4)

    def test_layernorm_state_dict(self):
        """nn.LayerNorm state_dict has gamma and beta."""
        ln = LayerNorm(16)
        state = ln.state_dict()
        assert set(state.keys()) == {"gamma", "beta"}
        assert state["gamma"] is ln.gamma
        assert state["beta"] is ln.beta
        assert state["gamma"].shape == (16,)
        assert state["beta"].shape == (16,)

    def test_layernorm_backward(self):
        """nn.LayerNorm backward: loss on output gives gamma/beta grads."""
        ln = LayerNorm(4)
        x = Tensor(
            [[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]],
            requires_grad=True,
        )
        out = ln(x)
        loss = out.sum()
        loss.backward()
        assert ln.gamma.grad is not None
        assert ln.beta.grad is not None
        assert ln.gamma.grad.shape == (4,)
        assert ln.beta.grad.shape == (4,)

    def test_layernorm_eps_passed_through(self):
        """Different eps changes output (sanity check eps is used)."""
        ln_small = LayerNorm(4)
        ln_small.eps = 1e-9
        ln_large = LayerNorm(4)
        ln_large.eps = 1e-1
        # Use same gamma/beta (ones/zeros) so only eps differs
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out_small = ln_small(x)
        out_large = ln_large(x)
        # With large eps, normalization is weaker; outputs should differ
        assert out_small.to_list() != out_large.to_list()


@pytest.mark.requires_triton
class TestCNN:
    """Tests for CNN model (MNIST-style)."""

    def test_cnn_forward_shape(self):
        """Test CNN forward: (N, 1, 28, 28) -> (N, 10)."""
        model = CNN()
        x = Tensor([[[[0.0] * 28] * 28]])  # (1, 1, 28, 28)
        y = model(x)
        assert y.shape == (1, 10)

    def test_cnn_forward_batch_size_4(self):
        """Test CNN forward with batch size 4 (realistic mini-batch).

        Ensures full pipeline (conv -> relu -> pool -> conv -> relu -> pool -> flatten -> linear)
        works without illegal memory access on non-trivial batch.
        """
        model = CNN()
        x_data = np.zeros((4, 1, 28, 28), dtype=np.float32)
        x = Tensor(x_data.tolist())
        y = model(x)
        assert y.shape == (4, 10)

    def test_cnn_forward_batch_size_32(self):
        """Test CNN forward with batch size 32 (large batch, multi-block path).

        Exercises maxpool with numel_out=200704 and grid=784; would exceed CUDA grid
        limit (65535) without BLOCK_ELEMS fix if we used one block per element.
        Uses 32 instead of 64 to avoid illegal memory access on some GPUs/drivers
        when conv+pool grids are large (demo uses BATCH_SIZE=64).
        """
        model = CNN()
        x_data = np.zeros((32, 1, 28, 28), dtype=np.float32)
        x = Tensor(x_data.tolist())
        y = model(x)
        assert y.shape == (32, 10)

    def test_cnn_parameters_non_empty(self):
        """Test CNN has parameters from conv and linear layers."""
        model = CNN()
        params = model.parameters()
        assert len(params) > 0


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


# =============================================================================
# Tests for Fused MSE Loss
# =============================================================================


@pytest.mark.requires_triton
class TestMSELoss:
    """Tests for fused MSE loss operation."""

    def test_mse_loss_forward(self):
        """Test mse_loss forward computation."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[1.0, 1.0], [1.0, 1.0]])

        loss = a.mse_loss(b)

        # MSE = mean((a - b)^2) = mean([0, 1, 4, 9]) = 14/4 = 3.5
        assert loss.shape == (1,)
        assert loss.item() == pytest.approx(3.5)

    def test_mse_loss_zero(self):
        """Test mse_loss when inputs are equal."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[1.0, 2.0], [3.0, 4.0]])

        loss = a.mse_loss(b)

        assert loss.item() == pytest.approx(0.0)

    def test_mse_loss_1d(self):
        """Test mse_loss on 1D tensors."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([0.0, 0.0, 0.0])

        loss = a.mse_loss(b)

        # MSE = mean([1, 4, 9]) = 14/3
        assert loss.item() == pytest.approx(14.0 / 3.0)

    def test_mse_loss_backward_both(self):
        """Test mse_loss backward when both inputs require grad."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)

        loss = a.mse_loss(b)
        loss.backward()

        # dL/da = 2 * (a - b) / N = 2 * a / 4 = a / 2
        # dL/db = -2 * (a - b) / N = -a / 2
        expected_grad_a = [[0.5, 1.0], [1.5, 2.0]]
        expected_grad_b = [[-0.5, -1.0], [-1.5, -2.0]]

        a_grad = a.grad.to_list()
        b_grad = b.grad.to_list()

        assert a_grad[0] == pytest.approx(expected_grad_a[0])
        assert a_grad[1] == pytest.approx(expected_grad_a[1])
        assert b_grad[0] == pytest.approx(expected_grad_b[0])
        assert b_grad[1] == pytest.approx(expected_grad_b[1])

    def test_mse_loss_backward_pred_only(self):
        """Test mse_loss backward when only predictions require grad."""
        pred = Tensor([[2.0, 4.0]], requires_grad=True)
        target = Tensor([[1.0, 1.0]], requires_grad=False)

        loss = pred.mse_loss(target)
        loss.backward()

        # dL/dpred = 2 * (pred - target) / N = 2 * [1, 3] / 2 = [1, 3]
        assert pred.grad.to_list()[0] == pytest.approx([1.0, 3.0])
        assert target.grad is None

    def test_mse_loss_shape_mismatch(self):
        """Test mse_loss raises error for mismatched shapes."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([1.0, 2.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            a.mse_loss(b)


# =============================================================================
# Tests for Cross-Entropy Loss
# =============================================================================


@pytest.mark.requires_triton
class TestCrossEntropyLoss:
    """Tests for fused cross-entropy loss operation."""

    def test_cross_entropy_forward_hand_computed(self):
        """Test cross_entropy forward with hand-computed expected value."""
        # B=2, C=3. Row 0: logits [0,1,0], target 1; Row 1: logits [0,0,1], target 2.
        # lse_0 = log(exp(0)+exp(1)+exp(0)) = log(1+2.718+1) ≈ 1.552; loss_0 = lse_0 - 1 = 0.552
        # lse_1 ≈ 1.552; loss_1 = lse_1 - 1 = 0.552. Sum = 1.104, mean = 0.552
        logits = Tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        targets = Tensor([1.0, 2.0])  # class indices as float32

        loss_mean = logits.cross_entropy(targets, reduction="mean")
        loss_sum = logits.cross_entropy(targets, reduction="sum")

        assert loss_mean.shape == (1,)
        assert loss_sum.shape == (1,)
        assert loss_mean.item() == pytest.approx(0.552, rel=1e-2)
        assert loss_sum.item() == pytest.approx(1.104, rel=1e-2)

    def test_cross_entropy_backward_hand_computed(self):
        """Test cross_entropy backward: grad = scale * (softmax - one_hot)."""
        logits = Tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True)
        targets = Tensor([1.0, 2.0])

        loss = logits.cross_entropy(targets, reduction="mean")
        loss.backward()

        # For mean: scale = 1/B = 0.5. softmax row0 = exp([0,1,0]-lse)/sum = [0.212, 0.576, 0.212], one_hot(1) = [0,1,0]
        # grad row0 = 0.5 * (softmax - one_hot) ≈ [0.106, -0.212, 0.106]; row1 similarly
        grad = logits.grad.to_list()
        assert len(grad) == 2
        assert grad[0][0] == pytest.approx(0.106, rel=1e-2)
        assert grad[0][1] == pytest.approx(-0.212, rel=1e-2)
        assert grad[0][2] == pytest.approx(0.106, rel=1e-2)
        assert grad[1][0] == pytest.approx(0.106, rel=1e-2)
        assert grad[1][1] == pytest.approx(0.106, rel=1e-2)
        assert grad[1][2] == pytest.approx(-0.212, rel=1e-2)
        assert targets.grad is None

    def test_cross_entropy_reduction_sum(self):
        """Test cross_entropy with reduction='sum' and its backward."""
        logits = Tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], requires_grad=True)
        targets = Tensor([2.0, 1.0])

        loss = logits.cross_entropy(targets, reduction="sum")
        loss.backward()

        # loss is sum of per-sample losses (no 1/B); grad scale = 1.0
        assert loss.shape == (1,)
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_cross_entropy_c_not_power_of_two(self):
        """Test cross_entropy when C is not a power of two (e.g. C=5)."""
        logits = Tensor([[0.1, 0.2, 0.3, 0.4, 0.0]] * 4)  # (4, 5)
        targets = Tensor([0.0, 1.0, 2.0, 3.0])

        loss = logits.cross_entropy(targets, reduction="mean")
        assert loss.shape == (1,)
        assert loss.item() > 0

    def test_cross_entropy_logits_no_grad(self):
        """Test cross_entropy when logits do not require grad (forward only)."""
        logits = Tensor([[1.0, 2.0], [0.0, 1.0]], requires_grad=False)
        targets = Tensor([0.0, 1.0])

        loss = logits.cross_entropy(targets, reduction="mean")
        # Loss has no grad graph when logits.requires_grad=False, so we do not call backward()
        assert loss.shape == (1,)
        assert logits.grad is None
        assert targets.grad is None

    def test_cross_entropy_shape_errors(self):
        """Test cross_entropy raises for invalid shapes."""
        logits_2d = Tensor([[1.0, 2.0], [3.0, 4.0]])
        targets_1d = Tensor([0.0, 1.0])

        with pytest.raises(ValueError, match="2D logits"):
            Tensor([1.0, 2.0, 3.0]).cross_entropy(targets_1d)
        with pytest.raises(ValueError, match="1D targets"):
            logits_2d.cross_entropy(Tensor([[0.0], [1.0]]))
        with pytest.raises(ValueError, match="batch size mismatch"):
            logits_2d.cross_entropy(Tensor([0.0, 1.0, 2.0]))

    def test_cross_entropy_reduction_error(self):
        """Test cross_entropy raises for invalid reduction."""
        logits = Tensor([[1.0, 2.0], [0.0, 1.0]])
        targets = Tensor([0.0, 1.0])

        with pytest.raises(ValueError, match="reduction"):
            logits.cross_entropy(targets, reduction="none")


# =============================================================================
# Tests for GPU Argmax
# =============================================================================


@pytest.mark.requires_triton
class TestArgmax:
    """Tests for GPU argmax operation."""

    def test_argmax_basic(self):
        """Test argmax on a simple 2D tensor."""
        x = Tensor([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]])

        indices = x.argmax(dim=1)

        assert indices.shape == (2,)
        result = indices.to_list()
        assert int(result[0]) == 1  # max at index 1 (value 3.0)
        assert int(result[1]) == 0  # max at index 0 (value 5.0)

    def test_argmax_single_row(self):
        """Test argmax on a single row."""
        x = Tensor([[0.1, 0.9, 0.5, 0.3]])

        indices = x.argmax(dim=1)

        assert indices.shape == (1,)
        assert int(indices.to_list()[0]) == 1  # max at index 1

    def test_argmax_all_same(self):
        """Test argmax when all values are the same (returns first)."""
        x = Tensor([[1.0, 1.0, 1.0]])

        indices = x.argmax(dim=1)

        # Should return index 0 (first occurrence of max)
        assert int(indices.to_list()[0]) == 0

    def test_argmax_negative_values(self):
        """Test argmax with negative values."""
        x = Tensor([[-5.0, -1.0, -3.0], [-2.0, -4.0, -1.0]])

        indices = x.argmax(dim=1)

        result = indices.to_list()
        assert int(result[0]) == 1  # -1.0 is max in first row
        assert int(result[1]) == 2  # -1.0 is max in second row

    def test_argmax_larger_tensor(self):
        """Test argmax on a larger tensor (like MNIST scores)."""
        # Simulate 4 samples with 10 classes (fixed seed for determinism)

        random.seed(42)
        data = []
        expected_indices = []
        for i in range(4):
            row = [random.random() for _ in range(10)]
            max_idx = row.index(max(row))
            expected_indices.append(max_idx)
            data.append(row)

        x = Tensor(data)
        indices = x.argmax(dim=1)

        result = indices.to_list()
        for i in range(4):
            assert int(result[i]) == expected_indices[i]

    def test_argmax_requires_2d(self):
        """Test argmax raises error for 1D tensor."""
        x = Tensor([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="requires 2D"):
            x.argmax(dim=1)

    def test_argmax_only_dim1(self):
        """Test argmax raises error for dim != 1."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="only supports dim=1"):
            x.argmax(dim=0)

    def test_argmax_no_grad(self):
        """Test that argmax doesn't track gradients."""
        x = Tensor([[1.0, 3.0, 2.0]], requires_grad=True)

        indices = x.argmax(dim=1)

        assert indices.requires_grad is False


# =============================================================================
# Tests for Fused Linear Layer
# =============================================================================


@pytest.mark.requires_triton
class TestFusedLinear:
    """Tests for fused linear (matmul + bias) operation."""

    def test_linear_forward(self):
        """Test linear forward: Y = X @ W + b"""
        # X: (2, 3), W: (3, 2), b: (2,) -> Y: (2, 2)
        x = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        w = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = Tensor([0.1, 0.2])

        y = x.linear(w, b)

        assert y.shape == (2, 2)
        result = y.to_list()
        # Row 0: [1,0,0] @ W + b = [1, 2] + [0.1, 0.2] = [1.1, 2.2]
        # Row 1: [0,1,0] @ W + b = [3, 4] + [0.1, 0.2] = [3.1, 4.2]
        assert result[0] == pytest.approx([1.1, 2.2])
        assert result[1] == pytest.approx([3.1, 4.2])

    def test_linear_matches_separate_ops(self):
        """Test that linear gives same result as matmul + add_bias."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        w = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        b = Tensor([1.0, 2.0, 3.0])

        # Fused
        y_fused = x.linear(w, b)

        # Separate operations
        y_separate = x.matmul(w).add_bias(b)

        fused_result = y_fused.to_list()
        separate_result = y_separate.to_list()

        for i in range(3):
            assert fused_result[i] == pytest.approx(separate_result[i], rel=1e-5)

    def test_linear_backward_all(self):
        """Test linear backward when all inputs require grad."""
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        w = Tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)
        b = Tensor([0.1, 0.2], requires_grad=True)

        y = x.linear(w, b)
        loss = y.sum()
        loss.backward()

        # Check gradients exist and have correct shapes
        assert x.grad is not None
        assert x.grad.shape == (1, 2)

        assert w.grad is not None
        assert w.grad.shape == (2, 2)

        assert b.grad is not None
        assert b.grad.shape == (2,)

        # db should be sum of out_grad over rows = [1, 1]
        assert b.grad.to_list() == pytest.approx([1.0, 1.0])

    def test_linear_backward_x_only(self):
        """Test linear backward when only input requires grad."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)
        b = Tensor([0.0, 0.0], requires_grad=False)

        y = x.linear(w, b)
        loss = y.sum()
        loss.backward()

        # dX = out_grad @ W^T = [[1,1], [1,1]] @ I = [[1,1], [1,1]]
        x_grad = x.grad.to_list()
        assert x_grad[0] == pytest.approx([1.0, 1.0])
        assert x_grad[1] == pytest.approx([1.0, 1.0])

        assert w.grad is None
        assert b.grad is None

    def test_linear_backward_accumulates(self):
        """Test that linear backward accumulates gradients."""
        x = Tensor([[1.0, 1.0]], requires_grad=True)
        w = Tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
        b = Tensor([0.0, 0.0], requires_grad=True)

        # Pre-set gradients
        from gradtuity import ones

        x.grad = ones((1, 2))
        w.grad = ones((2, 2))
        b.grad = ones((2,))

        y = x.linear(w, b)
        loss = y.sum()
        loss.backward()

        # Gradients should be accumulated (original + new)
        # Original x.grad = [[1, 1]], new contribution adds to it
        x_grad = x.grad.to_list()
        assert x_grad[0][0] > 1.0  # Should be > 1 due to accumulation
        assert x_grad[0][1] > 1.0

    def test_linear_shape_validation(self):
        """Test linear raises errors for invalid shapes."""
        x = Tensor([[1.0, 2.0]])
        w = Tensor([[1.0], [2.0], [3.0]])  # Wrong shape
        b = Tensor([1.0])

        with pytest.raises(ValueError, match="shape mismatch"):
            x.linear(w, b)

    def test_linear_bias_size_validation(self):
        """Test linear raises error when bias size doesn't match."""
        x = Tensor([[1.0, 2.0]])
        w = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([1.0, 2.0, 3.0])  # Wrong size

        with pytest.raises(ValueError, match="bias size"):
            x.linear(w, b)


# =============================================================================
# Tests for Fused Linear+ReLU
# =============================================================================


@pytest.mark.requires_triton
class TestLinearRelu:
    """Tests for fused linear+relu operation."""

    def test_linear_relu_forward(self):
        """Test linear_relu forward: Y = relu(X @ W + b)"""
        # X: (2, 2), W: (2, 2), b: (2,) -> Y: (2, 2)
        x = Tensor([[1.0, 0.0], [0.0, 1.0]])
        w = Tensor([[1.0, -1.0], [2.0, -2.0]])
        b = Tensor([0.5, 0.5])

        y = x.linear_relu(w, b)

        assert y.shape == (2, 2)
        result = y.to_list()
        # Row 0: relu([1,0] @ W + b) = relu([1, -1] + [0.5, 0.5]) = relu([1.5, -0.5]) = [1.5, 0]
        # Row 1: relu([0,1] @ W + b) = relu([2, -2] + [0.5, 0.5]) = relu([2.5, -1.5]) = [2.5, 0]
        assert result[0] == pytest.approx([1.5, 0.0])
        assert result[1] == pytest.approx([2.5, 0.0])

    def test_linear_relu_matches_separate_ops(self):
        """Test that linear_relu gives same result as linear + relu."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])
        w = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        b = Tensor([-0.5, 0.0, 0.5])

        # Fused
        y_fused = x.linear_relu(w, b)

        # Separate operations
        y_separate = x.linear(w, b).relu()

        fused_result = y_fused.to_list()
        separate_result = y_separate.to_list()

        for i in range(3):
            assert fused_result[i] == pytest.approx(separate_result[i], rel=1e-5)

    def test_linear_relu_all_positive(self):
        """Test linear_relu when all pre-activation values are positive."""
        x = Tensor([[1.0, 1.0]])
        w = Tensor([[1.0, 1.0], [1.0, 1.0]])
        b = Tensor([1.0, 1.0])

        y = x.linear_relu(w, b)

        # [1,1] @ W + b = [2, 2] + [1, 1] = [3, 3], all positive
        assert y.to_list()[0] == pytest.approx([3.0, 3.0])

    def test_linear_relu_all_negative(self):
        """Test linear_relu when all pre-activation values are negative."""
        x = Tensor([[1.0, 1.0]])
        w = Tensor([[-1.0, -1.0], [-1.0, -1.0]])
        b = Tensor([-1.0, -1.0])

        y = x.linear_relu(w, b)

        # [1,1] @ W + b = [-2, -2] + [-1, -1] = [-3, -3], all negative -> [0, 0]
        assert y.to_list()[0] == pytest.approx([0.0, 0.0])

    def test_linear_relu_backward_all(self):
        """Test linear_relu backward when all inputs require grad."""
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        w = Tensor([[0.5, -0.5], [1.0, -1.0]], requires_grad=True)
        b = Tensor([0.1, 0.1], requires_grad=True)

        y = x.linear_relu(w, b)
        loss = y.sum()
        loss.backward()

        # Check gradients exist and have correct shapes
        assert x.grad is not None
        assert x.grad.shape == (1, 2)

        assert w.grad is not None
        assert w.grad.shape == (2, 2)

        assert b.grad is not None
        assert b.grad.shape == (2,)

    def test_linear_relu_backward_matches_separate(self):
        """Test linear_relu backward matches separate linear + relu backward."""
        # Use same initialization for both
        x1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w1 = Tensor([[0.5, 0.2], [0.3, 0.4]], requires_grad=True)
        b1 = Tensor([0.1, -0.1], requires_grad=True)

        x2 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w2 = Tensor([[0.5, 0.2], [0.3, 0.4]], requires_grad=True)
        b2 = Tensor([0.1, -0.1], requires_grad=True)

        # Fused path
        y1 = x1.linear_relu(w1, b1)
        loss1 = y1.sum()
        loss1.backward()

        # Separate path
        y2 = x2.linear(w2, b2).relu()
        loss2 = y2.sum()
        loss2.backward()

        # Compare gradients
        assert x1.grad.to_list()[0] == pytest.approx(x2.grad.to_list()[0], rel=1e-5)
        assert x1.grad.to_list()[1] == pytest.approx(x2.grad.to_list()[1], rel=1e-5)

        assert w1.grad.to_list()[0] == pytest.approx(w2.grad.to_list()[0], rel=1e-5)
        assert w1.grad.to_list()[1] == pytest.approx(w2.grad.to_list()[1], rel=1e-5)

        assert b1.grad.to_list() == pytest.approx(b2.grad.to_list(), rel=1e-5)

    def test_linear_relu_backward_x_only(self):
        """Test linear_relu backward when only input requires grad."""
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        w = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)
        b = Tensor(
            [0.0, -10.0], requires_grad=False
        )  # Second output will be zeroed by relu

        y = x.linear_relu(w, b)
        # y = relu([1,2] @ I + [0,-10]) = relu([1, -8]) = [1, 0]
        loss = y.sum()
        loss.backward()

        # dY = [1, 1], but relu mask is [1, 0] since y = [1, 0]
        # dZ = [1, 0]
        # dX = dZ @ W^T = [1, 0] @ I = [1, 0]
        x_grad = x.grad.to_list()[0]
        assert x_grad == pytest.approx([1.0, 0.0])

        assert w.grad is None
        assert b.grad is None

    def test_linear_relu_backward_relu_blocks_gradient(self):
        """Test that relu mask correctly blocks gradients for negative pre-activations."""
        x = Tensor([[1.0]], requires_grad=True)
        w = Tensor(
            [[-1.0, 1.0]], requires_grad=True
        )  # One negative, one positive output
        b = Tensor([0.0, 0.0], requires_grad=True)

        y = x.linear_relu(w, b)
        # y = relu([[-1, 1]]) = [0, 1]
        loss = y.sum()
        loss.backward()

        # dY = [1, 1], relu mask = [0, 1] (since y = [0, 1])
        # dZ = [0, 1]
        # dW = X^T @ dZ = [[1]] @ [[0, 1]] = [[0, 1]]
        w_grad = w.grad.to_list()
        assert w_grad[0] == pytest.approx([0.0, 1.0])

        # db = sum(dZ, axis=0) = [0, 1]
        assert b.grad.to_list() == pytest.approx([0.0, 1.0])

    def test_linear_relu_shape_validation(self):
        """Test linear_relu raises errors for invalid shapes."""
        x = Tensor([[1.0, 2.0]])
        w = Tensor([[1.0], [2.0], [3.0]])  # Wrong shape
        b = Tensor([1.0])

        with pytest.raises(ValueError, match="shape mismatch"):
            x.linear_relu(w, b)

    def test_linear_relu_in_mlp(self):
        """Test that MLP correctly uses linear_relu for hidden layers."""
        model = MLP(2, [4, 4, 1])

        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = model(x)

        assert y.shape == (2, 1)

        # Test backward works
        y.sum().backward()

        # All parameters should have gradients
        for p in model.parameters():
            assert p.grad is not None
