"""
Tests for functional.py - Tensor factory functions.

These tests require a CUDA-enabled GPU to run.
Uses pytest parametrization to reduce test duplication.
"""

import pytest

from gradtuity.functional import (
    zeros,
    zeros_like,
    ones,
    ones_like,
    randn,
    full,
    full_like,
    zero_grad,
    sgd_step,
)
from gradtuity.tensor import Tensor


@pytest.mark.requires_cuda
class TestZeros:
    """Tests for zeros() function (CUDA only, no Triton needed)."""

    @pytest.mark.parametrize(
        "shape,expected_numel",
        [
            ((5,), 5),
            ((3, 4), 12),
            ((100, 200), 20000),
        ],
    )
    def test_zeros_shape_and_numel(self, shape, expected_numel):
        """Test zeros creates correct shape and numel."""
        t = zeros(shape)
        assert t.shape == shape
        assert t.numel == expected_numel

    def test_zeros_values_are_zero(self):
        """Test that all values are 0.0."""
        t = zeros((3, 4))
        data = t.to_list()
        for row in data:
            for val in row:
                assert val == 0.0

    def test_zeros_with_requires_grad(self):
        """Test zeros with requires_grad=True."""
        t = zeros((2, 3), requires_grad=True)
        assert t.requires_grad is True

    def test_zeros_rejects_rank_3(self):
        """Test that zeros rejects rank 3."""
        with pytest.raises(ValueError, match="rank"):
            zeros((2, 3, 4))


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestTritonFactories:
    """Tests for ones() and full() functions (require Triton)."""

    @pytest.mark.parametrize(
        "factory,fill_value,shape",
        [
            (ones, 1.0, (5,)),
            (ones, 1.0, (3, 4)),
            (ones, 1.0, (100, 200)),
            (lambda s, **kw: full(s, 3.14, **kw), 3.14, (5,)),
            (lambda s, **kw: full(s, -2.5, **kw), -2.5, (3, 4)),
            (lambda s, **kw: full(s, 0.0, **kw), 0.0, (3, 3)),
        ],
    )
    def test_factory_creates_correct_values(self, factory, fill_value, shape):
        """Test factory functions create tensors with correct values."""
        t = factory(shape)
        assert t.shape == shape

        data = t.to_list()
        # Flatten for uniform checking
        flat = data if isinstance(data[0], float) else [v for row in data for v in row]
        for val in flat:
            assert val == pytest.approx(fill_value)

    @pytest.mark.parametrize("factory", [ones, lambda s, **kw: full(s, 5.0, **kw)])
    def test_factory_with_requires_grad(self, factory):
        """Test factory functions with requires_grad=True."""
        t = factory((2, 3), requires_grad=True)
        assert t.requires_grad is True


@pytest.mark.requires_cuda
class TestZerosLike:
    """Tests for zeros_like() function (CUDA only)."""

    @pytest.mark.parametrize(
        "ref_data",
        [
            [1.0, 2.0, 3.0],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        ],
    )
    def test_zeros_like_matches_shape_with_zeros(self, ref_data):
        """Test zeros_like matches reference shape and fills with zeros."""
        ref = Tensor(ref_data)
        t = zeros_like(ref)

        assert t.shape == ref.shape
        assert t.numel == ref.numel

        data = t.to_list()
        flat = data if isinstance(data[0], float) else [v for row in data for v in row]
        for val in flat:
            assert val == 0.0

    def test_zeros_like_with_requires_grad(self):
        """Test zeros_like with requires_grad=True."""
        ref = Tensor([1.0, 2.0])
        t = zeros_like(ref, requires_grad=True)
        assert t.requires_grad is True


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestTritonLikeFunctions:
    """Tests for ones_like() and full_like() functions (require Triton)."""

    @pytest.mark.parametrize(
        "like_fn,fill_value,ref_data",
        [
            (ones_like, 1.0, [5.0, 6.0, 7.0]),
            (ones_like, 1.0, [[1.0, 2.0], [3.0, 4.0]]),
            (lambda t, **kw: full_like(t, 7.0, **kw), 7.0, [1.0, 2.0, 3.0]),
            (lambda t, **kw: full_like(t, -1.0, **kw), -1.0, [[1.0, 2.0], [3.0, 4.0]]),
        ],
    )
    def test_like_fn_matches_shape_with_fill_value(self, like_fn, fill_value, ref_data):
        """Test _like functions match reference shape and fill correctly."""
        ref = Tensor(ref_data)
        t = like_fn(ref)

        assert t.shape == ref.shape

        data = t.to_list()
        flat = data if isinstance(data[0], float) else [v for row in data for v in row]
        for val in flat:
            assert val == pytest.approx(fill_value)


@pytest.mark.requires_cuda
class TestRandn:
    """Tests for randn() function."""

    @pytest.mark.parametrize(
        "shape,expected_numel",
        [
            ((100,), 100),
            ((50, 20), 1000),
        ],
    )
    def test_randn_shape_and_numel(self, shape, expected_numel):
        """Test randn creates correct shape and numel."""
        t = randn(shape)
        assert t.shape == shape
        assert t.numel == expected_numel

    def test_randn_with_seed_reproducibility(self):
        """Test that same seed produces same values."""
        t1 = randn((10,), seed=42)
        t2 = randn((10,), seed=42)

        data1 = t1.to_list()
        data2 = t2.to_list()

        for v1, v2 in zip(data1, data2):
            assert v1 == pytest.approx(v2)

    def test_randn_different_seeds_differ(self):
        """Test that different seeds produce different values."""
        t1 = randn((10,), seed=42)
        t2 = randn((10,), seed=123)

        data1 = t1.to_list()
        data2 = t2.to_list()

        differences = sum(1 for v1, v2 in zip(data1, data2) if abs(v1 - v2) > 0.01)
        assert differences > 0

    def test_randn_with_requires_grad(self):
        """Test randn with requires_grad=True."""
        t = randn((5, 5), requires_grad=True)
        assert t.requires_grad is True

    def test_randn_statistics(self):
        """Test that randn produces approximately normal distribution."""
        t = randn((1000,), seed=42)
        data = t.to_list()

        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std = variance**0.5

        assert abs(mean) < 0.2  # Mean should be close to 0
        assert 0.8 < std < 1.2  # Std should be close to 1

    def test_randn_rejects_rank_3(self):
        """Test that randn rejects rank 3."""
        with pytest.raises(ValueError, match="rank"):
            randn((2, 3, 4))


@pytest.mark.requires_cuda
class TestIntegration:
    """Integration tests combining functional operations."""

    def test_zeros_creates_independent_tensors(self):
        """Test that zeros creates independent allocations."""
        t1 = zeros((3,))
        t2 = zeros((3,))
        assert t1.data_ptr() != t2.data_ptr()

    @pytest.mark.requires_triton
    def test_all_factory_functions_create_valid_tensors(self):
        """Test that all factory functions create valid tensors."""
        tensors = [
            zeros((3, 4)),
            ones((3, 4)),
            randn((3, 4), seed=42),
            full((3, 4), 2.5),
        ]

        for t in tensors:
            assert t.shape == (3, 4)
            assert t.numel == 12
            assert t.data_ptr() > 0

            data = t.to_list()
            assert len(data) == 3
            assert len(data[0]) == 4

    @pytest.mark.requires_triton
    def test_like_functions_match_reference_shape(self):
        """Test that _like functions match reference shape."""
        ref = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        for t in [zeros_like(ref), ones_like(ref), full_like(ref, 9.0)]:
            assert t.shape == ref.shape
            assert t.numel == ref.numel


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestZeroGrad:
    """Tests for zero_grad() function."""

    def test_zero_grad_allocates_grad_if_none(self):
        """Test that zero_grad allocates grad tensor if None."""
        p = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert p.grad is None

        zero_grad([p])

        assert p.grad is not None
        assert p.grad.shape == p.shape
        # Should be all zeros
        assert p.grad.to_list() == pytest.approx([0.0, 0.0, 0.0])

    def test_zero_grad_zeros_existing_grad(self):
        """Test that zero_grad zeros out existing grad tensor."""
        p = Tensor([1.0, 2.0], requires_grad=True)
        # Set a non-zero gradient
        p.grad = Tensor([5.0, 10.0])

        zero_grad([p])

        assert p.grad.to_list() == pytest.approx([0.0, 0.0])

    def test_zero_grad_multiple_params(self):
        """Test zero_grad with multiple parameters."""
        p1 = Tensor([1.0, 2.0], requires_grad=True)
        p2 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        # Set non-zero gradients
        p1.grad = Tensor([5.0, 10.0])
        p2.grad = Tensor([[1.0, 1.0], [1.0, 1.0]])

        zero_grad([p1, p2])

        assert p1.grad.to_list() == pytest.approx([0.0, 0.0])
        assert p2.grad.to_list() == [[0.0, 0.0], [0.0, 0.0]]

    def test_zero_grad_skips_non_requires_grad(self):
        """Test that zero_grad skips tensors without requires_grad."""
        p = Tensor([1.0, 2.0], requires_grad=False)

        # Should not raise, just skip
        zero_grad([p])

        assert p.grad is None


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestSgdStep:
    """Tests for sgd_step() function."""

    def test_sgd_step_basic(self):
        """Test basic SGD update: param -= lr * grad."""
        p = Tensor([10.0, 20.0], requires_grad=True)
        p.grad = Tensor([1.0, 2.0])

        sgd_step([p], lr=0.5)

        # Expected: [10 - 0.5*1, 20 - 0.5*2] = [9.5, 19.0]
        assert p.to_list() == pytest.approx([9.5, 19.0])

    def test_sgd_step_2d(self):
        """Test SGD update on 2D tensor."""
        p = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        p.grad = Tensor([[0.1, 0.2], [0.3, 0.4]])

        sgd_step([p], lr=1.0)

        # Expected: original - lr * grad
        expected = [[0.9, 1.8], [2.7, 3.6]]
        result = p.to_list()
        assert result[0] == pytest.approx(expected[0])
        assert result[1] == pytest.approx(expected[1])

    def test_sgd_step_multiple_params(self):
        """Test SGD step with multiple parameters."""
        p1 = Tensor([10.0], requires_grad=True)
        p1.grad = Tensor([2.0])

        p2 = Tensor([20.0], requires_grad=True)
        p2.grad = Tensor([4.0])

        sgd_step([p1, p2], lr=0.25)

        # p1: 10 - 0.25*2 = 9.5
        # p2: 20 - 0.25*4 = 19.0
        assert p1.to_list()[0] == pytest.approx(9.5)
        assert p2.to_list()[0] == pytest.approx(19.0)

    def test_sgd_step_raises_if_grad_is_none(self):
        """Test that sgd_step raises if gradient is None."""
        p = Tensor([1.0], requires_grad=True)
        # Don't set grad

        with pytest.raises(RuntimeError, match="None"):
            sgd_step([p], lr=0.1)

    def test_sgd_step_skips_non_requires_grad(self):
        """Test that sgd_step skips tensors without requires_grad."""
        p = Tensor([1.0, 2.0], requires_grad=False)

        # Should not raise, just skip
        sgd_step([p], lr=0.1)

        # Value unchanged
        assert p.to_list() == pytest.approx([1.0, 2.0])
