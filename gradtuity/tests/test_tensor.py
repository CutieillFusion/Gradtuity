"""
Tests for tensor.py - Tensor class with GPU storage.

These tests require a CUDA-enabled GPU to run.
"""

import pytest

from gradtuity.tensor import Tensor


class TestTensorConstruction:
    """Tests for Tensor construction from Python data."""

    def test_create_1d_tensor(self):
        """Test creating a 1D tensor from a list."""
        data = [1.0, 2.0, 3.0]
        t = Tensor(data)

        assert t.shape == (3,)
        assert t.numel == 3
        assert t.nbytes == 12  # 3 * 4 bytes
        assert t.ndim == 1
        assert t.requires_grad is False

    def test_create_2d_tensor(self):
        """Test creating a 2D tensor from nested lists."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        t = Tensor(data)

        assert t.shape == (2, 3)
        assert t.numel == 6
        assert t.nbytes == 24
        assert t.ndim == 2

    def test_create_with_explicit_shape(self):
        """Test creating tensor with explicit shape."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        t = Tensor(data, shape=(2, 3))

        assert t.shape == (2, 3)
        assert t.numel == 6

    def test_create_with_requires_grad(self):
        """Test creating tensor with requires_grad=True."""
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        assert t.requires_grad is True
        assert t.grad is None  # Not allocated until backward

    def test_create_with_name(self):
        """Test creating tensor with a name."""
        t = Tensor([1.0, 2.0], name="weights")

        assert t.name == "weights"
        assert "weights" in repr(t)

    def test_create_from_tuples(self):
        """Test creating tensor from tuples instead of lists."""
        data = ((1.0, 2.0), (3.0, 4.0))
        t = Tensor(data)

        assert t.shape == (2, 2)
        assert t.to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_create_with_integers(self):
        """Test that integers are converted to floats."""
        data = [1, 2, 3]
        t = Tensor(data)

        result = t.to_list()
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result)

    def test_scalar_tensor(self):
        """Test creating a scalar tensor (shape (1,))."""
        t = Tensor([42.0])

        assert t.shape == (1,)
        assert t.numel == 1
        assert t.item() == pytest.approx(42.0)


class TestTensorValidation:
    """Tests for Tensor validation and error handling."""

    def test_reject_rank_0(self):
        """Test that rank 0 (scalar without list) is rejected."""
        # A single float without wrapping list would need special handling
        # Our implementation requires at least a 1D list
        with pytest.raises((ValueError, TypeError)):
            Tensor(42.0)  # type: ignore

    def test_reject_rank_3(self):
        """Test that rank 3 tensors are rejected."""
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        with pytest.raises(ValueError, match="rank"):
            Tensor(data)

    def test_reject_mismatched_shape(self):
        """Test that mismatched shape raises error."""
        data = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="elements"):
            Tensor(data, shape=(2, 3))  # 6 elements needed, only 3 provided

    def test_reject_ragged_array(self):
        """Test that ragged arrays are rejected."""
        data = [[1.0, 2.0], [3.0, 4.0, 5.0]]  # Inconsistent row lengths
        with pytest.raises(ValueError, match="ragged|Inconsistent"):
            Tensor(data)

    def test_reject_zero_dimension(self):
        """Test that zero dimensions are rejected."""
        with pytest.raises(ValueError):
            Tensor([], shape=(0,))

    def test_reject_negative_dimension(self):
        """Test that negative dimensions are rejected."""
        with pytest.raises(ValueError):
            Tensor([1.0], shape=(-1,))


class TestTensorDataPtr:
    """Tests for data_ptr() method."""

    def test_data_ptr_returns_int(self):
        """Test that data_ptr returns an integer."""
        t = Tensor([1.0, 2.0, 3.0])
        ptr = t.data_ptr()

        assert isinstance(ptr, int)
        assert ptr > 0

    def test_data_ptr_consistent(self):
        """Test that data_ptr returns same value on multiple calls."""
        t = Tensor([1.0, 2.0, 3.0])

        ptr1 = t.data_ptr()
        ptr2 = t.data_ptr()

        assert ptr1 == ptr2

    def test_different_tensors_different_ptrs(self):
        """Test that different tensors have different pointers."""
        t1 = Tensor([1.0, 2.0, 3.0])
        t2 = Tensor([4.0, 5.0, 6.0])

        assert t1.data_ptr() != t2.data_ptr()


class TestTensorToList:
    """Tests for to_list() method."""

    def test_to_list_1d(self):
        """Test converting 1D tensor back to list."""
        data = [1.0, 2.5, -3.14, 0.0]
        t = Tensor(data)

        result = t.to_list()

        assert len(result) == len(data)
        for expected, actual in zip(data, result):
            assert actual == pytest.approx(expected, rel=1e-6)

    def test_to_list_2d(self):
        """Test converting 2D tensor back to nested list."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        t = Tensor(data)

        result = t.to_list()

        assert len(result) == 2
        assert len(result[0]) == 3
        assert result[0] == pytest.approx([1.0, 2.0, 3.0])
        assert result[1] == pytest.approx([4.0, 5.0, 6.0])

    def test_to_list_roundtrip(self):
        """Test data survives roundtrip through GPU."""
        original = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        t = Tensor(original)
        result = t.to_list()

        for i in range(len(original)):
            for j in range(len(original[0])):
                assert result[i][j] == pytest.approx(original[i][j], rel=1e-6)


class TestTensorItem:
    """Tests for item() method."""

    def test_item_scalar(self):
        """Test item() on single-element tensor."""
        t = Tensor([42.0])
        assert t.item() == pytest.approx(42.0)

    def test_item_negative(self):
        """Test item() with negative value."""
        t = Tensor([-3.14])
        assert t.item() == pytest.approx(-3.14, rel=1e-6)

    def test_item_rejects_multi_element(self):
        """Test that item() rejects tensors with multiple elements."""
        t = Tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="single-element"):
            t.item()


class TestTensorDetach:
    """Tests for detach() method."""

    def test_detach_shares_data(self):
        """Test that detached tensor shares the same GPU pointer."""
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        d = t.detach()

        assert d.data_ptr() == t.data_ptr()

    def test_detach_no_requires_grad(self):
        """Test that detached tensor has requires_grad=False."""
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        d = t.detach()

        assert t.requires_grad is True
        assert d.requires_grad is False

    def test_detach_same_shape(self):
        """Test that detached tensor has same shape."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        d = t.detach()

        assert d.shape == t.shape
        assert d.numel == t.numel

    def test_detach_same_data(self):
        """Test that detached tensor has same data values."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        t = Tensor(data)
        d = t.detach()

        # Compare row by row since pytest.approx doesn't support nested lists
        t_list = t.to_list()
        d_list = d.to_list()
        for t_row, d_row in zip(t_list, d_list):
            assert d_row == pytest.approx(t_row)

    def test_detach_does_not_own_memory(self):
        """Test that detached tensor doesn't own memory (no double-free)."""
        t = Tensor([1.0, 2.0, 3.0])
        d = t.detach()

        # Access internal attribute to verify
        assert d._owns_memory is False
        assert t._owns_memory is True


class TestTensorFromPtr:
    """Tests for _from_ptr class method."""

    def test_from_ptr_basic(self):
        """Test creating tensor from existing pointer."""
        # Create a tensor normally to get a valid pointer
        t1 = Tensor([1.0, 2.0, 3.0])
        ptr = t1.data_ptr()

        # Create another tensor from the same pointer (non-owning)
        t2 = Tensor._from_ptr(ptr, shape=(3,), owns_memory=False)

        assert t2.data_ptr() == ptr
        assert t2.shape == (3,)
        assert t2.to_list() == pytest.approx(t1.to_list())

    def test_from_ptr_different_shape(self):
        """Test creating tensor with different shape from same data."""
        t1 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ptr = t1.data_ptr()

        # View as 1D (same data, different shape)
        t2 = Tensor._from_ptr(ptr, shape=(6,), owns_memory=False)

        assert t2.shape == (6,)
        assert t2.to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_from_ptr_with_requires_grad(self):
        """Test _from_ptr with requires_grad=True."""
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor._from_ptr(
            t1.data_ptr(),
            shape=(2,),
            owns_memory=False,
            requires_grad=True,
        )

        assert t2.requires_grad is True

    def test_from_ptr_rejects_invalid_rank(self):
        """Test that _from_ptr rejects invalid ranks."""
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="rank"):
            Tensor._from_ptr(t.data_ptr(), shape=(2, 3, 1), owns_memory=False)


class TestTensorRepr:
    """Tests for __repr__ method."""

    def test_repr_basic(self):
        """Test basic repr output."""
        t = Tensor([1.0, 2.0, 3.0])
        r = repr(t)

        assert "Tensor" in r
        assert "(3,)" in r  # shape
        assert "0x" in r  # hex pointer

    def test_repr_with_requires_grad(self):
        """Test repr includes requires_grad when True."""
        t = Tensor([1.0, 2.0], requires_grad=True)
        r = repr(t)

        assert "requires_grad=True" in r

    def test_repr_with_name(self):
        """Test repr includes name when set."""
        t = Tensor([1.0], name="loss")
        r = repr(t)

        assert "loss" in r


class TestTensorGraphFields:
    """Tests for autograd graph fields."""

    def test_initial_graph_fields(self):
        """Test that graph fields are properly initialized."""
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        assert t._parents == ()
        assert t._backward is None
        assert t._ctx is None

    def test_grad_initially_none(self):
        """Test that grad is None until backward is called."""
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert t.grad is None


class TestTensorMemorySafety:
    """Tests for memory management and safety."""

    def test_many_allocations(self):
        """Test that many tensor allocations/deallocations work."""
        for _ in range(100):
            t = Tensor([[1.0, 2.0], [3.0, 4.0]])
            _ = t.to_list()
            # t goes out of scope and should be freed

    def test_large_tensor(self):
        """Test allocating a reasonably large tensor."""
        # 1000 x 100 = 100K floats = 400KB
        rows, cols = 1000, 100
        data = [[float(i * cols + j) for j in range(cols)] for i in range(rows)]

        t = Tensor(data)
        assert t.shape == (rows, cols)
        assert t.numel == rows * cols

        # Verify a few values survived the roundtrip
        result = t.to_list()
        assert result[0][0] == pytest.approx(0.0)
        assert result[0][99] == pytest.approx(99.0)
        assert result[999][99] == pytest.approx(99999.0)
