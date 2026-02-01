"""
Tests for tensor.py - Tensor class with GPU storage.

These tests require a CUDA-enabled GPU to run.
"""

import pytest

from gradtuity.tensor import Tensor

# Mark all tests in this module as requiring CUDA
pytestmark = pytest.mark.requires_cuda


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

    def test_create_4d_tensor(self):
        """Test creating a 4D tensor (e.g. NCHW for conv)."""
        data = [[[[1.0, 2.0], [3.0, 4.0]]]]  # (1, 1, 2, 2)
        t = Tensor(data)
        assert t.shape == (1, 1, 2, 2)
        assert t.numel == 4
        assert t.ndim == 4

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

    def test_reject_rank_5(self):
        """Test that rank 5 tensors are rejected (only 1-4 supported)."""
        data = [[[[[1.0]]]]]  # 5 levels
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

    def test_to_list_4d(self):
        """Test converting 4D tensor back to nested list."""
        data = [[[[1.0, 2.0], [3.0, 4.0]]]]  # (1, 1, 2, 2)
        t = Tensor(data)
        result = t.to_list()
        assert len(result) == 1
        assert len(result[0]) == 1
        assert len(result[0][0]) == 2
        assert result[0][0][0] == pytest.approx([1.0, 2.0])
        assert result[0][0][1] == pytest.approx([3.0, 4.0])


class TestTensorView:
    """Tests for view() and shape manipulation."""

    def test_view_2d_to_2d(self):
        """Test view with same numel."""
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v = t.view((3, 2))
        assert v.shape == (3, 2)
        assert v.numel == t.numel
        assert v.data_ptr() == t.data_ptr()

    def test_view_with_infer_dim(self):
        """Test view with -1 infer dimension."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        v = t.view((1, -1))
        assert v.shape == (1, 4)
        assert v.numel == 4

    def test_view_incompatible_numel_raises(self):
        """Test view with wrong numel raises."""
        t = Tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="elements"):
            t.view((2, 2))


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

        assert d.owns_memory is False
        assert t.owns_memory is True


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
        """Test that _from_ptr rejects invalid ranks (only 1-4 supported)."""
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="rank"):
            Tensor._from_ptr(t.data_ptr(), shape=(1, 1, 1, 1, 1), owns_memory=False)


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


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestBackward:
    """Tests for backward() method."""

    def test_backward_requires_scalar(self):
        """Test that backward() rejects non-scalar tensors."""
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        with pytest.raises(ValueError, match="scalar"):
            t.backward()

    def test_backward_requires_requires_grad(self):
        """Test that backward() requires requires_grad=True."""
        t = Tensor([1.0], requires_grad=False)
        with pytest.raises(RuntimeError, match="requires_grad"):
            t.backward()

    def test_backward_seeds_grad_with_ones(self):
        """Test that backward() seeds loss.grad with 1.0."""
        loss = Tensor([5.0], requires_grad=True)
        loss.backward()

        assert loss.grad is not None
        assert loss.grad.shape == (1,)
        assert loss.grad.to_list()[0] == pytest.approx(1.0)

    def test_backward_simple_linear_graph(self):
        """Test backward on a simple linear graph: a -> b -> c (scalar).

        Manually set up a graph where:
        - a is leaf with requires_grad=True
        - b = 2*a (simulated)
        - c = sum(b) = scalar

        Gradient should flow: dc/da = 2
        """
        from gradtuity.functional import zeros_like

        # Leaf tensor
        a = Tensor([1.0, 2.0], requires_grad=True)

        # Intermediate (pretend it's 2*a)
        b = Tensor([2.0, 4.0], requires_grad=False)

        # Manually simulate: b = 2*a
        # backward: a.grad += 2 * out_grad
        def b_backward(out_grad):
            if a.grad is None:
                a.grad = zeros_like(a)
            # Simulate accumulating 2 * out_grad into a.grad
            # For simplicity, we'll do this by reading out_grad and writing to a.grad
            grad_vals = out_grad.to_list()
            a_grad_vals = a.grad.to_list()
            new_vals = [av + 2.0 * gv for av, gv in zip(a_grad_vals, grad_vals)]
            a.grad = Tensor(new_vals)

        b._set_graph(parents=(a,), backward_fn=b_backward)

        # Output scalar (pretend it's sum(b) = 6.0)
        c = Tensor([6.0], requires_grad=False)

        # Manually simulate: c = sum(b)
        # backward: b.grad += broadcast(out_grad) = [1.0, 1.0]
        def c_backward(out_grad):
            if b.grad is None:
                b.grad = zeros_like(b)
            # sum backward broadcasts the scalar grad
            scalar_val = out_grad.to_list()[0]
            b_grad_vals = b.grad.to_list()
            new_vals = [bv + scalar_val for bv in b_grad_vals]
            b.grad = Tensor(new_vals)

        c._set_graph(parents=(b,), backward_fn=c_backward)

        # Run backward
        c.backward()

        # Check: b.grad should be [1.0, 1.0] from sum backward
        assert b.grad is not None
        assert b.grad.to_list() == pytest.approx([1.0, 1.0])

        # Check: a.grad should be [2.0, 2.0] (2 * b.grad)
        assert a.grad is not None
        assert a.grad.to_list() == pytest.approx([2.0, 2.0])


@pytest.mark.requires_cuda
class TestSetGraph:
    """Tests for _set_graph() conditional graph construction."""

    def test_set_graph_with_requires_grad_parent(self):
        """Test that _set_graph sets requires_grad=True when parent requires grad."""
        parent = Tensor([1.0, 2.0], requires_grad=True)
        child = Tensor([2.0, 3.0], requires_grad=False)

        child._set_graph(
            parents=(parent,),
            backward_fn=lambda out_grad: None,
        )

        assert child.requires_grad is True
        assert child._parents == (parent,)
        assert child._backward is not None

    def test_set_graph_without_requires_grad_parent(self):
        """Test that _set_graph keeps requires_grad=False when no parent requires grad."""
        parent = Tensor([1.0, 2.0], requires_grad=False)
        child = Tensor([2.0, 3.0], requires_grad=False)

        child._set_graph(
            parents=(parent,),
            backward_fn=lambda out_grad: None,
        )

        assert child.requires_grad is False
        # Graph fields should remain empty
        assert child._parents == ()
        assert child._backward is None

    def test_set_graph_multiple_parents_any_requires_grad(self):
        """Test _set_graph with multiple parents where one requires grad."""
        p1 = Tensor([1.0], requires_grad=False)
        p2 = Tensor([2.0], requires_grad=True)
        p3 = Tensor([3.0], requires_grad=False)

        child = Tensor([6.0], requires_grad=False)
        child._set_graph(
            parents=(p1, p2, p3),
            backward_fn=lambda out_grad: None,
        )

        # any() of requires_grad = True
        assert child.requires_grad is True
        assert child._parents == (p1, p2, p3)

    def test_set_graph_multiple_parents_none_requires_grad(self):
        """Test _set_graph with multiple parents where none require grad."""
        p1 = Tensor([1.0], requires_grad=False)
        p2 = Tensor([2.0], requires_grad=False)

        child = Tensor([3.0], requires_grad=False)
        child._set_graph(
            parents=(p1, p2),
            backward_fn=lambda out_grad: None,
        )

        # No parent requires grad
        assert child.requires_grad is False
        assert child._parents == ()
        assert child._backward is None

    def test_set_graph_with_ctx(self):
        """Test that _set_graph stores ctx when provided."""
        parent = Tensor([1.0], requires_grad=True)
        child = Tensor([2.0], requires_grad=False)

        ctx = {"saved_for_backward": parent}
        child._set_graph(
            parents=(parent,),
            backward_fn=lambda out_grad: None,
            ctx=ctx,
        )

        assert child._ctx == ctx


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


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestAdd:
    """Tests for elementwise add operation."""

    def test_add_1d_forward(self):
        """Test add forward on 1D tensors."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])

        c = a.add(b)

        assert c.shape == (3,)
        assert c.to_list() == pytest.approx([5.0, 7.0, 9.0])

    def test_add_2d_forward(self):
        """Test add forward on 2D tensors."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[0.1, 0.2], [0.3, 0.4]])

        c = a.add(b)

        assert c.shape == (2, 2)
        result = c.to_list()
        assert result[0] == pytest.approx([1.1, 2.2])
        assert result[1] == pytest.approx([3.3, 4.4])

    def test_add_shape_mismatch_raises(self):
        """Test that add raises error for mismatched shapes."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([1.0, 2.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            a.add(b)

    def test_add_backward_both_require_grad(self):
        """Test add backward when both inputs require grad."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

        c = a.add(b)
        # Simulate sum to get scalar
        # For now, manually set c.grad and call backward logic
        assert c.requires_grad is True

        # Create a scalar by adding all elements (simulated)
        # We'll test full backward through the graph
        from gradtuity.functional import ones_like

        c.grad = ones_like(c)
        c._backward(c.grad)

        # Both grads should be [1.0, 1.0, 1.0]
        assert a.grad is not None
        assert a.grad.to_list() == pytest.approx([1.0, 1.0, 1.0])
        assert b.grad is not None
        assert b.grad.to_list() == pytest.approx([1.0, 1.0, 1.0])

    def test_add_backward_one_requires_grad(self):
        """Test add backward when only one input requires grad."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=False)

        c = a.add(b)
        assert c.requires_grad is True

        from gradtuity.functional import ones_like

        c.grad = ones_like(c)
        c._backward(c.grad)

        assert a.grad is not None
        assert a.grad.to_list() == pytest.approx([1.0, 1.0])
        assert b.grad is None  # b doesn't require grad

    def test_add_no_grad(self):
        """Test add with no grad tracking."""
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])

        c = a.add(b)

        assert c.requires_grad is False
        assert c._parents == ()
        assert c._backward is None

    def test_add_accumulates_grad(self):
        """Test that add backward accumulates (doesn't replace) gradients."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)

        # Pre-set some gradients
        a.grad = Tensor([10.0, 20.0])
        b.grad = Tensor([30.0, 40.0])

        c = a.add(b)

        from gradtuity.functional import ones_like

        c.grad = ones_like(c)
        c._backward(c.grad)

        # Grads should be accumulated: original + 1.0
        assert a.grad.to_list() == pytest.approx([11.0, 21.0])
        assert b.grad.to_list() == pytest.approx([31.0, 41.0])


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestRelu:
    """Tests for ReLU operation."""

    def test_relu_forward_positive(self):
        """Test relu forward with positive values."""
        y = Tensor([1.0, 2.0, 3.0])
        z = y.relu()

        assert z.shape == (3,)
        assert z.to_list() == pytest.approx([1.0, 2.0, 3.0])

    def test_relu_forward_negative(self):
        """Test relu forward with negative values."""
        y = Tensor([-1.0, -2.0, -3.0])
        z = y.relu()

        assert z.to_list() == pytest.approx([0.0, 0.0, 0.0])

    def test_relu_forward_mixed(self):
        """Test relu forward with mixed positive/negative values."""
        y = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        z = y.relu()

        assert z.to_list() == pytest.approx([0.0, 0.0, 0.0, 1.0, 2.0])

    def test_relu_forward_2d(self):
        """Test relu forward on 2D tensor."""
        y = Tensor([[-1.0, 2.0], [3.0, -4.0]])
        z = y.relu()

        assert z.shape == (2, 2)
        result = z.to_list()
        assert result[0] == pytest.approx([0.0, 2.0])
        assert result[1] == pytest.approx([3.0, 0.0])

    def test_relu_backward_positive(self):
        """Test relu backward with positive input (grad passes through)."""
        y = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        z = y.relu()

        assert z.requires_grad is True

        from gradtuity.functional import ones_like

        z.grad = ones_like(z)
        z._backward(z.grad)

        # All positive -> mask is all 1s -> grad = [1, 1, 1]
        assert y.grad is not None
        assert y.grad.to_list() == pytest.approx([1.0, 1.0, 1.0])

    def test_relu_backward_negative(self):
        """Test relu backward with negative input (grad blocked)."""
        y = Tensor([-1.0, -2.0, -3.0], requires_grad=True)
        z = y.relu()

        from gradtuity.functional import ones_like

        z.grad = ones_like(z)
        z._backward(z.grad)

        # All negative -> mask is all 0s -> grad = [0, 0, 0]
        assert y.grad is not None
        assert y.grad.to_list() == pytest.approx([0.0, 0.0, 0.0])

    def test_relu_backward_mixed(self):
        """Test relu backward with mixed values."""
        y = Tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
        z = y.relu()

        from gradtuity.functional import ones_like

        z.grad = ones_like(z)
        z._backward(z.grad)

        # mask = [0, 1, 0, 1] -> grad = [0, 1, 0, 1]
        assert y.grad.to_list() == pytest.approx([0.0, 1.0, 0.0, 1.0])

    def test_relu_backward_zero_input(self):
        """Test relu backward at exactly zero (should be 0 gradient)."""
        y = Tensor([0.0, 1.0, -1.0], requires_grad=True)
        z = y.relu()

        from gradtuity.functional import ones_like

        z.grad = ones_like(z)
        z._backward(z.grad)

        # 0 is not > 0, so mask at index 0 is 0
        assert y.grad.to_list() == pytest.approx([0.0, 1.0, 0.0])

    def test_relu_no_grad(self):
        """Test relu with no grad tracking."""
        y = Tensor([1.0, -1.0, 2.0])
        z = y.relu()

        assert z.requires_grad is False
        assert z._parents == ()
        assert z._backward is None
        assert z.to_list() == pytest.approx([1.0, 0.0, 2.0])

    def test_relu_backward_accumulates(self):
        """Test that relu backward accumulates gradients."""
        y = Tensor([1.0, 2.0], requires_grad=True)
        y.grad = Tensor([10.0, 20.0])  # Pre-existing grad

        z = y.relu()

        from gradtuity.functional import ones_like

        z.grad = ones_like(z)
        z._backward(z.grad)

        # Accumulated: [10+1, 20+1] = [11, 21]
        assert y.grad.to_list() == pytest.approx([11.0, 21.0])

    def test_relu_backward_scales_with_out_grad(self):
        """Test that relu backward properly scales with output gradient."""
        y = Tensor([1.0, -1.0, 2.0], requires_grad=True)
        z = y.relu()

        # Use non-unit gradient
        z.grad = Tensor([2.0, 3.0, 4.0])
        z._backward(z.grad)

        # mask = [1, 0, 1], grad = out_grad * mask = [2, 0, 4]
        assert y.grad.to_list() == pytest.approx([2.0, 0.0, 4.0])


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestGELU:
    """Tests for GELU activation (tanh approximation)."""

    @staticmethod
    def _gelu_numpy_ref(x_np):
        """NumPy GELU (tanh approx) on float32 array; vectorized, no Python loop."""
        import numpy as np
        sqrt_2_over_pi = np.sqrt(np.float32(2.0 / np.pi))
        u = sqrt_2_over_pi * (x_np + np.float32(0.044715) * x_np**3)
        return np.float32(0.5) * x_np * (np.float32(1.0) + np.tanh(u))

    def _flatten(self, lst):
        """Flatten nested list to 1D for approx comparison."""
        out = []
        for x in lst:
            if isinstance(x, (list, tuple)):
                out.extend(self._flatten(x))
            else:
                out.append(x)
        return out

    def test_gelu_forward_1d(self):
        """Test GELU forward vs NumPy for 1D."""
        import numpy as np
        data = [1.0, -1.0, 0.0, 2.0, -2.0]
        x_np = np.array(data, dtype=np.float32)
        expected = self._gelu_numpy_ref(x_np)
        x = Tensor(data)
        y = x.gelu()
        assert y.shape == (5,)
        assert y.to_list() == pytest.approx(expected.tolist(), rel=1e-5, abs=1e-6)

    def test_gelu_forward_2d(self):
        """Test GELU forward vs NumPy for 2D."""
        import numpy as np
        data = [[1.0, -1.0], [0.5, -0.5]]
        x_np = np.array(data, dtype=np.float32)
        expected = self._gelu_numpy_ref(x_np)
        x = Tensor(data)
        y = x.gelu()
        assert y.shape == (2, 2)
        assert self._flatten(y.to_list()) == pytest.approx(
            self._flatten(expected.tolist()), rel=1e-5, abs=1e-6
        )

    def test_gelu_forward_4d(self):
        """Test GELU forward for 4D shape (1,2,3,4)."""
        import numpy as np
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 3, 4).astype(np.float32)
        expected = self._gelu_numpy_ref(x_np)
        x = Tensor(x_np.tolist())
        y = x.gelu()
        assert y.shape == (1, 2, 3, 4)
        assert self._flatten(y.to_list()) == pytest.approx(
            self._flatten(expected.tolist()), rel=1e-5, abs=2e-5
        )

    def test_gelu_rejects_approx(self):
        """Test that gelu(approx=...) only accepts 'tanh' in v1."""
        x = Tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="approx"):
            x.gelu(approx="exact")

    def test_gelu_backward_via_sum(self):
        """Test GELU backward: L = sum(gelu(x)), backward(), check grad."""
        import numpy as np
        x = Tensor([1.0, 2.0, -1.0, 0.5], requires_grad=True)
        y = x.gelu()
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4,)
        g = np.array(x.grad.to_list(), dtype=np.float32)
        assert np.all(np.isfinite(g))

    def test_gelu_backward_finite_difference(self):
        """Test GELU backward with finite-difference gradient check."""
        import numpy as np
        eps = 1e-3  # meaningful in float32; 1e-5 is too small
        np.random.seed(123)
        data = np.random.randn(32).astype(np.float32) * 0.5
        x = Tensor(data.tolist(), requires_grad=True)
        y = x.gelu()
        loss = y.sum()
        loss.backward()
        grad_gradtuity = np.array(x.grad.to_list(), dtype=np.float32)
        grad_numeric = np.zeros_like(data)
        for i in range(len(data)):
            data_plus = data.copy()
            data_plus[i] += eps
            data_minus = data.copy()
            data_minus[i] -= eps
            loss_plus = np.sum(self._gelu_numpy_ref(data_plus))
            loss_minus = np.sum(self._gelu_numpy_ref(data_minus))
            grad_numeric[i] = (loss_plus - loss_minus) / (2.0 * eps)
        assert grad_gradtuity == pytest.approx(grad_numeric, rel=0.05, abs=0.01)

    def test_gelu_no_grad(self):
        """Test GELU with requires_grad=False does not build graph."""
        x = Tensor([1.0, 2.0])
        y = x.gelu()
        assert y.requires_grad is False
        assert y.grad is None

    def test_gelu_no_nans_large(self):
        """GELU forward is finite for large-magnitude inputs (tanh stability)."""
        import numpy as np
        x = Tensor([50.0, -50.0, 20.0, -20.0])
        y = x.gelu()
        out = np.array(y.to_list(), dtype=np.float32)
        assert np.all(np.isfinite(out))

    def test_gelu_grad_accumulates(self):
        """Backward accumulates into x.grad when x is used twice (overwrite vs add)."""
        import numpy as np
        data = [0.1, -0.2, 0.3]
        x = Tensor(data, requires_grad=True)
        y1 = x.gelu().sum()
        y2 = x.gelu().sum()
        loss = y1.add(y2)
        loss.backward()
        x_ref = Tensor(data, requires_grad=True)
        y_ref = x_ref.gelu().sum()
        y_ref.backward()
        g = np.array(x.grad.to_list(), dtype=np.float32)
        g_ref = np.array(x_ref.grad.to_list(), dtype=np.float32)
        assert np.all(np.isfinite(g))
        assert g == pytest.approx(2.0 * g_ref, rel=1e-5, abs=1e-6)


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestSoftmax:
    """Tests for softmax(dim=-1)."""

    @staticmethod
    def _softmax_numpy(x_np):
        """Stable softmax over last axis (float32)."""
        import numpy as np
        x = np.asarray(x_np, dtype=np.float32)
        m = np.max(x, axis=-1, keepdims=True)
        e = np.exp(x - m)
        return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)

    def test_softmax_forward_2d(self):
        """Test softmax forward vs NumPy for 2D (rows, cols)."""
        import numpy as np
        data = [[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]]
        x_np = np.array(data, dtype=np.float32)
        expected = self._softmax_numpy(x_np)
        x = Tensor(data)
        y = x.softmax(dim=-1)
        assert y.shape == (2, 3)
        flat_y = np.array(y.to_list()).ravel()
        flat_expected = expected.ravel()
        assert flat_y == pytest.approx(flat_expected, rel=1e-5, abs=1e-6)

    def test_softmax_sum_to_one(self):
        """Per-row sum of softmax is 1."""
        import numpy as np
        x = Tensor([[1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]])
        y = x.softmax(dim=-1)
        rows = np.array(y.to_list())
        row_sums = np.sum(rows, axis=1)
        assert row_sums == pytest.approx(np.ones(2), abs=1e-4)

    def test_softmax_forward_4d(self):
        """Test softmax on 4D (B, H, S, S) last dim as cols."""
        import numpy as np
        np.random.seed(42)
        x_np = np.random.randn(2, 2, 3, 4).astype(np.float32) * 0.5
        expected = self._softmax_numpy(x_np)
        x = Tensor(x_np.tolist())
        y = x.softmax(dim=-1)
        assert y.shape == (2, 2, 3, 4)
        flat_y = np.array(y.to_list()).ravel()
        flat_expected = expected.ravel()
        assert flat_y == pytest.approx(flat_expected, rel=1e-4, abs=2e-5)

    def test_softmax_rejects_dim(self):
        """Test that softmax v1 only accepts dim=-1."""
        x = Tensor([[1.0, 2.0]])
        with pytest.raises(ValueError, match="dim=-1"):
            x.softmax(dim=0)

    def test_softmax_backward_via_sum(self):
        """Softmax backward: L = sum(softmax(x)), check grad finite."""
        import numpy as np
        x = Tensor([[1.0, 2.0, 0.5], [-0.5, 0.0, 1.0]], requires_grad=True)
        y = x.softmax(dim=-1)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 3)
        g = np.array(x.grad.to_list(), dtype=np.float32)
        assert np.all(np.isfinite(g))

    def test_softmax_no_grad(self):
        """Softmax with requires_grad=False does not build graph."""
        x = Tensor([[1.0, 2.0]])
        y = x.softmax(dim=-1)
        assert y.requires_grad is False
        assert y.grad is None


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestTranspose4D:
    """Tests for transpose4d_last2: (B,H,S,D) -> (B,H,D,S)."""

    def test_transpose4d_round_trip(self):
        """Round-trip: transpose4d_last2().transpose4d_last2() equals identity."""
        import numpy as np
        np.random.seed(42)
        x_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        x = Tensor(x_np.tolist())
        y = x.transpose4d_last2()
        assert y.shape == (2, 3, 5, 4)
        z = y.transpose4d_last2()
        assert z.shape == (2, 3, 4, 5)
        flat_x = np.array(x.to_list()).ravel()
        flat_z = np.array(z.to_list()).ravel()
        assert flat_z == pytest.approx(flat_x, rel=1e-5, abs=1e-6)

    def test_transpose4d_backward(self):
        """Backward: L = sum(transpose4d_last2(x)), grad should be ones (after inverse transpose)."""
        import numpy as np
        x = Tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]],  # (1,1,2,2)
            requires_grad=True,
        )
        y = x.transpose4d_last2()
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 1, 2, 2)
        # d(sum(y))/d(x) = ones in same layout as x; backward transposes out_grad (ones (1,1,2,2)) -> same
        g = np.array(x.grad.to_list(), dtype=np.float32)
        assert np.all(np.isfinite(g))
        assert g == pytest.approx(np.ones((1, 1, 2, 2), dtype=np.float32))

    def test_transpose4d_rejects_non_4d(self):
        """transpose4d_last2 rejects non-4D tensors."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="ndim=4"):
            x.transpose4d_last2()

    def test_transpose4d_no_grad(self):
        """transpose4d_last2 with requires_grad=False does not build graph."""
        x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1,1,2,2)
        y = x.transpose4d_last2()
        assert y.requires_grad is False
        assert y.grad is None


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestBMM:
    """Tests for batched matrix multiplication (bmm)."""

    def test_bmm_forward_rank3(self):
        """BMM forward rank-3: (B, M, K) @ (B, K, N) vs NumPy."""
        import numpy as np
        np.random.seed(42)
        B, M, K, N = 3, 4, 5, 2
        a_np = np.random.randn(B, M, K).astype(np.float32)
        b_np = np.random.randn(B, K, N).astype(np.float32)
        expected = np.matmul(a_np, b_np)
        a = Tensor(a_np.tolist())
        b = Tensor(b_np.tolist())
        y = a.bmm(b)
        assert y.shape == (B, M, N)
        flat_y = np.array(y.to_list()).ravel()
        flat_expected = expected.ravel()
        assert flat_y == pytest.approx(flat_expected, rel=0.01, abs=0.001)

    def test_bmm_forward_rank4(self):
        """BMM forward rank-4: (B, H, M, K) @ (B, H, K, N) vs NumPy."""
        import numpy as np
        np.random.seed(42)
        B, H, M, K, N = 2, 3, 4, 5, 6
        a_np = np.random.randn(B, H, M, K).astype(np.float32)
        b_np = np.random.randn(B, H, K, N).astype(np.float32)
        expected = np.matmul(a_np, b_np)
        a = Tensor(a_np.tolist())
        b = Tensor(b_np.tolist())
        y = a.bmm(b)
        assert y.shape == (B, H, M, N)
        flat_y = np.array(y.to_list()).ravel()
        flat_expected = expected.ravel()
        # Float32 matmul accumulation can have ~1e-2 level error
        assert flat_y == pytest.approx(flat_expected, rel=0.01, abs=0.01)

    def test_bmm_backward_finite_difference(self):
        """BMM backward: finite-difference gradient check (loose tolerance)."""
        import numpy as np
        eps = 1e-3
        B, M, K, N = 2, 2, 3, 2
        np.random.seed(123)
        a_np = np.random.randn(B, M, K).astype(np.float32) * 0.5
        b_np = np.random.randn(B, K, N).astype(np.float32) * 0.5
        a = Tensor(a_np.tolist(), requires_grad=True)
        b = Tensor(b_np.tolist(), requires_grad=True)
        y = a.bmm(b)
        loss = y.sum()
        loss.backward()
        # Numeric grad for a
        grad_a_numeric = np.zeros_like(a_np)
        for bi in range(B):
            for i in range(M):
                for j in range(K):
                    a_plus = a_np.copy()
                    a_plus[bi, i, j] += eps
                    a_minus = a_np.copy()
                    a_minus[bi, i, j] -= eps
                    loss_plus = np.matmul(a_plus, b_np).sum()
                    loss_minus = np.matmul(a_minus, b_np).sum()
                    grad_a_numeric[bi, i, j] = (loss_plus - loss_minus) / (2.0 * eps)
        grad_a = np.array(a.grad.to_list()).ravel()
        assert grad_a == pytest.approx(grad_a_numeric.ravel(), rel=0.1, abs=0.05)
        # Numeric grad for b
        grad_b_numeric = np.zeros_like(b_np)
        for bi in range(B):
            for i in range(K):
                for j in range(N):
                    b_plus = b_np.copy()
                    b_plus[bi, i, j] += eps
                    b_minus = b_np.copy()
                    b_minus[bi, i, j] -= eps
                    loss_plus = np.matmul(a_np, b_plus).sum()
                    loss_minus = np.matmul(a_np, b_minus).sum()
                    grad_b_numeric[bi, i, j] = (loss_plus - loss_minus) / (2.0 * eps)
        grad_b = np.array(b.grad.to_list()).ravel()
        assert grad_b == pytest.approx(grad_b_numeric.ravel(), rel=0.1, abs=0.05)

    def test_bmm_backward_accumulates(self):
        """BMM backward accumulates when used twice (L = y1.sum() + y2.sum())."""
        import numpy as np
        B, M, K, N = 2, 2, 2, 2
        data_a = [[[1.0, 0.5], [0.5, 1.0]], [[0.5, 1.0], [1.0, 0.5]]]
        data_b = [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]
        a = Tensor(data_a, requires_grad=True)
        b = Tensor(data_b, requires_grad=True)
        y1 = a.bmm(b).sum()
        y2 = a.bmm(b).sum()
        loss = y1.add(y2)
        loss.backward()
        a_ref = Tensor(data_a, requires_grad=True)
        b_ref = Tensor(data_b, requires_grad=True)
        y_ref = a_ref.bmm(b_ref).sum()
        y_ref.backward()
        g_a = np.array(a.grad.to_list()).ravel()
        g_a_ref = np.array(a_ref.grad.to_list()).ravel()
        assert g_a == pytest.approx(2.0 * g_a_ref, rel=1e-5, abs=1e-6)
        g_b = np.array(b.grad.to_list()).ravel()
        g_b_ref = np.array(b_ref.grad.to_list()).ravel()
        assert g_b == pytest.approx(2.0 * g_b_ref, rel=1e-5, abs=1e-6)

    def test_bmm_rejects_ndim(self):
        """bmm rejects ndim not 3 or 4."""
        a2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b2 = Tensor([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="ndim 3 or 4"):
            a2.bmm(b2)

    def test_bmm_rejects_mismatched_ndim(self):
        """bmm requires same ndim for both tensors."""
        a3 = Tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1,2,2)
        b2 = Tensor([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="same ndim"):
            a3.bmm(b2)

    def test_bmm_rejects_mismatched_batch(self):
        """bmm rejects mismatched batch dimension."""
        a = Tensor([[[1.0, 2.0]], [[3.0, 4.0]]])  # (2,1,2)
        b = Tensor([[[1.0], [2.0]]])  # (1,2,1) - batch 1 vs 2
        with pytest.raises(ValueError, match="batch"):
            a.bmm(b)

    def test_bmm_rejects_mismatched_inner(self):
        """bmm rejects mismatched inner dimension (K)."""
        a = Tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])  # (2,1,3) K=3
        b = Tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]])  # (2,2,2) K=2
        with pytest.raises(ValueError, match="inner"):
            a.bmm(b)


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestAddReluIntegration:
    """Integration tests combining add and relu operations."""

    def test_add_then_relu_forward(self):
        """Test forward: relu(a + b)."""
        a = Tensor([-2.0, -1.0, 0.0, 1.0])
        b = Tensor([1.0, 1.0, 1.0, 1.0])

        c = a.add(b)  # [-1, 0, 1, 2]
        z = c.relu()  # [0, 0, 1, 2]

        assert z.to_list() == pytest.approx([0.0, 0.0, 1.0, 2.0])

    def test_add_then_relu_backward(self):
        """Test backward through relu(a + b)."""
        a = Tensor([-2.0, 1.0], requires_grad=True)
        b = Tensor([1.0, 1.0], requires_grad=True)

        c = a.add(b)  # [-1, 2]
        z = c.relu()  # [0, 2]

        # Backward from z
        from gradtuity.functional import ones_like

        z.grad = ones_like(z)

        # Manual backward traversal (simulating loss.backward())
        z._backward(z.grad)  # c.grad = [0, 1] (relu mask)
        c._backward(c.grad)  # a.grad = c.grad, b.grad = c.grad

        # a.grad and b.grad should be [0, 1] (blocked at first element)
        assert a.grad.to_list() == pytest.approx([0.0, 1.0])
        assert b.grad.to_list() == pytest.approx([0.0, 1.0])

    def test_shared_input_add(self):
        """Test y = x + x (shared subgraph), grad should be 2*out_grad."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.add(x)

        from gradtuity.functional import ones_like

        y.grad = ones_like(y)
        y._backward(y.grad)

        # x appears twice, so grad = 1 + 1 = 2
        assert x.grad.to_list() == pytest.approx([2.0, 2.0, 2.0])


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestSum:
    """Tests for sum() reduction operation."""

    def test_sum_1d(self):
        """Test sum on 1D tensor."""
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        loss = x.sum()

        assert loss.shape == (1,)
        assert loss.item() == pytest.approx(10.0)

    def test_sum_2d(self):
        """Test sum on 2D tensor."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = x.sum()

        assert loss.shape == (1,)
        assert loss.item() == pytest.approx(10.0)

    def test_sum_large(self):
        """Test sum on larger tensor for correctness."""
        # 100 elements, each 1.0
        data = [1.0] * 100
        x = Tensor(data, shape=(100,))
        loss = x.sum()

        assert loss.item() == pytest.approx(100.0)

    def test_sum_negative_values(self):
        """Test sum with negative values."""
        x = Tensor([-1.0, 2.0, -3.0, 4.0])
        loss = x.sum()

        assert loss.item() == pytest.approx(2.0)

    def test_sum_backward(self):
        """Test sum backward broadcasts scalar gradient."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        loss = x.sum()

        assert loss.requires_grad is True

        # Backward
        loss.backward()

        # Gradient should be all ones (dloss/dx_i = 1 for sum)
        assert x.grad is not None
        assert x.grad.to_list() == pytest.approx([1.0, 1.0, 1.0])

    def test_sum_backward_2d(self):
        """Test sum backward on 2D tensor."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        loss = x.sum()

        loss.backward()

        # All gradients should be 1.0
        result = x.grad.to_list()
        assert result[0] == pytest.approx([1.0, 1.0])
        assert result[1] == pytest.approx([1.0, 1.0])

    def test_sum_no_grad(self):
        """Test sum with no gradient tracking."""
        x = Tensor([1.0, 2.0, 3.0])
        loss = x.sum()

        assert loss.requires_grad is False
        assert loss._backward is None


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestAddBias:
    """Tests for add_bias() operation."""

    def test_add_bias_forward(self):
        """Test add_bias forward computation."""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        b = Tensor([0.1, 0.2, 0.3])  # (3,)

        y = x.add_bias(b)

        assert y.shape == (2, 3)
        result = y.to_list()
        assert result[0] == pytest.approx([1.1, 2.2, 3.3])
        assert result[1] == pytest.approx([4.1, 5.2, 6.3])

    def test_add_bias_shape_validation_input_not_2d(self):
        """Test that add_bias rejects non-2D input."""
        x = Tensor([1.0, 2.0, 3.0])  # 1D
        b = Tensor([0.1, 0.2, 0.3])

        with pytest.raises(ValueError, match="2D"):
            x.add_bias(b)

    def test_add_bias_shape_validation_bias_not_1d(self):
        """Test that add_bias rejects non-1D bias."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[0.1, 0.2]])  # 2D instead of 1D

        with pytest.raises(ValueError, match="1D"):
            x.add_bias(b)

    def test_add_bias_shape_validation_size_mismatch(self):
        """Test that add_bias rejects mismatched sizes."""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        b = Tensor([0.1, 0.2])  # (2,) - wrong size

        with pytest.raises(ValueError, match="doesn't match"):
            x.add_bias(b)

    def test_add_bias_backward_x_grad(self):
        """Test add_bias backward for input gradient."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([0.1, 0.2], requires_grad=False)

        y = x.add_bias(b)

        from gradtuity.functional import ones_like

        y.grad = ones_like(y)
        y._backward(y.grad)

        # dX = dY (elementwise)
        result = x.grad.to_list()
        assert result[0] == pytest.approx([1.0, 1.0])
        assert result[1] == pytest.approx([1.0, 1.0])

    def test_add_bias_backward_bias_grad(self):
        """Test add_bias backward for bias gradient (sum over axis 0)."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        b = Tensor([0.1, 0.2], requires_grad=True)

        y = x.add_bias(b)

        from gradtuity.functional import ones_like

        y.grad = ones_like(y)
        y._backward(y.grad)

        # db = sum over rows of dY = [1+1, 1+1] = [2, 2]
        assert b.grad is not None
        assert b.grad.to_list() == pytest.approx([2.0, 2.0])

    def test_add_bias_backward_both_grads(self):
        """Test add_bias backward when both require grad."""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)  # (2, 3)
        b = Tensor([0.1, 0.2, 0.3], requires_grad=True)  # (3,)

        y = x.add_bias(b)

        from gradtuity.functional import ones_like

        y.grad = ones_like(y)
        y._backward(y.grad)

        # dX = dY (all ones)
        x_result = x.grad.to_list()
        assert x_result[0] == pytest.approx([1.0, 1.0, 1.0])
        assert x_result[1] == pytest.approx([1.0, 1.0, 1.0])

        # db = sum over batch = [2, 2, 2]
        assert b.grad.to_list() == pytest.approx([2.0, 2.0, 2.0])

    def test_add_bias_backward_non_unit_grad(self):
        """Test add_bias backward with non-unit output gradient."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([0.1, 0.2], requires_grad=True)

        y = x.add_bias(b)

        # Non-unit gradient
        y.grad = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y._backward(y.grad)

        # dX = dY
        x_result = x.grad.to_list()
        assert x_result[0] == pytest.approx([1.0, 2.0])
        assert x_result[1] == pytest.approx([3.0, 4.0])

        # db = sum over rows = [1+3, 2+4] = [4, 6]
        assert b.grad.to_list() == pytest.approx([4.0, 6.0])

    def test_add_bias_no_grad(self):
        """Test add_bias with no gradient tracking."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([0.1, 0.2])

        y = x.add_bias(b)

        assert y.requires_grad is False
        assert y._backward is None


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestEndToEndBackward:
    """End-to-end tests for complete backward pass through multiple ops."""

    def test_sum_of_add(self):
        """Test backward through sum(a + b)."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

        c = a.add(b)
        loss = c.sum()

        loss.backward()

        # Both gradients should be all ones
        assert a.grad.to_list() == pytest.approx([1.0, 1.0, 1.0])
        assert b.grad.to_list() == pytest.approx([1.0, 1.0, 1.0])

    def test_sum_of_relu(self):
        """Test backward through sum(relu(x))."""
        x = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)

        y = x.relu()  # [0, 0, 1, 2]
        loss = y.sum()  # 3.0

        loss.backward()

        # Gradient: relu mask * 1 = [0, 0, 1, 1]
        assert x.grad.to_list() == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_sum_of_relu_add(self):
        """Test backward through sum(relu(a + b))."""
        a = Tensor([-2.0, -1.0, 0.0, 1.0], requires_grad=True)
        b = Tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)

        c = a.add(b)  # [-1, 0, 1, 2]
        y = c.relu()  # [0, 0, 1, 2]
        loss = y.sum()  # 3.0

        loss.backward()

        # relu mask: [0, 0, 1, 1]
        # a.grad = b.grad = [0, 0, 1, 1]
        assert a.grad.to_list() == pytest.approx([0.0, 0.0, 1.0, 1.0])
        assert b.grad.to_list() == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_add_bias_relu_sum(self):
        """Test MLP-like forward: sum(relu(X + b))."""
        x = Tensor([[-1.0, 1.0], [0.0, 2.0]], requires_grad=True)  # (2, 2)
        b = Tensor([0.5, -0.5], requires_grad=True)  # (2,)

        # Forward: y = relu(x + b)
        # x + b = [[-0.5, 0.5], [0.5, 1.5]]
        # relu = [[0, 0.5], [0.5, 1.5]]
        y = x.add_bias(b).relu()
        loss = y.sum()  # 0 + 0.5 + 0.5 + 1.5 = 2.5

        assert loss.item() == pytest.approx(2.5)

        loss.backward()

        # relu mask on (x+b): [[0, 1], [1, 1]]
        # x.grad = relu_mask * 1 = [[0, 1], [1, 1]]
        x_result = x.grad.to_list()
        assert x_result[0] == pytest.approx([0.0, 1.0])
        assert x_result[1] == pytest.approx([1.0, 1.0])

        # b.grad = sum over rows of relu_mask = [0+1, 1+1] = [1, 2]
        assert b.grad.to_list() == pytest.approx([1.0, 2.0])


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestMatmul:
    """Tests for matrix multiplication operation."""

    def test_matmul_forward_small(self):
        """Test matmul forward with small matrices."""
        # A: (2, 3) @ B: (3, 2) -> C: (2, 2)
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        b = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)

        c = a.matmul(b)

        assert c.shape == (2, 2)
        # Manual calculation:
        # c[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        # c[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        # c[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        # c[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        result = c.to_list()
        assert result[0] == pytest.approx([22.0, 28.0])
        assert result[1] == pytest.approx([49.0, 64.0])

    def test_matmul_forward_identity(self):
        """Test matmul with identity matrix."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        identity = Tensor([[1.0, 0.0], [0.0, 1.0]])

        c = a.matmul(identity)

        result = c.to_list()
        assert result[0] == pytest.approx([1.0, 2.0])
        assert result[1] == pytest.approx([3.0, 4.0])

    def test_matmul_forward_batch_vector(self):
        """Test matmul typical case: batch of vectors times weight matrix."""
        # X: (batch=4, in=3) @ W: (in=3, out=2) -> (4, 2)
        x = Tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )  # (4, 3)
        w = Tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )  # (3, 2)

        y = x.matmul(w)

        assert y.shape == (4, 2)
        # Row 0: [1,0,0] @ W = [1, 2]
        # Row 1: [0,1,0] @ W = [3, 4]
        # Row 2: [0,0,1] @ W = [5, 6]
        # Row 3: [1,1,1] @ W = [1+3+5, 2+4+6] = [9, 12]
        result = y.to_list()
        assert result[0] == pytest.approx([1.0, 2.0])
        assert result[1] == pytest.approx([3.0, 4.0])
        assert result[2] == pytest.approx([5.0, 6.0])
        assert result[3] == pytest.approx([9.0, 12.0])

    def test_matmul_shape_validation_not_2d(self):
        """Test that matmul rejects non-2D inputs."""
        a = Tensor([1.0, 2.0, 3.0])  # 1D
        b = Tensor([[1.0], [2.0], [3.0]])  # 2D

        with pytest.raises(ValueError, match="2D"):
            a.matmul(b)

    def test_matmul_shape_validation_inner_mismatch(self):
        """Test that matmul rejects incompatible inner dimensions."""
        a = Tensor([[1.0, 2.0]])  # (1, 2)
        b = Tensor([[1.0, 2.0, 3.0]])  # (1, 3) - inner dim mismatch

        with pytest.raises(ValueError, match="mismatch"):
            a.matmul(b)

    def test_matmul_backward_a_only(self):
        """Test matmul backward when only A requires grad."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2, 2)
        b = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)  # identity

        c = a.matmul(b)
        loss = c.sum()

        loss.backward()

        # dC = ones(2,2)
        # dA = dC @ B^T = ones @ I = ones
        result = a.grad.to_list()
        assert result[0] == pytest.approx([1.0, 1.0])
        assert result[1] == pytest.approx([1.0, 1.0])
        assert b.grad is None

    def test_matmul_backward_b_only(self):
        """Test matmul backward when only B requires grad."""
        a = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)  # identity
        b = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2, 2)

        c = a.matmul(b)
        loss = c.sum()

        loss.backward()

        # dC = ones(2,2)
        # dB = A^T @ dC = I @ ones = ones
        result = b.grad.to_list()
        assert result[0] == pytest.approx([1.0, 1.0])
        assert result[1] == pytest.approx([1.0, 1.0])
        assert a.grad is None

    def test_matmul_backward_both(self):
        """Test matmul backward when both require grad."""
        # A: (2, 3), B: (3, 2) -> C: (2, 2)
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        b = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)

        c = a.matmul(b)
        loss = c.sum()

        loss.backward()

        # dC = ones(2, 2)
        # dA = dC @ B^T: (2,2) @ (2,3) -> (2,3)
        # B^T = [[1,3,5], [2,4,6]]
        # dA = ones @ B^T = [[1+2, 3+4, 5+6], [1+2, 3+4, 5+6]] = [[3,7,11], [3,7,11]]
        a_result = a.grad.to_list()
        assert a_result[0] == pytest.approx([3.0, 7.0, 11.0])
        assert a_result[1] == pytest.approx([3.0, 7.0, 11.0])

        # dB = A^T @ dC: (3,2) @ (2,2) -> (3,2)
        # A^T = [[1,4], [2,5], [3,6]]
        # dB = A^T @ ones = [[1+4, 1+4], [2+5, 2+5], [3+6, 3+6]] = [[5,5], [7,7], [9,9]]
        b_result = b.grad.to_list()
        assert b_result[0] == pytest.approx([5.0, 5.0])
        assert b_result[1] == pytest.approx([7.0, 7.0])
        assert b_result[2] == pytest.approx([9.0, 9.0])

    def test_matmul_no_grad(self):
        """Test matmul with no gradient tracking."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[1.0, 0.0], [0.0, 1.0]])

        c = a.matmul(b)

        assert c.requires_grad is False
        assert c._backward is None


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
class TestMLPForwardBackward:
    """Test full MLP forward and backward pass."""

    def test_mlp_forward_backward(self):
        """Test complete MLP: loss = sum(relu(X @ W + b))."""
        # Input: (batch=2, in_features=3)
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=False)

        # Weights: (in_features=3, out_features=2)
        w = Tensor([[0.1, 0.2], [-0.1, 0.1], [0.0, -0.1]], requires_grad=True)

        # Bias: (out_features=2)
        b = Tensor([0.5, -0.5], requires_grad=True)

        # Forward pass
        # h = X @ W: (2,3) @ (3,2) -> (2,2)
        h = x.matmul(w)

        # y = h + b (broadcast)
        y = h.add_bias(b)

        # z = relu(y)
        z = y.relu()

        # loss = sum(z)
        loss = z.sum()

        # Should run without error
        loss.backward()

        # Check that gradients were computed
        assert w.grad is not None
        assert b.grad is not None
        assert w.grad.shape == w.shape
        assert b.grad.shape == b.shape
