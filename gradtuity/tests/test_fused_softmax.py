"""Test the fused causal-mask + softmax kernel against the unfused reference."""

import pytest

from gradtuity.tensor import Tensor


def _compare_tensors(a: Tensor, b: Tensor, atol: float = 1e-6) -> None:
    assert a.shape == b.shape
    a_list = a.to_list()
    b_list = b.to_list()

    def walk(x, y):
        if isinstance(x, list):
            for xi, yi in zip(x, y):
                walk(xi, yi)
        else:
            assert abs(x - y) < atol, f"{x} vs {y}"

    walk(a_list, b_list)


@pytest.mark.requires_triton
def test_fused_matches_unfused_forward():
    # Small (B, H, S, S) with explicit values
    import random
    random.seed(0)
    B, H, S = 1, 2, 8
    data = [
        [
            [[random.uniform(-2, 2) for _ in range(S)] for _ in range(S)]
            for _ in range(H)
        ]
        for _ in range(B)
    ]
    x_ref = Tensor(data, requires_grad=False)
    x_fused = Tensor(data, requires_grad=False)

    ref = x_ref.apply_causal_mask().softmax(dim=-1)
    fused = x_fused.softmax_with_causal_mask()

    _compare_tensors(ref, fused, atol=1e-5)


@pytest.mark.requires_triton
def test_fused_backward_matches_unfused():
    import random
    random.seed(1)
    B, H, S = 1, 2, 8
    data = [
        [
            [[random.uniform(-2, 2) for _ in range(S)] for _ in range(S)]
            for _ in range(H)
        ]
        for _ in range(B)
    ]
    x_ref = Tensor(data, requires_grad=True)
    x_fused = Tensor(data, requires_grad=True)

    out_ref = x_ref.apply_causal_mask().softmax(dim=-1).sum()
    out_fused = x_fused.softmax_with_causal_mask().sum()

    out_ref.backward()
    out_fused.backward()

    _compare_tensors(x_ref.grad, x_fused.grad, atol=1e-5)


@pytest.mark.requires_triton
def test_fused_masked_positions_are_zero():
    """For position (i, j > i), the fused output should be 0."""
    B, H, S = 1, 1, 4
    data = [[[[1.0, 2.0, 3.0, 4.0]] * S]]  # arbitrary scores
    out = Tensor(data, requires_grad=False).softmax_with_causal_mask()
    out_list = out.to_list()
    for i in range(S):
        for j in range(S):
            if j > i:
                assert out_list[0][0][i][j] == 0.0, f"pos ({i},{j}) not masked: {out_list[0][0][i][j]}"
