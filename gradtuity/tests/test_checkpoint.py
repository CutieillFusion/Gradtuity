"""Test gradient checkpointing: gradients must match the non-checkpointed path."""

import pytest

from gradtuity import Tensor, checkpoint


def _walk_compare(a, b, atol):
    if isinstance(a, list):
        for ai, bi in zip(a, b):
            _walk_compare(ai, bi, atol)
    else:
        assert abs(a - b) < atol, f"{a} vs {b}"


@pytest.mark.requires_triton
def test_checkpoint_gradient_matches_unwrapped():
    # f(x) = ((x * 2) + 3) * x
    def f(x: Tensor) -> Tensor:
        y = x.scale(2.0)
        # Use scale + add via simple ops: just chain a few ops to make a non-trivial graph
        z = y.scale(1.0).gelu()  # arbitrary differentiable chain
        return z

    data = [[1.0, -0.5, 2.0, 0.3], [0.1, -2.0, 1.5, -1.0]]
    x_ref = Tensor(data, requires_grad=True)
    x_chk = Tensor(data, requires_grad=True)

    out_ref = f(x_ref).sum()
    out_chk = checkpoint(f, x_chk).sum()

    out_ref.backward()
    out_chk.backward()

    _walk_compare(x_ref.grad.to_list(), x_chk.grad.to_list(), atol=1e-5)


@pytest.mark.requires_triton
def test_checkpoint_attention_block_matches_unwrapped():
    """End-to-end: GPT-style block with checkpointing should give same gradients."""
    from gradtuity.nn import CausalSelfAttention, LayerNorm

    B, S, E = 1, 8, 32
    H = 4

    # Build a small block deterministically
    ln = LayerNorm(E)
    attn = CausalSelfAttention(embed_dim=E, num_heads=H)

    def block(x: Tensor) -> Tensor:
        return x.add(attn(ln(x)))

    import random
    random.seed(123)
    data = [[[random.uniform(-1, 1) for _ in range(E)] for _ in range(S)] for _ in range(B)]

    x_ref = Tensor(data, requires_grad=True)
    x_chk = Tensor(data, requires_grad=True)

    out_ref = block(x_ref).sum()
    out_chk = checkpoint(block, x_chk).sum()

    out_ref.backward()
    out_chk.backward()

    _walk_compare(x_ref.grad.to_list(), x_chk.grad.to_list(), atol=1e-4)
