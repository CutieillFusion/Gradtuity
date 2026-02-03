"""
Tests for dropout (tensor.dropout, nn.Dropout, deterministic RNG).

These tests require a CUDA-enabled GPU to run.
"""

import pytest

from gradtuity import Dropout, Tensor, ones, zeros
from gradtuity.random import (
    DropoutRNG,
    default_rng,
    dropout_rng_state_dict,
    load_dropout_rng_state,
)

pytestmark = pytest.mark.requires_cuda


class TestDropoutEvalNoOp:
    """Eval no-op: drop(x) == x when training=False or model.eval()."""

    def test_tensor_dropout_training_false_returns_self(self):
        x = ones((4, 8), requires_grad=True)
        y = x.dropout(p=0.1, training=False)
        assert y.ptr == x.ptr
        assert y.shape == x.shape

    def test_nn_dropout_eval_returns_same_values(self):
        drop = Dropout(p=0.1)
        drop.eval()
        x = ones((4, 8), requires_grad=True)
        y = drop(x)
        assert y.ptr == x.ptr
        # Values unchanged
        x_list = x.to_list()
        y_list = y.to_list()
        for i in range(len(x_list)):
            for j in range(len(x_list[i])):
                assert x_list[i][j] == y_list[i][j]


class TestDropoutTrainScaling:
    """Train behavior: inverted dropout, expected mean output ≈ 1 for ones input."""

    def test_dropout_ones_mean_approx_one(self):
        rng = DropoutRNG(seed=42, counter=0)
        x = ones((100, 100), requires_grad=True)
        y = x.dropout(p=0.1, training=True, rng=rng)
        flat = y.to_list()
        total = sum(sum(row) for row in flat)
        n = 100 * 100
        mean = total / n
        # Inverted dropout: kept elements are x/(1-p) = 1/0.9; dropped are 0.
        # Expected mean = (1-p) * (1/(1-p)) + p * 0 = 1.0
        assert 0.95 <= mean <= 1.05


class TestDropoutDeterminism:
    """Determinism: same seed+counter -> same output; counter advance -> different output."""

    def test_same_seed_counter_same_output(self):
        x = ones((10, 10), requires_grad=True)
        rng1 = DropoutRNG(seed=123, counter=0)
        y1 = x.dropout(p=0.2, training=True, rng=rng1)
        rng2 = DropoutRNG(seed=123, counter=0)
        y2 = x.dropout(p=0.2, training=True, rng=rng2)
        a = y1.to_list()
        b = y2.to_list()
        for i in range(len(a)):
            for j in range(len(a[i])):
                assert a[i][j] == b[i][j]

    def test_counter_advance_different_output(self):
        x = ones((10, 10), requires_grad=True)
        rng = DropoutRNG(seed=456, counter=0)
        y1 = x.dropout(p=0.2, training=True, rng=rng)
        y2 = x.dropout(p=0.2, training=True, rng=rng)
        a = y1.to_list()
        b = y2.to_list()
        # At least one element should differ (very unlikely to be identical)
        found_diff = False
        for i in range(len(a)):
            for j in range(len(a[i])):
                if a[i][j] != b[i][j]:
                    found_diff = True
                    break
            if found_diff:
                break
        assert found_diff


class TestDropoutBackward:
    """Backward: grad is 0 or 1/(1-p) and matches regenerated mask."""

    def test_backward_grad_scaling(self):
        rng = DropoutRNG(seed=789, counter=0)
        x = ones((20, 20), requires_grad=True)
        y = x.dropout(p=0.2, training=True, rng=rng)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        scale = 1.0 / (1.0 - 0.2)
        flat = x.grad.to_list()
        for row in flat:
            for v in row:
                assert v == 0.0 or abs(v - scale) < 1e-5


class TestDropoutEdgeCases:
    """p=0 no-op, p>=1 zeros."""

    def test_p_zero_returns_self(self):
        x = ones((2, 3), requires_grad=True)
        y = x.dropout(p=0.0, training=True)
        assert y.ptr == x.ptr

    def test_p_one_returns_zeros(self):
        x = ones((2, 3), requires_grad=True)
        y = x.dropout(p=1.0, training=True)
        flat = y.to_list()
        for row in flat:
            for v in row:
                assert v == 0.0


class TestDropoutRNGStateDict:
    """Checkpoint RNG save/load."""

    def test_dropout_rng_state_dict_round_trip(self):
        rng = default_rng()
        rng.counter = 100
        state = dropout_rng_state_dict()
        assert "seed" in state
        assert "counter" in state
        assert state["counter"] == 100
        load_dropout_rng_state(state)
        rng2 = default_rng()
        assert rng2.counter == 100
