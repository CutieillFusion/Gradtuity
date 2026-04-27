"""
Tests for nn.TiedLMHead (weight tying with embedding).

These tests require a CUDA-enabled GPU to run.
"""

import pytest

from gradtuity import Embedding, TiedLMHead, randn

pytestmark = pytest.mark.requires_cuda


class TestTiedLMHeadWeightRef:
    """id(wte.weight) == id(lm_head.weight_ref)."""

    def test_weight_ref_is_same_tensor(self):
        wte = Embedding(100, 32)
        lm_head = TiedLMHead(wte)
        assert lm_head.weight_ref is wte.weight
        assert id(lm_head.weight_ref) == id(wte.weight)


class TestTiedLMHeadGradient:
    """Gradient check: logits.sum().backward() gives nonzero wte.weight.grad."""

    def test_backward_into_shared_weight(self):
        wte = Embedding(10, 4)
        lm_head = TiedLMHead(wte)
        h = randn((2, 3, 4), requires_grad=True)
        logits = lm_head(h)
        loss = logits.sum()
        loss.backward()
        assert wte.weight.grad is not None
        flat = wte.weight.grad.to_list()
        total = sum(sum(row) for row in flat)
        assert total != 0.0


class TestTiedLMHeadParameters:
    """Parameter list does not double-count tied weight."""

    def test_tied_lm_head_parameters_empty(self):
        wte = Embedding(20, 8)
        lm_head = TiedLMHead(wte)
        assert lm_head.parameters() == []

    def test_model_parameters_no_double_count(self):
        wte = Embedding(50, 16)
        lm_head = TiedLMHead(wte)
        params = lm_head.parameters()
        wte_params = wte.parameters()
        assert len(params) == 0
        assert len(wte_params) == 1
        all_tensors = {id(p) for p in wte_params}
        assert id(wte.weight) in all_tensors
