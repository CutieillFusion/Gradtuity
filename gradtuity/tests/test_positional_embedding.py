"""
Tests for nn.PositionalEmbedding.

These tests require a CUDA-enabled GPU to run.
"""

import pytest

from gradtuity import PositionalEmbedding, Tensor

pytestmark = pytest.mark.requires_cuda


class TestPositionalEmbeddingShape:
    """PositionalEmbedding returns (B, S, E)."""

    def test_shape_b_s_e(self):
        wpe = PositionalEmbedding(max_positions=128, embed_dim=8)
        out = wpe(seq_len=10, batch_size=4)
        assert out.shape == (4, 10, 8)

    def test_small_max_positions(self):
        wpe = PositionalEmbedding(max_positions=5, embed_dim=2)
        out = wpe(seq_len=3, batch_size=2)
        assert out.shape == (2, 3, 2)


class TestPositionalEmbeddingSamePositionAcrossBatch:
    """Same position vector repeated across batch rows."""

    def test_same_position_vector_per_row(self):
        wpe = PositionalEmbedding(max_positions=10, embed_dim=4)
        out = wpe(seq_len=3, batch_size=2)
        data = out.to_list()
        # Row 0 and row 1 should have same position vectors (positions 0,1,2)
        for pos in range(3):
            assert data[0][pos] == pytest.approx(data[1][pos])


class TestPositionalEmbeddingStartPos:
    """start_pos offset works."""

    def test_start_pos_offset(self):
        wpe = PositionalEmbedding(max_positions=20, embed_dim=2)
        out_zero = wpe(seq_len=2, batch_size=1, start_pos=0)
        out_five = wpe(seq_len=2, batch_size=1, start_pos=5)
        zero_list = out_zero.to_list()
        five_list = out_five.to_list()
        # First position: start_pos=0 looks up position 0; start_pos=5 looks up position 5
        assert zero_list[0][0] != pytest.approx(five_list[0][0])
        # Second position: 1 vs 6
        assert zero_list[0][1] != pytest.approx(five_list[0][1])


class TestPositionalEmbeddingStateDict:
    """PositionalEmbedding state_dict includes embed.weight."""

    def test_state_dict_has_embed_weight(self):
        wpe = PositionalEmbedding(max_positions=16, embed_dim=4)
        state = wpe.state_dict()
        assert "embed.weight" in state
        assert state["embed.weight"].shape == (16, 4)
