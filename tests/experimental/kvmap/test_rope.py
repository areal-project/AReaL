# SPDX-License-Identifier: Apache-2.0
"""Removing the rotary transform inverts applying it, and applying it matches Hugging Face Qwen2."""

import pytest
import torch

from areal.experimental.kvmap.rope import (
    apply_rotary,
    remove_rotary,
    rotary_cos_sin,
)

HEAD_DIM = 8
THETA = 1_000_000.0


def test_remove_rotary_inverts_apply_rotary_at_small_and_large_positions():
    generator = torch.Generator().manual_seed(0)
    keys = torch.randn(2, 3, 4, HEAD_DIM, dtype=torch.float64, generator=generator)
    positions = torch.tensor([[0, 1, 2, 3], [10_000, 10_001, 31_000, 32_767]])
    cos, sin = rotary_cos_sin(
        positions, head_dim=HEAD_DIM, theta=THETA, dtype=torch.float64
    )

    restored = remove_rotary(apply_rotary(keys, cos, sin), cos, sin)

    torch.testing.assert_close(restored, keys, rtol=0.0, atol=1e-12)
    # Witness that the transform is not the identity, so the round trip means something.
    assert not torch.allclose(apply_rotary(keys, cos, sin), keys)


def test_apply_rotary_matches_hugging_face_qwen2_rotary_embedding():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    config = Qwen2Config(
        hidden_size=HEAD_DIM * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=THETA,
    )
    positions = torch.tensor([[0, 5, 77, 4096]])
    keys = torch.randn(1, 1, 4, HEAD_DIM, generator=torch.Generator().manual_seed(1))
    hf_cos, hf_sin = Qwen2RotaryEmbedding(config)(keys, positions)
    _, expected = apply_rotary_pos_emb(keys, keys, hf_cos, hf_sin)

    cos, sin = rotary_cos_sin(
        positions, head_dim=HEAD_DIM, theta=THETA, dtype=torch.float32
    )
    actual = apply_rotary(keys, cos, sin)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "positions_shape, head_dim, message",
    [((4,), HEAD_DIM, "positions must be"), ((1, 4), 7, "head_dim must be even")],
)
def test_rotary_cos_sin_rejects_bad_shapes(positions_shape, head_dim, message):
    with pytest.raises(ValueError, match=message):
        rotary_cos_sin(
            torch.zeros(positions_shape, dtype=torch.long),
            head_dim=head_dim,
            theta=THETA,
            dtype=torch.float32,
        )


def test_apply_rotary_rejects_token_count_mismatch():
    keys = torch.zeros(1, 1, 4, HEAD_DIM)
    cos, sin = rotary_cos_sin(
        torch.zeros(1, 5, dtype=torch.long),
        head_dim=HEAD_DIM,
        theta=THETA,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="disagree on batch, tokens or d"):
        apply_rotary(keys, cos, sin)
