# SPDX-License-Identifier: Apache-2.0

import base64

import numpy as np
import pytest
import torch

from areal.engine.r3.asserts import R3Error
from areal.engine.r3.preprocess import (
    decode_sglang_routed_experts,
    preprocess_routed_experts_batch,
)


def _encode_int32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.int32).tobytes()).decode("utf-8")


def test_decode_sglang_routed_experts_valid_payload_returns_2d_array():
    raw = np.arange(12, dtype=np.int32).reshape(3, 4)

    decoded = decode_sglang_routed_experts(
        _encode_int32(raw),
        prompt_tokens=2,
        completion_tokens=2,
        source="test",
    )

    np.testing.assert_array_equal(decoded, raw)


def test_decode_sglang_routed_experts_bad_shape_raises_clear_error():
    raw = np.arange(5, dtype=np.int32)

    with pytest.raises(R3Error, match="cannot be reshaped"):
        decode_sglang_routed_experts(
            _encode_int32(raw),
            prompt_tokens=2,
            completion_tokens=2,
            source="test",
        )


def test_preprocess_routed_experts_flat_dim_mismatch_raises():
    raw = np.ones((3, 5), dtype=np.int32)

    with pytest.raises(R3Error, match="flat_dim"):
        preprocess_routed_experts_batch(
            [raw],
            seq_lens=[4],
            num_moe_layers=2,
            topk=2,
        )


def test_preprocess_routed_experts_drops_leading_dense_layer_slots():
    moe_layers = np.arange(1, 13, dtype=np.int32).reshape(3, 2, 2)
    dense_layer = np.zeros((3, 1, 2), dtype=np.int32)
    raw = np.concatenate([dense_layer, moe_layers], axis=1).reshape(3, -1)

    out = preprocess_routed_experts_batch(
        [raw],
        seq_lens=[4],
        num_moe_layers=2,
        topk=2,
    )

    routed = out["routed_experts"]
    valid = out["r3_routing_valid"]
    assert routed.shape == (1, 4, 2, 2)
    assert valid.tolist() == [True]
    torch.testing.assert_close(
        routed[0, :3],
        torch.as_tensor(moe_layers, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(routed[0, 3], routed[0, 2], rtol=0, atol=0)


def test_preprocess_routed_experts_fills_legal_missing_last_token():
    raw = np.arange(1, 13, dtype=np.int32).reshape(3, 4)

    out = preprocess_routed_experts_batch(
        [raw],
        seq_lens=[4],
        num_moe_layers=2,
        topk=2,
    )

    routed = out["routed_experts"]
    valid = out["r3_routing_valid"]
    assert routed.shape == (1, 4, 2, 2)
    assert valid.tolist() == [True]
    torch.testing.assert_close(routed[0, 3], routed[0, 2], rtol=0, atol=0)


def test_preprocess_routed_experts_internal_zero_row_marks_invalid():
    raw = np.array(
        [
            [1, 2, 3, 4],
            [0, 0, 0, 0],
            [5, 6, 7, 8],
        ],
        dtype=np.int32,
    )

    out = preprocess_routed_experts_batch(
        [raw],
        seq_lens=[4],
        num_moe_layers=2,
        topk=2,
    )

    assert out["r3_routing_valid"].tolist() == [False]
    torch.testing.assert_close(
        out["routed_experts"][0, 1],
        out["routed_experts"][0, 0],
        rtol=0,
        atol=0,
    )


def test_preprocess_routed_experts_none_marks_invalid_without_fake_routing():
    out = preprocess_routed_experts_batch(
        [None],
        seq_lens=[3],
        num_moe_layers=2,
        topk=2,
    )

    assert out["r3_routing_valid"].tolist() == [False]
    assert torch.count_nonzero(out["routed_experts"]) == 0
