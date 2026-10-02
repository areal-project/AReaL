# SPDX-License-Identifier: Apache-2.0
"""Feature packing order is layer major, then head, then dimension, and bad selections are rejected."""

import pytest
import torch

from areal.experimental.kvmap.contract import (
    FeatureContract,
    pack_features,
    unpack_target,
)


def test_pack_features_orders_layer_then_head_then_dim_and_rows_batch_then_token():
    # value = 1000 * layer + 100 * head + 10 * token + dim, with batch as the sign
    layers = []
    for layer in range(2):
        x = torch.zeros(2, 2, 3, 4)
        for head in range(2):
            for token in range(3):
                for dim in range(4):
                    x[:, head, token, dim] = (
                        1000 * layer + 100 * head + 10 * token + dim
                    )
        x[1] *= -1
        layers.append(x)

    rows = pack_features(layers, (1, 0))

    assert rows.shape == (6, 16)
    assert rows[0].tolist() == [
        1000 + 100 * h + d for h in range(2) for d in range(4)
    ] + [100 * h + d for h in range(2) for d in range(4)]
    assert rows[1, 0] == 1010  # second token of batch 0
    assert rows[3, 0] == -1000  # first token of batch 1


def test_unpack_target_inverts_single_layer_packing():
    x = torch.randn(3, 2, 5, 4, generator=torch.Generator().manual_seed(0))
    assert torch.equal(
        unpack_target(
            pack_features([x], (0,)), batch=3, tokens=5, kv_heads=2, head_dim=4
        ),
        x,
    )


def test_same_layer_contract_selects_each_layer_once():
    contract = FeatureContract.same_layer(num_layers=4, kv_heads=2, head_dim=8)
    assert contract.source_layers_by_target == ((0,), (1,), (2,), (3,))
    assert contract.feature_dim(2) == 16 and contract.target_dim == 16
    assert FeatureContract.from_json(contract.to_json()) == contract


@pytest.mark.parametrize(
    "selection, message",
    [
        (((0, 0),), "twice"),
        (((5,),), "outside"),
        (((),), "no source layer"),
        ((), "at least one target layer"),
    ],
)
def test_invalid_selections_are_rejected(selection, message):
    with pytest.raises(ValueError, match=message):
        FeatureContract(
            source_layers_by_target=selection,
            source_num_layers=3,
            source_kv_heads=1,
            source_head_dim=2,
            target_kv_heads=1,
            target_head_dim=2,
        )


def test_from_json_rejects_unknown_keys():
    raw = FeatureContract.same_layer(num_layers=1, kv_heads=1, head_dim=2).to_json()
    raw["extra"] = 1
    with pytest.raises(ValueError, match="must be exactly"):
        FeatureContract.from_json(raw)
