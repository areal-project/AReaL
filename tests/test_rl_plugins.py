# SPDX-License-Identifier: Apache-2.0

import json
import math

import pytest
import torch

from areal.api.cli_args import (
    GenerationHyperparameters,
    MicroBatchSpec,
    PluginConfig,
    RejectionSamplingConfig,
)
from areal.api.rl_plugins import PolicyDistribution
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils.data import (
    concat_batch,
    pack_tensor_dict,
    split_batch,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import ppo_actor_loss_fn


def test_plugin_config_clone_and_rpc_preserve_type():
    config = GenerationHyperparameters(
        request_plugin=PluginConfig("examples.vlm.pacman.policy.TokenSetRequest")
    )
    restored = deserialize_value(
        json.loads(
            json.dumps(serialize_value(config.new(temperature=0.7)), allow_nan=False)
        )
    )
    assert isinstance(restored.request_plugin, PluginConfig)
    assert restored.request_plugin == config.request_plugin
    assert "request_plugin" not in GenerationHyperparameters().to_openai_args_dict()
    with pytest.raises(TypeError, match="must implement"):
        restored.request_plugin.build(PolicyDistribution)


def test_rpc_nonfinite_bounds_survive_strict_json():
    values = [float("inf"), float("-inf"), float("nan"), 0.7]
    restored = deserialize_value(
        json.loads(json.dumps(serialize_value(values), allow_nan=False))
    )
    assert restored[:2] == values[:2] and restored[3] == values[3]
    assert math.isnan(restored[2])
    with pytest.raises(ValueError, match="Invalid nonfinite"):
        deserialize_value({"type": "nonfinite_float", "value": "oops"})


def test_token_metadata_split_pack_and_unpad_preserve_feature_axis():
    rows = [
        {
            "input_ids": torch.arange(length).view(1, -1),
            "attention_mask": torch.ones((1, length), dtype=torch.bool),
            "features": torch.arange(length * 7).view(1, length, 7),
        }
        for length in (3, 5)
    ]
    data, meta = concat_batch(rows)
    recovered = split_batch(data, meta)
    for original, result in zip(rows, recovered, strict=True):
        torch.testing.assert_close(
            result["features"], original["features"], rtol=0, atol=0
        )
    mbs = split_padded_tensor_dict_into_mb_list(
        data, MicroBatchSpec(max_tokens_per_mb=5)
    )
    for mb in mbs.mbs:
        assert mb["features"].shape[:2] == mb["attention_mask"].shape
        packed = pack_tensor_dict(mb)
        assert packed["features"].shape == (int(mb["attention_mask"].sum()), 7)
        for row in range(mb["input_ids"].shape[0]):
            length = int(mb["attention_mask"][row].sum())
            torch.testing.assert_close(
                mb["features"][row, :length],
                torch.arange(length * 7).view(length, 7),
                rtol=0,
                atol=0,
            )


def test_weighted_ppo_rejection_keeps_original_denominator():
    current = torch.zeros((2, 2), requires_grad=True)
    prox = torch.zeros_like(current)
    behavior = torch.tensor([[0.0, 0.0], [-2.0, -2.0]])
    mask = torch.tensor([[True, False], [True, True]])
    weights = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    loss, _ = ppo_actor_loss_fn(
        current,
        prox,
        behavior,
        torch.ones_like(current),
        0.05,
        mask,
        rejection_sampling=RejectionSamplingConfig(
            level="sequence",
            action="mask",
            metric="ratio",
            agg="sum",
            lower=0.8,
            upper=1.25,
        ),
        loss_reduction_weights=weights,
    )
    torch.testing.assert_close(loss, torch.tensor(-0.5), rtol=0, atol=1e-6)
    loss.backward()
    torch.testing.assert_close(
        current.grad, torch.tensor([[-0.5, 0.0], [0.0, 0.0]]), rtol=0, atol=1e-6
    )
