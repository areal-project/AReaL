# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.api.data_hook import HookConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data_hook import load_data_hook


class _Hook:
    def __init__(self, value):
        self.value = value


class _ComputeHook:
    def pre_compute_logp(self, data):
        transformed = [
            {**item, "input_ids": item["input_ids"] + 1} for item in data
        ]
        return transformed, "aligned"

    def post_compute_logp(self, outputs, state):
        assert state == "aligned"
        return [output - 1 for output in outputs]


def test_load_data_hook_constructs_configured_class(monkeypatch):
    monkeypatch.setattr("areal.utils.data_hook.import_from_string", lambda _: _Hook)
    hook = load_data_hook(HookConfig(class_path="example.Hook", kwargs={"value": 3}))
    assert isinstance(hook, _Hook)
    assert hook.value == 3

    monkeypatch.setattr("areal.utils.data_hook.import_from_string", lambda _: dict)
    assert load_data_hook(HookConfig(class_path="example.Mapping")) == {}
    assert load_data_hook(None) is None

    monkeypatch.setattr("areal.utils.data_hook.import_from_string", lambda _: 3)
    with pytest.raises(TypeError, match="must name a class"):
        load_data_hook(HookConfig(class_path="example.Invalid"))


def test_ppo_actor_compute_logp_calls_hook_around_batched_forward():
    actor = PPOActor.__new__(PPOActor)
    actor.data_hook = _ComputeHook()
    actor._compute_logp = lambda batch: batch["input_ids"].float() * 2
    data = [
        {
            "input_ids": torch.tensor([[2]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.bool),
        }
    ]

    result = actor.compute_logp(data)

    torch.testing.assert_close(result[0], torch.tensor([[5.0]]), rtol=0, atol=0)
    torch.testing.assert_close(data[0]["input_ids"], torch.tensor([[2]]), rtol=0, atol=0)
