# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor


class _FakeR3Engine:
    _r3_enabled = True
    _r3_pending_routed_experts = None
    _r3_pending_valid = None

    def eval(self):
        self.eval_called = True

    def forward(self, input_, aggregate_fn):
        assert "routed_experts" not in input_
        assert "r3_routing_valid" not in input_
        assert self._r3_pending_routed_experts is not None
        assert self._r3_pending_valid is not None
        return aggregate_fn([torch.ones(1), torch.ones(1)])


def test_compute_logp_uses_r3_side_channel_without_forwarding_keys():
    engine = _FakeR3Engine()
    actor = PPOActor(PPOActorConfig(), engine)
    routed = torch.ones(2, 4, 3, 2, dtype=torch.int32)
    valid = torch.tensor([True, False])
    data = {
        "input_ids": torch.ones(2, 4, dtype=torch.int64),
        "attention_mask": torch.ones(2, 4, dtype=torch.bool),
        "routed_experts": routed,
        "r3_routing_valid": valid,
    }

    result = actor._compute_logp(data)

    torch.testing.assert_close(result, torch.ones(2), rtol=0, atol=0)
    assert engine._r3_pending_routed_experts is None
    assert engine._r3_pending_valid is None
    assert data["routed_experts"] is routed
    assert data["r3_routing_valid"] is valid


def test_split_r3_minibatches_follows_forward_indices():
    actor = PPOActor(PPOActorConfig(), _FakeR3Engine())
    routed = torch.arange(3 * 4 * 2 * 2).reshape(3, 4, 2, 2)
    valid = torch.tensor([True, False, True])
    mb_inputs = SimpleNamespace(
        forward_indices=[2, 0, 1],
        mbs=[
            {"input_ids": torch.ones(2, 4)},
            {"input_ids": torch.ones(1, 4)},
        ],
    )

    r3_mbs = actor._split_r3_minibatches(routed, valid, mb_inputs)

    torch.testing.assert_close(r3_mbs[0][0], routed[[2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(r3_mbs[0][1], valid[[2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(r3_mbs[1][0], routed[[1]], rtol=0, atol=0)
    torch.testing.assert_close(r3_mbs[1][1], valid[[1]], rtol=0, atol=0)
