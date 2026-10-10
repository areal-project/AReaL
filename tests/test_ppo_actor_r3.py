# SPDX-License-Identifier: Apache-2.0

import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils import stats_tracker


class _FakeR3Engine:
    def __init__(self, forward_result=None, *, r3_enabled=True, require_r3=True):
        self._r3_enabled = r3_enabled
        self._r3_pending_routed_experts = None
        self._r3_pending_valid = None
        self.forward_result = forward_result
        self.require_r3 = require_r3

    def eval(self):
        self.eval_called = True

    def forward(self, input_, aggregate_fn):
        assert "routed_experts" not in input_
        assert "r3_routing_valid" not in input_
        if self.require_r3:
            assert self._r3_pending_routed_experts is not None
            assert self._r3_pending_valid is not None
        if self.forward_result is not None:
            return self.forward_result
        return aggregate_fn([torch.ones(1), torch.ones(1)])


def _reset_stats_tracker():
    stats_tracker.export(reset=True)


def test_compute_logp_uses_r3_side_channel_without_forwarding_keys():
    _reset_stats_tracker()
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


def test_compute_logp_logs_rollout_train_r3_metrics():
    _reset_stats_tracker()
    train_logp = torch.tensor([[-0.1, -1.0, 0.0, 0.0]], dtype=torch.float32)
    engine = _FakeR3Engine(forward_result=train_logp)
    actor = PPOActor(PPOActorConfig(), engine)
    data = {
        "input_ids": torch.ones(1, 4, dtype=torch.int64),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "logprobs": torch.tensor([[9.0, -0.3, -0.6, 8.0]], dtype=torch.float32),
        "loss_mask": torch.tensor([[False, True, True, False]]),
        "routed_experts": torch.ones(1, 4, 3, 2, dtype=torch.int32),
        "r3_routing_valid": torch.tensor([True]),
    }

    result = actor._compute_logp(data)

    torch.testing.assert_close(result, train_logp, rtol=0, atol=0)
    # Compact diagnostics retain scalar sufficient statistics, not token arrays.
    tracker = stats_tracker.DEFAULT_TRACKER
    assert "compute_logp/r3/rollout_train_k3_kl" in tracker.compact_keys
    exported = stats_tracker.export(reset=True)
    assert exported["compute_logp/r3/n_valid_tokens"] == 2.0
    delta = torch.tensor([0.2, -0.4], dtype=torch.float32)
    assert exported["compute_logp/r3/enabled"] == 1.0
    torch.testing.assert_close(
        torch.tensor(exported["compute_logp/r3/rollout_train_logp_abs_diff/avg"]),
        delta.abs().mean(),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(exported["compute_logp/r3/rollout_train_logp_sq_diff/avg"]),
        delta.square().mean(),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(exported["compute_logp/r3/rollout_train_k3_kl/avg"]),
        (torch.expm1(delta) - delta).mean(),
        rtol=0,
        atol=1e-6,
    )
    assert exported["compute_logp/r3/rollout_train_extreme_frac_tau2/avg"] == 0.0
    assert exported["compute_logp/r3/rollout_train_extreme_frac_tau5/avg"] == 0.0


def test_compute_logp_r3_metrics_do_not_wrap_loss_mask_last_token():
    _reset_stats_tracker()
    train_logp = torch.tensor([[-0.5, 10.0, 10.0, 10.0]], dtype=torch.float32)
    engine = _FakeR3Engine(forward_result=train_logp)
    actor = PPOActor(PPOActorConfig(), engine)
    data = {
        "input_ids": torch.ones(1, 4, dtype=torch.int64),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "logprobs": torch.tensor([[99.0, -0.25, 77.0, 88.0]], dtype=torch.float32),
        "loss_mask": torch.tensor([[True, True, False, False]]),
        "routed_experts": torch.ones(1, 4, 3, 2, dtype=torch.int32),
        "r3_routing_valid": torch.tensor([True]),
    }

    actor._compute_logp(data)

    exported = stats_tracker.export(reset=True)
    torch.testing.assert_close(
        torch.tensor(exported["compute_logp/r3/rollout_train_logp_abs_diff/avg"]),
        torch.tensor(0.25),
        rtol=0,
        atol=1e-6,
    )


def test_compute_logp_logs_r3_disabled_for_rollout_train_metrics():
    _reset_stats_tracker()
    engine = _FakeR3Engine(
        forward_result=torch.zeros(1, 3),
        r3_enabled=False,
        require_r3=False,
    )
    actor = PPOActor(PPOActorConfig(), engine)
    data = {
        "input_ids": torch.ones(1, 3, dtype=torch.int64),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "logprobs": torch.zeros(1, 3),
        "loss_mask": torch.tensor([[False, True, False]]),
    }

    actor._compute_logp(data)

    exported = stats_tracker.export(reset=True)
    assert exported["compute_logp/r3/enabled"] == 0.0


def test_split_r3_minibatches_reorders_rows_and_skips_transport_dummy():
    actor = PPOActor(PPOActorConfig(), _FakeR3Engine())
    routed = torch.arange(3 * 4 * 2 * 2).reshape(3, 4, 2, 2)
    valid = torch.tensor([True, False, True])
    mb_inputs = [
        {
            "input_ids": torch.ones(2, 4),
            "_r3_row_indices": torch.tensor([[2] * 4, [0] * 4]),
        },
        {"input_ids": torch.ones(1, 4), "_r3_row_indices": torch.tensor([[1] * 4])},
        {
            "input_ids": torch.ones(1, 1),
            "_transport_dummy": True,
            "_r3_row_indices": torch.zeros(1, 1, dtype=torch.long),
        },
    ]

    r3_mbs = actor._split_r3_minibatches(routed, valid, mb_inputs)

    torch.testing.assert_close(r3_mbs[0][0], routed[[2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(r3_mbs[0][1], valid[[2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(r3_mbs[1][0], routed[[1]], rtol=0, atol=0)
    torch.testing.assert_close(r3_mbs[1][1], valid[[1]], rtol=0, atol=0)

    assert r3_mbs[2] == (None, None)
    assert all("_r3_row_indices" not in mb for mb in mb_inputs)


def test_split_r3_minibatches_real_splitter_preserves_variable_length_rows():
    """The balanced PPO split must carry each row's original routing."""
    from areal.utils.data import split_training_batch_into_microbatches

    actor = PPOActor(PPOActorConfig(), _FakeR3Engine())
    routed = torch.arange(4 * 6 * 2 * 2).reshape(4, 6, 2, 2)
    valid = torch.tensor([True, False, True, True])
    row_ids = torch.arange(4, dtype=torch.long).unsqueeze(1).expand(-1, 6)
    data = {
        "input_ids": row_ids.clone(),
        "attention_mask": torch.arange(6).unsqueeze(0)
        < torch.tensor([6, 1, 4, 2]).unsqueeze(1),
        "_r3_row_indices": row_ids,
    }
    minibatches = split_training_batch_into_microbatches(data, n_mbs=3)
    r3_minibatches = actor._split_r3_minibatches(routed, valid, minibatches)
    seen = []
    for mb, (routes, validity) in zip(minibatches, r3_minibatches, strict=True):
        indices = mb["input_ids"][:, 0]
        seen.extend(indices.tolist())
        torch.testing.assert_close(routes, routed[indices], rtol=0, atol=0)
        torch.testing.assert_close(validity, valid[indices], rtol=0, atol=0)
        assert "_r3_row_indices" not in mb
    assert sorted(seen) == [0, 1, 2, 3]
