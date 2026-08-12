# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor, grpo_loss_fn


def test_compute_advantages_masks_stale_rollout_tokens():
    """Mask generated tokens produced by an older rollout policy version."""
    engine = MagicMock()
    engine.get_version.return_value = 2
    actor = PPOActor(
        PPOActorConfig(
            kl_ctl=0.0,
            mask_stale_rollout_tokens=True,
        ),
        engine,
    )
    data = {
        "input_ids": torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int64),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
        "logprobs": torch.zeros((1, 5), dtype=torch.float32),
        "versions": torch.tensor([[-1, -1, 1, 2, 2]], dtype=torch.int32),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
    }

    out = actor._compute_advantages(data)

    torch.testing.assert_close(
        out["loss_mask"],
        torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        out["versions"],
        torch.tensor([[-1, -1, 1, 2, 2]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        out["_aligned_versions"],
        torch.tensor([[-1, 1, 2, 2, -1]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_compute_advantages_keeps_stale_rollout_tokens_by_default():
    """Keep stale rollout tokens and versions when stale masking is disabled."""
    engine = MagicMock()
    engine.get_version.return_value = 2
    actor = PPOActor(PPOActorConfig(kl_ctl=0.0), engine)
    data = {
        "input_ids": torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int64),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
        "logprobs": torch.zeros((1, 5), dtype=torch.float32),
        "versions": torch.tensor([[-1, -1, 1, 2, 2]], dtype=torch.int32),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
    }

    out = actor._compute_advantages(data)

    torch.testing.assert_close(
        out["loss_mask"],
        torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        out["versions"],
        torch.tensor([[-1, -1, 1, 2, 2]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    assert "_aligned_versions" not in out


def test_compute_advantages_masking_is_noop_without_versions():
    """Leave the loss mask unchanged when rollout versions are unavailable."""
    engine = MagicMock()
    engine.get_version.return_value = 2
    actor = PPOActor(
        PPOActorConfig(
            kl_ctl=0.0,
            mask_stale_rollout_tokens=True,
        ),
        engine,
    )
    data = {
        "input_ids": torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int64),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
        "logprobs": torch.zeros((1, 5), dtype=torch.float32),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
    }

    out = actor._compute_advantages(data)

    torch.testing.assert_close(
        out["loss_mask"],
        torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_compute_advantages_masking_is_noop_with_none_versions():
    """Treat an explicitly unavailable rollout version tensor as missing."""
    engine = MagicMock()
    actor = PPOActor(
        PPOActorConfig(
            kl_ctl=0.0,
            mask_stale_rollout_tokens=True,
        ),
        engine,
    )
    data = {
        "input_ids": torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int64),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
        "logprobs": torch.zeros((1, 5), dtype=torch.float32),
        "versions": None,
        "rewards": torch.tensor([1.0], dtype=torch.float32),
    }

    out = actor._compute_advantages(data)

    torch.testing.assert_close(
        out["loss_mask"],
        torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert out["versions"] is None


def test_grpo_loss_skips_staleness_metrics_with_none_versions():
    """Skip version staleness metrics when rollout versions are unavailable."""
    input_data = {
        "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.int64),
        "logprobs": torch.zeros((1, 3), dtype=torch.float32),
        "advantages": torch.ones((1, 3), dtype=torch.float32),
        "loss_mask": torch.ones((1, 3), dtype=torch.bool),
        "prox_logp": torch.zeros((1, 3), dtype=torch.float32),
        "versions": None,
    }

    with (
        patch("areal.trainer.ppo.actor.stats_tracker"),
        patch("areal.trainer.ppo.actor._log_version_staleness_stats") as log_staleness,
    ):
        loss = grpo_loss_fn(
            logprobs=torch.zeros((1, 3), dtype=torch.float32),
            entropy=torch.zeros((1, 3), dtype=torch.float32),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            current_version=2,
        )

    assert torch.isfinite(loss)
    log_staleness.assert_not_called()


def test_grpo_loss_uses_aligned_versions_when_available():
    """Use next-token-aligned versions for proximal loss and version metrics."""
    raw_versions = torch.tensor([[-1, -1, 1, 2]], dtype=torch.int32)
    aligned_versions = torch.tensor([[-1, 1, 2, -1]], dtype=torch.int32)
    input_data = {
        "input_ids": torch.tensor([[11, 12, 13, 14]], dtype=torch.int64),
        "logprobs": torch.zeros((1, 4), dtype=torch.float32),
        "advantages": torch.ones((1, 4), dtype=torch.float32),
        "loss_mask": torch.tensor([[0, 1, 1, 0]], dtype=torch.bool),
        "prox_logp": torch.zeros((1, 4), dtype=torch.float32),
        "versions": raw_versions,
        "_aligned_versions": aligned_versions,
    }

    with (
        patch("areal.trainer.ppo.actor.stats_tracker"),
        patch(
            "areal.trainer.ppo.actor._resolve_proximal_logp",
            return_value=torch.zeros((1, 4), dtype=torch.float32),
        ) as resolve_proximal_logp,
        patch(
            "areal.trainer.ppo.actor._log_proximal_approximation_stats"
        ) as log_proximal_stats,
        patch("areal.trainer.ppo.actor._log_version_staleness_stats") as log_staleness,
    ):
        loss = grpo_loss_fn(
            logprobs=torch.zeros((1, 4), dtype=torch.float32),
            entropy=torch.zeros((1, 4), dtype=torch.float32),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            current_version=2,
        )

    assert torch.isfinite(loss)
    assert resolve_proximal_logp.call_args.kwargs["versions"] is aligned_versions
    assert log_proximal_stats.call_args.kwargs["versions"] is aligned_versions
    assert log_staleness.call_args.kwargs["versions"] is aligned_versions


def test_compute_advantages_keeps_fully_stale_trajectory_without_fresh_suffix():
    """Keep a fully stale trajectory rather than creating a zero-weight batch."""
    engine = MagicMock()
    engine.get_version.return_value = 2
    actor = PPOActor(
        PPOActorConfig(
            kl_ctl=0.0,
            mask_stale_rollout_tokens=True,
        ),
        engine,
    )
    data = {
        "input_ids": torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int64),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
        "logprobs": torch.zeros((1, 5), dtype=torch.float32),
        "versions": torch.tensor([[-1, -1, 1, 1, 1]], dtype=torch.int32),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
    }

    out = actor._compute_advantages(data)

    torch.testing.assert_close(
        out["loss_mask"],
        torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
