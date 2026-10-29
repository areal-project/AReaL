from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import (
    PPOActor,
    _policy_version_loss_mask,
    grpo_loss_fn,
)
from areal.trainer.ppo.loss_reduction import prepare_policy_gradient_batch
from areal.utils.data import pack_tensor_dict
from areal.utils.stats_tracker import DistributedStatsTracker


def test_version_mask_preserves_fresh_tokens_in_mixed_concat_history():
    versions = torch.tensor([[-1, -1, 44, 48, -1, 49, -1]])
    mask = torch.tensor([[0, 1, 1, 0, 1, 0, 0]], dtype=torch.bool)
    original = versions.clone()
    keep, aligned = _policy_version_loss_mask(versions, mask, 49, 1)
    assert keep.tolist() == [[False, False, True, False, True, False, False]]
    assert aligned.tolist() == [[-1, 44, 48, -1, 49, -1, -1]]
    torch.testing.assert_close(versions, original, rtol=0, atol=0)
    assert mask.sum() == 3


def test_version_mask_zero_lag_keeps_only_current_targets():
    keep, _ = _policy_version_loss_mask(
        torch.tensor([[-1, 48, 49, -1]]),
        torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        49,
        0,
    )
    assert keep.tolist() == [[False, True, False, False]]


@pytest.mark.parametrize("versions", [None, torch.ones(1, 2), torch.ones(1, 3)])
def test_version_mask_invalid_metadata_raises(versions):
    with pytest.raises(ValueError, match="versions"):
        _policy_version_loss_mask(versions, torch.ones(1, 3).bool(), 5, 1)


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
@pytest.mark.parametrize("all_expired", [False, True])
def test_expired_tokens_have_zero_gradient_without_dropping_trajectory(
    mode, all_expired
):
    original_mask = torch.ones((1, 3), dtype=torch.bool)
    version_mask = torch.tensor([[False, not all_expired, False]])
    data = {
        "loss_mask": original_mask,
        "policy_version_mask": version_mask,
        "logprobs": torch.zeros((1, 3)),
        "advantages": torch.ones((1, 3)),
        "input_ids": torch.tensor([[10, 11, 12]]),
    }
    prepared = prepare_policy_gradient_batch(
        data, mode=mode, divisor=3 if mode == "constant" else None, group_sizes=[1]
    )
    step = prepared.for_steps([data])[0]
    logits = torch.zeros((1, 3), requires_grad=True)
    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        loss = grpo_loss_fn(
            logprobs=logits,
            entropy=torch.zeros_like(logits),
            input_data=data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            prox_logp_method="reuse_train_logp",
            loss_step=step,
        )
    loss.backward()
    expected = torch.tensor([[0.0, 0.0 if all_expired else -1 / 3, 0.0]])
    torch.testing.assert_close(logits.grad, expected, rtol=1e-6, atol=1e-7)
    assert step.loss_weight(data) > 0
    assert data["input_ids"].tolist() == [[10, 11, 12]]
    assert original_mask.all()
    assert torch.isfinite(loss)


def test_actor_masks_before_packing_and_logs_both_version_scopes():
    engine = MagicMock()
    engine.get_version.return_value = 49
    engine.train_batch.return_value = {}
    actor = PPOActor(PPOActorConfig(backend="fsdp:d1"), engine)
    shape = (1, 4)
    data = {
        "input_ids": torch.tensor([[10, 11, 12, 13]]),
        "attention_mask": torch.ones(shape, dtype=torch.bool),
        "loss_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        "input_token_loss_mask": torch.tensor([[0, 1, 1, 1]], dtype=torch.bool),
        "versions": torch.tensor([[-1, 44, 48, 49]], dtype=torch.int32),
        "advantages": torch.ones(shape),
        "kl_rewards": torch.zeros(shape),
        "tot_rewards": torch.zeros(shape),
        "rewards": torch.ones(1),
        "is_truncated": torch.tensor([False]),
    }
    tracker = DistributedStatsTracker()
    with (
        patch("areal.trainer.ppo.actor.stats_tracker", tracker),
        patch("areal.trainer.ppo.actor.stage_batch_for_engine"),
        patch(
            "areal.trainer.ppo.actor.split_training_batch_into_microbatches",
            return_value=[data],
        ),
    ):
        actor._ppo_update(data, max_head_offpolicyness=1)
    assert engine.train_batch.call_count == 1
    assert data["policy_version_mask"].tolist() == [[False, True, True, False]]
    assert data["input_token_loss_mask"].tolist() == [[False, True, True, True]]
    metrics = tracker.export()
    assert metrics["update/version_stats/sample_staleness_theta_max"] == 5
    assert (
        metrics["update/version_filtered/version_stats/sample_staleness_theta_max"] == 1
    )
    assert metrics["update/version_filtered_fraction"] == pytest.approx(1 / 3)
    packed = pack_tensor_dict(data)
    assert packed["policy_version_mask"].tolist() == [False, True, True, False]


@pytest.mark.parametrize("all_expired", [False, True])
def test_pure_distillation_masks_expired_tokens_with_original_denominator(all_expired):
    from areal.api.cli_args import MOPDLossConfig

    logits = torch.zeros((1, 3), requires_grad=True)
    mask = torch.tensor([[False, not all_expired, False]])
    data = {
        "loss_mask": torch.ones((1, 3), dtype=torch.bool),
        "policy_version_mask": mask,
        "logprobs": torch.zeros((1, 3)),
        "mopd_behavior_logprobs": torch.zeros((1, 3)),
        "mopd_teacher_logp_sum": torch.ones((1, 3)),
        "mopd_teacher_weight_sum": torch.ones((1, 3)),
    }
    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        loss = grpo_loss_fn(
            logprobs=logits,
            entropy=torch.zeros_like(logits),
            input_data=data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            mopd_loss_config=MOPDLossConfig(),
        )
    loss.backward()
    expected = torch.tensor([[0.0, 0.0 if all_expired else -1 / 3, 0.0]])
    torch.testing.assert_close(logits.grad, expected, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(loss)
