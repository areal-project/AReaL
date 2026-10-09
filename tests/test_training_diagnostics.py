import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor, _log_version_staleness_stats
from areal.trainer.ppo.stats import log_train_inference_stats
from areal.utils.stats_tracker import DistributedStatsTracker


def test_staleness_unequal_batches_uses_token_weights_and_global_extrema():
    """Unequal microbatches must not turn max/min into averages of extrema."""
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        _log_version_staleness_stats(
            torch.tensor([0, -1]), 5, torch.tensor([True, True])
        )
        _log_version_staleness_stats(
            torch.tensor([4, 4, 4, 0]), 5, torch.tensor([True, True, True, False])
        )
    result = tracker.export()
    prefix = "version_stats/"
    assert result[prefix + "sample_staleness_theta_avg"] == 2
    assert result[prefix + "sample_staleness_theta_max"] == 5
    assert result[prefix + "sample_staleness_theta_min"] == 1
    assert result[prefix + "sample_staleness_proximal_avg"] == 2
    assert result[prefix + "n_valid_generated_tokens"] == 4
    assert result[prefix + "stale_token_fraction"] == 1


def test_staleness_no_generated_tokens_has_no_fabricated_average():
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        _log_version_staleness_stats(
            torch.tensor([-1, 0]), 5, torch.tensor([True, False])
        )
    result = tracker.export()
    assert result["version_stats/n_valid_generated_tokens"] == 0
    assert "version_stats/sample_staleness_theta_avg" not in result


def test_train_infer_masks_nonfinite_padding_and_weights_unequal_batches():
    """Masked padding must not leak into moments, KL, or ratio tails."""
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.ppo.stats.stats_tracker", tracker):
        log_train_inference_stats(
            torch.tensor([-1.0, float("nan")]),
            torch.tensor([-3.0, float("inf")]),
            torch.tensor([True, False]),
        )
        log_train_inference_stats(
            torch.full((3,), -1.0), torch.full((3,), -1.0), torch.ones(3).bool()
        )
    result = tracker.export()
    prefix = "ppo_actor/train_infer/"
    assert result[prefix + "logp_diff/avg"] == 0.5
    assert result[prefix + "logp_diff_squared"] == 1
    assert result[prefix + "logp_abs_diff/max"] == 2
    assert result[prefix + "ratio_outside_2"] == 0.25
    assert result[prefix + "ratio_outside_10"] == 0
    assert result[prefix + "trainer_nll"] == 1
    assert result[prefix + "rollout_nll"] == 1.5
    assert result[prefix + "kl_k1"] == -0.5
    assert result[prefix + "kl_k2"] == 0.5
    assert result[prefix + "kl_k3"] == pytest.approx(
        (torch.exp(torch.tensor(2.0)).item() - 3) / 4
    )
    assert result[prefix + "nonfinite_logp_fraction"] == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("trainer_invalid", [True, False])
def test_invalid_valid_logprob_does_not_report_healthy_ratio_tails(
    invalid, trainer_invalid
):
    tracker = DistributedStatsTracker()
    trainer = torch.tensor([invalid if trainer_invalid else -1.0, -1.0])
    rollout = torch.tensor([-1.0 if trainer_invalid else invalid, -1.0])
    with patch("areal.trainer.ppo.stats.stats_tracker", tracker):
        log_train_inference_stats(trainer, rollout, torch.ones(2).bool())
    stats = tracker.export()
    assert stats["ppo_actor/train_infer/nonfinite_logp_fraction"] == 0.5
    for threshold in (1.5, 2, 3, 5, 10):
        assert math.isnan(stats[f"ppo_actor/train_infer/ratio_outside_{threshold}"])


@pytest.mark.parametrize("decoupled", [False, True])
def test_advantages_logs_original_rollout_before_recompute_overwrite(decoupled):
    """The same raw mismatch is reported with standard and decoupled PPO."""
    actor = PPOActor(
        PPOActorConfig(
            backend="fsdp:d1",
            recompute_logprob=True,
            use_decoupled_loss=decoupled,
            kl_ctl=0.0,
        ),
        MagicMock(),
    )
    data = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.bool),
        "logprobs": torch.tensor([[0.0, 0.0, -3.0, -4.0]]),
        "prox_logp": torch.tensor([[0.0, -2.0, -2.0, 0.0]]),
        "rewards": torch.tensor([1.0]),
        "is_truncated": torch.tensor([False]),
    }
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.ppo.stats.stats_tracker", tracker):
        actor._compute_advantages(data)
    result = tracker.export()
    assert result["ppo_actor/train_infer/logp_diff/avg"] == 1.5
    assert result["ppo_actor/train_infer/n_valid_tokens"] == 2


def test_ppo_update_advantage_sign_fractions_ignore_padding():
    engine = MagicMock()
    engine.train_batch.return_value = {}
    actor = PPOActor(PPOActorConfig(backend="fsdp:d1"), engine)
    data = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        "advantages": torch.tensor([[2.0, -1.0, 0.0, 9.0]]),
        "kl_rewards": torch.zeros((1, 4)),
        "tot_rewards": torch.zeros((1, 4)),
        "rewards": torch.tensor([1.0]),
        "is_truncated": torch.tensor([False]),
    }
    tracker = DistributedStatsTracker()
    with (
        patch("areal.trainer.ppo.actor.stats_tracker", tracker),
        patch("areal.trainer.ppo.actor.stage_batch_for_engine"),
        patch(
            "areal.trainer.ppo.actor.split_training_batch_into_microbatches",
            return_value=[],
        ),
    ):
        actor._ppo_update(data)
    result = tracker.export()
    for sign in ("positive", "negative", "zero"):
        assert result[f"advantage_{sign}_fraction"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "versions,mask,expected_age,expected_stale,expected_future,count",
    [
        ([-1, -1, 4, 5], [0, 1, 1, 0], 0.5, 0.5, 0.0, 2),
        ([-1, -1, 5], [0, 1, 0], 0.0, 0.0, 0.0, 1),
        ([-1, 4, -1, -1, 5, 6], [1, 0, 0, 1, 1, 0], 0.0, 1 / 3, 1 / 3, 3),
    ],
)
def test_ppo_update_aligns_versions_before_counting_checkpoint_age(
    versions, mask, expected_age, expected_stale, expected_future, count
):
    """Include every turn's first token and use the last published version."""
    engine = MagicMock()
    engine.get_version.return_value = 5
    engine.train_batch.return_value = {}
    actor = PPOActor(PPOActorConfig(backend="fsdp:d1"), engine)
    raw_versions = torch.tensor([versions], dtype=torch.int32)
    shape = raw_versions.shape
    data = {
        "input_ids": torch.ones(shape, dtype=torch.long),
        "attention_mask": torch.ones(shape, dtype=torch.bool),
        "loss_mask": torch.tensor([mask], dtype=torch.bool),
        "versions": raw_versions.clone(),
        "advantages": torch.zeros(shape),
        "kl_rewards": torch.zeros(shape),
        "tot_rewards": torch.zeros(shape),
        "rewards": torch.tensor([1.0]),
        "is_truncated": torch.tensor([False]),
    }
    tracker = DistributedStatsTracker()
    with (
        patch("areal.trainer.ppo.actor.stats_tracker", tracker),
        patch("areal.trainer.ppo.actor.stage_batch_for_engine"),
        patch(
            "areal.trainer.ppo.actor.split_training_batch_into_microbatches",
            return_value=[data, data],
        ),
    ):
        actor._ppo_update(data)
    result = tracker.export()
    prefix = "update/version_stats/"
    assert result[prefix + "sample_staleness_proximal_avg"] == pytest.approx(
        expected_age
    )
    assert result[prefix + "sample_staleness_theta_avg"] == pytest.approx(expected_age)
    assert result[prefix + "stale_token_fraction"] == pytest.approx(expected_stale)
    assert result[prefix + "future_token_fraction"] == pytest.approx(expected_future)
    assert result[prefix + "n_valid_generated_tokens"] == count
    assert engine.train_batch.call_count == 2
    assert all(
        value.numel() <= 4
        for key, values in tracker.stats.items()
        if "version_stats" in key
        for value in values
    )
    torch.testing.assert_close(
        raw_versions,
        torch.tensor([versions], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_pure_mopd_uses_raw_versions_for_checkpoint_age_diagnostics():
    """The pure-MOPD preprocessing route skips _compute_advantages."""
    engine = MagicMock()
    engine.get_version.return_value = 5
    actor = PPOActor(PPOActorConfig(backend="fsdp:d1"), engine)
    actor._mopd_loss_config = SimpleNamespace(rl_coefficient=0)
    batch = {
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1]], dtype=torch.bool),
        "versions": torch.tensor([[-1, -1, 4]], dtype=torch.int32),
        "logprobs": torch.zeros((1, 3)),
        "rewards": torch.ones(1),
        "mopd_teacher_logp_sum": torch.zeros((1, 3)),
        "is_truncated": torch.tensor([False]),
    }
    raw_versions = batch["versions"].clone()
    batch = actor._prepare_mopd_batch(batch)
    assert batch["versions"].equal(raw_versions)
    tracker = DistributedStatsTracker()
    with (
        patch("areal.trainer.ppo.actor.stats_tracker", tracker),
        patch("areal.trainer.ppo.actor.stage_batch_for_engine"),
        patch(
            "areal.trainer.ppo.actor.split_training_batch_into_microbatches",
            return_value=[],
        ),
    ):
        actor._ppo_update(batch)
    result = tracker.export()
    assert result["update/version_stats/n_valid_generated_tokens"] == 1
    assert result["update/version_stats/sample_staleness_theta_avg"] == 1


def test_kl_estimators_match_rollout_to_trainer_direction():
    """An exact finite distribution checks KL direction, not just the formula."""
    q = torch.tensor([0.75, 0.25], dtype=torch.float64)
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    # Three occurrences of token 0 and one of token 1 exactly represent q.
    tokens = torch.tensor([0, 0, 0, 1])
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.ppo.stats.stats_tracker", tracker):
        log_train_inference_stats(
            p[tokens].log(), q[tokens].log(), torch.ones(4).bool()
        )
    result = tracker.export()
    expected = (q * (q.log() - p.log())).sum().item()
    reverse = (p * (p.log() - q.log())).sum().item()
    assert result["ppo_actor/train_infer/kl_k1"] == pytest.approx(expected)
    assert result["ppo_actor/train_infer/kl_k3"] == pytest.approx(expected)
    assert result["ppo_actor/train_infer/kl_k3"] != pytest.approx(reverse)


def test_train_infer_rejects_broadcastable_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        log_train_inference_stats(
            torch.zeros(2, 3), torch.zeros(3), torch.ones(2, 3).bool()
        )


def test_large_finite_logprob_gap_keeps_k3_finite():
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.ppo.stats.stats_tracker", tracker):
        log_train_inference_stats(
            torch.tensor([-1.0]), torch.tensor([-101.0]), torch.tensor([True])
        )
    result = tracker.export()
    assert result["ppo_actor/train_infer/nonfinite_logp_fraction"] == 0
    assert result["ppo_actor/train_infer/kl_k3_overflow_fraction"] == 0
    assert result["ppo_actor/train_infer/kl_k3"] == pytest.approx(math.expm1(100) - 100)


def test_train_infer_keeps_only_compact_summaries_for_long_sequences():
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.ppo.stats.stats_tracker", tracker):
        log_train_inference_stats(
            torch.full((20000,), -1.0),
            torch.full((20000,), -2.0),
            torch.ones(20000, dtype=torch.bool),
        )
    assert all(
        value.numel() <= 4 for values in tracker.stats.values() for value in values
    )
    assert tracker.export()["ppo_actor/train_infer/n_valid_tokens"] == 20000


def test_ratio_tail_is_symmetric_and_decreases_with_threshold():
    trainer = torch.tensor([-5.0, -4.0, -6.0, -2.0, -8.0])
    rollout = torch.full_like(trainer, -5.0)
    results = []
    for p, q in ((trainer, rollout), (rollout, trainer)):
        tracker = DistributedStatsTracker()
        with patch("areal.trainer.ppo.stats.stats_tracker", tracker):
            log_train_inference_stats(p, q, torch.ones(5).bool())
        stats = tracker.export()
        tails = [
            stats[f"ppo_actor/train_infer/ratio_outside_{tau}"]
            for tau in (1.5, 2, 3, 5, 10)
        ]
        assert tails == sorted(tails, reverse=True)
        assert tails == pytest.approx([0.8, 0.8, 0.4, 0.4, 0.4])
        results.append(tails)
    assert results[0] == results[1]
