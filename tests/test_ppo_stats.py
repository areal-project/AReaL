from unittest.mock import MagicMock, patch

import torch

from areal.api.cli_args import PPOActorConfig, RejectionSamplingConfig
from areal.trainer.ppo.actor import PPOActor, grpo_loss_fn
from areal.trainer.ppo.critic import ppo_loss_fn
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils.data import concat_batch
from areal.utils.stats_tracker import DistributedStatsTracker


def test_infer_token_denominator_prefers_attention_mask():
    input_data = {
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "input_ids": torch.tensor([[11, 12], [13, 14]]),
    }

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(5))

    assert n_tokens.shape == torch.Size([2, 3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_uses_input_ids_when_attention_mask_missing():
    input_data = {"input_ids": torch.tensor([[11, 12, 13], [14, 15, 16]])}

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(2, 3))

    assert n_tokens.shape == torch.Size([2, 3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_falls_back_for_padded_tree_input_ids():
    input_data = {"input_ids": torch.tensor([11, 12, 13, 0])}

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(3))

    assert n_tokens.shape == torch.Size([3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_falls_back_when_metadata_is_missing():
    fallback = torch.zeros(4)

    n_tokens = infer_token_denominator({"logprobs": torch.zeros(2)}, fallback=fallback)

    assert n_tokens.shape == torch.Size([4])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_grpo_loss_fn_uses_full_cu_seqlens_for_n_tokens():
    input_data = {
        "input_ids": torch.tensor([11, 12]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "logprobs": torch.zeros(2),
        "advantages": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
        "prox_logp": torch.zeros(2),
        "versions": torch.zeros(2, dtype=torch.int32),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()
        mock_tracker.scope = MagicMock()
        mock_tracker.scope.return_value.__enter__ = MagicMock()
        mock_tracker.scope.return_value.__exit__ = MagicMock()

        grpo_loss_fn(
            logprobs=torch.zeros(2),
            entropy=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )

    n_tokens = next(
        call.kwargs["n_tokens"]
        for call in mock_tracker.denominator.call_args_list
        if "n_tokens" in call.kwargs
    )
    assert n_tokens.shape == torch.Size([4])
    assert torch.all(n_tokens)


def test_critic_loss_fn_uses_full_cu_seqlens_for_n_tokens():
    input_data = {
        "input_ids": torch.tensor([11, 12]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "values": torch.zeros(2),
        "returns": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
    }

    with patch("areal.trainer.ppo.critic.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()

        ppo_loss_fn(
            value=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
        )

    n_tokens = mock_tracker.denominator.call_args.kwargs["n_tokens"]
    assert n_tokens.shape == torch.Size([4])
    assert torch.all(n_tokens)


def test_grpo_loss_fn_uses_packed_denominator_for_tree_vocab_stats():
    tracker = DistributedStatsTracker()
    input_data = {
        "input_ids": torch.tensor([11, 12, 13, 0]),
        "logprobs": torch.zeros(3),
        "advantages": torch.ones(3),
        "loss_mask": torch.ones(3, dtype=torch.bool),
        "prox_logp": torch.zeros(3),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        grpo_loss_fn(
            logprobs=torch.zeros(3),
            entropy=torch.zeros(3),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            vocab_min_logits=torch.zeros(3),
            vocab_max_logits=torch.zeros(3),
        )

    stats = tracker.export(reset=True)
    assert "n_tokens" in stats


def test_grpo_loss_fn_logs_rollout_probs_diff_with_original_valid_mask():
    """Measure behavior/proximal probability differences before rejection."""
    tracker = DistributedStatsTracker()
    behavior_probs = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    proximal_probs = torch.tensor([[0.5, 0.01, 0.25, 0.9]])
    original_mask = torch.tensor([[True, True, True, False]])
    input_data = {
        "logprobs": torch.log(behavior_probs),
        "prox_logp": torch.log(proximal_probs),
        "advantages": torch.ones_like(behavior_probs),
        "loss_mask": original_mask,
    }
    rejection_sampling = RejectionSamplingConfig(
        level="token",
        action="mask",
        metric="ratio",
        lower=0.1,
        upper=2.0,
    )

    with (
        patch("areal.trainer.ppo.actor.stats_tracker", tracker),
        tracker.scope("ppo_actor"),
        tracker.scope("update"),
    ):
        grpo_loss_fn(
            logprobs=torch.log(proximal_probs),
            entropy=torch.zeros_like(behavior_probs),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            rejection_sampling=rejection_sampling,
            m2_threshold=0.3,
            use_decoupled_loss=True,
        )

    stats = tracker.export(reset=True)
    # The second token is rejected by its 0.02 ratio but must remain in this
    # diagnostic; the fourth token is padding and must remain excluded.
    expected_diff = torch.tensor([0.0, 0.49, 0.25])
    torch.testing.assert_close(
        torch.tensor(stats["ppo_actor/update/rollout_probs_diff/avg"]),
        expected_diff.mean(),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(stats["ppo_actor/update/rollout_probs_diff/min"]),
        expected_diff.min(),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(stats["ppo_actor/update/rollout_probs_diff/max"]),
        expected_diff.max(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert stats["ppo_actor/update/rollout_probs_diff_tokens"] == 3.0


def test_compute_advantages_preserves_behavior_logp_before_standard_recompute():
    """Keep aligned rollout logprobs when standard PPO overwrites old logprobs."""
    actor = PPOActor(
        PPOActorConfig(
            backend="fsdp:d1",
            use_decoupled_loss=False,
            recompute_logprob=True,
            kl_ctl=0.0,
        ),
        MagicMock(),
    )
    behavior_probs = torch.tensor([[1.0, 0.5, 0.25, 1.0]])
    proximal_probs = torch.tensor([[0.6, 0.2, 0.9, 1.0]])
    data = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[True, True, True, False]]),
        "loss_mask": torch.tensor([[False, True, True, False]]),
        "logprobs": torch.log(behavior_probs),
        "prox_logp": torch.log(proximal_probs),
        "rewards": torch.zeros(1),
    }

    with patch("areal.trainer.ppo.stats.stats_tracker"):
        result = actor._compute_advantages(data, concat_batch([data])[1])

    expected_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    expected_behavior_logp = (
        torch.roll(torch.log(behavior_probs), shifts=-1, dims=-1) * expected_mask
    )
    expected_proximal_logp = torch.log(proximal_probs) * expected_mask
    torch.testing.assert_close(
        result["behavior_logp"], expected_behavior_logp, rtol=0, atol=0
    )
    torch.testing.assert_close(
        result["logprobs"], expected_proximal_logp, rtol=0, atol=0
    )


def test_grpo_loss_fn_logs_rollout_probs_diff_for_standard_recompute():
    """Measure on-policy train/inference differences after standard recompute."""
    tracker = DistributedStatsTracker()
    behavior_probs = torch.tensor([0.5, 0.25, 0.9])
    proximal_probs = torch.tensor([0.6, 0.2, 0.1])
    input_data = {
        "logprobs": torch.log(proximal_probs),
        "behavior_logp": torch.log(behavior_probs),
        "prox_logp": torch.log(proximal_probs),
        "advantages": torch.ones(3),
        "loss_mask": torch.tensor([True, True, False]),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        grpo_loss_fn(
            logprobs=torch.log(proximal_probs),
            entropy=torch.zeros(3),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            use_decoupled_loss=False,
        )

    stats = tracker.export(reset=True)
    expected_diff = torch.tensor([0.1, 0.05])
    torch.testing.assert_close(
        torch.tensor(stats["rollout_probs_diff/avg"]),
        expected_diff.mean(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_grpo_loss_fn_skips_rollout_probs_diff_when_standard_recompute_is_bypassed():
    """Do not report the diagnostic when standard PPO bypasses recomputation."""
    tracker = DistributedStatsTracker()
    input_data = {
        "logprobs": torch.zeros(2),
        "prox_logp": torch.zeros(2),
        "advantages": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        grpo_loss_fn(
            logprobs=torch.zeros(2),
            entropy=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            use_decoupled_loss=False,
        )

    stats = tracker.export(reset=True)
    assert not any("rollout_probs_diff" in key for key in stats)


def test_grpo_loss_fn_skips_rollout_probs_diff_without_recomputed_proximal():
    """Do not report an exact-policy diagnostic for a loglinear approximation."""
    tracker = DistributedStatsTracker()
    input_data = {
        "logprobs": torch.zeros(2),
        "prox_logp": None,
        "advantages": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
        "versions": torch.zeros(2, dtype=torch.int32),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        grpo_loss_fn(
            logprobs=torch.zeros(2),
            entropy=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            current_version=2,
            prox_logp_method="loglinear",
            use_decoupled_loss=True,
        )

    stats = tracker.export(reset=True)
    assert not any("rollout_probs_diff" in key for key in stats)


def test_grpo_loss_fn_logs_rollout_probs_diff_for_reused_proximal():
    """Use detached training-forward logprobs for the reuse diagnostic."""
    tracker = DistributedStatsTracker()
    behavior_probs = torch.tensor([0.5, 0.25, 0.9])
    current_probs = torch.tensor([0.6, 0.2, 0.1])
    input_data = {
        "logprobs": torch.log(behavior_probs),
        "advantages": torch.ones(3),
        "loss_mask": torch.tensor([True, True, False]),
    }

    with (
        patch("areal.trainer.ppo.actor.stats_tracker", tracker),
        patch("areal.trainer.ppo.stats.stats_tracker", tracker),
    ):
        grpo_loss_fn(
            logprobs=torch.log(current_probs),
            entropy=torch.zeros(3),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            prox_logp_method="reuse",
            use_decoupled_loss=True,
        )

    stats = tracker.export(reset=True)
    expected_diff = torch.tensor([0.1, 0.05])
    torch.testing.assert_close(
        torch.tensor(stats["rollout_probs_diff/avg"]),
        expected_diff.mean(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert "prox_logp" not in input_data
