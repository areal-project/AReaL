from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import MOPDLossConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor, _infer_prompt_lens, grpo_loss_fn
from areal.trainer.ppo.critic import ppo_loss_fn
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils.functional.loss_aggregation import (
    make_policy_gradient_reduction,
    prepare_prompt_token_weights,
)
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


@pytest.mark.parametrize("mode", ["seq_mean", "prompt_mean", "constant"])
@pytest.mark.parametrize("teacher_field", ["teacher_logp", "mopd_teacher_logp_sum"])
def test_actor_rejects_non_token_distillation_before_batch_processing(
    mode, teacher_field
):
    actor = object.__new__(PPOActor)
    actor.config = PPOActorConfig(
        loss_aggregation=mode,
        loss_aggregation_divisor=4.0 if mode == "constant" else None,
    )
    actor._mopd_loss_config = None

    with pytest.raises(ValueError, match="Distillation is only supported"):
        actor._ppo_update({teacher_field: torch.zeros(1, 2)})


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


def _rolled_loss_mask(prompt_len: int, answer_len: int) -> torch.Tensor:
    """Build the loss_mask as _ppo_update sees it: rolled left by one."""
    mask = torch.tensor([[0] * prompt_len + [1] * answer_len], dtype=torch.float)
    return torch.roll(mask, shifts=-1, dims=-1)


def test_infer_prompt_lens_recovers_the_prompt_length():
    prompt_len, answer_len = 5, 4
    loss_mask = _rolled_loss_mask(prompt_len, answer_len)
    attention_mask = torch.ones(1, prompt_len + answer_len, dtype=torch.long)

    assert _infer_prompt_lens(attention_mask, loss_mask).tolist() == [prompt_len]


def test_infer_prompt_lens_matches_the_sum_formula_without_rejection():
    prompt_len, answer_len = 3, 7
    loss_mask = _rolled_loss_mask(prompt_len, answer_len)
    attention_mask = torch.ones(1, prompt_len + answer_len, dtype=torch.long)

    legacy = attention_mask.sum(-1) - loss_mask.sum(-1)

    assert _infer_prompt_lens(attention_mask, loss_mask).tolist() == legacy.tolist()


def test_infer_prompt_lens_handles_a_batch_of_mixed_prompt_lengths():
    specs = [(2, 6), (5, 3), (1, 7)]
    masks = torch.cat([_rolled_loss_mask(p, a) for p, a in specs])
    attention_mask = torch.ones(len(specs), 8, dtype=torch.long)

    got = _infer_prompt_lens(attention_mask, masks).tolist()

    assert got == [p for p, _ in specs]


def test_infer_prompt_lens_falls_back_to_seqlen_when_nothing_is_trained():
    loss_mask = torch.zeros(1, 9)
    attention_mask = torch.ones(1, 9, dtype=torch.long)

    assert _infer_prompt_lens(attention_mask, loss_mask).tolist() == [9]


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
@pytest.mark.parametrize("filtered", [False, True])
def test_actor_objective_metric_matches_weighted_loss_across_microbatches(
    mode, filtered
):
    reduction = make_policy_gradient_reduction(
        mode, divisor=4.0 if mode == "constant" else None
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    advantages = -torch.tensor([[1.0, 3.0, 0.0], [2.0, 4.0, 6.0], [10.0, 0.0, 0.0]])
    prompt_weights = prepare_prompt_token_weights(mask, [2, 1])
    # This fixed selection isolates aggregation from M2's own selection scope.
    retained = torch.tensor([[1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.bool)

    def evaluate(partitions):
        tracker = DistributedStatsTracker()
        losses, weights = [], []
        for rows in partitions:
            local_mask = mask[rows]
            logprobs = torch.zeros_like(advantages[rows])
            inputs = {
                "logprobs": logprobs,
                "prox_logp": logprobs,
                "advantages": advantages[rows],
                "loss_mask": local_mask,
                "prompt_token_weights": prompt_weights[rows],
            }
            with (
                patch("areal.trainer.ppo.actor.stats_tracker", tracker),
                patch(
                    "areal.trainer.ppo.actor._apply_m2po_masking",
                    return_value=retained[rows] if filtered else local_mask,
                ),
            ):
                losses.append(
                    grpo_loss_fn(
                        logprobs=logprobs,
                        entropy=torch.zeros_like(logprobs),
                        input_data=inputs,
                        eps_clip=0.2,
                        eps_clip_higher=None,
                        c_clip=None,
                        pg_reduction=reduction,
                        m2_threshold=0.1,
                    )
                )
            weights.append(
                reduction.normalizer(
                    local_mask, prompt_token_weights=prompt_weights[rows]
                )
            )
        expected = sum(loss * weight for loss, weight in zip(losses, weights)) / sum(
            weights
        )
        exported = tracker.export()
        assert exported["actor_loss/avg"] == pytest.approx(expected.item())
        assert "actor_loss_token_mean/avg" in exported
        return exported

    full = evaluate([slice(None)])
    split = evaluate([slice(0, 1), slice(1, 3)])
    assert split["actor_loss/avg"] == pytest.approx(full["actor_loss/avg"])
    if mode != "token_mean":
        assert full["actor_loss/avg"] != pytest.approx(
            full["actor_loss_token_mean/avg"]
        )


@pytest.mark.parametrize("rl_coefficient", [0.0, 0.5])
def test_actor_objective_metric_includes_mopd_composition(rl_coefficient):
    tracker = DistributedStatsTracker()
    logprobs = torch.tensor([[-0.2, -0.4]])
    inputs = {
        "logprobs": torch.zeros_like(logprobs),
        "prox_logp": torch.zeros_like(logprobs),
        "advantages": torch.ones_like(logprobs),
        "loss_mask": torch.ones_like(logprobs, dtype=torch.bool),
        "mopd_teacher_logp_sum": torch.full_like(logprobs, -1.0),
        "mopd_teacher_weight_sum": torch.ones_like(logprobs),
        "mopd_behavior_logprobs": torch.zeros_like(logprobs),
    }
    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=inputs,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            mopd_loss_config=MOPDLossConfig(rl_coefficient=rl_coefficient),
        )
    assert tracker.export()["actor_loss/avg"] == pytest.approx(loss.item())
