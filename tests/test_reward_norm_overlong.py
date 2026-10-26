# SPDX-License-Identifier: Apache-2.0

"""Row-length penalties survive logical trajectory reward normalization."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import Normalization


def _batch(rewards, lengths, starts):
    width = max(lengths) + 1
    attention_mask = torch.zeros(len(lengths), width, dtype=torch.bool)
    loss_mask = torch.zeros_like(attention_mask)
    for row, length in enumerate(lengths):
        attention_mask[row, : length + 1] = True
        loss_mask[row, 1 : length + 1] = True
    return {
        "input_ids": torch.ones_like(loss_mask, dtype=torch.long),
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "logprobs": torch.zeros_like(loss_mask, dtype=torch.float32),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "begin_of_trajectory": torch.tensor(starts, dtype=torch.int32),
    }


def _actor(norm, mode="traj-mean", **kwargs):
    return PPOActor(
        PPOActorConfig(
            path="dummy",
            loss_aggregation=mode,
            reward_norm=norm,
            recompute_logprob=False,
            kl_ctl=0.0,
            overlong_reward_penalty=True,
            overlong_tokens=2,
            overlong_penalty_factor=1.0,
            max_new_tokens=4,
            **kwargs,
        ),
        engine=None,
    )


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean", "traj-mean", "prompt-mean"])
@pytest.mark.parametrize("long_row", [0, 1])
@pytest.mark.parametrize(
    "mean_level,std_level",
    [("group", None), ("group", "group"), ("batch", "group"), ("group", "batch")],
)
def test_actor_overlong_penalty_preserves_each_split_row(
    mode, long_row, mean_level, std_level
):
    """The same logical references apply regardless of which branch is long."""
    norm = NormConfig(
        mean_level=mean_level, std_level=std_level, group_size=2, std_unbiased=False
    )
    lengths = [1, 1, 1]
    lengths[long_row] = 4
    batch = _batch([1.0, 1.0, 0.0], lengths, [1, 0, 1])

    result = _actor(norm, mode).compute_advantages([batch])[0]

    # References are [1, 0], hence mean 0.5 and population std 0.5.
    expected = torch.tensor([0.5, 0.5, -0.5])
    expected[long_row] -= 1.0
    if std_level is not None:
        expected /= 0.5 + norm.eps
    torch.testing.assert_close(
        result["tot_rewards"].sum(-1), expected, rtol=1e-6, atol=1e-6
    )
    expected_advantages = expected[:, None].expand_as(result["advantages"])
    torch.testing.assert_close(
        result["advantages"][result["loss_mask"].bool()],
        expected_advantages[result["loss_mask"].bool()],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean", "traj-mean", "prompt-mean"])
def test_actor_overlong_reference_shares_reward_bias_scaling_and_clipping(mode):
    """Scores and logical references use identical reward transformations."""
    norm = NormConfig(
        mean_level="group", std_level="group", group_size=2, std_unbiased=False
    )
    batch = _batch([1.0, 1.0, -1.0], [1, 4, 1], [1, 0, 1])

    result = _actor(
        norm, mode, reward_bias=2.0, reward_scaling=3.0, reward_clip=7.0
    ).compute_advantages([batch])[0]

    # Scores [7, 6, 3], references [7, 3]: mean 5, population std 2.
    expected = torch.tensor([2.0, 1.0, -2.0]) / (2.0 + norm.eps)
    torch.testing.assert_close(
        result["tot_rewards"].sum(-1), expected, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("std_level", [None, "group"])
def test_actor_single_row_overlong_normalization_keeps_penalized_baseline(std_level):
    """Ordinary single-row GRPO still normalizes rewards after the penalty."""
    norm = NormConfig(mean_level="group", std_level=std_level, group_size=2)
    batch = _batch([1.0, 0.0], [4, 1], [1, 1])

    result = _actor(norm).compute_advantages([batch])[0]

    # Penalized rewards are [0, 0]; using raw references [1, 0] would be wrong.
    torch.testing.assert_close(
        result["tot_rewards"].sum(-1), torch.zeros(2), rtol=0, atol=0
    )


def test_actor_mixed_row_counts_preserves_singleton_penalized_reference():
    """A split outcome uses its shared reference; a singleton keeps its score."""
    norm = NormConfig(mean_level="group", std_level=None, group_size=2)
    batch = _batch([1.0, 1.0, 1.0], [1, 4, 4], [1, 0, 1])

    result = _actor(norm).compute_advantages([batch])[0]

    # References [1, 0] include the singleton's penalty, scores are [1, 0, 0].
    torch.testing.assert_close(
        result["tot_rewards"].sum(-1),
        torch.tensor([0.5, -0.5, -0.5]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("long_row", [0, 1])
def test_actor_constant_references_do_not_amplify_row_penalty_by_epsilon(long_row):
    """A zero reference std retains the penalty without division by epsilon."""
    norm = NormConfig(mean_level="group", std_level="group", group_size=2)
    lengths = [1, 1, 1, 1]
    lengths[long_row] = 4
    batch = _batch([1.0] * 4, lengths, [1, 0, 1, 0])

    result = _actor(norm).compute_advantages([batch])[0]

    expected = torch.zeros(4)
    expected[long_row] = -1.0
    torch.testing.assert_close(result["tot_rewards"].sum(-1), expected, rtol=0, atol=0)


@pytest.mark.parametrize("norm", [None, NormConfig(mean_level=None, std_level=None)])
def test_actor_identity_normalization_keeps_penalized_row_scores(norm):
    """An explicitly disabled normalization must not collapse row scores."""
    batch = _batch([1.0, 1.0, 0.0], [1, 4, 1], [1, 0, 1])

    result = _actor(norm).compute_advantages([batch])[0]

    torch.testing.assert_close(
        result["tot_rewards"].sum(-1), torch.tensor([1.0, 0.0, 0.0]), rtol=0, atol=0
    )


@pytest.mark.parametrize("supplied_reference", [False, True])
def test_split_row_rewards_require_a_shared_original_reference(supplied_reference):
    """Fail explicitly rather than select an arbitrary branch as the outcome."""
    norm = Normalization(NormConfig(mean_level="group", std_level=None))
    with pytest.raises(RuntimeError, match="row rewards differ"):
        norm(
            torch.tensor([1.0, 0.0]),
            traj_starts=torch.tensor([1, 0]),
            unpenalized_rewards=torch.tensor([1.0, 2.0])
            if supplied_reference
            else None,
        )


def test_singleton_tiny_variance_keeps_existing_scale():
    """The split-reference scale fallback does not change singleton normalization."""
    norm = Normalization(
        NormConfig(mean_level="group", std_level="group", group_size=2)
    )
    scores = torch.tensor([1.0, 1.0 + 1e-7])

    expected = norm(scores)
    actual = norm(
        scores,
        traj_starts=torch.tensor([1, 1]),
        unpenalized_rewards=torch.tensor([2.0, 2.0]),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("partitions", [1, 2])
def test_actor_overlong_with_short_prompt_groups_preserves_real_boundaries(partitions):
    """Actual group sizes and references survive whole-prompt DP dispatch."""
    norm = NormConfig(
        mean_level="group", std_level="group", group_size=8, std_unbiased=False
    )
    batches, expected = [], []
    for count in [7, 8, 8, 8]:
        references = torch.arange(count, dtype=torch.float32)
        scores = references.repeat_interleave(2) - torch.tensor([0.0, 1.0] * count)
        expected.append(
            (scores - references.mean()) / (references.std(unbiased=False) + norm.eps)
        )
        batches.append(
            _batch(
                references.repeat_interleave(2).tolist(), [1, 4] * count, [1, 0] * count
            )
        )
    actor = _actor(norm, "prompt-mean")

    result = []
    shard_size = len(batches) // partitions
    for start in range(0, len(batches), shard_size):
        result.extend(actor.compute_advantages(batches[start : start + shard_size]))

    for computed, target in zip(result, expected, strict=True):
        torch.testing.assert_close(
            computed["tot_rewards"].sum(-1), target, rtol=1e-6, atol=1e-6
        )


def _distributed_split_reference_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        scores = torch.tensor([0.0, 1.0] if rank == 0 else [0.0])
        original = torch.tensor([1.0, 1.0] if rank == 0 else [0.0])
        starts = torch.tensor([1, 0] if rank == 0 else [1])
        for mean_level, std_level in [
            ("batch", "batch"),
            ("batch", "group"),
            ("group", "batch"),
        ]:
            norm = Normalization(
                NormConfig(
                    mean_level=mean_level, std_level=std_level, std_unbiased=False
                )
            )
            actual = norm(
                scores,
                traj_starts=starts,
                group_sizes=[scores.numel()],
                unpenalized_rewards=original,
                reduce_group=dist.group.WORLD,
            )
            if mean_level == "batch":
                expected = torch.tensor([-0.5, 0.5] if rank == 0 else [-0.5]) / (
                    0.5 + norm.eps
                )
            else:
                expected = torch.tensor([-1.0, 0.0] if rank == 0 else [0.0])
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

        # One malformed rank must not leave its peer in the mean/std collectives.
        norm = Normalization(NormConfig(mean_level="batch", std_level="batch"))
        invalid_original = torch.tensor([1.0, 2.0] if rank == 0 else [0.0])
        with pytest.raises(RuntimeError, match="row rewards differ"):
            norm(
                scores,
                traj_starts=starts,
                unpenalized_rewards=invalid_original,
                reduce_group=dist.group.WORLD,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo"
)
def test_batch_normalization_split_references_match_real_dp_collectives(tmp_path):
    """Ragged logical units and invalid-reference checks agree across two ranks."""
    mp.spawn(
        _distributed_split_reference_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )
