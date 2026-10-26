# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import Normalization


def _expected_normalization(groups, config):
    """Compute the configured moments from explicit logical prompt groups."""
    values = torch.cat(groups).double()
    populations = groups if config.mean_level == "group" else [values]
    means = []
    for population in populations:
        population = population.double()
        if config.mean_level is None:
            mean = torch.zeros_like(population)
        elif config.mean_leave1out:
            mean = (
                (population.sum() - population) / (population.numel() - 1)
                if population.numel() > 1
                else torch.zeros_like(population)
            )
        else:
            mean = population.mean().expand_as(population)
        means.append(mean)
    centered = values - torch.cat(means)
    if config.std_level is None:
        return centered.float()

    populations = (
        list(centered.split([g.numel() for g in groups]))
        if config.std_level == "group"
        else [centered]
    )
    normalized = []
    for population in populations:
        divisor = population.numel() - int(config.std_unbiased)
        std = (
            torch.linalg.vector_norm(population) / divisor**0.5
            if divisor > 0
            else population.new_tensor(1.0)
        )
        normalized.append(population / (std + config.eps))
    return torch.cat(normalized).float()


@pytest.mark.parametrize("group_counts", [[7, 8], [8, 7, 8], [8, 8, 7], [1, 8]])
@pytest.mark.parametrize("rows", ["single", "marked_single", "multiple"])
@pytest.mark.parametrize(
    "mean_level,std_level",
    [
        ("group", "group"),
        ("batch", "group"),
        ("group", "batch"),
        ("group", None),
        (None, "group"),
        ("batch", "batch"),
    ],
)
@pytest.mark.parametrize("leave_one_out,unbiased", [(False, False), (True, True)])
def test_reward_normalization_variable_groups_matches_logical_moments(
    group_counts, rows, mean_level, std_level, leave_one_out, unbiased
):
    """Short groups keep their boundaries and fan-out rows carry no extra weight."""
    config = NormConfig(
        mean_level=mean_level,
        std_level=std_level,
        group_size=8,
        mean_leave1out=leave_one_out,
        std_unbiased=unbiased,
    )
    groups = [
        torch.arange(count, dtype=torch.float32) + index * 10 + 1
        for index, count in enumerate(group_counts)
    ]
    repeats = (
        torch.arange(sum(group_counts)) % 3 + 1
        if rows == "multiple"
        else torch.ones(sum(group_counts), dtype=torch.long)
    )
    first_rows = repeats.cumsum(0) - repeats
    rewards = torch.cat(groups).repeat_interleave(repeats)
    starts = torch.zeros_like(rewards, dtype=torch.int32)
    starts[first_rows] = 1
    row_counts = [int(r.sum()) for r in repeats.split(group_counts)]

    actual = Normalization(config)(
        rewards,
        traj_starts=starts if rows != "single" else None,
        group_sizes=row_counts,
    )
    expected = _expected_normalization(groups, config).repeat_interleave(repeats)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("with_starts", [False, True])
@pytest.mark.parametrize("leave_one_out", [False, True])
def test_reward_normalization_without_group_metadata_keeps_partial_tail(
    with_starts, leave_one_out
):
    """The fixed-size fallback still normalizes an incomplete final group."""
    config = NormConfig(
        mean_level="group",
        std_level="group",
        group_size=8,
        mean_leave1out=leave_one_out,
    )
    groups = [torch.arange(8).float(), torch.arange(7).float() + 10]
    rewards = torch.cat(groups)

    actual = Normalization(config)(
        rewards, traj_starts=torch.ones_like(rewards) if with_starts else None
    )

    torch.testing.assert_close(
        actual, _expected_normalization(groups, config), rtol=1e-6, atol=1e-6
    )


def test_reward_normalization_explicit_groups_override_configured_singleton():
    """Explicit groups must also override the configured singleton fast paths."""
    config = NormConfig(
        mean_level="group",
        std_level="group",
        group_size=1,
        mean_leave1out=True,
        std_unbiased=True,
    )
    groups = [torch.tensor([1.0, 3.0]), torch.tensor([7.0, 11.0, 13.0])]

    actual = Normalization(config)(torch.cat(groups), group_sizes=[2, 3])

    torch.testing.assert_close(
        actual, _expected_normalization(groups, config), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("group_sizes", [[0, 3], [-1, 4], [1, 1]])
@pytest.mark.parametrize("with_starts", [False, True])
def test_reward_normalization_invalid_group_sizes_raises(group_sizes, with_starts):
    """Reject metadata that cannot describe the supplied rows."""
    norm = Normalization(NormConfig(mean_level="group", std_level="group"))

    with pytest.raises(ValueError, match="group_sizes"):
        norm(
            torch.arange(3, dtype=torch.float32),
            traj_starts=torch.ones(3) if with_starts else None,
            group_sizes=group_sizes,
        )


def test_reward_normalization_trajectory_crossing_group_boundary_raises():
    """A prompt boundary must not split a logical trajectory."""
    norm = Normalization(NormConfig(mean_level="group", std_level="group"))

    with pytest.raises(ValueError, match="first row of each group"):
        norm(
            torch.tensor([1.0, 1.0, 1.0, 3.0]),
            traj_starts=torch.tensor([1, 0, 0, 1]),
            group_sizes=[2, 2],
        )


@pytest.mark.parametrize("group_counts", [[7, 8, 8, 8], [8, 7, 8, 8]])
@pytest.mark.parametrize("multiple_rows", [False, True])
@pytest.mark.parametrize("partitions", [1, 2])
@pytest.mark.parametrize(
    "loss_aggregation", ["token-mean", "seq-mean", "traj-mean", "prompt-mean"]
)
def test_actor_advantages_short_group_preserves_prompt_boundaries(
    group_counts, multiple_rows, partitions, loss_aggregation
):
    """Each actor shard uses actual prompt groups and returns original row shapes."""
    config = PPOActorConfig(
        path="dummy",
        loss_aggregation=loss_aggregation,
        reward_norm=NormConfig(mean_level="group", std_level="group", group_size=8),
        recompute_logprob=False,
        kl_ctl=0.0,
    )
    actor = PPOActor(config, engine=None)
    batch = []
    for group_index, count in enumerate(group_counts):
        repeats = (
            torch.arange(count) % 3 + 1
            if multiple_rows
            else torch.ones(count, dtype=torch.long)
        )
        n_rows = int(repeats.sum())
        shape = (n_rows, group_index + 3)
        item = {
            "input_ids": torch.ones(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.bool),
            "loss_mask": torch.ones(shape, dtype=torch.bool),
            "logprobs": torch.zeros(shape),
            "rewards": torch.full((n_rows,), float(group_index)),
        }
        item["loss_mask"][:, 0] = False
        if multiple_rows:
            starts = torch.zeros(n_rows, dtype=torch.int32)
            starts[repeats.cumsum(0) - repeats] = 1
            item["begin_of_trajectory"] = starts
        batch.append(item)

    # Emulate whole-prompt DP dispatch. Group normalization needs no collective.
    result = []
    shard_size = len(batch) // partitions
    for start in range(0, len(batch), shard_size):
        result.extend(actor.compute_advantages(batch[start : start + shard_size]))

    assert len(result) == len(batch)
    for original, computed in zip(batch, result):
        for key in ("advantages", "tot_rewards"):
            torch.testing.assert_close(
                computed[key],
                torch.zeros_like(original["logprobs"]),
                rtol=0,
                atol=0,
            )
