# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from tests.test_grouped_rollout_workflow import _interaction, _ListWorkflow, _Logger
from tests.test_token_rewards_advantage import _actor

from areal.api.cli_args import NormConfig
from areal.experimental.openai.proxy.server import (
    deserialize_interactions,
    serialize_interactions,
)
from areal.experimental.openai.types import concat_tensor_interactions
from areal.infra.dist_rollout import _pack_gathered_trajectories
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.workflow_executor import WorkflowContractError
from areal.trainer.ppo.actor import _group_training_metrics
from areal.utils.data import (
    Normalization,
    RolloutGroup,
    TrajBatchMeta,
    concat_batch,
    normalize_rollout_rewards,
    split_batch,
)


def _norm(**kwargs):
    return Normalization(
        NormConfig(
            mean_level=kwargs.pop("mean_level", "group"),
            std_level=kwargs.pop("std_level", "group"),
            std_unbiased=kwargs.pop("std_unbiased", False),
            eps=kwargs.pop("eps", 0.0),
            **kwargs,
        )
    )


def _meta(counts=(2, 1), rewards=()):
    return TrajBatchMeta(1, [sum(counts)], [3], [RolloutGroup(counts, rewards)])


@pytest.mark.parametrize("counts", [(1, 1), (3, 1), (1, 4)])
def test_split_count_does_not_reweight_rollout_reward_statistics(counts):
    values = torch.tensor([1.0] * counts[0] + [3.0] * counts[1])
    actual = normalize_rollout_rewards(values, _norm(), _meta(counts))
    expected = torch.tensor([-1.0] * counts[0] + [1.0] * counts[1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("leave_out", [False, True])
@pytest.mark.parametrize("unbiased", [False, True])
def test_differing_row_rewards_use_explicit_reference_and_existing_std(
    leave_out, unbiased
):
    # References [1, 3]; row 1's own reward is 2, not the reference 1.
    values = torch.tensor([1.0, 2.0, 3.0])
    norm = _norm(mean_leave1out=leave_out, std_unbiased=unbiased)
    actual = normalize_rollout_rewards(values, norm, _meta(rewards=(1.0, None)))
    center = torch.tensor([3.0, 3.0, 1.0]) if leave_out else torch.tensor([2.0] * 3)
    # AReaL computes variance around the selected baseline, including LOO.
    scale = (2.0 if leave_out else 1.0) * (2**0.5 if unbiased else 1.0)
    torch.testing.assert_close(actual, (values - center) / scale, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("mean_level", "std_level"),
    [("batch", "group"), ("group", "batch"), ("batch", "batch")],
)
def test_mixed_levels_count_logical_references(mean_level, std_level):
    refs = torch.tensor([1.0, 3.0, 6.0, 10.0])
    values = torch.tensor([1.0, 2.0, 3.0, 6.0, 10.0, 11.0])
    meta = TrajBatchMeta(
        2,
        [3, 3],
        [3, 3],
        [RolloutGroup((2, 1), (1.0, 3.0)), RolloutGroup((1, 2), (6.0, 10.0))],
    )
    mean = (
        refs.mean().expand(4)
        if mean_level == "batch"
        else torch.tensor([2.0, 2.0, 8.0, 8.0])
    )
    residual_sq = (refs - mean).square()
    scale = (
        residual_sq.mean().sqrt().expand(4)
        if std_level == "batch"
        else residual_sq.reshape(2, 2).mean(1).sqrt().repeat_interleave(2)
    )
    member = torch.tensor([0, 0, 1, 2, 3, 3])
    expected = (values - mean[member]) / scale[member]
    actual = normalize_rollout_rewards(
        values, _norm(mean_level=mean_level, std_level=std_level), meta
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_differing_row_rewards_require_explicit_reference():
    with pytest.raises(RuntimeError, match="explicit rollout_reward"):
        normalize_rollout_rewards(torch.tensor([1.0, 2.0, 3.0]), _norm(), _meta())


@pytest.mark.parametrize("std_level", ["group", "batch"])
@pytest.mark.parametrize("deviation", [0.0, 2**-10, 2**-9])
def test_degenerate_reference_scale_preserves_centered_rows(std_level, deviation):
    eps = 2**-10
    rows = torch.tensor([-1.0, -deviation, deviation])
    actual = normalize_rollout_rewards(
        rows,
        _norm(std_level=std_level, eps=eps),
        _meta(rewards=(-deviation, deviation)),
    )
    divisor = 1.0 if deviation <= eps else deviation + eps
    torch.testing.assert_close(actual, rows / divisor, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mean_level,std_level,divisor",
    [("group", "group", 1.0), ("group", "batch", 1.0), ("batch", "group", 2.0)],
)
def test_tied_group_references_use_configured_scale_scope(
    mean_level, std_level, divisor
):
    rows = torch.tensor([0.0, 1.0, 1.0, 4.0, 5.0, 5.0])
    meta = TrajBatchMeta(
        2,
        [3, 3],
        [3, 3],
        [RolloutGroup((2, 1), (1.0, 1.0)), RolloutGroup((2, 1), (5.0, 5.0))],
    )
    actual = normalize_rollout_rewards(
        rows, _norm(mean_level=mean_level, std_level=std_level, eps=0.0), meta
    )
    mean = 3.0 if mean_level == "batch" else torch.tensor([1.0] * 3 + [5.0] * 3)
    torch.testing.assert_close(actual, (rows - mean) / divisor, rtol=0, atol=0)


@pytest.mark.parametrize("deviation", [0.0, 1e-8, 2e-8])
def test_rollout_reference_scale_fallback_agrees_for_scalars_and_cached_rows(deviation):
    from areal.experimental.openai.types import normalize_logical_rollout_rewards

    first, last = _interaction(-1.0), _interaction(deviation)
    first.rollout_reward, last.rollout_reward = -deviation, deviation
    first._cache["rewards"] = torch.tensor([-1.0, -0.5])
    last._cache = None
    assert normalize_logical_rollout_rewards([{"first": first}, {"last": last}])
    divisor = 1.0 if deviation <= 1e-8 else deviation + 1e-8
    assert first.reward == pytest.approx(-1.0 / divisor)
    assert last.reward == pytest.approx(deviation / divisor)
    assert first.rollout_reward == pytest.approx(-deviation / divisor)
    assert last.rollout_reward == pytest.approx(deviation / divisor)
    torch.testing.assert_close(
        first._cache["rewards"],
        torch.tensor([-1.0, -0.5]) / divisor,
        rtol=1e-6,
        atol=0,
    )


def test_advantage_statistics_remain_masked_token_weighted():
    x = torch.tensor([[1.0, 2.0, 100.0], [3.0, 100.0, 100.0], [4.0, 5.0, 6.0]])
    mask = x < 100
    norm = _norm(mean_leave1out=True)
    actual = norm(x, mask, group_sizes=[3], group_member_counts=[2])
    baseline = (x[mask].sum() - x) / 5
    scale = ((x - baseline)[mask].square().mean()).sqrt()
    torch.testing.assert_close(
        actual, (x - baseline) * mask / scale, rtol=1e-6, atol=1e-6
    )


def test_split_singleton_uses_existing_logical_singleton_fallback():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actual = _norm(mean_leave1out=True, std_unbiased=True)(
        x, torch.ones_like(x), group_sizes=[2], group_member_counts=[1]
    )
    torch.testing.assert_close(actual, torch.zeros_like(x), rtol=0, atol=0)


@pytest.mark.asyncio
async def test_interactions_survive_wrapper_proxy_rpc_packing_and_batch_roundtrip():
    first, second, third = _interaction(1.0), _interaction(2.0), _interaction(3.0)
    second.rollout_reward = 1.0
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": first, "b": second}, None, {"c": third}]),
        group_size=3,
        min_usable_group_size=2,
        logger=_Logger(),
    )
    interactions = await workflow.arun_episode(None, {})
    interactions = deserialize_interactions(serialize_interactions(interactions))
    trajectory = concat_tensor_interactions(interactions)
    trajectory = deserialize_value(serialize_value(trajectory))
    packed = _pack_gathered_trajectories([[trajectory]], world_size=1, rank=0).data
    batch, meta = concat_batch(packed)
    assert "rollout_group" not in batch
    assert meta.logical_group_sizes == [2]
    assert meta.rollout_groups == [RolloutGroup((2, 1), (1.0, None))]
    restored = split_batch(batch, meta)[0]
    assert restored["rollout_group"] == RolloutGroup((2, 1), (1.0, None))
    torch.testing.assert_close(
        restored["rewards"], torch.tensor([1.0, 2.0, 3.0]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        split_batch(batch["rewards"], meta)[0], trajectory["rewards"], rtol=0, atol=0
    )
    starts, sizes, weights = _group_training_metrics(
        batch["loss_mask"], meta.traj_group_sizes, meta.logical_group_sizes
    )
    assert sizes[starts].tolist() == [2.0]
    assert weights[starts].tolist() == [3.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_rollout_time_normalization_preserves_row_rewards_and_cache(explicit):
    first, second, third = _interaction(1.0), _interaction(2.0), _interaction(3.0)
    if explicit:
        second.rollout_reward = 1.0
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": first, "b": second}, {"c": third}]),
        group_size=2,
        min_usable_group_size=2,
        reward_normalization=True,
        logger=_Logger(),
    )
    if not explicit:
        with pytest.raises(WorkflowContractError, match="explicit rollout_reward"):
            await workflow.arun_episode(None, {})
        return
    result = concat_tensor_interactions(await workflow.arun_episode(None, {}))
    torch.testing.assert_close(
        result["rewards"], torch.tensor([-1.0, 0.0, 1.0]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        result["original_rewards"], torch.tensor([1.0, 2.0, 3.0]), rtol=0, atol=0
    )
    assert result["rollout_group"].rewards == pytest.approx((-1.0, 1.0))


def test_actor_penalizes_rows_but_transforms_explicit_reference_without_length_penalty():
    actor = _actor(direct=False)
    actor.reward_norm = _norm()
    actor.reward_bias, actor.reward_scaling, actor.reward_clip = 1.0, 2.0, 8.0
    actor.config.overlong_reward_penalty = True
    actor.config.overlong_tokens = 2
    actor.config.overlong_penalty_factor = 1.0
    actor.config.max_new_tokens = 3
    data = {
        "input_ids": torch.ones(3, 4, dtype=torch.long),
        "attention_mask": torch.ones(3, 4, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]]),
        "logprobs": torch.zeros(3, 4),
        "rewards": torch.tensor([1.0, 2.0, 4.0]),
    }
    # Penalized rows [0, 2, 3.5] become [2, 6, 8]; refs [1, 4] become [4, 8].
    output = actor._compute_advantages(copy.deepcopy(data), _meta(rewards=(1.0, 4.0)))
    torch.testing.assert_close(
        output["tot_rewards"].sum(1), torch.tensor([-2.0, 0.0, 1.0]), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(
    "counts,rewards",
    [((0, 1), ()), ((True, 1), ()), ((1,), (float("nan"),)), ((1, 1), (1.0,))],
)
def test_rollout_group_validates_constructor_and_wire(counts, rewards):
    with pytest.raises(ValueError):
        RolloutGroup(counts, rewards)


def test_concat_rejects_wrong_row_count_and_retains_source_metadata():
    trajectory = _interaction(1.0).to_tensor_dict()
    trajectory["rollout_group"] = RolloutGroup((2,))
    with pytest.raises(ValueError, match="expected 1"):
        concat_batch([trajectory])
    assert trajectory["rollout_group"] == RolloutGroup((2,))


def test_disabled_reward_statistics_do_not_require_a_reference():
    rows = torch.tensor([1.0, 2.0, 3.0])
    actual = normalize_rollout_rewards(
        rows, _norm(mean_level=None, std_level=None), _meta()
    )
    torch.testing.assert_close(actual, rows, rtol=0, atol=0)


def test_overlong_penalty_cannot_hide_differing_raw_rewards():
    with pytest.raises(RuntimeError, match="explicit rollout_reward"):
        normalize_rollout_rewards(
            torch.tensor([0.0, 0.0, 3.0]),
            _norm(),
            _meta(),
            unpenalized_rewards=torch.tensor([1.0, 0.0, 3.0]),
        )


@pytest.mark.asyncio
async def test_cached_split_row_ambiguity_is_a_nonretryable_contract_error():
    first = _interaction(1.0)
    first._cache["rewards"] = torch.tensor([1.0, 2.0])
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": first}, {"b": _interaction(3.0)}]),
        group_size=2,
        reward_normalization=True,
        logger=_Logger(),
    )
    with pytest.raises(WorkflowContractError, match="explicit rollout_reward"):
        await workflow.arun_episode(None, {})


def test_concat_revalidates_rpc_restored_metadata():
    trajectory = _interaction(1.0).to_tensor_dict()
    wire = serialize_value(RolloutGroup((1,)))
    wire["data"]["row_counts"] = [0, 1]
    trajectory["rollout_group"] = deserialize_value(wire)
    with pytest.raises(ValueError, match="positive integers"):
        concat_batch([trajectory])


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.float64])
async def test_constant_nonbinary_reward_normalizes_to_zero_with_or_without_cache(
    cache_dtype,
):
    first, second = _interaction(0.1), _interaction(0.1)
    first._cache["rewards"] = torch.tensor([0.1], dtype=cache_dtype)
    second._cache = None
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": first}, {"b": second}]),
        group_size=2,
        reward_normalization=True,
        logger=_Logger(),
    )
    await workflow.arun_episode(None, {})
    assert first.reward == second.reward == 0.0
    assert first.rollout_reward == second.rollout_reward == 0.0
    torch.testing.assert_close(first._cache["rewards"], torch.zeros(1), rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("interaction_output", [False, True])
async def test_executor_single_rollout_tracks_rows_with_optional_reference(
    interaction_output,
):
    from tests.test_workflow_executor_filtering import _executor, _TrajectoryWorkflow

    from areal.infra.workflow_executor import _RolloutTaskInput

    executor, _ = _executor()
    executor.config.check_trajectory_format = True
    for task_id, reference in enumerate([1.0, None]):
        first, second = _interaction(1.0), _interaction(1.0)
        if interaction_output:
            first.rollout_reward = reference
            trajectory = {"a": first, "b": second}
        else:
            trajectory = concat_tensor_interactions({"a": first, "b": second})
            trajectory.pop("rollout_group")
            if reference is not None:
                trajectory["rollout_reward"] = reference
        result = await executor._create_workflow_task(
            _RolloutTaskInput(
                task_id=task_id,
                data={},
                workflow=_TrajectoryWorkflow(trajectory),
            )
        )()
        assert result is not None
        assert result.trajectory["rollout_group"] == RolloutGroup((2,), (reference,))
        assert "rollout_reward" not in result.trajectory
        assert result.trajectory["input_ids"].shape[0] == 2


def test_v2_terminal_reward_normalization_keeps_existing_contract():
    from areal.experimental.openai.types import normalize_group_rewards

    first, second, third = _interaction(0.5), _interaction(1.0), _interaction(3.0)
    assert normalize_group_rewards([{"a": first, "b": second}, {"c": third}])
    assert [first.reward, second.reward, third.reward] == [-1.0, -1.0, 1.0]
    assert all(v.rollout_reward is None for v in [first, second, third])


def test_integer_row_tensors_preserve_fractional_explicit_reference():
    actual = normalize_rollout_rewards(
        torch.tensor([1, 2, 3]), _norm(), _meta(rewards=(1.5, None))
    )
    torch.testing.assert_close(
        actual, torch.tensor([-5 / 3, -1 / 3, 1.0]), rtol=1e-6, atol=1e-6
    )
