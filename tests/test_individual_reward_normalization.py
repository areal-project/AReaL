# SPDX-License-Identifier: Apache-2.0

"""Individual turn scores retain one explicit normalization reference per episode."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api import ModelResponse, RolloutWorkflow
from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.proxy import workflow as workflow_module
from areal.experimental.openai.proxy.server import (
    deserialize_interactions,
    serialize_interactions,
)
from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.ppo import actor as actor_module
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import (
    Normalization,
    TrajBatchMeta,
    concat_batch,
    normalize_rollout_rewards,
    split_batch,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["token-mean", "seq-mean", "traj-mean", "prompt-mean"])
@pytest.mark.parametrize("export_mode", ["inline", "online"])
@pytest.mark.parametrize(
    "explicit_reference",
    [
        pytest.param(None, id="default-reference"),
        pytest.param(2.0, id="explicit-reference"),
    ],
)
@pytest.mark.parametrize(
    "long_row",
    [
        pytest.param(None, id="no-penalty"),
        pytest.param(0, id="long-first"),
        pytest.param(1, id="long-last"),
    ],
)
async def test_discounted_individual_rows_keep_episode_reference(
    mode, export_mode, explicit_reference, long_row, monkeypatch
):
    """Exercise export, proxy transport, grouped markers, RPC and actor rewards."""
    device = "cpu"

    cache = InteractionCache()
    for index, key in enumerate(("first", "last")):
        length = 4 if long_row == index else 1
        interaction = InteractionWithTokenLogpReward(
            reward=0.0 if index == 0 else 1.0,
            output_message_list=[],
            model_response=ModelResponse(
                input_tokens=[1],
                output_tokens=[2] * length,
                output_logprobs=[0.0] * length,
                output_versions=[0] * length,
            ),
        )
        interaction.interaction_id = key
        if index == 0 and explicit_reference is not None:
            interaction.rollout_reward = explicit_reference
        cache[key] = interaction

    class ProxyClient:
        session_api_key = "test-session"
        session_id = "session-1"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def set_last_reward(self, reward):
            cache.set_last_reward(reward)

        async def export_interactions(self, *, discount, style):
            interactions = cache.export_interactions(
                style=style, reward_discount=discount
            )
            return deserialize_interactions(serialize_interactions(interactions))

    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda **kwargs: ProxyClient()
    )
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow = OpenAIProxyWorkflow(
        mode=export_mode,
        agent=SimpleNamespace(run=AsyncMock()),
        discount=0.9,
        proxy_gateway_addr="http://localhost",
    )
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow,
        "_run_agent",
        AsyncMock(
            return_value=1.0
            if export_mode == "inline"
            else SimpleNamespace(session_id="session-1")
        ),
    )
    context = workflow_context.get()
    workflow_context.set(workflow_context.WorkflowContext(task_id=1))
    try:
        exported = await workflow.arun_episode(engine=None, data={})
        assert [v.reward for v in exported.values()] == pytest.approx([0.9, 1.0])

        peer = InteractionWithTokenLogpReward(
            reward=3.0,
            model_response=ModelResponse(
                input_tokens=[1],
                output_tokens=[2],
                output_logprobs=[0.0],
                output_versions=[0],
            ),
        )

        class Episodes(RolloutWorkflow):
            async def arun_episode(self, engine, data):
                return (
                    exported
                    if workflow_context.get().sample_idx == 0
                    else {"peer": peer}
                )

        data = await GroupedRolloutWorkflow(Episodes(), 2, Mock()).arun_episode(
            None, {}
        )
    finally:
        workflow_context.set(context)

    data = deserialize_value(serialize_value(data))
    torch.testing.assert_close(
        data["begin_of_trajectory"],
        torch.tensor([1, 0, 1], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    data = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }
    norm = NormConfig(
        mean_level="group", std_level="group", group_size=2, std_unbiased=False
    )
    actor = PPOActor(
        PPOActorConfig(
            path="dummy",
            loss_aggregation=mode,
            reward_norm=norm,
            recompute_logprob=False,
            kl_ctl=0.0,
            overlong_reward_penalty=long_row is not None,
            overlong_tokens=2,
            overlong_penalty_factor=1.0,
            max_new_tokens=4,
        ),
        engine=None,
    )

    result = actor.compute_advantages([data])[0]

    reference = 1.0 if explicit_reference is None else explicit_reference
    references = torch.tensor([reference, 3.0], device=device)
    scores = torch.tensor([0.9, 1.0, 3.0], device=device)
    if long_row is not None:
        scores[long_row] -= 1.0
    expected = (scores - references.mean()) / (
        references.std(unbiased=False) + norm.eps
    )
    torch.testing.assert_close(
        result["tot_rewards"].sum(-1), expected, rtol=1e-6, atol=1e-6
    )
    active = result["loss_mask"].bool()
    torch.testing.assert_close(
        result["advantages"][active],
        expected[:, None].expand_as(result["advantages"])[active],
        rtol=1e-6,
        atol=1e-6,
    )


def _reward_batch():
    return {
        "input_ids": torch.ones(3, 3, dtype=torch.long),
        "attention_mask": torch.ones(3, 3, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1]] * 3, dtype=torch.bool),
        "logprobs": torch.zeros(3, 3),
        "rewards": torch.tensor([0.9, 1.0, 3.0]),
        "begin_of_trajectory": torch.tensor([1, 0, 1], dtype=torch.int32),
        "rollout_rewards": [1.0, 1.0, None],
    }


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean", "traj-mean", "prompt-mean"])
def test_rollout_references_stay_out_of_model_batches(mode, monkeypatch):
    """Reference metadata survives advantages but never reaches forward/train inputs."""
    calls = []

    def forward(input_, aggregate_fn):
        assert "rollout_rewards" not in input_
        calls.append("forward")
        return input_["logprobs"].clone()

    def train_batch(batch, loss_fn, loss_weight_fn):
        assert "rollout_rewards" not in batch
        calls.append("train")
        return {}

    engine = SimpleNamespace(
        eval=lambda: None,
        train=lambda: None,
        forward=forward,
        train_batch=train_batch,
        get_version=lambda: 0,
        data_parallel_group=None,
        device="cpu",
    )
    actor = PPOActor(
        PPOActorConfig(
            path="dummy",
            loss_aggregation=mode,
            reward_norm=NormConfig(
                mean_level="group", std_level="group", std_unbiased=False
            ),
            recompute_logprob=False,
            kl_ctl=0.0,
            ppo_n_minibatches=2,
        ),
        engine,
    )
    monkeypatch.setattr(actor_module, "stats_tracker", MagicMock())
    data = [_reward_batch()]

    actor.compute_logp(data)
    advantages = actor.compute_advantages(data)
    assert advantages[0]["rollout_rewards"] == [1.0, 1.0, None]
    actor.ppo_update(advantages)

    assert calls == ["forward", "train", "train"]
    assert data[0]["rollout_rewards"] == [1.0, 1.0, None]


@pytest.mark.parametrize("with_metadata", [False, True])
def test_missing_references_keep_legacy_unmarked_row_normalization(with_metadata):
    data = _reward_batch()
    data.pop("begin_of_trajectory")
    if with_metadata:
        data["rollout_rewards"] = [None] * 3
    else:
        data.pop("rollout_rewards")
    norm = NormConfig(mean_level="group", std_level="group", std_unbiased=False)
    actor = PPOActor(
        PPOActorConfig(
            path="dummy", reward_norm=norm, recompute_logprob=False, kl_ctl=0.0
        ),
        engine=None,
    )
    scores = data["rewards"].clone()

    result = actor.compute_advantages([data])[0]

    expected = (scores - scores.mean()) / (scores.std(unbiased=False) + norm.eps)
    torch.testing.assert_close(
        result["tot_rewards"].sum(-1), expected, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("bias,scale,clip", [(0.0, 2.0, 5.0), (2.0, 3.0, 10.0)])
def test_explicit_rollout_reference_shares_row_reward_transform(bias, scale, clip):
    data = _reward_batch()
    norm = NormConfig(mean_level="group", std_level="group", std_unbiased=False)
    actor = PPOActor(
        PPOActorConfig(
            path="dummy",
            reward_norm=norm,
            recompute_logprob=False,
            kl_ctl=0.0,
            reward_bias=bias,
            reward_scaling=scale,
            reward_clip=clip,
        ),
        engine=None,
    )
    scores = ((data["rewards"].clone() + bias) * scale).clamp(-clip, clip)
    references = ((torch.tensor([1.0, 3.0]) + bias) * scale).clamp(-clip, clip)

    result = actor.compute_advantages([data])[0]

    expected = (scores - references.mean()) / (
        references.std(unbiased=False) + norm.eps
    )
    torch.testing.assert_close(
        result["tot_rewards"].sum(-1), expected, rtol=1e-6, atol=1e-6
    )


def _distributed_explicit_reference_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        scores = torch.tensor([0.9, 1.0] if rank == 0 else [3.0])
        starts = torch.tensor([1, 0] if rank == 0 else [1])
        meta = TrajBatchMeta(
            1, [scores.numel()], [1], [[1.0, 1.0]] if rank == 0 else [None]
        )
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

            actual = normalize_rollout_rewards(
                scores,
                norm,
                meta,
                traj_starts=starts,
                reduce_group=dist.group.WORLD,
            )

            if mean_level == "batch":
                expected = torch.tensor([-1.1, -1.0] if rank == 0 else [1.0]) / (
                    1.0 + norm.eps
                )
            else:
                expected = torch.tensor([-0.1, 0.0] if rank == 0 else [0.0])
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

        # Actual actor dispatch: one rank uses the helper, its peer the legacy class.
        data = {
            "input_ids": torch.ones(scores.numel(), 3, dtype=torch.long),
            "attention_mask": torch.ones(scores.numel(), 3, dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1, 1]] * scores.numel(), dtype=torch.bool),
            "logprobs": torch.zeros(scores.numel(), 3),
            "rewards": scores,
            "begin_of_trajectory": starts,
        }
        if rank == 0:
            data["rollout_rewards"] = [1.0, 1.0]
        actor = PPOActor(
            PPOActorConfig(
                path="dummy",
                reward_norm=NormConfig(
                    mean_level="batch", std_level="batch", std_unbiased=False
                ),
                recompute_logprob=False,
                kl_ctl=0.0,
            ),
            engine=None,
        )

        actual = actor.compute_advantages([data])[0]["tot_rewards"].sum(-1)

        expected = torch.tensor([-1.1, -1.0] if rank == 0 else [1.0]) / (
            1.0 + actor.reward_norm.eps
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

        norm = Normalization(NormConfig(mean_level="batch", std_level="batch"))
        if rank == 0:
            # Discounted rows require a reference; synchronize this error to peers.
            meta.rollout_rewards = [[None, None]]
        with pytest.raises(RuntimeError, match="explicit rollout_reward"):
            normalize_rollout_rewards(
                scores,
                norm,
                meta,
                traj_starts=starts,
                reduce_group=dist.group.WORLD,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo"
)
def test_explicit_reference_batch_normalization_matches_dp_collectives(tmp_path):
    """Discounted row scores and one-rank reference errors agree across real DP."""
    mp.spawn(
        _distributed_explicit_reference_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_source", ["first", "second", "all-none", "absent"])
async def test_grouped_tensor_references_are_collected_outside_concat(reference_source):
    """Mixed reference columns do not change markers or mutate episode inputs."""
    episodes = [
        {
            "input_ids": torch.ones(rows, 3, dtype=torch.long),
            "attention_mask": torch.ones(rows, 3, dtype=torch.bool),
            "rewards": torch.ones(rows),
        }
        for rows in (2, 1)
    ]
    expected = None
    if reference_source == "first":
        episodes[0]["rollout_rewards"] = [1.0, 1.0]
        expected = [1.0, 1.0, None]
    elif reference_source == "second":
        episodes[1]["rollout_rewards"] = [3.0]
        expected = [None, None, 3.0]
    elif reference_source == "all-none":
        episodes[0]["rollout_rewards"] = [None, None]
        expected = [None, None, None]
    keys = [set(episode) for episode in episodes]

    class Episodes(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            return episodes[workflow_context.get().sample_idx]

    result = await GroupedRolloutWorkflow(Episodes(), 2, Mock()).arun_episode(None, {})

    torch.testing.assert_close(
        result["begin_of_trajectory"],
        torch.tensor([1, 0, 1], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    if expected is None:
        assert "rollout_rewards" not in result
    else:
        assert result["rollout_rewards"] == expected
    assert [set(episode) for episode in episodes] == keys


@pytest.mark.parametrize("second_has_metadata", [False, True])
def test_concat_split_restores_each_prompt_reference_column(second_has_metadata):
    first = _reward_batch()
    second = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "loss_mask": torch.ones(1, 4, dtype=torch.bool),
        "logprobs": torch.zeros(1, 4),
        "rewards": torch.tensor([5.0]),
        "begin_of_trajectory": torch.ones(1, dtype=torch.int32),
    }
    if second_has_metadata:
        second["rollout_rewards"] = [5.0]
    originals = [first, second]

    batched, meta = concat_batch(originals)
    assert "rollout_rewards" not in batched
    restored = split_batch(batched, meta)

    for original, result in zip(originals, restored, strict=True):
        assert set(result) == set(original)
        for key, value in original.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(result[key], value, rtol=0, atol=0)
            else:
                assert result[key] == value


@pytest.mark.parametrize("bias,scale,clip", [(0.0, 1.0, float("inf")), (1.0, 2.0, 9.0)])
def test_shaped_normalization_rewards_use_upstream_logical_unit_contract(
    bias, scale, clip
):
    """The override contains raw shaped references per trajectory, not per row."""
    data = _reward_batch()
    batch, meta = concat_batch([data])
    norm = Normalization(
        NormConfig(mean_level="group", std_level="group", std_unbiased=False)
    )
    shaped = torch.tensor([2.0, 4.0])

    actual = normalize_rollout_rewards(
        batch["rewards"],
        norm,
        meta,
        reward_bias=bias,
        reward_scaling=scale,
        reward_clip=clip,
        normalization_rewards=shaped,
        traj_starts=batch["begin_of_trajectory"],
    )

    scores = ((batch["rewards"] + bias) * scale).clamp(-clip, clip)
    references = ((shaped + bias) * scale).clamp(-clip, clip)
    expected = (scores - references.mean()) / (
        references.std(unbiased=False) + norm.eps
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "bias,scale,clip", [(0.0, 1.0, float("inf")), (2.0, 2.0, 11.0)]
)
def test_mixed_references_keep_existing_split_and_singleton_penalty_policy(
    bias, scale, clip
):
    """Explicit outcomes coexist with the existing split/singleton penalty references."""
    scores = torch.tensor([0.9, 1.0, 1.0, 2.0, 3.0])
    unpenalized = torch.tensor([0.9, 1.0, 2.0, 2.0, 4.0])
    starts = torch.tensor([1, 0, 1, 0, 1])
    meta = TrajBatchMeta(1, [5], [1], [[1.0, 1.0, None, None, None]])
    norm = Normalization(
        NormConfig(mean_level="group", std_level="group", std_unbiased=False)
    )

    actual = normalize_rollout_rewards(
        scores,
        norm,
        meta,
        reward_bias=bias,
        reward_scaling=scale,
        reward_clip=clip,
        unpenalized_rewards=unpenalized,
        traj_starts=starts,
    )

    # Explicit reference 1; split reference 2 before penalties; singleton score 3.
    references = ((torch.tensor([1.0, 2.0, 3.0]) + bias) * scale).clamp(-clip, clip)
    transformed = ((scores + bias) * scale).clamp(-clip, clip)
    expected = (transformed - references.mean()) / (
        references.std(unbiased=False) + norm.eps
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_shaped_override_ignores_unused_raw_reference_transform():
    """A shaped override replaces raw references before their affine transform."""
    data = _reward_batch()
    data["rollout_rewards"] = [1e300, 1e300, None]
    batch, meta = concat_batch([data])
    norm = Normalization(
        NormConfig(mean_level="group", std_level="group", std_unbiased=False)
    )

    actual = normalize_rollout_rewards(
        batch["rewards"],
        norm,
        meta,
        normalization_rewards=torch.tensor([2.0, 4.0]),
        traj_starts=batch["begin_of_trajectory"],
    )

    expected = (batch["rewards"] - 3.0) / (1.0 + norm.eps)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
