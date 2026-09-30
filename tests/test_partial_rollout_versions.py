# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    NormConfig,
    PPOActorConfig,
)
from areal.api.io_struct import ModelRequest
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.trainer.ppo.actor import PPOActor


def trajectory():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1, 1]]),
        "versions": torch.tensor([[-1, -1, 0, 1, 2, 3]]),
        "logprobs": torch.zeros(1, 6),
        "rewards": torch.tensor([1.0]),
        "is_truncated": torch.tensor([False]),
    }


@pytest.mark.parametrize(
    "gap,expected",
    [(0, [0, 0, 0, 0, 1, 0]), (1, [0, 0, 0, 1, 1, 0]), (2, [0, 0, 1, 1, 1, 0])],
)
def test_stale_tokens_strict_threshold_before_gae(gap, expected):
    """Old prefix stays in context while only permitted actions get advantages."""
    engine = MagicMock()
    engine.get_version.return_value = 3
    actor = PPOActor(
        PPOActorConfig(
            mask_stale_tokens=True, max_token_version_gap=gap, recompute_logprob=False
        ),
        engine,
    )
    data = trajectory()
    context = data["input_ids"].clone()
    result = actor._compute_advantages(data)
    torch.testing.assert_close(
        result["loss_mask"],
        torch.tensor([expected], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result["advantages"], result["loss_mask"], rtol=0, atol=0
    )
    torch.testing.assert_close(result["input_ids"], context, rtol=0, atol=0)
    assert result["attention_mask"].all()


def test_disabled_mask_preserves_existing_behavior():
    """The opt-out path needs no version metadata and trains every action."""
    engine = MagicMock()
    actor = PPOActor(PPOActorConfig(recompute_logprob=False), engine)
    data = trajectory()
    del data["versions"]
    result = actor._compute_advantages(data)
    torch.testing.assert_close(
        result["loss_mask"],
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]),
        rtol=0,
        atol=0,
    )
    engine.get_version.assert_not_called()


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_threshold_raises(value):
    with pytest.raises(ValueError, match="max_token_version_gap"):
        PPOActorConfig(max_token_version_gap=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_abort_buffer_resumes_context_with_request_versions(monkeypatch, enabled):
    """Exercise actual agenerate through abort, buffer, and resumed completion."""
    backend = MagicMock()
    engine = RemoteInfEngine(
        InferenceEngineConfig(enable_partial_rollout=enabled), backend
    )
    engine.addresses = ["localhost:1234"]
    engine.workflow_executor = MagicMock()
    engine.workflow_executor.is_paused.return_value = False
    backend.parse_generation_response.side_effect = [
        SimpleNamespace(
            output_tokens=[3, 4],
            output_logprobs=[-0.1, -0.2],
            stop_reason="abort",
            routed_experts=None,
            spec_accept_rate=None,
            spec_accept_length=None,
        ),
        SimpleNamespace(
            output_tokens=[5],
            output_logprobs=[-0.3],
            stop_reason="stop",
            routed_experts=None,
            spec_accept_rate=None,
            spec_accept_length=None,
        ),
    ]
    calls = []

    def build(req, **kwargs):
        calls.append(
            (list(req.input_ids), req.gconfig.max_new_tokens, kwargs["version"])
        )
        return SimpleNamespace(endpoint="/generate", payload={}, method="POST")

    backend.build_generation_request.side_effect = build
    monkeypatch.setattr(
        "areal.infra.remote_inf_engine.workflow_context.get_aiohttp_session",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "areal.infra.remote_inf_engine.arequest_with_retry", AsyncMock(return_value={})
    )
    req = ModelRequest(
        input_ids=[1, 2],
        gconfig=GenerationHyperparameters(max_new_tokens=4, max_tokens=10),
    )
    task = asyncio.create_task(engine.agenerate(req))
    if enabled:
        for _ in range(100):
            if req.rid in engine.partial_rollout_buffer:
                break
            await asyncio.sleep(0.001)
        assert req.rid in engine.partial_rollout_buffer
        assert not task.done()
        assert engine.partial_rollout_buffer[req.rid]["generation_segments"] == [
            (0, 2, 0)
        ]
        engine.set_version(2)
    result = await asyncio.wait_for(task, timeout=2)
    version = 2 if enabled else 0
    assert calls == [([1, 2], 4, 0), ([1, 2, 3, 4], 2, version)]
    assert result.output_versions == [0, 0, version]
    assert result.generation_segments == [(0, 2, 0), (2, 3, version)]
    assert result.output_tokens == [3, 4, 5]
    assert engine.partial_rollout_buffer == {}
    assert req.input_ids == [1, 2]


def test_stale_actions_have_zero_ppo_gradient():
    """Masked actions must not contribute to the actual PPO objective."""
    from areal.utils.functional import ppo_actor_loss_fn

    engine = MagicMock()
    engine.get_version.return_value = 3
    actor = PPOActor(
        PPOActorConfig(mask_stale_tokens=True, recompute_logprob=False), engine
    )
    result = actor._compute_advantages(trajectory())
    logp = torch.zeros(1, 6, requires_grad=True)
    loss, _ = ppo_actor_loss_fn(
        logprobs=logp,
        proximal_logprobs=torch.zeros_like(logp),
        old_logprobs=result["logprobs"],
        advantages=result["advantages"],
        eps_clip=0.2,
        loss_mask=result["loss_mask"].bool(),
    )
    loss.backward()
    torch.testing.assert_close(
        logp.grad, torch.tensor([[0.0, 0.0, 0.0, -0.5, -0.5, 0.0]]), rtol=0, atol=0
    )


def test_all_stale_trajectory_has_zero_advantage():
    """An exhausted row retains context and contributes no actor/critic target."""
    engine = MagicMock()
    engine.get_version.return_value = 5
    actor = PPOActor(
        PPOActorConfig(mask_stale_tokens=True, recompute_logprob=False), engine
    )
    result = actor._compute_advantages(trajectory())
    for key in ("loss_mask", "advantages", "returns"):
        torch.testing.assert_close(
            result[key], torch.zeros_like(result[key]), rtol=0, atol=0
        )
    assert result["attention_mask"].all()


def test_mask_statistics_count_original_tokens_and_weighted_ratio(monkeypatch):
    """Aggregate token counts, not an unweighted mean of per-trajectory ratios."""
    from areal.utils.stats_tracker import DistributedStatsTracker

    tracker = DistributedStatsTracker()
    monkeypatch.setattr("areal.trainer.ppo.actor.stats_tracker.get", lambda: tracker)
    engine = MagicMock()
    engine.get_version.return_value = 3
    actor = PPOActor(PPOActorConfig(mask_stale_tokens=True), engine)
    actor._mask_stale_tokens(trajectory())
    second = trajectory()
    second["loss_mask"][:, 2:4] = 0
    actor._mask_stale_tokens(second)
    metrics = tracker.export()
    assert metrics["stale_generated_tokens"] == 6
    assert metrics["stale_masked_tokens"] == 2
    assert metrics["stale_partially_masked_trajectories"] == 1
    assert metrics["stale_masked_ratio"] == pytest.approx(2 / 6)


@pytest.mark.parametrize("critic", [False, True])
def test_globally_stale_minibatch_skips_optimizer(monkeypatch, critic):
    """All-stale actions cannot reach the engine's positive-weight assertion."""
    from areal.api.cli_args import PPOCriticConfig
    from areal.trainer.ppo.critic import PPOCritic

    engine = MagicMock()
    engine.get_version.return_value = 5
    actor = PPOActor(PPOActorConfig(mask_stale_tokens=True), engine)
    data = actor._compute_advantages(trajectory())
    target = PPOCritic(PPOCriticConfig(), engine) if critic else actor
    module = "critic" if critic else "actor"
    monkeypatch.setattr(
        f"areal.trainer.ppo.{module}.stage_batch_for_engine", lambda *_: None
    )
    monkeypatch.setattr(
        f"areal.trainer.ppo.{module}.split_training_batch_into_microbatches",
        lambda data, **_: [data],
    )
    target._ppo_update(data)
    engine.train_batch.assert_not_called()


def test_local_empty_rank_participates_when_peer_has_tokens(monkeypatch):
    """Every rank makes the same update decision; local emptiness is not enough."""
    from areal.trainer.ppo.masking import has_global_loss_tokens

    group = object()
    monkeypatch.setattr("areal.trainer.ppo.masking.dist.is_initialized", lambda: True)

    def reduce(present, *, op, group):
        present.fill_(1)

    collective = MagicMock(side_effect=reduce)
    monkeypatch.setattr("areal.trainer.ppo.masking.dist.all_reduce", collective)
    assert has_global_loss_tokens(torch.zeros(1, 4, dtype=torch.bool), group)
    assert collective.call_args.kwargs["group"] is group


def test_actor_retains_mixed_minibatch_and_skips_only_empty_one(monkeypatch):
    """Do not discard a usable trajectory or call an empty optimizer update."""
    engine = MagicMock()
    engine.get_version.return_value = 3
    engine.train_batch.return_value = {}
    actor = PPOActor(PPOActorConfig(mask_stale_tokens=True), engine)
    data = actor._compute_advantages(trajectory())
    empty = {**data, "loss_mask": torch.zeros_like(data["loss_mask"])}
    monkeypatch.setattr(
        "areal.trainer.ppo.actor.stage_batch_for_engine", lambda *_: None
    )
    monkeypatch.setattr(
        "areal.trainer.ppo.actor.split_training_batch_into_microbatches",
        lambda data, **_: [empty, data],
    )
    actor._ppo_update(data)
    engine.train_batch.assert_called_once()
    assert engine.train_batch.call_args.args[0] is data
    assert "stale_token_mask" in data


def _normalize_with_empty_rank(rank, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        engine = SimpleNamespace(
            get_version=lambda: 3, data_parallel_group=dist.group.WORLD
        )
        actor = PPOActor(
            PPOActorConfig(
                mask_stale_tokens=True,
                recompute_logprob=False,
                adv_norm=NormConfig(mean_level="batch", std_level="batch"),
            ),
            engine,
        )
        data = trajectory()
        if rank == 1:
            data["versions"][:, 2:] = 0
        result = actor._compute_advantages(data)
        assert torch.isfinite(result["advantages"]).all()
        torch.testing.assert_close(
            result["advantages"],
            torch.zeros_like(result["advantages"]),
            rtol=0,
            atol=0,
        )
        assert result["loss_mask"].sum() == (2 if rank == 0 else 0)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_advantage_normalization_empty_rank_joins_real_collectives(tmp_path):
    """One all-stale rank must not strand a peer in mean/std all-reduces."""
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_normalize_with_empty_rank, args=(rank, str(tmp_path / "init"))
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=40)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
