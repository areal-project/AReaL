"""CPU regressions for configured PRM scoring through the v2 export endpoint."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx
import pytest
import pytest_asyncio
import torch
from pydantic import TypeAdapter

from areal.api import ModelResponse
from areal.api.cli_args import PRMConfig, PRMScorerConfig
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.proxy.proxy_rollout_server import _score_prm_branches
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra import workflow_context
from areal.infra.rpc.serialization import deserialize_value
from areal.reward.prm import (
    BaseScorer,
    BaseTrajectoryScorer,
    PRMMetricObservation,
    PRMRunner,
    PRMScorerResult,
)
from areal.reward.prm.export import PRMExportStats
from areal.utils import stats_tracker
from areal.v2.inference_service.controller.workflow import InferenceServiceWorkflow
from areal.v2.inference_service.data_proxy.app import create_app
from areal.v2.inference_service.data_proxy.config import DataProxyConfig

_HEADERS = {"Authorization": "Bearer areal-admin-key"}


@pytest.fixture(autouse=True)
def clear_stats():
    """Keep process-local statistics and workflow context isolated between cases."""
    stats_tracker.export_all(reduce_group=None)
    previous = workflow_context.get()
    yield
    stats_tracker.export_all(reduce_group=None)
    workflow_context.set(previous)


@pytest_asyncio.fixture
async def proxy_factory(monkeypatch):
    """Use real session/scoring/export code, without a model or remote tensor host."""
    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app._remotize_trajectory",
        lambda traj, node_addr: traj,
    )

    async with AsyncExitStack() as lifespans:

        async def make(scorers, *, enabled=True):
            app = create_app(
                DataProxyConfig(
                    backend_addr="",
                    chat_template_type="concat",
                    prm=PRMConfig(enabled=enabled, scorers=scorers),
                )
            )
            await lifespans.enter_async_context(app.router.lifespan_context(app))
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            )
            return client, app.state.session_store

        yield make


def _interaction(name, *, parent=None, branch="a", output_len=2):
    input_tokens = [1, 2]
    messages = [{"role": "user", "content": "task"}]
    if parent is not None:
        input_tokens = (
            parent.model_response.input_tokens
            + parent.model_response.output_tokens
            + [9]
        )
        messages = (
            parent.messages
            + parent.output_message_list
            + [{"role": "tool", "content": branch}]
        )
    result = InteractionWithTokenLogpReward(
        model_response=ModelResponse(
            input_tokens=input_tokens,
            output_tokens=[3] * output_len,
            output_logprobs=[-0.5] * output_len,
            output_versions=[2] * output_len,
        ),
        chat_template_type="concat",
        parent=parent,
        reward=1.0,
        messages=messages,
        output_message_list=[{"role": "assistant", "content": name}],
    )
    result._interaction_id = name
    return result


def _add_session(store, task, interactions):
    sid, _ = store.start_session(task)
    session = store.get_session(sid)
    session._active_completions = InteractionCache.from_dict(
        {x.interaction_id: x for x in interactions}, session_id=sid
    )
    session.set_reward(interactions[-1].interaction_id, interactions[-1].reward)
    return sid


async def _export(client, session_ids, **kwargs):
    return await client.post(
        "/export_trajectories",
        headers=_HEADERS,
        json={"session_ids": session_ids, "style": "concat", **kwargs},
    )


class _BranchScorer(BaseScorer):
    name = "branch"

    def __init__(self, *, dense=False, **kwargs):
        super().__init__(**kwargs)
        self.dense = dense

    async def evaluate(self, interaction, ctx):
        branch = next(m["content"] for m in ctx["messages"] if m["role"] == "tool")
        value = 1.0 if branch == "a" else 2.0
        if interaction.parent is not None:
            value *= 10
        if self.dense:
            return torch.arange(1, interaction.model_response.output_len + 1) * value
        return value

    def prepare_result(self, interaction, result, ctx):
        return PRMScorerResult(
            result,
            (
                PRMMetricObservation(
                    metric_id="has/parent",
                    scope="turn",
                    target_id=interaction.interaction_id,
                    value=interaction.parent is not None,
                    value_type="boolean",
                    aggregations=("count", "rate"),
                ),
            ),
        )


class _TrajectoryScorer(BaseTrajectoryScorer):
    name = "trajectory"

    async def evaluate_trajectory(self, interactions, ctx):
        return {x.interaction_id: float(i) for i, x in enumerate(interactions)}


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize("dense", [False, True])
async def test_v2_export_matches_v1_branch_tensors_and_json_metrics(
    proxy_factory, dense, is_eval
):
    """Shared ancestors get branch-local scores; transport preserves v1 contracts."""
    root = _interaction("root")
    a = _interaction("a", parent=root, branch="a")
    b = _interaction("b", parent=root, branch="b", output_len=3)
    scorers = [_BranchScorer(dense=dense, weight=0.5), _TrajectoryScorer()]
    v1 = await _score_prm_branches(
        {"a": a, "b": b},
        PRMRunner(PRMConfig(scorers=scorers)),
        session_id="v1",
        is_eval=is_eval,
    )
    expected_metrics = stats_tracker.export_all(reduce_group=None)
    client, store = await proxy_factory(scorers)
    async with client:
        sid = _add_session(store, "v2", [root, a, b])
        cached_root = root.to_tensor_dict()
        response = await _export(client, [sid], is_eval=is_eval)
    assert response.status_code == 200
    data = response.json()
    actual = deserialize_value(data["traj"])
    for index, leaf in enumerate(v1.values()):
        expected = leaf.to_tensor_dict()
        length = expected["input_ids"].shape[-1]
        for key in ("input_ids", "loss_mask", "logprobs", "versions", "token_rewards"):
            torch.testing.assert_close(
                actual[key][index, :length], expected[key].squeeze(0), rtol=0, atol=0
            )
        assert not actual["token_rewards"][index, length:].any()
    assert root._cache is cached_root
    assert all(x.token_rewards is None for x in (root, a, b))
    assert not stats_tracker.export_all(reduce_group=None)

    # Exercise the real workflow consumer, including pydantic JSON reconstruction.
    workflow_context.set(workflow_context.WorkflowContext(is_eval=is_eval))
    workflow = InferenceServiceWorkflow(controller=MagicMock())
    workflow._request_export = AsyncMock(return_value=data)
    await workflow._export_interactions(MagicMock(), [sid])
    assert workflow._request_export.call_args.args[1]["is_eval"] is is_eval
    assert stats_tracker.export_all(reduce_group=None) == pytest.approx(
        expected_metrics
    )
    assert store.session_count == 0


class _RewardScorer(BaseScorer):
    name = "outcome"

    def __init__(self):
        super().__init__()
        self.seen = []

    async def evaluate(self, interaction, ctx):
        self.seen.append((interaction.reward, ctx["is_eval"]))
        return interaction.reward


@pytest.mark.asyncio
async def test_scoring_precedes_outcome_normalization(proxy_factory):
    """Normalize outcomes as before, without normalizing the process signals."""
    scorer = _RewardScorer()
    client, store = await proxy_factory([scorer])
    async with client:
        sessions = []
        for reward in (1.0, 3.0):
            interaction = _interaction(str(reward))
            interaction.reward = reward
            sessions.append(_add_session(store, "group", [interaction]))
        response = await _export(
            client, sessions, reward_normalization=True, is_eval=True
        )
    assert response.status_code == 200
    assert scorer.seen == [(1.0, True), (3.0, True)]
    traj = deserialize_value(response.json()["traj"])
    torch.testing.assert_close(
        traj["rewards"], torch.tensor([-1.0, 1.0]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        traj["token_rewards"],
        torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 3.0, 3.0]]),
        rtol=0,
        atol=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "timeout", "shape", "nan"])
@pytest.mark.parametrize("normalize", [False, True])
async def test_failed_scorer_rejects_group_but_retains_successful_session_stats(
    proxy_factory, failure, normalize
):
    """Unknown scores are not zero; successful-session observations retain v1 meaning."""

    class Scorer(BaseScorer):
        name = "failure"

        async def evaluate(self, interaction, ctx):
            if interaction.interaction_id == "bad":
                if failure == "timeout":
                    raise TimeoutError("judge timed out")
                if failure == "shape":
                    return torch.zeros(3)
                if failure == "nan":
                    return float("nan")
                return None
            return 0.0

    client, store = await proxy_factory([Scorer()])
    async with client:
        sessions = [
            _add_session(store, "group", [_interaction(name)])
            for name in ("good", "bad")
        ]
        response = await _export(client, sessions, reward_normalization=normalize)
    assert response.status_code == 200
    assert response.json()["traj"] == {}
    assert store.session_count == 0
    workflow = InferenceServiceWorkflow(controller=MagicMock())
    workflow._request_export = AsyncMock(return_value=response.json())
    assert await workflow._export_interactions(MagicMock(), sessions) == {}
    metrics = stats_tracker.export_all(reduce_group=None)
    assert metrics["rollout/prm_turn_reward/failure"] == 0.0
    assert metrics["rollout/prm_turn_reward/failure__count"] == 1


@pytest.mark.asyncio
async def test_later_branch_failure_returns_no_partial_session_stats(proxy_factory):
    """A successful first branch cannot leak observations from a rejected session."""

    class Scorer(_BranchScorer):
        async def evaluate(self, interaction, ctx):
            if any(m.get("content") == "b" for m in ctx["messages"]):
                raise RuntimeError("bad branch")
            return await super().evaluate(interaction, ctx)

    root = _interaction("root")
    client, store = await proxy_factory([Scorer()])
    async with client:
        sid = _add_session(
            store,
            "branched",
            [
                root,
                _interaction("a", parent=root),
                _interaction("b", parent=root, branch="b"),
            ],
        )
        response = await _export(client, [sid])
    assert response.json() == {
        "traj": {},
        "prm_stats": {"turns": [], "trajectory_rewards": []},
    }
    assert root.token_rewards is None


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [False, True])
async def test_unconfigured_or_disabled_prm_keeps_original_response(
    proxy_factory, disabled
):
    """No configuration or disabled scoring is a no-op, including response shape."""
    scorer = _RewardScorer()
    client, store = await proxy_factory(
        [scorer] if disabled else [], enabled=not disabled
    )
    async with client:
        sid = _add_session(store, "plain", [_interaction("plain")])
        response = await _export(client, [sid])
    assert set(response.json()) == {"traj"}
    assert "token_rewards" not in deserialize_value(response.json()["traj"])
    assert not scorer.seen


@pytest.mark.asyncio
async def test_discard_skips_scoring_and_cleans_sessions(proxy_factory):
    """A group already rejected by its Agent calls does not invoke judges."""
    scorer = _RewardScorer()
    client, store = await proxy_factory([scorer])
    async with client:
        sid = _add_session(store, "discard", [_interaction("unused")])
        response = await _export(client, [sid], discard_trajectory=True)
    assert response.json() == {"traj": {}}
    assert not scorer.seen
    assert store.session_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["style", "duplicates"])
async def test_invalid_prm_export_does_not_consume_trajectory(proxy_factory, invalid):
    """Invalid protocol requests fail before claiming any ready trajectory."""
    scorer = _RewardScorer()
    client, store = await proxy_factory([scorer])
    async with client:
        sid = _add_session(store, "validation", [_interaction("turn")])
        response = await _export(
            client,
            [sid, sid] if invalid == "duplicates" else [sid],
            **({"style": "individual"} if invalid == "style" else {}),
        )
        assert response.status_code == 400
        assert not scorer.seen
        assert store.get_session(sid).has_ready_trajectories
        response = await _export(client, [sid])
    assert response.status_code == 200
    assert len(scorer.seen) == 1


@pytest.mark.asyncio
async def test_overlapping_exports_score_once_and_preserve_reused_session(
    proxy_factory,
):
    """An awaiting export owns its old session, not a new session with the same ID."""
    entered, release = asyncio.Event(), asyncio.Event()

    class Scorer(_RewardScorer):
        async def evaluate(self, interaction, ctx):
            entered.set()
            await release.wait()
            return await super().evaluate(interaction, ctx)

    scorer = Scorer()
    client, store = await proxy_factory([scorer])
    async with client:
        sid = _add_session(store, "reused", [_interaction("first")])
        pending = asyncio.create_task(_export(client, [sid]))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            repeated = await _export(client, [sid])
            assert repeated.json()["traj"] == {}
            assert repeated.json()["prm_stats"]["turns"] == []
            new_sid, _ = store.start_session("reused")
            assert new_sid == sid
            replacement = store.get_session(new_sid)
        finally:
            release.set()
        first = await asyncio.wait_for(pending, timeout=2)
    assert first.status_code == 200
    assert first.json()["traj"]
    assert scorer.seen == [(1.0, False)]
    assert store.get_session(sid) is replacement


@pytest.mark.asyncio
async def test_cancelled_scoring_cleans_owned_sessions(proxy_factory):
    """Cancellation waits for both single-turn sessions before releasing them."""
    entered = asyncio.Event()
    active = []
    sessions_during_cleanup = []

    class Scorer(_RewardScorer):
        async def evaluate(self, interaction, ctx):
            active.append(interaction.interaction_id)
            if len(active) == 2:
                entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                sessions_during_cleanup.append(store.session_count)

    client, store = await proxy_factory([Scorer()])
    async with client:
        ids = [
            _add_session(store, "cancel", [_interaction(name)])
            for name in ("first", "second")
        ]
        pending = asyncio.create_task(_export(client, ids))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
        finally:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert sessions_during_cleanup == [2, 2]
    assert store.session_count == 0
    assert not stats_tracker.export_all(reduce_group=None)


@pytest.mark.asyncio
async def test_concurrent_sessions_keep_context_and_statistics_isolated(proxy_factory):
    """One request's JSON report cannot drain or mix another request's observations."""

    class Scorer(_RewardScorer):
        async def evaluate(self, interaction, ctx):
            await asyncio.sleep(0)
            return await super().evaluate(interaction, ctx)

    client, store = await proxy_factory([Scorer()])
    async with client:
        one, two = _interaction("one"), _interaction("two")
        one.reward, two.reward = 1.0, 3.0
        ids = [_add_session(store, "concurrent", [x]) for x in (one, two)]
        responses = await asyncio.gather(
            _export(client, [ids[0]]), _export(client, [ids[1]], is_eval=True)
        )
    reports = [
        TypeAdapter(PRMExportStats).validate_python(r.json()["prm_stats"])
        for r in responses
    ]
    assert [[t.reward for t in s.turns] for s in reports] == [[1.0], [3.0]]
    assert not stats_tracker.export_all(reduce_group=None)
    assert store.session_count == 0


@pytest.mark.asyncio
async def test_group_sessions_score_concurrently_and_retain_request_order(
    proxy_factory,
):
    """Independent judges overlap, but completion order cannot reorder samples."""
    both_entered = asyncio.Event()
    second_finished = asyncio.Event()
    entered = []

    class Scorer(_RewardScorer):
        async def evaluate(self, interaction, ctx):
            entered.append(interaction.interaction_id)
            if len(entered) == 2:
                both_entered.set()
            await both_entered.wait()
            if interaction.interaction_id == "first":
                await second_finished.wait()
            else:
                second_finished.set()
            return await super().evaluate(interaction, ctx)

    scorer = Scorer()
    client, store = await proxy_factory([scorer])
    async with client:
        first, second = _interaction("first"), _interaction("second")
        first.reward, second.reward = 1.0, 3.0
        ids = [_add_session(store, "parallel", [x]) for x in (first, second)]
        response = await asyncio.wait_for(_export(client, ids), timeout=2)
    assert response.status_code == 200
    assert scorer.seen == [(3.0, False), (1.0, False)]
    stats = TypeAdapter(PRMExportStats).validate_python(response.json()["prm_stats"])
    assert [turn.reward for turn in stats.turns] == [1.0, 3.0]
    trajectory = deserialize_value(response.json()["traj"])
    torch.testing.assert_close(
        trajectory["rewards"], torch.tensor([1.0, 3.0]), rtol=0, atol=0
    )
    assert store.session_count == 0


@pytest.mark.asyncio
async def test_persistent_session_scores_each_ready_trajectory_only_once(proxy_factory):
    """A retained session can produce subsequent complete trajectories, not re-score old ones."""
    scorer = _RewardScorer()
    client, store = await proxy_factory([scorer])
    async with client:
        sid = _add_session(store, "persistent", [_interaction("first")])
        first = await _export(client, [sid], trajectory_id=0, remove_session=False)
        repeated = await _export(client, [sid], trajectory_id=0, remove_session=False)
        assert first.json()["traj"]
        assert repeated.json()["traj"] == {}
        session = store.get_session(sid)
        second_turn = _interaction("second")
        session._active_completions = InteractionCache.from_dict(
            {"second": second_turn}
        )
        session.set_reward("second", 2.0)
        second = await _export(client, [sid], trajectory_id=1)
    assert second.json()["traj"]
    assert scorer.seen == [(1.0, False), (2.0, False)]
    assert store.session_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("next_ready", [False, True])
@pytest.mark.parametrize("first_result", ["success", "failure", "cancelled"])
async def test_online_scoring_preserves_next_hitl_trajectory(
    proxy_factory, next_ready, first_result
):
    """Finishing an export cannot delete concurrent work in a persistent session."""
    entered, release = asyncio.Event(), asyncio.Event()

    class Scorer(_RewardScorer):
        async def evaluate(self, interaction, ctx):
            if interaction.interaction_id == "first":
                entered.set()
                await release.wait()
                if first_result == "failure":
                    return None
            return await super().evaluate(interaction, ctx)

    client, store = await proxy_factory([Scorer()])
    session = store.get_or_create_hitl_session()
    session.active_completions["first"] = _interaction("first")
    session.set_reward("first", 1.0)
    controller = MagicMock()
    controller.wait_for_online_trajectory = AsyncMock(
        side_effect=[
            {"session_id": session.session_id, "trajectory_id": index}
            for index in (0, 1)
        ]
    )
    workflow = InferenceServiceWorkflow(controller=controller, export_style="concat")

    async def request_export(http_session, payload):
        response = await client.post(
            "/export_trajectories", headers=_HEADERS, json=payload
        )
        response.raise_for_status()
        return response.json()

    workflow._request_export = AsyncMock(side_effect=request_export)
    async with client:
        pending = asyncio.create_task(workflow._run_online(MagicMock()))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            session.active_completions["second"] = _interaction("second")
            if next_ready:
                session.set_reward("second", 2.0)
            if first_result == "cancelled":
                pending.cancel()
        finally:
            release.set()
        if first_result == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            first = await asyncio.wait_for(pending, timeout=2)
            assert bool(first) is (first_result == "success")

        assert store.get_session(session.session_id) is session
        if not next_ready:
            session.set_reward("second", 2.0)
        second = await workflow._run_online(MagicMock())
        torch.testing.assert_close(
            second["rewards"], torch.tensor([2.0]), rtol=0, atol=0
        )
    assert not session.has_ready_trajectories
    assert store.session_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("hitl", [False, True])
@pytest.mark.parametrize("prm_enabled", [False, True])
async def test_online_export_retains_only_persistent_sessions(
    proxy_factory, hitl, prm_enabled
):
    """HITL IDs keep advancing; ordinary session keys remain refreshable."""
    client, store = await proxy_factory([_RewardScorer()], enabled=prm_enabled)
    if hitl:
        session = store.get_or_create_hitl_session()
        sid, api_key = session.session_id, None
    else:
        sid, api_key = store.start_session("online")
        session = store.get_session(sid)
    controller = MagicMock()
    controller.wait_for_online_trajectory = AsyncMock()
    workflow = InferenceServiceWorkflow(controller=controller, export_style="concat")

    async def request_export(http_session, payload):
        response = await client.post(
            "/export_trajectories", headers=_HEADERS, json=payload
        )
        response.raise_for_status()
        return response.json()

    workflow._request_export = AsyncMock(side_effect=request_export)
    async with client:
        for index in range(2):
            turn = _interaction(f"turn-{index}")
            session.active_completions[turn.interaction_id] = turn
            reward = session.set_reward(turn.interaction_id, float(index + 1))
            assert reward.trajectory_id == (index if hitl else 0)
            controller.wait_for_online_trajectory.return_value = {
                "session_id": sid,
                "trajectory_id": reward.trajectory_id,
            }
            trajectory = await workflow._run_online(MagicMock())
            torch.testing.assert_close(
                trajectory["rewards"], torch.tensor([float(index + 1)]), rtol=0, atol=0
            )
            if hitl:
                assert store.get_session(sid) is session
            else:
                assert store.get_session(sid) is None
                assert store.get_session_by_api_key(api_key) is None
                if index == 0:
                    sid, refreshed_key = store.start_session("online", api_key=api_key)
                    assert refreshed_key == api_key
                    session = store.get_session(sid)
    assert store.session_count == int(hitl)


@pytest.mark.asyncio
async def test_missing_member_rejects_prm_group_without_normalization(proxy_factory):
    """PRM cannot silently shrink a group when one requested session is unavailable."""
    client, store = await proxy_factory([_RewardScorer()])
    async with client:
        sid = _add_session(store, "missing", [_interaction("present")])
        response = await _export(client, [sid, "not-present"])
    assert response.json()["traj"] == {}
    assert len(response.json()["prm_stats"]["turns"]) == 1
    assert store.session_count == 0


@pytest.mark.asyncio
async def test_http_export_retry_records_received_metrics_once(proxy_factory):
    """Retry connection setup without duplicating a successfully received report."""
    client, store = await proxy_factory([_RewardScorer()])
    async with client:
        sid = _add_session(store, "retry", [_interaction("turn")])
        exported = await _export(client, [sid])

    failed = MagicMock()
    failed.__aenter__ = AsyncMock(side_effect=aiohttp.ServerDisconnectedError())
    response = MagicMock()
    response.json = AsyncMock(return_value=exported.json())
    succeeded = MagicMock()
    succeeded.__aenter__ = AsyncMock(return_value=response)
    session = MagicMock()
    session.post.side_effect = [failed, succeeded]

    workflow = InferenceServiceWorkflow(controller=MagicMock())
    trajectory = await workflow._export_interactions(session, [sid])
    assert trajectory
    assert session.post.call_count == 2
    metrics = stats_tracker.export_all(reduce_group=None)
    assert metrics["rollout/prm_turn_reward/outcome__count"] == 1
    assert metrics["rollout/prm_trajectory_reward/outcome__count"] == 1


@pytest.mark.asyncio
async def test_metric_recording_errors_do_not_retry_export(monkeypatch):
    """Local metric errors cannot repeat HTTP export after partial publication."""
    workflow = InferenceServiceWorkflow(controller=MagicMock())
    workflow._request_export = AsyncMock(
        return_value={"traj": {}, "prm_stats": {"turns": [], "trajectory_rewards": []}}
    )
    monkeypatch.setattr(
        PRMExportStats, "record", MagicMock(side_effect=RuntimeError("invalid metric"))
    )
    with pytest.raises(RuntimeError, match="invalid metric"):
        await workflow._export_interactions(MagicMock(), ["session"])
    workflow._request_export.assert_awaited_once()


def test_data_proxy_cli_reconstructs_typed_prm_config(monkeypatch):
    """The subprocess CLI preserves scorer weights, kwargs and shaping settings."""
    from areal.v2.inference_service.data_proxy import __main__ as cli

    config = PRMConfig(
        scorers=[
            PRMScorerConfig(
                path="examples.prm.scorers.LengthBudgetScorer",
                weight=0.25,
                kwargs={"max_output_tokens": 7},
            )
        ]
    )
    config.advantage_shaping.mode = "process_weighted"
    monkeypatch.setattr(
        "sys.argv",
        [
            "data_proxy",
            "--host",
            "127.0.0.1",
            "--tokenizer-path",
            "mock",
            "--chat-template-type",
            "concat",
            "--prm-config",
            json.dumps(asdict(config)),
        ],
    )
    factory = MagicMock()
    monkeypatch.setattr(cli, "create_app", factory)
    monkeypatch.setattr(cli.uvicorn, "run", MagicMock())
    cli.main()
    received = factory.call_args.args[0].prm
    assert received == config
    assert isinstance(received.scorers[0], PRMScorerConfig)


def test_data_proxy_rejects_unsupported_prm_chat_template():
    """Standalone service configuration enforces the same concat requirement."""
    with pytest.raises(ValueError, match="chat_template_type='concat'"):
        DataProxyConfig(prm=PRMConfig(scorers=[PRMScorerConfig(path="unused.Scorer")]))


@pytest.mark.asyncio
async def test_data_proxy_loads_configured_scorer_during_startup():
    """A real configured plugin loads without an inference backend or model download."""
    app = create_app(
        DataProxyConfig(
            backend_addr="",
            chat_template_type="concat",
            prm=PRMConfig(
                scorers=[
                    PRMScorerConfig(
                        path="examples.prm.scorers.LengthBudgetScorer",
                        kwargs={"max_output_tokens": 1},
                    )
                ]
            ),
        )
    )
    async with app.router.lifespan_context(app):
        sid = _add_session(app.state.session_store, "plugin", [_interaction("long")])
        # A failing length budget is a legitimate zero, not a rejected trajectory.
        from unittest.mock import patch

        with patch(
            "areal.v2.inference_service.data_proxy.app._remotize_trajectory",
            side_effect=lambda traj, node_addr: traj,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await _export(client, [sid])
    assert response.status_code == 200
    assert response.json()["traj"]
    stats = TypeAdapter(PRMExportStats).validate_python(response.json()["prm_stats"])
    assert stats.turns[0].reward == 0.0
    assert stats.turns[0].observations[0].value is False
