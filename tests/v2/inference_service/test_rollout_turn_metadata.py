# SPDX-License-Identifier: Apache-2.0

"""Per-session classification transport from V2 agents to exported turn stats."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import torch

from areal.api import ModelResponse
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils.rollout_stats import (
    ROLLOUT_HARNESS_OUTCOME,
    ROLLOUT_TURN_COUNT,
    record_rollout_turn_stats,
)
from areal.utils.stats_tracker import DistributedStatsTracker
from areal.v2.inference_service.controller.workflow import InferenceServiceWorkflow
from areal.v2.inference_service.data_proxy.app import create_app
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.session import (
    SessionStore,
)


def _interaction(turns: int) -> InteractionWithTokenLogpReward:
    # Deliberately trigger non-growing-prompt fallback: tensor IDs alone lose
    # parent history, but the retained branch still has `turns` real generations.
    leaf = None
    for _ in range(turns):
        leaf = InteractionWithTokenLogpReward(
            model_response=ModelResponse(
                input_tokens=[1],
                output_tokens=[2],
                output_logprobs=[-0.1],
                output_versions=[0],
            ),
            parent=leaf,
            chat_template_type="concat",
            reward=0.0,
            metadata={"thinking_option": "on"},
        )
    assert leaf is not None
    leaf.to_tensor_dict()["loss_mask"].zero_()
    return leaf


@pytest.fixture
def tracker(monkeypatch):
    tracker = DistributedStatsTracker("rollout")
    monkeypatch.setattr("areal.utils.stats_tracker.get", lambda scope: tracker)
    return tracker


def _workflow(agent, monkeypatch):
    workflow = InferenceServiceWorkflow(controller=Mock(), agent=agent, group_size=2)
    workflow._start_session = AsyncMock(
        return_value=("group", [("s0", "k0"), ("s1", "k1")])
    )
    workflow._set_last_reward = AsyncMock()
    workflow._export_interactions = AsyncMock(return_value={"rewards": torch.zeros(2)})
    monkeypatch.setattr(
        "areal.v2.inference_service.controller.workflow.workflow_context.get",
        lambda: SimpleNamespace(task_id=1, is_eval=False),
    )
    monkeypatch.setattr(
        "areal.v2.inference_service.controller.workflow.workflow_context.get_httpx_client",
        AsyncMock(return_value=None),
    )
    return workflow


@pytest.mark.asyncio
@pytest.mark.parametrize("async_getter", [False, True])
async def test_concurrent_agent_metadata_stays_in_its_session(
    monkeypatch, tracker, async_getter
):
    """Sibling agents keep separate ContextVar results across await points."""
    result = ContextVar("episode_result", default={})
    expected = {
        "s0": {"arena_status": "DONE"},
        "s1": {
            "arena_status": "HARNESS_FAILED",
            "harness_outcome_code": "LLM_CONTEXT_WINDOW_EXCEEDED",
        },
    }

    async def run(data, *, api_key, **kwargs):
        result.set({**expected["s" + api_key[-1]], "unrelated_state": object()})
        await asyncio.sleep(0)
        return 0.0

    async def get_async():
        await asyncio.sleep(0)
        return result.get()

    agent = SimpleNamespace(
        run=run, get_episode_metadata=get_async if async_getter else result.get
    )
    workflow = _workflow(agent, monkeypatch)

    trajectory = await workflow._run_offline(None, {})

    assert trajectory is not None
    assert (
        workflow._export_interactions.await_args.kwargs["session_metadata"] == expected
    )
    assert result.get() == {}  # The parent task never received the child contexts.
    assert tracker.export()["rollout/reward__count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "getter_mode", ["absent", "raises", "invalid", "invalid_fields"]
)
async def test_bad_optional_metadata_does_not_reject_reward(
    monkeypatch, tracker, getter_mode
):
    """A failed optional metadata hook leaves status unknown, not a rollout failure."""
    agent = SimpleNamespace(run=AsyncMock(return_value=1.0))
    if getter_mode == "raises":
        agent.get_episode_metadata = Mock(
            side_effect=RuntimeError("metadata unavailable")
        )
    elif getter_mode == "invalid":
        agent.get_episode_metadata = Mock(return_value=["not a dict"])
    elif getter_mode == "invalid_fields":
        agent.get_episode_metadata = Mock(return_value={"arena_status": object()})
    workflow = _workflow(agent, monkeypatch)

    trajectory = await workflow._run_offline(None, {})

    assert trajectory is not None
    assert "session_metadata" not in workflow._export_interactions.await_args.kwargs
    assert tracker.export()["rollout/reward"] == 1.0


@pytest.mark.asyncio
async def test_export_request_transports_metadata_with_trajectory(monkeypatch):
    """The controller puts per-session classification into the actual POST body."""
    metadata = {"s0": {"arena_status": "HARNESS_FAILED"}}
    response = SimpleNamespace(
        raise_for_status=Mock(),
        json=AsyncMock(
            return_value={"traj": serialize_value({"rewards": torch.zeros(1)})}
        ),
    )
    context_manager = Mock()
    context_manager.__aenter__ = AsyncMock(return_value=response)
    context_manager.__aexit__ = AsyncMock(return_value=False)
    session = SimpleNamespace(post=Mock(return_value=context_manager))
    workflow = InferenceServiceWorkflow(controller=Mock(), gateway_addr="http://test")

    result = await workflow._export_interactions(
        session, ["s0"], session_metadata=metadata
    )

    assert session.post.call_args.kwargs["json"]["session_metadata"] == metadata
    torch.testing.assert_close(result["rewards"], torch.zeros(1), rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_metadata", [False, True])
async def test_data_proxy_stamps_each_session_before_merging(
    monkeypatch, tracker, include_metadata
):
    """Exercise request validation, per-session export, padding, and serialization."""
    config = DataProxyConfig(backend_addr="")
    app = create_app(config)
    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    app.state.session_store = store
    branches = {
        "s0": {"a": _interaction(2)},
        "s1": {"b": _interaction(4)},
        "s2": {"c": _interaction(6)},
    }
    sessions = {
        sid: SimpleNamespace(
            export_trajectory=Mock(return_value=(0, interactions)),
            snapshot_generated_turns=Mock(return_value=1),
        )
        for sid, interactions in branches.items()
    }
    monkeypatch.setattr(store, "get_session", sessions.get)
    monkeypatch.setattr(RTensor, "remotize", lambda trajectory, node_addr: trajectory)
    payload = {"session_ids": ["s0", "s1", "s2"], "remove_session": False}
    if include_metadata:
        payload["session_metadata"] = {
            "s0": {"arena_status": "DONE"},
            "s1": {
                "arena_status": "HARNESS_FAILED",
                "harness_outcome_code": "LLM_CONTEXT_WINDOW_EXCEEDED",
            },
        }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/export_trajectories",
            headers={"Authorization": f"Bearer {config.admin_api_key}"},
            json=payload,
        )

    assert response.status_code == 200, response.text
    trajectory = deserialize_value(response.json()["traj"])
    torch.testing.assert_close(
        trajectory[ROLLOUT_TURN_COUNT], torch.tensor([2.0, 4.0, 6.0]), rtol=0, atol=0
    )
    if not include_metadata:
        torch.testing.assert_close(
            trajectory[ROLLOUT_HARNESS_OUTCOME],
            torch.full((3,), -1, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        assert all(
            leaf.metadata == {"thinking_option": "on"}
            for branch in branches.values()
            for leaf in branch.values()
        )
        return
    assert trajectory[ROLLOUT_HARNESS_OUTCOME][2] == -1
    assert branches["s0"]["a"].metadata == {
        "thinking_option": "on",
        "arena_status": "DONE",
    }
    assert branches["s2"]["c"].metadata == {"thinking_option": "on"}
    record_rollout_turn_stats([trajectory], scope="rollout")
    metrics = tracker.export()
    assert metrics["rollout/num_turns_count"] == 3
    assert metrics["rollout/num_turns_no_harness_err/avg"] == 2
    assert metrics["rollout/num_turns_harness_err/LLM_CONTEXT_WINDOW_EXCEEDED/avg"] == 4
    assert metrics["rollout/num_turns_no_harness_err_count"] == 1
    assert metrics["rollout/num_turns_harness_err_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("discard", [False, True])
async def test_export_observes_excluded_failed_session_before_cleanup(
    monkeypatch, discard
):
    from areal.v2.inference_service.data_proxy.session import SessionData

    config = DataProxyConfig(backend_addr="")
    app = create_app(config)
    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    app.state.session_store = store
    good, failed = SessionData("good"), SessionData("failed")
    good.active_completions["g"] = _interaction(1)
    good.active_completions["g"]._interaction_id = "g"
    good.active_completions["g"].output_message_list = [
        {"role": "assistant", "content": "ok"}
    ]
    good.set_reward(None, 1.0)
    for i in range(3):
        failed.active_completions[str(i)] = _interaction(1)
    # Failed member has generated responses, but no reward and no ready trajectory.
    sessions = {"good": good, "failed": failed}
    monkeypatch.setattr(store, "get_session", sessions.get)
    monkeypatch.setattr(RTensor, "remotize", lambda trajectory, node_addr: trajectory)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/export_trajectories",
            headers={"Authorization": f"Bearer {config.admin_api_key}"},
            json={
                "session_ids": ["good", "failed"],
                "export_session_ids": ["good"],
                "discard_trajectory": discard,
                "remove_session": False,
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["episode_turns"] == {"good": 1, "failed": 3}
    trajectory = deserialize_value(response.json()["traj"])
    assert bool(trajectory) is not discard


def test_online_turn_snapshot_is_limited_to_target_trajectory():
    from areal.v2.inference_service.data_proxy.session import SessionData

    session = SessionData("online")
    session.active_completions["first"] = _interaction(1)
    first = session.set_reward(None, 1.0)
    for i in range(2):
        session.active_completions[f"second-{i}"] = _interaction(1)
    second = session.set_reward(None, 1.0)
    assert session.snapshot_generated_turns(first.trajectory_id) == 1
    assert session.snapshot_generated_turns(second.trajectory_id) == 2
    assert session.snapshot_generated_turns(999) is None
    session.active_completions["unfinished"] = _interaction(1)
    assert session.snapshot_generated_turns() == 4
    assert session.snapshot_generated_turns(first.trajectory_id) == 1


@pytest.mark.asyncio
async def test_export_empty_trajectory_still_carries_turn_metrics_until_consumption(
    monkeypatch,
):
    from areal.infra import workflow_context
    from areal.infra.rpc.serialization import serialize_value
    from areal.infra.workflow_context import WorkflowContext
    from areal.infra.workflow_executor import is_consumed_rollout_metric
    from areal.utils.episode_turns import track_episode_turns
    from areal.utils.rollout_metrics import ConsumedMetrics
    from areal.utils.scalar_capture import capture_scalars

    workflow_context.set(WorkflowContext(is_eval=False))
    workflow = _workflow(SimpleNamespace(run=AsyncMock(return_value=0.0)), monkeypatch)
    workflow._request_export = AsyncMock(
        return_value={"traj": serialize_value({}), "episode_turns": {"s0": 2, "s1": 7}}
    )
    # _workflow stubs the method for agent tests; exercise the real response path.
    with capture_scalars(select=is_consumed_rollout_metric) as metrics:
        with track_episode_turns():
            result = await InferenceServiceWorkflow._export_interactions(
                workflow, AsyncMock(), ["s0", "s1"], discard_trajectory=True
            )
    assert result == {}
    buffer = ConsumedMetrics()
    assert buffer.export() == {}
    buffer.record(metrics)
    stats = buffer.export()
    assert stats["rollout/num_turns_count"] == 2
    assert stats["rollout/num_turns/avg"] == 4.5
    assert stats["rollout/num_turns_missing_count"] == 0
