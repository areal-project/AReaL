# SPDX-License-Identifier: Apache-2.0

"""CPU observations: estimates are advisory and cannot admit a training batch."""

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.elastic.swe.progress import EpisodeProgress, SWEProgress


@pytest.mark.asyncio
@pytest.mark.parametrize("serial", [False, True])
async def test_native_v2_members_without_sample_indices_have_observable_eta(
    tmp_path, monkeypatch, serial
):
    """Run the actual v2 group wrapper, which only supplies native task_id."""
    from areal.infra import workflow_context
    from areal.v2.inference_service.controller.workflow import InferenceServiceWorkflow

    for sample in range(4):
        write_progress(tmp_path, 0, sample, stage="completed")

    async def run(data, **kwargs):
        context = workflow_context.get()
        directory = tmp_path / "swe-episodes" / uuid.uuid4().hex
        directory.mkdir()
        progress = EpisodeProgress(
            directory,
            task_id="same-task",
            group_id=context.task_id,
            sample_idx=context.sample_idx,
        )
        await progress.mark("grading")
        return 0.0

    workflow = InferenceServiceWorkflow(
        SimpleNamespace(get_version=lambda: 0),
        agent=SimpleNamespace(run=run),
        group_size=4,
        serialize_group_samples=serial,
    )
    workflow._start_session = AsyncMock(
        return_value=("group", [(str(i), "test-key") for i in range(4)])
    )
    workflow._set_last_reward = AsyncMock()
    workflow._export_interactions = AsyncMock(return_value={})
    monkeypatch.setattr(workflow_context, "get_httpx_client", AsyncMock())
    previous = workflow_context.get()
    workflow_context.set(workflow_context.WorkflowContext(task_id=7))
    try:
        await workflow._run_offline(AsyncMock(), {})
    finally:
        workflow_context.set(previous)
    observation = SWEProgress(tmp_path)(
        {"group_size": 4, "batch_size": 1, "counts": {"ready": 0}}
    )
    assert observation["batch_remaining_seconds"] is not None
    members = observation["groups"][0]["samples"]
    assert len({m["episode_id"] for m in members}) == 4
    assert {m["sample_idx"] for m in members} == {None}
    assert observation["groups"][0]["group_id"] == 7
    assert (
        workflow._export_interactions.await_args.kwargs["discard_trajectory"] is False
    )


@pytest.mark.asyncio
async def test_progress_tracks_phases_without_resetting_rollout_age(
    tmp_path, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(
        "examples.elastic.swe.progress.time.monotonic", lambda: clock[0]
    )
    progress = EpisodeProgress(tmp_path, task_id="task", group_id=9, sample_idx=2)
    await progress.mark("initializing")
    clock[0] = 10
    await progress.mark("rollout", phase="generation", turn=1)
    clock[0] = 40
    await progress.mark("rollout", phase="tool", turn=1)
    assert json.loads(progress.path.read_text())["stage_started"] == 10
    clock[0] = 110
    await progress.mark("grading", turn=4)
    clock[0] = 190
    await progress.mark("cleanup", turn=4)
    assert json.loads(progress.path.read_text())["stage"] == "cleanup"
    clock[0] = 220
    await progress.mark("completed", turn=4)
    record = json.loads(progress.path.read_text())
    assert record["group_id"] == 9 and record["sample_idx"] == 2
    assert record["stage_seconds"] == {
        "initializing": 10,
        "rollout": 100,
        "grading": 80,
        "cleanup": 30,
    }


@pytest.mark.asyncio
async def test_progress_write_failure_is_advisory_but_cancellation_propagates(
    tmp_path, monkeypatch
):
    progress = EpisodeProgress(tmp_path, task_id="task", group_id=1, sample_idx=0)

    async def fail(*args):
        raise OSError("observation volume unavailable")

    monkeypatch.setattr("examples.elastic.swe.progress.aiofiles.os.replace", fail)
    await progress.mark("grading")
    assert "unavailable" in progress.error

    async def cancel(*args):
        raise asyncio.CancelledError()

    monkeypatch.setattr("examples.elastic.swe.progress.aiofiles.os.replace", cancel)
    with pytest.raises(asyncio.CancelledError):
        await progress.mark("cleanup")


def write_progress(root, group, sample, *, stage, elapsed=0):
    directory = root / "swe-episodes" / f"{group}-{sample}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "progress.json").write_text(
        json.dumps(
            {
                "task_id": "same-task",
                "group_id": group,
                "sample_idx": sample,
                "episode_id": directory.name,
                "stage": stage,
                "phase": stage,
                "turn": 3,
                "stage_started": 1000 - elapsed,
                "stage_seconds": {
                    "initializing": 10,
                    "rollout": 100,
                    "grading": 80,
                    "cleanup": 30,
                },
            }
        )
    )


def test_batch_eta_uses_slowest_sample_and_native_group_readiness(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("examples.elastic.swe.progress.time.monotonic", lambda: 1000)
    for sample in range(4):
        write_progress(tmp_path, 0, sample, stage="completed")
        write_progress(tmp_path, 1, sample, stage="cleanup", elapsed=10)  # 20s
        write_progress(tmp_path, 2, sample, stage="completed")
    write_progress(tmp_path, 2, 3, stage="grading", elapsed=20)  # slowest 90s
    status = {
        "group_size": 4,
        "batch_size": 2,
        "counts": {"ready": 0},
        "last_measured_durations": {
            "startup_seconds": 50,
            "checkpoint_seconds": 20,
            "release_seconds": 5,
        },
    }
    observer = SWEProgress(tmp_path, margin_seconds=10)
    result = observer(status)
    assert result["batch_remaining_seconds"] == {"low": 90, "high": 90}
    assert len(result["groups"]) == 2  # Same SWE task, distinct native groups.
    assert result["start_within_seconds"] == 30
    assert result["estimated_free_window_seconds"] == 5
    status["actor_running"] = False
    assert observer(status)["estimated_free_window_seconds"] == 30
    assert result["lifecycle_costs"]["startup_seconds"]["source"] == "current_run"
    status["counts"]["ready"] = 1
    assert observer(status)["batch_remaining_seconds"] == {"low": 20, "high": 20}
    status["counts"]["ready"] = 2
    assert observer(status)["batch_remaining_seconds"] == {"low": 0, "high": 0}


@pytest.mark.parametrize(
    "case", ["no_history", "overdue", "missing_member", "failed_member"]
)
def test_eta_unknown_does_not_mean_ready(tmp_path, monkeypatch, case):
    monkeypatch.setattr("examples.elastic.swe.progress.time.monotonic", lambda: 1000)
    if case != "no_history":
        for sample in range(4):
            write_progress(tmp_path, 0, sample, stage="completed")
    for sample in range(3 if case == "missing_member" else 4):
        write_progress(
            tmp_path,
            1,
            sample,
            stage="failed" if case == "failed_member" and sample == 0 else "grading",
            elapsed=100 if case == "overdue" else 20,
        )
    result = SWEProgress(tmp_path)(
        {"group_size": 4, "batch_size": 1, "counts": {"ready": 0}}
    )
    assert result["batch_remaining_seconds"] is None
    assert result["start_within_seconds"] is None
    assert result["eta_confidence"] == "unknown"
    assert result["lifecycle_costs"]["startup_seconds"] == {
        "seconds": 90,
        "source": "configured_prior",
    }
