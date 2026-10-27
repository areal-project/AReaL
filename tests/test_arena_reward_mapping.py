"""Per-task reward snapshots, startup validation, and wire-level launch routing."""

import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest

from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient
from examples.swe.arena_config import parse_arena_stream_config
from examples.swe.arena_preflight import _recent_health
from examples.swe.arena_reward_mapping import (
    load_reward_mapping,
    prepare_reward_mapping,
)
from examples.swe.arena_types import ArenaRewardRefConfig, ArenaStreamConfig


@pytest.fixture
def arena(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    return ArenaOpenAPIClient(base_url="https://arena.example", request_retries=0)


@pytest.fixture
def rows():
    return [{"data_id": f"env:task-{name}@1"} for name in ("a", "b")]


def version_payload(kind, key, version, **kwargs):
    return {
        "key": key,
        "version": version,
        "status": "PUBLISHED",
        "content_hash": "env-hash",
        "spec": {"metadata": {"source_reward_ref": f"grader-{key}@reward-v2"}},
    }


def test_prepare_mapping_two_graders_preserves_snapshot(
    tmp_path, arena, rows, monkeypatch
):
    """Startup pins distinct graders, and recovery never resolves Env metadata again."""
    version = Mock(side_effect=version_payload)
    monkeypatch.setattr(arena, "get_published_version", version)
    path = tmp_path / "run" / "rewards.json"

    mapping = prepare_reward_mapping(arena, "stream", rows, str(path))

    assert mapping == {
        "env:task-a@1": ArenaRewardRefConfig("grader-task-a", "reward-v2"),
        "env:task-b@1": ArenaRewardRefConfig("grader-task-b", "reward-v2"),
    }
    assert path.stat().st_mode & 0o777 == 0o600
    snapshot = path.read_bytes()
    version.reset_mock()
    assert prepare_reward_mapping(arena, "stream", rows, str(path)) == mapping
    assert path.read_bytes() == snapshot
    assert [call.args[0] for call in version.call_args_list] == ["rewards", "rewards"]

    with pytest.raises(ArenaAPIError, match="coverage mismatch"):
        prepare_reward_mapping(arena, "stream", rows[:1], str(path))
    assert path.read_bytes() == snapshot


@pytest.mark.parametrize("source_ref", [None, "", "grader", "grader@", "a@b@c", "a@ b"])
def test_prepare_mapping_invalid_ref_leaves_no_snapshot(
    tmp_path, arena, rows, monkeypatch, source_ref
):
    """Missing or malformed references cannot fall back to a Stream default."""
    monkeypatch.setattr(
        arena,
        "get_published_version",
        lambda *args, **kwargs: {
            "spec": {"metadata": {"source_reward_ref": source_ref}}
        },
    )
    path = tmp_path / "rewards.json"
    with pytest.raises(ArenaAPIError, match="source_reward_ref"):
        prepare_reward_mapping(arena, "stream", rows, str(path))
    assert not path.exists()


def test_prepare_mapping_unpublished_reward_leaves_no_snapshot(tmp_path, arena, rows):
    """A real API response for a draft reward fails before the map is published."""

    def handler(request):
        kind, key, _, version = request.url.path.split("/")[-4:]
        data = version_payload(kind, key, version)
        if kind == "rewards":
            data["status"] = "DRAFT"
        return httpx.Response(200, json=data)

    path = tmp_path / "rewards.json"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArenaAPIError, match="PUBLISHED"):
            prepare_reward_mapping(arena, "stream", rows, str(path), client=client)
    assert not path.exists()


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_stream", "count", "errors"])
def test_load_mapping_corrupt_snapshot_rejected(tmp_path, mutation):
    """Do not accept incomplete snapshots or silently overwrite duplicate IDs."""
    item = {"data_id": "env:a@1", "reward_ref": {"key": "r", "version": "v1"}}
    payload = {"stream_id": "stream", "count": 1, "items": [item]}
    if mutation == "duplicate":
        payload.update(count=2, items=[item, item])
    elif mutation == "wrong_stream":
        payload["stream_id"] = "another-stream"
    elif mutation == "count":
        payload["count"] = 2
    else:
        payload["errors"] = ["incomplete"]
    path = tmp_path / "rewards.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ArenaAPIError):
        load_reward_mapping(str(path), "stream")


def test_mapping_config_with_default_reward_rejected():
    """Per-task mapping and Stream-default pinning are separate explicit modes."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_arena_stream_config(
            {
                "name": "s",
                "stream_id": "stream",
                "reward_mapping_file": "map.json",
                "expected_reward_ref": {"key": "default", "version": "1"},
            }
        )


def test_dataset_mapping_preserves_per_row_reward(tmp_path, arena, rows, monkeypatch):
    """A Stream default must not replace per-task rewards during dataset loading."""
    from examples.swe.train_swe_rl import _resolve_arena_stream

    monkeypatch.setattr(arena, "get_published_version", version_payload)
    monkeypatch.setattr(
        arena,
        "resolve_stream",
        lambda _: {
            "stream_id": "stream",
            "default_reward_ref": {"key": "default", "version": "1"},
        },
    )
    monkeypatch.setattr(arena, "get_all_dataset_rows", lambda *args: rows)
    config = ArenaStreamConfig(
        name="s", stream_id="stream", reward_mapping_file=str(tmp_path / "map.json")
    )
    resolved, dataset = _resolve_arena_stream(arena, config)
    assert resolved.expected_reward_ref == ArenaRewardRefConfig()
    assert [row["reward_ref_key"] for row in dataset] == [
        "grader-task-a",
        "grader-task-b",
    ]
    assert all(row["reward_ref_version"] == "reward-v2" for row in dataset)


def test_workflow_mapping_launches_matching_reward_and_audits(
    tmp_path, arena, rows, monkeypatch
):
    """Two tasks in one Stream send their own grader; unknown or swapped rows fail."""
    from examples.swe.arena_agent import ArenaStreamAgentWorkflow

    def version_with_harness(kind, key, version, **kwargs):
        payload = version_payload(kind, key, version)
        if kind == "envs":
            payload["spec"]["metadata"]["harness_ref"] = f"h-{key}@1"
        return payload

    monkeypatch.setattr(arena, "get_published_version", version_with_harness)
    mapping_path = tmp_path / "map.json"
    prepare_reward_mapping(arena, "stream", rows, str(mapping_path))
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_llm_route_mode": "direct",
            "arena_result_dump_dir": str(tmp_path / "results"),
            "arena_streams": [
                {
                    "name": "s",
                    "stream_id": "stream",
                    "reward_mapping_file": str(mapping_path),
                    "harness": "wrong-stream-default@1",
                }
            ],
        }
    )
    seen = []

    def handler(request):
        assert request.url.path.endswith("/launch_one_task")
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(
            200, json={"task_id": f"task-{len(seen)}", "status": "DONE", "score": 0.7}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for row in rows:
                key = row["data_id"].removeprefix("env:").split("@")[0]
                data = {
                    **row,
                    "stream_id": "stream",
                    "arena_stream_name": "s",
                    "reward_ref_key": f"grader-{key}",
                    "reward_ref_version": "reward-v2",
                }
                kwargs = dict(
                    base_url="http://proxy",
                    api_key="session-key",
                    session_id="session-1",
                    arena_http_client=client,
                )
                result = await workflow.run(data, **kwargs)
                assert result == 0.7
                await workflow.persist_episode_result(data, result)
                with pytest.raises(ValueError, match="reward_ref mismatch"):
                    await workflow.run({**data, "reward_ref_key": "wrong"}, **kwargs)
                with pytest.raises(ValueError, match="Harness mismatch"):
                    await workflow.run({**data, "harness": "wrong@1"}, **kwargs)
                with pytest.raises(ValueError, match="no data_id"):
                    await workflow.run({**data, "data_id": "env:unknown@1"}, **kwargs)

    asyncio.run(run())
    assert [item["reward"] for item in seen] == [
        "grader-task-a@reward-v2",
        "grader-task-b@reward-v2",
    ]
    assert [item["harness"] for item in seen] == ["h-task-a@1", "h-task-b@1"]
    records = [
        json.loads(line)
        for path in (tmp_path / "results").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [record["expected_reward_ref"]["key"] for record in records] == [
        "grader-task-a",
        "grader-task-b",
    ]


def test_recent_health_mapping_excludes_wrong_task_grader():
    """Historical scores only count when their data-id/reward pairing matches."""
    tasks = [
        {
            "data_id": "env:a@1",
            "harness_ref": {"key": "h", "version": "1"},
            "reward_ref": {"key": grader, "version": "1"},
            "status": "DONE",
        }
        for grader in ("correct", "wrong")
    ]
    assert _recent_health(
        tasks,
        harness="h@1",
        reward_ref=ArenaRewardRefConfig(),
        reward_mapping={"env:a@1": ArenaRewardRefConfig("correct", "1")},
    ) == (1, 0)


def test_preflight_mapping_creates_complete_snapshot(tmp_path, arena, rows):
    """Preflight resolves the full dataset, even though its count probe uses one row."""
    import yaml

    from examples.swe.arena_preflight import _validate_streams_with_client

    path = tmp_path / "mapping.json"
    config_path = tmp_path / "streams.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "streams": [
                    {
                        "name": "s",
                        "stream_id": "stream",
                        "harness": "h@1",
                        "reward_mapping_file": str(path),
                    }
                ]
            }
        )
    )
    data_ids = [row["data_id"] for row in rows]

    def handler(request):
        url = request.url.path
        if url.endswith("/streams/stream"):
            payload = {"stream": {"stream_id": "stream", "status": "ACTIVE"}}
        elif url.endswith("/harnesses/h/versions/1"):
            payload = {"status": "PUBLISHED"}
        elif url.endswith("/dataset"):
            limit = int(request.url.params["limit"])
            offset = int(request.url.params.get("offset", 0))
            selected = data_ids[offset : offset + limit]
            payload = {
                "data_ids": selected,
                "total": 2,
                "count": len(selected),
                "limit": limit,
                "offset": offset,
            }
        elif url.endswith("/tasks"):
            payload = {"items": []}
        else:
            kind, key, _, version = url.split("/")[-4:]
            payload = version_payload(kind, key, version)
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        summaries = _validate_streams_with_client(
            str(config_path),
            base_url=arena.base_url,
            retries=0,
            min_terminal_tasks=0,
            client=client,
        )
    assert summaries[0]["reward_mapping_count"] == 2
    assert summaries[0]["expected_reward_ref"] == {"key": "", "version": ""}
    assert set(load_reward_mapping(str(path), "stream")) == set(data_ids)


@pytest.mark.parametrize("method", ["launch_one_task", "launch_one_task_result"])
def test_launch_api_explicit_reward_preserved(arena, method):
    """Both launch APIs encode the Online CLI's reward string wire format."""

    def handler(request):
        assert json.loads(request.content)["reward"] == "grader@reward-v1"
        return httpx.Response(
            200, json={"task_id": "t", "status": "DONE", "score": 1.0}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await getattr(arena, method)(
                "stream",
                "env:a@1",
                "model",
                "http://proxy",
                "dummy",
                reward="grader@reward-v1",
                client=client,
            )

    asyncio.run(run())


def test_mapping_stream_bindings_pin_harnesses_and_reject_drift(
    tmp_path, arena, rows, monkeypatch
):
    """Freeze per-task Harnesses and validate both new and recovered snapshots."""
    from examples.swe.arena_reward_mapping import load_harness_mapping

    bindings = {
        row["data_id"]: {
            "reward_ref": {"key": f"grader-{i}", "version": "1"},
            "harness_ref": {"key": "areal-harness", "version": f"v{i}"},
        }
        for i, row in enumerate(rows)
    }
    stream = {"metadata": {"per_task_bindings": json.dumps(bindings)}}
    calls = []

    def version(kind, key, ver, **kwargs):
        calls.append((kind, key, ver))
        return {"status": "PUBLISHED", "spec": {"metadata": {}}}

    monkeypatch.setattr(arena, "get_published_version", version)
    path = str(tmp_path / "map.json")
    prepare_reward_mapping(arena, "stream", rows, path, stream=stream)
    expected = {row["data_id"]: f"areal-harness@v{i}" for i, row in enumerate(rows)}
    assert load_harness_mapping(path, "stream") == expected
    assert ("harnesses", "areal-harness", "v0") in calls
    assert ("harnesses", "areal-harness", "v1") in calls
    prepare_reward_mapping(arena, "stream", rows, path, stream=stream)
    bindings[rows[0]["data_id"]]["harness_ref"]["version"] = "changed"
    stream["metadata"]["per_task_bindings"] = json.dumps(bindings)
    with pytest.raises(ArenaAPIError, match="Frozen Arena Harness binding mismatch"):
        prepare_reward_mapping(arena, "stream", rows, path, stream=stream)


def test_mapping_partial_harnesses_rejected(tmp_path, arena, rows, monkeypatch):
    """Mixed pinning must not silently route missing tasks to a default Harness."""

    def version(kind, key, ver, **kwargs):
        payload = version_payload(kind, key, ver)
        if key == "task-a":
            payload["spec"]["metadata"]["harness_ref"] = "h@1"
        return payload

    monkeypatch.setattr(arena, "get_published_version", version)
    path = tmp_path / "map.json"
    with pytest.raises(ArenaAPIError, match="cover every data_id"):
        prepare_reward_mapping(arena, "stream", rows, str(path))
    assert not path.exists()


def test_health_filters_each_task_harness():
    """Count both Harness variants, excluding a result with the wrong binding."""
    rows = [
        {"data_id": "a", "harness_ref": {"key": "h", "version": "1"}, "status": "DONE"},
        {
            "data_id": "b",
            "harness_ref": {"key": "h", "version": "2"},
            "status": "HARNESS_FAILED",
        },
        {"data_id": "b", "harness_ref": {"key": "h", "version": "1"}, "status": "DONE"},
    ]
    assert _recent_health(
        rows,
        harness="h@default",
        reward_ref=ArenaRewardRefConfig(),
        harness_mapping={"a": "h@1", "b": "h@2"},
    ) == (2, 1)


@pytest.mark.parametrize("status", ["PUBLISHED", "DRAFT"])
def test_harness_registry_requires_published_version(arena, status):
    """Harness pins use the real registry path and reject unpublished versions."""

    def handler(request):
        assert request.url.path == "/openapi/v1/harnesses/h/versions/1"
        return httpx.Response(200, json={"key": "h", "version": "1", "status": status})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        if status == "DRAFT":
            with pytest.raises(ArenaAPIError, match="PUBLISHED"):
                arena.get_published_version("harnesses", "h", "1", client=client)
        else:
            assert (
                arena.get_published_version("harnesses", "h", "1", client=client)[
                    "status"
                ]
                == status
            )


@pytest.mark.parametrize("field", ["reward_ref", "harness_ref"])
def test_optional_stream_binding_is_frozen_and_explicit_drift_rejected(
    tmp_path, arena, rows, monkeypatch, field
):
    """Reward-only and Harness-only pins work for both creation and recovery."""
    from examples.swe.arena_reward_mapping import load_harness_mapping

    version = Mock(side_effect=version_payload)
    monkeypatch.setattr(arena, "get_published_version", version)
    bindings = {
        row["data_id"]: {
            field: f"grader-task-{name}@reward-v2" if field == "reward_ref" else "h@1"
        }
        for row, name in zip(rows, ("a", "b"), strict=True)
    }
    stream = {"metadata": {"per_task_bindings": bindings}}
    path = tmp_path / "map.json"
    first = prepare_reward_mapping(arena, "stream", rows, str(path), stream=stream)
    snapshot = path.read_bytes()
    harnesses = load_harness_mapping(str(path), "stream")
    assert harnesses == (
        {row["data_id"]: "h@1" for row in rows} if field == "harness_ref" else {}
    )
    version.reset_mock()
    assert (
        prepare_reward_mapping(arena, "stream", rows, str(path), stream=stream) == first
    )
    assert all(call.args[0] != "envs" for call in version.call_args_list)
    bindings[rows[0]["data_id"]][field] = "changed@1"
    with pytest.raises(ArenaAPIError, match="Frozen Arena .* binding mismatch"):
        prepare_reward_mapping(arena, "stream", rows, str(path), stream=stream)
    assert path.read_bytes() == snapshot
