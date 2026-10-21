"""Per-task reward snapshots, startup validation, and wire-level launch routing."""

import json
from unittest.mock import Mock

import pytest

from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient
from examples.swe.arena_reward_mapping import (
    prepare_reward_mapping,
)
from examples.swe.arena_types import ArenaRewardRefConfig


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
