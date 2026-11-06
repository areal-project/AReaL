"""Per-task reward snapshots, startup validation, and wire-level launch routing."""

from unittest.mock import Mock

import pytest

from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient
from examples.swe.arena_reward_mapping import (
    prepare_reward_mapping,
)


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
