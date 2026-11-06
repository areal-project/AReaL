"""Pinned dataset recovery must preserve order and reject publication drift."""

import json
from dataclasses import replace

import httpx
import pytest

from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient
from examples.swe.arena_config import parse_arena_stream_config
from examples.swe.arena_dataset_snapshot import load_arena_dataset_snapshot
from examples.swe.arena_types import ArenaStreamConfig
from examples.swe.train_swe_rl import _resolve_arena_stream


@pytest.fixture
def snapshot(tmp_path):
    """Intentionally use non-sorted task order, as required for checkpoint replay."""
    payload = {
        "format_version": 1,
        "stream_id": "stream",
        "count": 2,
        "rows": [
            {
                "data_id": f"env:{key}@old",
                "arena_task_type": "swe",
                "env_content_hash": f"hash-{key}",
            }
            for key in ("z", "a")
        ],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


@pytest.fixture
def arena_http(monkeypatch):
    """Exercise real API response validation through an HTTP mock transport."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    requests = []
    overrides = {}

    def handler(request):
        requests.append(request.url.path)
        if request.url.path == "/openapi/v1/streams/stream":
            return httpx.Response(200, json={"stream_id": "stream"})
        kind, key, _, version = request.url.path.split("/")[-4:]
        assert kind in {"envs", "rewards"}
        payload = {
            "key": key,
            "version": version,
            "status": "PUBLISHED",
            "content_hash": f"hash-{key}",
        }
        payload.update(overrides.get(key, {}))
        return httpx.Response(200, json={"data": payload})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    arena = ArenaOpenAPIClient(base_url="https://arena.example", request_retries=0)
    return arena, requests, overrides


def test_snapshot_exact_versions_preserve_order_and_bytes(snapshot, arena_http):
    """The ordered snapshot is read-only and does not query latest dataset rows."""
    path, payload = snapshot
    arena, requests, _ = arena_http
    original = path.read_bytes()

    rows = load_arena_dataset_snapshot(arena, str(path), "stream", "responses")

    assert [row["data_id"] for row in rows] == [
        item["data_id"] for item in payload["rows"]
    ]
    assert all(row["llm_protocol"] == "responses" for row in rows)
    assert all(row["stream_id"] == "stream" for row in rows)
    assert set(requests) == {
        "/openapi/v1/envs/z/versions/old",
        "/openapi/v1/envs/a/versions/old",
    }
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "override,error",
    [
        ({"content_hash": "changed"}, "content hash mismatch"),
        ({"status": "DRAFT"}, "exact PUBLISHED"),
        ({"version": "new"}, "exact PUBLISHED"),
        ({"key": "other"}, "exact PUBLISHED"),
    ],
)
def test_snapshot_publication_drift_fails_without_rewriting(
    snapshot, arena_http, override, error
):
    """Published version, key and content hash must all match the explicit pin."""
    path, _ = snapshot
    arena, _, overrides = arena_http
    overrides["z"] = override
    original = path.read_bytes()

    with pytest.raises(ArenaAPIError, match=error):
        load_arena_dataset_snapshot(arena, str(path), "stream", "responses")

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda p: p.update(stream_id="other"), "Stream mismatch"),
        (lambda p: p.update(format_version=True), "format_version"),
        (lambda p: p.update(count=True), "count"),
        (lambda p: p.update(count=3), "count"),
        (lambda p: p.update(errors=["incomplete"]), "complete non-empty"),
        (lambda p: p.update(rows=[]), "complete non-empty"),
        (lambda p: p["rows"].__setitem__(1, p["rows"][0]), "Duplicate"),
        (lambda p: p["rows"][0].pop("arena_task_type"), "task types"),
        (lambda p: p["rows"][0].pop("env_content_hash"), "content hashes"),
        (
            lambda p: p["rows"][0].update(data_id="env:z@latest"),
            "exact Env versions",
        ),
        (lambda p: p["rows"][0].update(data_id="z@old"), "env:key@version"),
    ],
)
def test_snapshot_invalid_input_fails_before_http(
    snapshot, arena_http, mutation, error
):
    """Reject incomplete or ambiguous snapshots before consulting publications."""
    path, payload = snapshot
    arena, requests, _ = arena_http
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ArenaAPIError, match=error):
        load_arena_dataset_snapshot(arena, str(path), "stream", "responses")

    assert not requests
    assert path.read_bytes() == original


def test_resolver_snapshot_keeps_frozen_reward_and_default_drift_guard(
    snapshot, arena_http, tmp_path, monkeypatch
):
    """Explicit recovery keeps old IDs; default live selection still rejects drift."""
    path, payload = snapshot
    arena, requests, _ = arena_http
    reward_path = tmp_path / "rewards.json"
    reward_path.write_text(
        json.dumps(
            {
                "stream_id": "stream",
                "count": 2,
                "items": [
                    {
                        "data_id": row["data_id"],
                        "reward_ref": {"key": "grader", "version": "original"},
                    }
                    for row in payload["rows"]
                ],
            }
        ),
        encoding="utf-8",
    )
    original = reward_path.read_bytes()
    config = ArenaStreamConfig(
        name="tasks",
        stream_id="stream",
        llm_protocol="responses",
        reward_mapping_file=str(reward_path),
        dataset_snapshot_file=str(path),
    )
    live_calls = []

    def live_rows(*args, **kwargs):
        live_calls.append(True)
        return [{"data_id": "env:z@new"}, {"data_id": "env:a@new"}]

    monkeypatch.setattr(arena, "get_all_dataset_rows", live_rows)

    _, rows = _resolve_arena_stream(arena, config)

    assert [row["data_id"] for row in rows] == ["env:z@old", "env:a@old"]
    assert all(row["reward_ref_version"] == "original" for row in rows)
    assert "/openapi/v1/rewards/grader/versions/original" in requests
    assert not live_calls
    with pytest.raises(ArenaAPIError, match="dataset coverage mismatch"):
        _resolve_arena_stream(arena, replace(config, dataset_snapshot_file=""))
    assert live_calls == [True]
    assert reward_path.read_bytes() == original


@pytest.mark.parametrize("value", [["wrong"], [], 0, False, None])
def test_config_non_string_snapshot_path_is_rejected(value):
    """Do not coerce structured snapshot configuration into a filesystem path."""
    with pytest.raises(ValueError, match="dataset_snapshot_file"):
        parse_arena_stream_config(
            {"name": "tasks", "stream_id": "stream", "dataset_snapshot_file": value}
        )
