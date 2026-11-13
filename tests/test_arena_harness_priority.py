"""Harness precedence stays consistent across startup and rollout workers."""

import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml

from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient
from examples.swe.arena_preflight import validate_streams


@pytest.fixture(params=["override", "override_protocol", "fallback"])
def harness_case(request, tmp_path, monkeypatch):
    """Reuse a frozen snapshot with two different task Harnesses."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("ARENA_LLM_BASE_URL", "https://arena.example/api")
    entry = {
        "name": "tasks",
        "stream_id": "stream",
        "reward_mapping_file": str(tmp_path / "map.json"),
    }
    if request.param != "fallback":
        entry["harness"] = "codex@override"
    if request.param == "override_protocol":
        entry["llm_protocol"] = "chat_completions"
    items = [
        {
            "data_id": f"env:task-{index}@1",
            "reward_ref": {"key": "grader", "version": str(index)},
            "harness_ref": {"key": key, "version": "task"},
        }
        for index, key in enumerate(("claude-code", "codex"))
    ]
    stream = {
        "stream_id": "stream",
        "status": "ACTIVE",
        "default_harness_ref": {"key": "remote", "version": "default"},
    }
    monkeypatch.setattr(
        ArenaOpenAPIClient, "resolve_stream", lambda *_args, **_kwargs: stream
    )

    def dataset_rows(_self, stream_id, llm_protocol="chat_completions", **_kwargs):
        return [
            {
                "data_id": item["data_id"],
                "stream_id": stream_id,
                "llm_protocol": llm_protocol,
            }
            for item in items
        ]

    monkeypatch.setattr(ArenaOpenAPIClient, "get_all_dataset_rows", dataset_rows)
    monkeypatch.setattr(
        ArenaOpenAPIClient,
        "get_published_version",
        lambda *_args, **_kwargs: {"status": "PUBLISHED"},
    )
    return SimpleNamespace(
        entry=entry,
        items=items,
        stream=stream,
        tmp_path=tmp_path,
        mode=request.param,
    )


def write_snapshot(case, with_harnesses):
    """Write a preexisting snapshot and return its original bytes."""
    items = [
        {
            key: value
            for key, value in item.items()
            if with_harnesses or key != "harness_ref"
        }
        for item in case.items
    ]
    path = case.tmp_path / "map.json"
    path.write_text(
        json.dumps({"stream_id": "stream", "count": len(items), "items": items}),
        encoding="utf-8",
    )
    return path.read_bytes()


def expected_routing(case, with_harnesses):
    """Describe the public precedence contract for both dataset rows."""
    if case.mode != "fallback":
        protocol = (
            "chat_completions" if case.mode == "override_protocol" else "responses"
        )
        return [("codex@override", protocol)] * 2
    if with_harnesses:
        return [("claude-code@task", "anthropic"), ("codex@task", "responses")]
    return [("common@fallback", "chat_completions")] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("with_harnesses", [False, True])
async def test_dataset_and_rollout_config_harness_priority_is_consistent(
    harness_case, with_harnesses, monkeypatch
):
    """Serialized workers launch and audit the same Harness selected at startup."""
    from examples.swe.arena_agent import ArenaStreamAgentWorkflow
    from examples.swe.train_swe_rl import get_arena_mixture_dataset

    case = harness_case
    snapshot = write_snapshot(case, with_harnesses)
    config = {
        "arena_base_url": "https://arena.example",
        "arena_request_timeout": 1.0,
        "arena_request_retries": 0,
        "arena_poll_interval": 0.0,
        "arena_harness": "common@fallback",
        "arena_streams": [case.entry],
        "arena_result_dump_dir": str(case.tmp_path / "results"),
    }
    dataset, resolved = get_arena_mixture_dataset(SimpleNamespace(**config))
    expected = expected_routing(case, with_harnesses)
    assert [row["llm_protocol"] for row in dataset] == [p for _, p in expected]
    config["arena_streams"] = [asdict(stream) for stream in resolved]
    workflow = ArenaStreamAgentWorkflow(econfig=config)
    register = AsyncMock(wraps=workflow.client.register_llm_proxy_async)
    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", register)
    launched = []

    def handler(request):
        if request.method == "POST" and request.url.path.endswith("/llm/models"):
            return httpx.Response(200, json=json.loads(request.content))
        if request.url.path.endswith("/launch_one_task"):
            launched.append(json.loads(request.content))
            return httpx.Response(202, json={"task_id": "task", "status": "PENDING"})
        if request.url.path.endswith("/tasks/task/result"):
            return httpx.Response(200, json={"status": "DONE", "score": 0.5})
        if request.method == "DELETE":
            return httpx.Response(204)
        raise AssertionError(request.url)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        for row in dataset:
            reward = await workflow.run(
                row,
                base_url="http://rollout-proxy",
                api_key="session-key",
                arena_http_client=client,
            )
            assert reward == 0.5
            await workflow.persist_episode_result(row, reward)

    assert [task["harness"] for task in launched] == [h for h, _ in expected]
    assert [task["reward"] for task in launched] == ["grader@0", "grader@1"]
    assert [call.kwargs["protocol"] for call in register.await_args_list] == [
        p for _, p in expected
    ]
    records = [
        json.loads(line)
        for path in (case.tmp_path / "results").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["expected_harness"] for record in records] == [
        h for h, _ in expected
    ]
    assert (case.tmp_path / "map.json").read_bytes() == snapshot

    stale_row = {**dataset[0], "harness": "stale@1"}
    with pytest.raises(ValueError, match="Harness mismatch"):
        workflow._expected_harness(stale_row, workflow.stream_configs["tasks"])
    stale_protocol = "responses" if expected[0][1] == "anthropic" else "anthropic"
    with pytest.raises(ValueError, match="LLM protocol mismatch"):
        await workflow.run(
            {**dataset[0], "llm_protocol": stale_protocol},
            base_url="http://rollout-proxy",
            api_key="session-key",
        )


@pytest.mark.parametrize("with_harnesses", [False, True])
def test_preflight_config_harness_priority_checks_effective_history(
    harness_case, with_harnesses, monkeypatch
):
    """Preflight validates the override and counts only matching task history."""
    case = harness_case
    snapshot = write_snapshot(case, with_harnesses)
    streams_file = case.tmp_path / "streams.yaml"
    streams_file.write_text(yaml.safe_dump({"streams": [case.entry]}), encoding="utf-8")
    expected = expected_routing(case, with_harnesses)
    checked = []
    history = []
    for item, (harness, _) in zip(case.items, expected, strict=True):
        key, version = harness.split("@")
        history.append(
            {
                "data_id": item["data_id"],
                "status": "DONE",
                "harness_ref": {"key": key, "version": version},
                "reward_ref": item["reward_ref"],
            }
        )
        history.append(
            {
                **history[-1],
                "status": "FAILED",
                "harness_ref": (
                    item["harness_ref"]
                    if case.mode != "fallback" and with_harnesses
                    else {"key": "remote", "version": "default"}
                ),
            }
        )

    def arena_json(_base_url, args, **_kwargs):
        if args[:2] == ["stream", "get"]:
            return {"data": {"stream": case.stream}}
        if args[:2] == ["stream", "dataset"]:
            return {"data": {"total": len(case.items)}}
        if args[:2] == ["harness", "version-get"]:
            checked.append("@".join(args[2:]))
            return {"data": {"status": "PUBLISHED"}}
        if args[:2] == ["stream", "tasks"]:
            return {"data": history}
        raise AssertionError(args)

    monkeypatch.setattr("examples.swe.arena_preflight._arena_json", arena_json)
    summaries = validate_streams(
        str(streams_file),
        base_url="https://arena.example",
        default_harness="common@fallback",
        retries=0,
    )

    assert checked == (
        [] if case.mode == "fallback" and with_harnesses else [expected[0][0]]
    )
    assert summaries[0]["recent_terminal_tasks"] == 2
    assert summaries[0]["recent_failures"] == 0
    assert (case.tmp_path / "map.json").read_bytes() == snapshot

    if case.mode != "fallback":

        def unpublished(_base_url, args, **kwargs):
            if args[:2] == ["harness", "version-get"]:
                return {"data": {"status": "DRAFT"}}
            return arena_json(_base_url, args, **kwargs)

        monkeypatch.setattr("examples.swe.arena_preflight._arena_json", unpublished)
        with pytest.raises(ArenaAPIError, match="not PUBLISHED"):
            validate_streams(
                str(streams_file),
                base_url="https://arena.example",
                default_harness="common@fallback",
                retries=0,
            )
