"""Unit tests for Arena multi-Stream terminal-reward routing."""

import asyncio
import base64
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import (
    ArenaAPIError,
    ArenaOpenAPIClient,
    ArenaTaskFailedError,
    ArenaTaskResult,
)
from examples.swe.arena_config import (
    build_weighted_arena_rows,
    load_arena_stream_configs,
    parse_arena_stream_config,
)
from examples.swe.arena_preflight import _arena_json, _recent_health, validate_streams
from examples.swe.train_swe_rl import get_arena_dataset, get_arena_mixture_dataset
from examples.swe.utils import ArenaRewardRefConfig, ArenaStreamConfig

from areal.utils import stats_tracker

REPO_ROOT = Path(__file__).resolve().parents[1]
MULTI_STREAM_CONFIG = REPO_ROOT / "examples/swe/arena_multi_stream.yaml"


def test_load_arena_stream_configs_file_parses_reward_specs(tmp_path):
    """File-backed entries should retain independent reward and routing config."""
    streams_file = tmp_path / "streams.yaml"
    streams_file.write_text(
        """
streams:
  - name: astra
    stream_id: stream-astra
    sampling_weight: 3
    harness: claude-code@1.0.0
    expected_reward_ref:
      key: astrocode-bench-reward
      version: 1.0.14
    reward_threshold: 0.98
    reward_transform_fn: examples.swe.reward_transforms.astra_partial_reward
  - name: tbench
    stream_id: stream-tbench
    sampling_weight: 1
    harness: codex@2.0.0
    expected_reward_ref:
      key: tb2-reward
      version: 4.0.0
""".strip(),
        encoding="utf-8",
    )

    streams = load_arena_stream_configs(
        {
            "arena_streams_file": str(streams_file),
            "arena_task_envs": {"COMMON": "1"},
        }
    )

    assert [stream.name for stream in streams] == ["astra", "tbench"]
    assert streams[0].expected_reward_ref == ArenaRewardRefConfig(
        key="astrocode-bench-reward", version="1.0.14"
    )
    assert streams[0].task_envs == {"COMMON": "1"}
    assert streams[1].reward_transform_fn == ""


def test_load_arena_stream_configs_parses_inline_yaml():
    """The launcher can inject Stream configuration without a runtime file path."""
    streams_yaml = """
streams:
  - name: astra
    stream_id: stream-astra
    sampling_weight: 3
    harness: claude-code@1.0.0
    expected_reward_ref:
      key: astrocode-bench-reward
      version: 1.0.14
""".strip()
    streams = load_arena_stream_configs(
        {"arena_streams_yaml_b64": base64.b64encode(streams_yaml.encode()).decode()}
    )

    assert len(streams) == 1
    assert streams[0].name == "astra"
    assert streams[0].sampling_weight == 3.0
    assert streams[0].expected_reward_ref == ArenaRewardRefConfig(
        key="astrocode-bench-reward", version="1.0.14"
    )


def test_arena_stream_profile_preserves_raw_rewards(monkeypatch):
    """Streams without a transform should retain their raw Arena rewards."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    streams = yaml.safe_load(MULTI_STREAM_CONFIG.read_text())["econfig"][
        "arena_streams"
    ]
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_streams": streams,
        }
    )
    data = {
        "arena_stream_name": "first",
        "stream_id": "your-first-stream",
    }

    transformed = [
        workflow._transform_reward(raw_reward, data) for raw_reward in (0.2, 0.75, 0.98)
    ]

    assert transformed == pytest.approx([0.2, 0.75, 0.98])


def test_load_arena_stream_configs_rejects_invalid_base64():
    """Corrupted launcher transport must fail instead of becoming legacy config."""
    with pytest.raises(ValueError, match="not valid base64 UTF-8"):
        load_arena_stream_configs({"arena_streams_yaml_b64": "not-base64%%%"})


def test_preflight_openapi_maps_read_only_operations(monkeypatch):
    """Multi-Stream preflight should not require an external Arena CLI."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    real_client = httpx.Client
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(200, json={"path": request.url.path})

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("examples.swe.arena_preflight.httpx.Client", client_factory)

    _arena_json(
        "https://arena.example",
        ["stream", "get", "stream/a"],
        retries=0,
    )
    _arena_json(
        "https://arena.example",
        ["harness", "version-get", "claude/code", "1.2.4"],
        retries=0,
    )
    _arena_json(
        "https://arena.example",
        ["stream", "dataset", "stream/a", "--limit", "1"],
        retries=0,
    )
    _arena_json(
        "https://arena.example",
        [
            "stream",
            "tasks",
            "stream/a",
            "--limit",
            "100",
            "--offset",
            "23",
        ],
        retries=0,
    )

    assert [
        (request.method, request.url.raw_path.decode().split("?", 1)[0])
        for request in requests
    ] == [
        ("GET", "/openapi/v1/streams/stream%2Fa"),
        ("GET", "/openapi/v1/harnesses/claude%2Fcode/versions/1.2.4"),
        ("POST", "/openapi/v1/streams/stream%2Fa/dataset"),
        ("GET", "/openapi/v1/streams/stream%2Fa/tasks"),
    ]
    assert dict(requests[-1].url.params) == {"limit": "100", "offset": "23"}


def test_preflight_retries_known_transient_403(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    monkeypatch.setattr("examples.swe.arena_preflight.time.sleep", lambda _delay: None)
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                403,
                text="spanner-http-ant-group-watch-all transient rejection",
            )
        return httpx.Response(200, json={"data": {"status": "PUBLISHED"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = _arena_json(
            "https://arena.example",
            ["harness", "version-get", "claude", "1"],
            retries=1,
            client=client,
        )

    assert payload == {"data": {"status": "PUBLISHED"}}
    assert attempts == 2


def test_parse_arena_stream_config_rejects_metric_unsafe_name():
    """Stream names must be safe to embed in metric keys and dump metadata."""
    with pytest.raises(ValueError, match="may only contain"):
        parse_arena_stream_config({"name": "bad/name", "stream_id": "stream-a"})


def test_load_arena_stream_configs_rejects_duplicate_stream_ids():
    """A copied Stream id must not silently receive multiple mixture weights."""
    with pytest.raises(ValueError, match="ids must be unique"):
        load_arena_stream_configs(
            {
                "arena_streams": [
                    {"name": "a", "stream_id": "stream-a"},
                    {"name": "b", "stream_id": "stream-a"},
                ]
            }
        )


def test_build_weighted_arena_rows_uses_attempted_group_weights():
    """An explicit subset should realize weights without repeating rows."""
    streams = [
        ArenaStreamConfig(name="astra", stream_id="a", sampling_weight=3.0),
        ArenaStreamConfig(name="tbench", stream_id="b", sampling_weight=1.0),
    ]
    rows = build_weighted_arena_rows(
        {
            "astra": [
                {"arena_stream_name": "astra", "data_id": f"a-{index}"}
                for index in range(9)
            ],
            "tbench": [
                {"arena_stream_name": "tbench", "data_id": f"b-{index}"}
                for index in range(9)
            ],
        },
        streams,
        epoch_size=12,
    )

    assert Counter(row["arena_stream_name"] for row in rows) == {
        "astra": 9,
        "tbench": 3,
    }
    assert len({row["data_id"] for row in rows}) == 12
    assert [row["arena_stream_name"] for row in rows[:4]] == [
        "astra",
        "astra",
        "astra",
        "tbench",
    ]


def test_build_weighted_arena_rows_equal_weights_concatenate_without_repeats():
    """Equal weights should include each source row exactly once."""
    streams = [
        ArenaStreamConfig(name="large", stream_id="a", sampling_weight=1.0),
        ArenaStreamConfig(name="small", stream_id="b", sampling_weight=1.0),
    ]
    rows = build_weighted_arena_rows(
        {
            "large": [
                {"arena_stream_name": "large", "data_id": f"a-{index}"}
                for index in range(4)
            ],
            "small": [{"arena_stream_name": "small", "data_id": "b-0"}],
        },
        streams,
    )

    assert len(rows) == 5
    assert Counter(row["arena_stream_name"] for row in rows) == {
        "large": 4,
        "small": 1,
    }
    assert Counter(row["data_id"] for row in rows) == {
        "a-0": 1,
        "a-1": 1,
        "a-2": 1,
        "a-3": 1,
        "b-0": 1,
    }
    assert [row["arena_stream_name"] for row in rows[:2]] == [
        "large",
        "small",
    ]


def test_build_weighted_arena_rows_equal_weights_keep_all_unique_ids():
    streams = [
        ArenaStreamConfig(name="a", stream_id="a", sampling_weight=1.0),
        ArenaStreamConfig(name="b", stream_id="b", sampling_weight=1.0),
        ArenaStreamConfig(name="c", stream_id="c", sampling_weight=1.0),
    ]
    rows_by_stream = {
        "a": [
            {"arena_stream_name": "a", "data_id": f"a-{index}"} for index in range(56)
        ],
        "b": [
            {"arena_stream_name": "b", "data_id": f"b-{index}"} for index in range(441)
        ],
        "c": [
            {"arena_stream_name": "c", "data_id": f"c-{index}"} for index in range(20)
        ],
    }

    rows = build_weighted_arena_rows(rows_by_stream, streams, size_multiple=4)

    assert len(rows) == 517
    assert Counter(row["arena_stream_name"] for row in rows) == {
        "a": 56,
        "b": 441,
        "c": 20,
    }
    assert len({row["data_id"] for row in rows}) == 517
    assert [row["arena_stream_name"] for row in rows[:4]] == ["a", "b", "c", "a"]


def test_build_weighted_arena_rows_rejects_oversized_static_epoch():
    """An explicit static epoch may not repeat source rows."""
    streams = [
        ArenaStreamConfig(name="large", stream_id="a", sampling_weight=1.0),
        ArenaStreamConfig(name="small", stream_id="b", sampling_weight=1.0),
    ]
    with pytest.raises(ValueError, match="cannot exceed the 5 unique source rows"):
        build_weighted_arena_rows(
            {
                "large": [
                    {"arena_stream_name": "large", "data_id": f"a-{index}"}
                    for index in range(4)
                ],
                "small": [{"arena_stream_name": "small", "data_id": "b-0"}],
            },
            streams,
            epoch_size=6,
        )


def test_build_weighted_arena_rows_applies_stream_level_weight():
    """The full-union default should never repeat rows for a Stream weight."""
    streams = [
        ArenaStreamConfig(name="weighted", stream_id="a", sampling_weight=2.0),
        ArenaStreamConfig(name="plain", stream_id="b", sampling_weight=1.0),
    ]
    rows_by_stream = {
        "weighted": [
            {"arena_stream_name": "weighted", "data_id": "a-0"},
            {"arena_stream_name": "weighted", "data_id": "a-1"},
        ],
        "plain": [
            {"arena_stream_name": "plain", "data_id": f"b-{index}"}
            for index in range(3)
        ],
    }

    rows = build_weighted_arena_rows(rows_by_stream, streams)

    assert Counter(row["arena_stream_name"] for row in rows) == {
        "weighted": 2,
        "plain": 3,
    }
    assert Counter(row["data_id"] for row in rows) == {
        "a-0": 1,
        "a-1": 1,
        "b-0": 1,
        "b-1": 1,
        "b-2": 1,
    }


def test_build_weighted_arena_rows_common_weight_scale_is_invariant():
    """Only relative Stream weights should affect the virtual mixture."""
    rows_by_stream = {
        "a": [{"arena_stream_name": "a", "data_id": "a-0"}],
        "b": [
            {"arena_stream_name": "b", "data_id": "b-0"},
            {"arena_stream_name": "b", "data_id": "b-1"},
        ],
    }

    baseline = build_weighted_arena_rows(
        rows_by_stream,
        [
            ArenaStreamConfig(name="a", stream_id="a", sampling_weight=0.5),
            ArenaStreamConfig(name="b", stream_id="b", sampling_weight=1.0),
        ],
    )
    scaled = build_weighted_arena_rows(
        rows_by_stream,
        [
            ArenaStreamConfig(name="a", stream_id="a", sampling_weight=1.0),
            ArenaStreamConfig(name="b", stream_id="b", sampling_weight=2.0),
        ],
    )

    assert scaled == baseline


def test_build_weighted_arena_rows_rejects_mass_overflow():
    """Finite per-Stream weights must not produce an infinite aggregate mass."""
    with pytest.raises(ValueError, match="overflowed the mixture mass"):
        build_weighted_arena_rows(
            {
                "a": [{"arena_stream_name": "a", "data_id": "a-0"}],
                "b": [{"arena_stream_name": "b", "data_id": "b-0"}],
            },
            [
                ArenaStreamConfig(name="a", stream_id="a", sampling_weight=1e308),
                ArenaStreamConfig(name="b", stream_id="b", sampling_weight=1e308),
            ],
        )


def test_build_weighted_arena_rows_does_not_pad_default_to_training_batch():
    streams = [
        ArenaStreamConfig(name="a", stream_id="a", sampling_weight=1.0),
        ArenaStreamConfig(name="b", stream_id="b", sampling_weight=1.0),
        ArenaStreamConfig(name="c", stream_id="c", sampling_weight=1.0),
    ]
    rows_by_stream = {
        "a": [{"arena_stream_name": "a", "data_id": "a-0"}],
        "b": [{"arena_stream_name": "b", "data_id": f"b-{i}"} for i in range(5)],
        "c": [{"arena_stream_name": "c", "data_id": "c-0"}],
    }

    rows = build_weighted_arena_rows(rows_by_stream, streams, size_multiple=4)

    assert len(rows) == 7
    assert Counter(row["arena_stream_name"] for row in rows) == {
        "a": 1,
        "b": 5,
        "c": 1,
    }
    assert len({row["data_id"] for row in rows}) == 7


def test_get_arena_mixture_dataset_resolves_and_mixes_streams(monkeypatch):
    """Dataset rows should carry Stream and reward-ref provenance."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    configured = [
        ArenaStreamConfig(
            name="astra",
            stream_id="stream-a",
            sampling_weight=2.0,
            harness="claude-code@1.0.0",
            expected_reward_ref=ArenaRewardRefConfig("reward-a", "1.0.0"),
        ),
        ArenaStreamConfig(
            name="tbench",
            stream_id="stream-b",
            sampling_weight=1.0,
            harness="codex@2.0.0",
            expected_reward_ref=ArenaRewardRefConfig("reward-b", "2.0.0"),
        ),
    ]

    def resolve_stream(_self, stream_id, **_kwargs):
        suffix = stream_id[-1]
        return {
            "stream_id": stream_id,
            "default_reward_ref": {
                "key": f"reward-{suffix}",
                "version": "1.0.0" if suffix == "a" else "2.0.0",
            },
        }

    def dataset_rows(_self, stream_id, llm_protocol, **_kwargs):
        return [
            {
                "stream_id": stream_id,
                "data_id": f"{stream_id}-{index}",
                "llm_protocol": llm_protocol,
                "arena_task_type": "swe",
            }
            for index in range(6)
        ]

    monkeypatch.setattr(ArenaOpenAPIClient, "resolve_stream", resolve_stream)
    monkeypatch.setattr(ArenaOpenAPIClient, "get_all_dataset_rows", dataset_rows)
    econfig = SimpleNamespace(
        arena_base_url="https://arena.example",
        arena_request_timeout=1.0,
        arena_request_retries=0,
        arena_streams=configured,
        arena_streams_file="",
        arena_mixture_epoch_size=9,
    )

    dataset, resolved = get_arena_mixture_dataset(econfig)

    assert Counter(dataset["arena_stream_name"]) == {"astra": 6, "tbench": 3}
    assert set(dataset["reward_ref_key"]) == {"reward-a", "reward-b"}
    assert [stream.llm_protocol for stream in resolved] == ["anthropic", "responses"]


def test_get_arena_dataset_retains_legacy_stream_id_return(monkeypatch):
    """The original single-Stream helper remains scalar-return compatible."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setattr(
        ArenaOpenAPIClient,
        "resolve_stream",
        lambda _self, stream_id, **_kwargs: {
            "stream_id": stream_id,
            "default_reward_ref": None,
        },
    )
    monkeypatch.setattr(
        ArenaOpenAPIClient,
        "get_all_dataset_rows",
        lambda _self, stream_id, llm_protocol, **_kwargs: [
            {
                "stream_id": stream_id,
                "data_id": "data-1",
                "llm_protocol": llm_protocol,
                "arena_task_type": "swe",
            }
        ],
    )
    econfig = SimpleNamespace(
        arena_base_url="https://arena.example",
        arena_request_timeout=1.0,
        arena_request_retries=0,
        arena_streams=[],
        arena_streams_file="",
        arena_mixture_epoch_size=0,
        stream_id="stream-a",
        arena_harness="",
        arena_llm_protocol="chat_completions",
        arena_task_envs={},
        arena_reward_threshold="0.98",
        arena_reward_transform_fn="",
    )

    dataset, stream_id = get_arena_dataset(econfig)

    assert len(dataset) == 1
    assert stream_id == "stream-a"


def test_multi_stream_preflight_pins_reward_ref_and_checks_health(
    monkeypatch, tmp_path
):
    """Preflight should validate every routing tuple without exposing env values."""
    streams_file = tmp_path / "streams.yaml"
    streams_file.write_text(
        """
streams:
  - name: astra
    stream_id: stream-a
    harness: claude@1
    task_envs:
      PRIVATE_VALUE: hidden
    expected_reward_ref:
      key: reward-a
      version: "1"
""".strip(),
        encoding="utf-8",
    )

    def arena_json(_base_url, args, *, retries, client):
        assert retries == 0
        assert isinstance(client, httpx.Client)
        if args[:2] == ["stream", "get"]:
            return {
                "data": {
                    "stream": {
                        "stream_id": "stream-a",
                        "status": "ACTIVE",
                        "default_reward_ref": {"key": "reward-a", "version": "1"},
                    }
                }
            }
        if args[:2] == ["harness", "version-get"]:
            return {"data": {"status": "PUBLISHED"}}
        if args[:2] == ["stream", "dataset"]:
            return {"data": {"total": 10}}
        if args[:2] == ["stream", "tasks"]:
            assert args[-4:] == ["--limit", "100", "--offset", "0"]
            return {
                "data": [
                    {
                        "task": {
                            "status": "DONE",
                            "harness_ref": {"key": "claude", "version": "1"},
                            "reward_ref": {"key": "reward-a", "version": "1"},
                        }
                    }
                ]
            }
        raise AssertionError(args)

    monkeypatch.setattr("examples.swe.arena_preflight._arena_json", arena_json)

    summaries = validate_streams(
        str(streams_file),
        base_url="https://arena.example",
        retries=0,
    )

    assert summaries[0]["expected_reward_ref"] == {
        "key": "reward-a",
        "version": "1",
    }
    assert summaries[0]["llm_protocol"] == "anthropic"
    assert summaries[0]["task_envs"] == ["PRIVATE_VALUE"]
    assert summaries[0]["recent_failure_rate"] == 0.0


def test_preflight_health_ignores_tasks_without_exact_provenance():
    """Health must not mix tasks from another or unidentifiable grader tuple."""
    rows = [
        {"status": "DONE"},
        {
            "status": "DONE",
            "harness_ref": {"key": "claude", "version": "1"},
        },
        {
            "status": "FAILED",
            "harness_ref": {"key": "claude", "version": "1"},
            "reward_ref": {"key": "reward-a", "version": "1"},
        },
    ]

    terminal, failures = _recent_health(
        rows,
        harness="claude@1",
        reward_ref=ArenaRewardRefConfig("reward-a", "1"),
    )

    assert (terminal, failures) == (1, 1)


def test_workflow_routes_harness_transform_and_raw_metrics(monkeypatch, tmp_path):
    """A row should select all rollout and reward settings from its Stream."""
    stats_tracker.export_all(reset=True)
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "llm-key")
    transforms = {
        "transform.a": lambda reward, _data: reward * 0.1,
        "transform.b": lambda reward, _data: reward + 0.25,
    }
    monkeypatch.setattr(
        "examples.swe.arena_agent.import_from_string", transforms.__getitem__
    )
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_result_dump_dir": str(tmp_path / "arena-results"),
            "arena_streams": [
                {
                    "name": "a",
                    "stream_id": "stream-a",
                    "harness": "claude@1",
                    "reward_transform_fn": "transform.a",
                },
                {
                    "name": "b",
                    "stream_id": "stream-b",
                    "harness": "codex@2",
                    "task_envs": {"MODE": "b"},
                    "expected_reward_ref": {"key": "reward-b", "version": "2"},
                    "reward_transform_fn": "transform.b",
                },
            ],
        }
    )
    launched = {}

    async def resolve_stream(stream_id, **_kwargs):
        return {
            "stream_id": stream_id,
            "default_reward_ref": {"key": "reward-b", "version": "2"},
        }

    monkeypatch.setattr(
        workflow.client,
        "resolve_stream_async",
        resolve_stream,
    )

    async def register(**_kwargs):
        return "https://arena.example/api", "model-b"

    async def launch(**kwargs):
        launched.update(kwargs)
        kwargs["on_launch_success"]("task-b")
        return ArenaTaskResult(
            task_id="task-b",
            status="DONE",
            score=0.5,
            raw={"exit_code": 0, "reward_txt": "ok"},
            trace_id="trace-b",
        )

    async def delete(*_args, **_kwargs):
        return None

    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", register)
    monkeypatch.setattr(workflow.client, "launch_one_task_result", launch)
    monkeypatch.setattr(workflow.client, "delete_llm_proxy_async", delete)
    data = {
        "arena_stream_name": "b",
        "stream_id": "stream-b",
        "data_id": "data-b",
        "reward_ref_key": "reward-b",
        "reward_ref_version": "2",
    }

    async def run_workflow():
        transport = httpx.MockTransport(lambda _: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as http_client:
            reward = await workflow.run(
                data,
                base_url="http://rollout-proxy",
                api_key="session-key",
                arena_http_client=http_client,
            )
            await workflow.persist_episode_result(data, reward)
            workflow.record_episode_metrics(data, reward)
            return reward

    reward = asyncio.run(run_workflow())

    assert reward == pytest.approx(0.75)
    assert launched["stream_id"] == "stream-b"
    assert launched["harness"] == "codex@2"
    assert launched["task_envs"] == {"MODE": "b"}
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena_score"] == pytest.approx(0.5)
    assert stats["rollout/training_score"] == pytest.approx(0.75)
    assert stats["rollout/arena_raw_present"] == 1.0
    assert stats["rollout/stream/b/arena_score"] == pytest.approx(0.5)
    dump_files = list((tmp_path / "arena-results").glob("arena_results_*.jsonl"))
    assert len(dump_files) == 1
    dump_record = json.loads(dump_files[0].read_text(encoding="utf-8"))
    assert dump_record["raw"] == {"exit_code": 0, "reward_txt": "ok"}
    assert dump_record["expected_reward_ref"] == {
        "key": "reward-b",
        "version": "2",
    }
    assert dump_files[0].stat().st_mode & 0o777 == 0o600


def test_workflow_rejects_row_reward_ref_mismatch(monkeypatch):
    """A row produced under another grader version must fail before launch."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "llm-key")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_streams": [
                {
                    "name": "a",
                    "stream_id": "stream-a",
                    "expected_reward_ref": {"key": "reward-a", "version": "1"},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="reward_ref mismatch"):
        asyncio.run(
            workflow.run(
                {
                    "arena_stream_name": "a",
                    "stream_id": "stream-a",
                    "data_id": "data-a",
                    "reward_ref_key": "reward-a",
                    "reward_ref_version": "2",
                },
                base_url="http://rollout-proxy",
                api_key="session-key",
            )
        )


def test_workflow_rejects_live_reward_ref_drift(monkeypatch):
    """A grader default changed after dataset discovery must block launch."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "llm-key")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_streams": [
                {
                    "name": "a",
                    "stream_id": "stream-a",
                    "llm_protocol": "anthropic",
                    "expected_reward_ref": {"key": "reward-a", "version": "1"},
                }
            ],
        }
    )

    async def resolve_stream(_stream_id, **_kwargs):
        return {
            "stream_id": "stream-a",
            "default_reward_ref": {"key": "reward-a", "version": "2"},
        }

    monkeypatch.setattr(
        workflow.client,
        "resolve_stream_async",
        resolve_stream,
    )

    with pytest.raises(ArenaAPIError, match="drifted before launch"):
        asyncio.run(
            workflow.run(
                {
                    "arena_stream_name": "a",
                    "stream_id": "stream-a",
                    "data_id": "data-a",
                    "llm_protocol": "anthropic",
                    "reward_ref_key": "reward-a",
                    "reward_ref_version": "1",
                },
                base_url="http://rollout-proxy",
                api_key="session-key",
            )
        )


@pytest.mark.parametrize(
    ("status", "context_overflow", "interaction_count", "expected"),
    [
        ("HARNESS_FAILED", True, 1, "model_failure_zero"),
        ("HARNESS_FAILED", True, 0, "unknown_failure_reject"),
        ("HARNESS_FAILED", False, 1, "unknown_failure_reject"),
        ("SETUP_FAILED", True, 1, "system_failure_reject"),
        ("TIMEOUT", False, 1, "model_failure_zero"),
        ("TIMEOUT", True, 0, "unknown_failure_reject"),
        ("NO_OUTPUT", False, 1, "model_failure_zero"),
        ("NO_OUTPUT", False, 0, "unknown_failure_reject"),
    ],
)
def test_arena_failure_classifier_is_conservative(
    status, context_overflow, interaction_count, expected
):
    """Only a typed, recoverable Arena model overflow may become reward zero."""
    error = ArenaTaskFailedError(task_id="task-1", status=status)

    disposition = ArenaStreamAgentWorkflow.classify_proxy_failure(
        error,
        context_overflow=context_overflow,
        interaction_count=interaction_count,
    )

    assert disposition == expected


def test_arena_failure_classifier_rejects_non_terminal_api_error():
    """Transport and result-processing errors are system failures."""
    disposition = ArenaStreamAgentWorkflow.classify_proxy_failure(
        ArenaAPIError("request failed"),
        context_overflow=True,
        interaction_count=1,
    )

    assert disposition == "system_failure_reject"


def test_arena_failure_classifier_keeps_explicit_claude_agent_phase_failure():
    """A Harness-attributed Claude agent failure is a trainable zero reward."""
    error = ArenaTaskFailedError(
        task_id="task-1",
        status="HARNESS_FAILED",
        result=ArenaTaskResult(
            task_id="task-1",
            status="HARNESS_FAILED",
            score=0.0,
            raw={
                "error": "harness: harness agent phase exited with code 1: "
                "harness: agent phase error: claude reported error:"
            },
        ),
    )

    disposition = ArenaStreamAgentWorkflow.classify_proxy_failure(
        error,
        context_overflow=False,
        interaction_count=3,
    )

    assert disposition == "model_failure_zero"


def test_arena_failure_classifier_rejects_ambiguous_harness_failure_raw():
    """Unattributed Harness text must not silently become training data."""
    error = ArenaTaskFailedError(
        task_id="task-1",
        status="HARNESS_FAILED",
        result=ArenaTaskResult(
            task_id="task-1",
            status="HARNESS_FAILED",
            score=0.0,
            raw={"error": "harness failed"},
        ),
    )

    disposition = ArenaStreamAgentWorkflow.classify_proxy_failure(
        error,
        context_overflow=False,
        interaction_count=3,
    )

    assert disposition == "unknown_failure_reject"


def test_workflow_audits_recovered_failure_with_final_zero_reward(
    monkeypatch, tmp_path
):
    """The audit row must agree with proxy-level model-failure recovery."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_result_dump_dir": str(tmp_path / "arena-results"),
            "stream_id": "stream-a",
        }
    )
    data = {"stream_id": "stream-a", "data_id": "data-a"}
    error = ArenaTaskFailedError(
        task_id="task-a",
        status="HARNESS_FAILED",
        result=ArenaTaskResult(
            task_id="task-a",
            status="HARNESS_FAILED",
            score=0.0,
            raw={"failure": "context_overflow"},
        ),
    )

    async def persist_result() -> None:
        await workflow.record_failure_disposition(data, error, "model_failure_zero")
        await workflow.persist_episode_result(data, 0.0)

    asyncio.run(persist_result())

    dump_file = next((tmp_path / "arena-results").glob("arena_results_*.jsonl"))
    records = [json.loads(line) for line in dump_file.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "HARNESS_FAILED"
    assert records[0]["arena_score"] == 0.0
    assert records[0]["training_score"] == 0.0
    assert records[0]["raw"] == {"failure": "context_overflow"}


def test_workflow_audits_rejected_failure_without_training_reward(
    monkeypatch, tmp_path
):
    """System failures remain distinguishable from trainable zero rewards."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_result_dump_dir": str(tmp_path / "arena-results"),
            "stream_id": "stream-a",
        }
    )
    data = {"stream_id": "stream-a", "data_id": "data-a"}
    error = ArenaTaskFailedError(
        task_id="task-a",
        status="SETUP_FAILED",
        result=ArenaTaskResult(
            task_id="task-a",
            status="SETUP_FAILED",
            score=0.0,
            raw={"failure": "sandbox"},
        ),
    )

    asyncio.run(
        workflow.record_failure_disposition(data, error, "system_failure_reject")
    )

    dump_file = next((tmp_path / "arena-results").glob("arena_results_*.jsonl"))
    record = json.loads(dump_file.read_text())
    assert record["status"] == "SETUP_FAILED"
    assert record["training_score"] is None


def test_workflow_audits_terminal_result_when_transform_fails(monkeypatch, tmp_path):
    """A bad local transform must not discard an already-computed Arena result."""
    stats_tracker.export_all(reset=True)
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "llm-key")
    monkeypatch.setattr(
        "examples.swe.arena_agent.import_from_string",
        lambda _path: lambda _reward, _data: float("nan"),
    )
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_result_dump_dir": str(tmp_path / "arena-results"),
            "arena_streams": [
                {
                    "name": "a",
                    "stream_id": "stream-a",
                    "reward_transform_fn": "transform.bad",
                }
            ],
        }
    )

    async def register(**_kwargs):
        return "https://arena.example/api", "model-a"

    async def launch(**kwargs):
        kwargs["on_launch_success"]("task-a")
        return ArenaTaskResult(
            task_id="task-a",
            status="DONE",
            score=0.5,
            raw={"grader": "complete"},
        )

    async def delete(*_args, **_kwargs):
        return None

    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", register)
    monkeypatch.setattr(workflow.client, "launch_one_task_result", launch)
    monkeypatch.setattr(workflow.client, "delete_llm_proxy_async", delete)

    with pytest.raises(ValueError, match="non-finite"):
        asyncio.run(
            workflow.run(
                {
                    "arena_stream_name": "a",
                    "stream_id": "stream-a",
                    "data_id": "data-a",
                },
                base_url="http://rollout-proxy",
                api_key="session-key",
            )
        )

    dump_file = next((tmp_path / "arena-results").glob("arena_results_*.jsonl"))
    dump_record = json.loads(dump_file.read_text(encoding="utf-8"))
    assert dump_record["raw"] == {"grader": "complete"}
    assert dump_record["training_score"] is None
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena/terminal_success"] == 1.0
    assert stats["rollout/arena/result_processing_success"] == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"astrocode_report": {"case": {"resolved": True}}},
        {"exit_code": 0, "reward_txt": "1"},
        ["grader", "specific"],
    ],
)
def test_task_result_preserves_heterogeneous_raw(monkeypatch, raw):
    """Different grader payloads should not affect top-level score parsing."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "task-1", "status": "PENDING"})
        return httpx.Response(
            200,
            json={
                "task_id": "task-1",
                "status": "DONE",
                "score": 0.25,
                "raw": raw,
                "artifacts_uri": "oss://artifact",
                "trace_id": "trace-1",
                "computed_at": "2026-08-17T00:00:00Z",
            },
        )

    async def launch():
        client = ArenaOpenAPIClient(base_url="https://arena.example", poll_interval=0.0)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            return await client.launch_one_task_result(
                stream_id="stream-1",
                data_id="data-1",
                model_name="model-1",
                proxy_base_url="http://proxy",
                proxy_api_key="key",
                client=http_client,
            )

    result = asyncio.run(launch())

    assert result.score == pytest.approx(0.25)
    assert result.raw == raw
    assert result.trace_id == "trace-1"


@pytest.mark.parametrize(
    "payload",
    [
        0.25,
        {"data": {"status": "DONE", "score": 0.25}},
        {"result": {"output": {"status": "DONE", "score": 0.25}}},
    ],
)
def test_task_result_accepts_legacy_control_envelopes(monkeypatch, payload):
    """Known control envelopes remain compatible without inspecting raw."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def launch():
        client = ArenaOpenAPIClient(base_url="https://arena.example")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            return await client.launch_one_task_result(
                stream_id="stream-1",
                data_id="data-1",
                model_name="model-1",
                proxy_base_url="http://proxy",
                proxy_api_key="key",
                client=http_client,
            )

    result = asyncio.run(launch())

    assert result.status == "DONE"
    assert result.score == pytest.approx(0.25)


def test_result_dump_rejects_symlinked_directory(monkeypatch, tmp_path):
    """A sudo-launched audit writer must not follow attacker-controlled links."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    stats_tracker.export_all(reset=True)
    target = tmp_path / "target"
    target.mkdir()
    dump_link = tmp_path / "arena-results"
    dump_link.symlink_to(target, target_is_directory=True)
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_result_dump_dir": str(dump_link),
            "stream_id": "stream-a",
        }
    )

    asyncio.run(
        workflow._dump_task_result(
            {"stream_id": "stream-a", "data_id": "data-a"},
            workflow.stream_configs["default"],
            ArenaTaskResult(task_id="task-a", status="DONE", score=1.0),
            training_score=1.0,
        )
    )

    assert list(target.iterdir()) == []
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena/result_dump_success"] == 0.0


def test_task_result_does_not_extract_score_from_raw(monkeypatch):
    """A grader-local score field must not replace a missing envelope score."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "task-1", "status": "PENDING"})
        return httpx.Response(
            200,
            json={
                "task_id": "task-1",
                "status": "DONE",
                "raw": {"score": 1.0, "reward": 1.0},
            },
        )

    async def launch():
        client = ArenaOpenAPIClient(base_url="https://arena.example", poll_interval=0.0)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await client.launch_one_task(
                stream_id="stream-1",
                data_id="data-1",
                model_name="model-1",
                proxy_base_url="http://proxy",
                proxy_api_key="key",
                client=http_client,
            )

    with pytest.raises(ArenaAPIError, match="top-level 'score'"):
        asyncio.run(launch())


def test_failed_task_retains_raw_in_typed_error(monkeypatch):
    """Failure diagnostics should survive without becoming reward zero."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "task-1", "status": "PENDING"})
        return httpx.Response(
            200,
            json={
                "task_id": "task-1",
                "status": "HARNESS_FAILED",
                "score": 0,
                "raw": {"error": "grader failed", "arca": {}},
            },
        )

    async def launch():
        client = ArenaOpenAPIClient(base_url="https://arena.example", poll_interval=0.0)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await client.launch_one_task(
                stream_id="stream-1",
                data_id="data-1",
                model_name="model-1",
                proxy_base_url="http://proxy",
                proxy_api_key="key",
                client=http_client,
            )

    with pytest.raises(ArenaTaskFailedError) as exc_info:
        asyncio.run(launch())

    assert exc_info.value.result is not None
    assert exc_info.value.result.raw == {"error": "grader failed", "arca": {}}
