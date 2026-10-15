# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import pytest

from examples.prefix_replay.agent import (
    PrefixReplayAgent,
    PrefixReplayEmptyAction,
    PrefixReplayLengthTruncated,
)
from examples.prefix_replay.config import (
    build_prefix_replay_workflow_kwargs,
)

from areal.api.cli_args import GenerationHyperparameters
from areal.dataset.prefix_replay import (
    PREFIX_REPLAY_MAX_TOTAL_TOKENS_FIELD,
    PREFIX_REPLAY_METADATA_KEY,
    PrefixReplayIndexedDataset,
    build_prefix_replay_dataset,
    build_prefix_replay_indexed_dataset,
    expand_teacher_trajectory,
    load_prefix_replay_dataset,
    load_prefix_replay_indexed_dataset,
    load_prefix_replay_instance_ids,
)
from areal.dataset.prefix_replay_cache import (
    build_prefix_replay_cache_metadata,
    preflight_prefix_replay_cache,
    read_prefix_replay_cache,
    write_prefix_replay_cache,
)


def _teacher_trajectory() -> dict[str, Any]:
    return {
        "instance_id": "task-1",
        "metadata": {"task": "coding"},
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command":"pytest"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "1 failed",
            },
            {"role": "assistant", "content": "The fix is ready."},
        ],
    }


def test_expand_teacher_trajectory_replays_prefix_without_target_action():
    """Each candidate ends before the selected teacher assistant action."""
    rows = expand_teacher_trajectory(
        _teacher_trajectory(),
        kappa=1.0,
        route_field="task_type",
    )

    assert len(rows) == 2
    assert [message["role"] for message in rows[0]["messages"]] == [
        "system",
        "user",
    ]
    assert [message["role"] for message in rows[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert rows[0]["task_type"] == "coding"
    assert rows[0]["instance_id"] == "task-1"
    assert rows[0][PREFIX_REPLAY_METADATA_KEY] == {
        "assistant_turn_index": 0,
        "total_assistant_turns": 2,
        "sampling_probability": 1.0,
    }
    assert rows[1][PREFIX_REPLAY_METADATA_KEY]["assistant_turn_index"] == 1
    assert rows[1][PREFIX_REPLAY_METADATA_KEY]["sampling_probability"] == 1.0


def test_build_prefix_replay_dataset_step_decay_is_seeded():
    """The first turn is always kept and later Bernoulli draws are reproducible."""
    trajectory = {
        "messages": [
            {"role": "user", "content": "turn 0"},
            {"role": "assistant", "content": "answer 0"},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "answer 2"},
            {"role": "user", "content": "turn 3"},
            {"role": "assistant", "content": "answer 3"},
        ]
    }

    first = build_prefix_replay_dataset([trajectory], kappa=0.6, seed=7)
    second = build_prefix_replay_dataset([trajectory], kappa=0.6, seed=7)

    assert first == second
    assert first[0][PREFIX_REPLAY_METADATA_KEY]["assistant_turn_index"] == 0
    assert all(row["messages"][-1]["role"] in ("user", "tool") for row in first)


def test_build_prefix_replay_indexed_dataset_slices_prefixes_lazily():
    """Compact replay stores one trajectory and lightweight prefix indices."""
    dataset = build_prefix_replay_indexed_dataset(
        [_teacher_trajectory()],
        kappa=1.0,
        route_field="task_type",
    )

    assert isinstance(dataset, PrefixReplayIndexedDataset)
    assert len(dataset.trajectories) == 1
    assert len(dataset.indices) == 2
    assert [spec["message_end"] for spec in dataset.indices] == [2, 4]
    assert [message["role"] for message in dataset[0]["messages"]] == [
        "system",
        "user",
    ]
    assert [message["role"] for message in dataset[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert dataset[0]["task_type"] == "coding"


def test_indexed_prefix_replay_reserves_experience_tokens_per_instance():
    """Experience-aware limits filter prefixes and cap the final trajectory."""

    class PrefixLengthTokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            if tokenize:
                return {"input_ids": list(range(len(messages)))}
            return "x" * 2048

    record = {
        "instance_id": "task-1",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "tool", "content": "observation"},
            {"role": "assistant", "content": "final"},
        ],
    }

    dataset = build_prefix_replay_indexed_dataset(
        [record],
        kappa=1.0,
        tokenizer=PrefixLengthTokenizer(),
        max_length=128,
        max_total_tokens_by_instance_id={"task-1": 3},
    )

    assert len(dataset) == 1
    assert dataset[0]["messages"] == [{"role": "user", "content": "first"}]
    assert (
        dataset[0][PREFIX_REPLAY_METADATA_KEY][PREFIX_REPLAY_MAX_TOTAL_TOKENS_FIELD]
        == 3
    )


def test_indexed_prefix_replay_preserves_global_sampling_stream():
    """Compact preprocessing stays identical to the original expanded builder."""
    records = [_teacher_trajectory() for _ in range(5)]

    expanded = build_prefix_replay_dataset(records, kappa=0.6, seed=19)
    indexed = build_prefix_replay_indexed_dataset(records, kappa=0.6, seed=19)

    assert indexed.to_expanded_rows() == expanded


def test_indexed_sampling_consumes_rng_for_overlength_turns():
    """Length-filtered turns still advance RNG before the next trajectory."""

    class SelectiveLengthTokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            if not tokenize:
                return "x" * 2048
            if messages[0]["content"] == "truncate":
                return {"input_ids": list(range(len(messages)))}
            return {"input_ids": [0]}

    def trajectory(first_content: str) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": first_content},
                {"role": "assistant", "content": "answer-0"},
                {"role": "user", "content": "question-1"},
                {"role": "assistant", "content": "answer-1"},
                {"role": "tool", "content": "observation"},
                {"role": "assistant", "content": "answer-2"},
            ]
        }

    records = [trajectory("truncate"), trajectory("keep")]
    tokenizer = SelectiveLengthTokenizer()
    expanded = build_prefix_replay_dataset(
        records,
        kappa=0.6,
        seed=23,
        tokenizer=tokenizer,
        max_length=1,
    )
    indexed = build_prefix_replay_indexed_dataset(
        records,
        kappa=0.6,
        seed=23,
        tokenizer=tokenizer,
        max_length=1,
    )

    assert indexed.to_expanded_rows() == expanded


def test_build_prefix_replay_dataset_normalizes_official_pool_row():
    """Official ReOPD `prompt` rows and metadata routes work without rebuilding."""
    row = {
        "prompt": [{"role": "user", "content": "Question"}],
        "label": "teacher answer",
        "metadata": {
            "task": "math",
            "assistant_turn_index": 0,
            "total_assistant_turns": 3,
        },
    }

    result = build_prefix_replay_dataset(
        [row],
        input_mode="prefix",
        route_field="task_type",
    )

    assert result[0]["messages"] == row["prompt"]
    assert result[0]["task_type"] == "math"
    assert result[0][PREFIX_REPLAY_METADATA_KEY]["assistant_turn_index"] == 0
    assert "prompt" not in result[0]


def test_load_prefix_replay_dataset_reads_jsonl(tmp_path):
    """The loader accepts trajectory JSONL and preserves experience identifiers."""
    path = tmp_path / "teacher.jsonl"
    path.write_text(json.dumps(_teacher_trajectory()) + "\n", encoding="utf-8")

    rows = load_prefix_replay_dataset(path, kappa=1.0)

    assert len(rows) == 2
    assert {row["instance_id"] for row in rows} == {"task-1"}


def test_load_prefix_replay_dataset_supports_swe_nested_conversations(tmp_path):
    """SWE records preserve nested tools and consecutive user messages."""
    path = tmp_path / "swe.jsonl"
    tools = [{"type": "function", "function": {"name": "shell"}}]
    record = {
        "instance_id": "game-1",
        "conversations": [
            {
                "messages": [
                    {"role": "system", "content": "Use the skill."},
                    {"role": "user", "content": "Task context"},
                    {"role": "user", "content": "Please solve it"},
                    {"role": "assistant", "content": "First action"},
                    {"role": "tool", "content": "Observation"},
                    {"role": "assistant", "content": "Final action"},
                ],
                "tools": tools,
                "model": "teacher",
            }
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    rows = load_prefix_replay_dataset(
        path,
        kappa=1.0,
        input_mode="trajectory",
        route_field="task_type",
        route_default_value="game",
    )

    assert len(rows) == 2
    assert [message["role"] for message in rows[0]["messages"]] == [
        "system",
        "user",
        "user",
    ]
    assert [message["role"] for message in rows[1]["messages"]] == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
    ]
    assert rows[0]["tools"] == tools
    assert rows[0]["instance_id"] == "game-1"
    assert rows[0]["task_type"] == "game"
    assert "conversations" not in rows[0]


def test_load_prefix_replay_indexed_dataset_supports_swe_nested_conversations(
    tmp_path,
):
    """The compact loader preserves SWE trajectory fields without row expansion."""
    path = tmp_path / "swe.jsonl"
    tools = [{"type": "function", "function": {"name": "shell"}}]
    record = {
        "instance_id": "game-1",
        "conversations": [
            {
                "messages": [
                    {"role": "system", "content": "Use the skill."},
                    {"role": "user", "content": "Task context"},
                    {"role": "user", "content": "Please solve it"},
                    {"role": "assistant", "content": "First action"},
                    {"role": "tool", "content": "Observation"},
                    {"role": "assistant", "content": "Final action"},
                ],
                "tools": tools,
            }
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = load_prefix_replay_indexed_dataset(
        path,
        kappa=1.0,
        input_mode="trajectory",
        route_field="task_type",
        route_default_value="game",
    )

    assert len(dataset.trajectories) == 1
    assert len(dataset) == 2
    assert dataset[0]["tools"] == tools
    assert dataset[0]["instance_id"] == "game-1"
    assert dataset[0]["task_type"] == "game"
    assert "conversations" not in dataset[0]


def test_load_prefix_replay_dataset_stringifies_tool_arguments_for_openai(tmp_path):
    """Cached replay messages use OpenAI-compatible JSON-string arguments."""
    path = tmp_path / "swe.jsonl"
    record = {
        "conversations": [
            {
                "messages": [
                    {"role": "user", "content": "Inspect the file"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": {"file_path": "/tmp/a"},
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": "contents"},
                    {"role": "assistant", "content": "Done"},
                ]
            }
        ]
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    rows = load_prefix_replay_dataset(
        path,
        kappa=1.0,
        input_mode="trajectory",
        parse_tool_call_args=True,
    )

    arguments = rows[1]["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"file_path": "/tmp/a"}


def test_prefix_replay_length_filter_renders_parsed_tool_arguments():
    """Length checks parse JSON-string arguments without mutating cached rows."""

    class ToolArgumentTokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
            assert arguments == {"file_path": "/tmp/a"}
            return {"input_ids": [0]}

    row = {
        "messages": [
            {"role": "user", "content": "Inspect the file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path":"/tmp/a"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "contents"},
        ]
    }

    rows = build_prefix_replay_dataset(
        [row],
        input_mode="prefix",
        tokenizer=ToolArgumentTokenizer(),
        max_length=4,
    )

    arguments = rows[0]["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"file_path": "/tmp/a"}


def test_build_prefix_replay_dataset_prefers_existing_route_over_default():
    """A fixed single-task route only fills records whose route is absent."""
    row = {
        "task_type": "special",
        "messages": [{"role": "user", "content": "Question"}],
    }

    result = build_prefix_replay_dataset(
        [row],
        input_mode="prefix",
        route_field="task_type",
        route_default_value="game",
    )

    assert result[0]["task_type"] == "special"


def test_build_prefix_replay_dataset_filters_rendered_prefix_length():
    """Length filtering uses the same rendered generation prompt as rollout."""

    class LengthTokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            assert tokenize is True
            return {"input_ids": list(range(len(messages)))}

    records = [
        {"messages": [{"role": "user", "content": "short"}]},
        {
            "messages": [
                {"role": "user", "content": "long"},
                {"role": "assistant", "content": "answer"},
                {"role": "tool", "content": "observation"},
            ]
        },
    ]

    rows = build_prefix_replay_dataset(
        records,
        input_mode="prefix",
        tokenizer=LengthTokenizer(),
        max_length=1,
    )

    assert len(rows) == 1
    assert rows[0]["messages"] == records[0]["messages"]


def test_build_prefix_replay_dataset_fast_paths_trajectory_length_filter():
    """If the longest sampled prefix fits, earlier prefixes need no tokenization."""

    class CountingTokenizer:
        def __init__(self) -> None:
            self.rendered_lengths: list[int] = []
            self.tokenized_lengths: list[int] = []

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            if tokenize:
                self.tokenized_lengths.append(len(messages))
                return {"input_ids": list(range(len(messages)))}
            self.rendered_lengths.append(len(messages))
            return "x" * len(messages)

    tokenizer = CountingTokenizer()
    trajectory = {
        "messages": [
            {"role": "user", "content": "turn 0"},
            {"role": "assistant", "content": "answer 0"},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "tool", "content": "observation"},
            {"role": "assistant", "content": "answer 2"},
        ]
    }

    rows = build_prefix_replay_dataset(
        [trajectory],
        input_mode="trajectory",
        kappa=1.0,
        tokenizer=tokenizer,
        max_length=2048,
    )

    assert len(rows) == 3
    assert tokenizer.rendered_lengths == [6]
    assert tokenizer.tokenized_lengths == []


def test_prefix_replay_processed_cache_roundtrips_and_rebuilds_stale_cache(tmp_path):
    """Processed cache is reused only when metadata matches current settings."""
    source_path = tmp_path / "teacher.jsonl"
    source_path.write_text(json.dumps(_teacher_trajectory()) + "\n", encoding="utf-8")
    cache_dir = tmp_path / "processed_prefix_replay_train"
    meta = build_prefix_replay_cache_metadata(
        str(source_path),
        split="train",
        tokenizer_path="/model",
        max_length=128,
        kappa=1.0,
        seed=42,
        input_mode="trajectory",
        drop_system_messages=False,
        route_field="task_type",
        route_metadata_field="task",
        route_default_value="game",
        parse_tool_call_args=True,
        chat_template_kwargs={"thinking_option": "off"},
    )
    rows = build_prefix_replay_indexed_dataset(
        [_teacher_trajectory()],
        kappa=1.0,
        route_field="task_type",
        route_metadata_field="task",
        route_default_value="game",
        parse_tool_call_args=True,
    )

    assert preflight_prefix_replay_cache(cache_dir, meta)[0] is False
    write_prefix_replay_cache(cache_dir, meta, rows)
    assert (cache_dir / "trajectories").is_dir()
    assert (cache_dir / "indices").is_dir()
    assert preflight_prefix_replay_cache(cache_dir, meta) == (
        True,
        "complete cache is compatible",
    )
    cached_rows = read_prefix_replay_cache(cache_dir, meta)
    assert isinstance(cached_rows, PrefixReplayIndexedDataset)
    assert cached_rows.to_expanded_rows() == rows.to_expanded_rows()

    stale_meta = dict(meta)
    stale_meta["max_length"] = 64
    cache_valid, reason = preflight_prefix_replay_cache(cache_dir, stale_meta)

    assert cache_valid is False
    assert "metadata" in reason
    assert cache_dir.is_dir()
    assert not (cache_dir / "trajectories").exists()
    assert not (cache_dir / "indices").exists()


def test_prefix_replay_cache_metadata_tracks_experience_source_and_field(tmp_path):
    source_path = tmp_path / "teacher.jsonl"
    source_path.write_text(json.dumps(_teacher_trajectory()) + "\n", encoding="utf-8")
    experience_path = tmp_path / "hints.jsonl"
    experience_path.write_text(
        json.dumps({"instance_id": "task-1", "hint": "first"}) + "\n",
        encoding="utf-8",
    )

    def metadata(field: str) -> dict[str, Any]:
        return build_prefix_replay_cache_metadata(
            str(source_path),
            split="train",
            tokenizer_path="/model",
            max_length=128,
            kappa=1.0,
            seed=42,
            input_mode="trajectory",
            drop_system_messages=False,
            route_field=None,
            route_metadata_field=None,
            route_default_value=None,
            parse_tool_call_args=True,
            chat_template_kwargs={},
            experience_path=str(experience_path),
            experience_field=field,
            experience_context_length=256,
        )

    original = metadata("hint")
    different_field = metadata("experience")
    experience_path.write_text(
        json.dumps({"instance_id": "task-1", "hint": "updated"}) + "\n",
        encoding="utf-8",
    )
    changed_source = metadata("hint")

    assert original["experience"]["field"] == "hint"
    assert original["experience"]["context_length"] == 256
    assert different_field != original
    assert changed_source != original


def test_load_prefix_replay_instance_ids_rejects_missing_id(tmp_path):
    source_path = tmp_path / "teacher.jsonl"
    source_path.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="instance_id.*non-empty string"):
        load_prefix_replay_instance_ids(source_path)


def test_build_prefix_replay_dataset_preserves_explicit_thinking_option():
    """Bailing V3 thinking_option is not shadowed by enable_thinking=False."""

    class ThinkingTokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            assert tokenize is True
            assert kwargs["thinking_option"] == "on"
            assert "enable_thinking" not in kwargs
            return {"input_ids": [1]}

    rows = build_prefix_replay_dataset(
        [{"messages": [{"role": "user", "content": "Question"}]}],
        input_mode="prefix",
        tokenizer=ThinkingTokenizer(),
        max_length=8,
        chat_template_kwargs={"thinking_option": "on"},
    )

    assert len(rows) == 1


def test_build_prefix_replay_dataset_requires_tokenizer_for_length_filter():
    """A token-length filter cannot run without a tokenizer and chat template."""
    with pytest.raises(ValueError, match="tokenizer is required"):
        build_prefix_replay_dataset(
            [{"messages": [{"role": "user", "content": "Question"}]}],
            input_mode="prefix",
            max_length=8,
        )


def test_prefix_replay_grouped_sampling_keeps_proxy_request_single_sample():
    """The rollout controller, rather than one proxy request, expands the group."""
    gconfig = GenerationHyperparameters(
        n_samples=4,
        max_new_tokens=8,
        drop_incomplete_group=True,
    )

    workflow_kwargs = build_prefix_replay_workflow_kwargs(gconfig)

    assert gconfig.n_samples == 4
    assert workflow_kwargs["n"] == 1
    assert workflow_kwargs["max_completion_tokens"] == 8


def test_expand_teacher_trajectory_does_not_generate_invalid_candidates():
    """Split assistant records are excluded before kappa sampling."""
    trajectory = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "tool", "content": "Observation"},
            {"role": "assistant", "content": "Third"},
        ]
    }

    rows = expand_teacher_trajectory(trajectory, kappa=1.0)

    assert len(rows) == 2
    assert rows[0]["messages"] == [{"role": "user", "content": "Question"}]
    assert rows[0][PREFIX_REPLAY_METADATA_KEY]["assistant_turn_index"] == 0
    assert rows[0][PREFIX_REPLAY_METADATA_KEY]["total_assistant_turns"] == 2
    assert [message["role"] for message in rows[1]["messages"]] == [
        "user",
        "assistant",
        "assistant",
        "tool",
    ]
    assert rows[1][PREFIX_REPLAY_METADATA_KEY]["assistant_turn_index"] == 1


@pytest.mark.parametrize("kappa", [0.0, -0.1, 1.1, float("inf")])
def test_build_prefix_replay_dataset_rejects_invalid_kappa(kappa):
    """The step-decay base is a probability and must stay in (0, 1]."""
    with pytest.raises(ValueError, match="kappa"):
        build_prefix_replay_dataset([_teacher_trajectory()], kappa=kappa)


class _FakeUsage:
    def __init__(self, *, prompt_tokens: int = 3, completion_tokens: int = 2) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChoice:
    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(
        self,
        *,
        response_id: str = "chatcmpl-prefix",
        finish_reason: str = "stop",
        completion_tokens: int = 2,
    ) -> None:
        self.id = response_id
        self.choices = [_FakeChoice(finish_reason)]
        self.usage = _FakeUsage(completion_tokens=completion_tokens)


def _patch_fake_openai(monkeypatch, response: _FakeResponse) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> _FakeResponse:
            requests.append(kwargs)
            return response

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr("examples.prefix_replay.agent.AsyncOpenAI", FakeAsyncOpenAI)
    return requests


@pytest.mark.asyncio
async def test_prefix_replay_agent_generates_one_zero_reward_action(monkeypatch):
    """Agent sends the replay prefix to the proxy and rewards only that response."""
    response = _FakeResponse(response_id="chatcmpl-ok", finish_reason="tool_calls")
    requests = _patch_fake_openai(monkeypatch, response)
    agent = PrefixReplayAgent(temperature=0.7, max_completion_tokens=8)
    tools = [{"type": "function", "function": {"name": "shell"}}]

    result = await agent.run(
        {
            "messages": [{"role": "user", "content": "Question"}],
            "tools": tools,
        },
        base_url="http://proxy",
        api_key="session-key",
    )

    assert result == {"chatcmpl-ok": 0.0}
    assert requests == [
        {
            "messages": [{"role": "user", "content": "Question"}],
            "model": "default",
            "temperature": 0.7,
            "max_completion_tokens": 8,
            "tools": tools,
        }
    ]


@pytest.mark.asyncio
async def test_prefix_replay_agent_applies_experience_safe_total_limit(monkeypatch):
    """The proxy caps the completed trajectory before teacher injection."""
    requests = _patch_fake_openai(monkeypatch, _FakeResponse())
    agent = PrefixReplayAgent(max_completion_tokens=128)

    await agent.run(
        {
            "messages": [{"role": "user", "content": "Question"}],
            PREFIX_REPLAY_METADATA_KEY: {PREFIX_REPLAY_MAX_TOTAL_TOKENS_FIELD: 80},
        },
        base_url="http://proxy",
    )

    assert requests[0]["max_completion_tokens"] == 128
    assert requests[0]["extra_body"] == {"max_total_tokens": 80}


@pytest.mark.asyncio
async def test_prefix_replay_agent_rejects_length_truncated_action(monkeypatch):
    """Length-truncated actions are rejected so batch collection can retry."""
    requests = _patch_fake_openai(
        monkeypatch,
        _FakeResponse(finish_reason="length", completion_tokens=8),
    )
    agent = PrefixReplayAgent(max_completion_tokens=8)

    with pytest.raises(PrefixReplayLengthTruncated):
        await agent.run(
            {"messages": [{"role": "user", "content": "Question"}]},
            base_url="http://proxy",
        )

    assert requests[0]["max_completion_tokens"] == 8


@pytest.mark.asyncio
async def test_prefix_replay_agent_rejects_empty_action(monkeypatch):
    """An empty proxy completion carries no distillation tokens."""
    _patch_fake_openai(monkeypatch, _FakeResponse(completion_tokens=0))
    agent = PrefixReplayAgent(max_completion_tokens=8)

    with pytest.raises(PrefixReplayEmptyAction):
        await agent.run(
            {"messages": [{"role": "user", "content": "Question"}]},
            base_url="http://proxy",
        )
