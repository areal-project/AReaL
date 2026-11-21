# SPDX-License-Identifier: Apache-2.0

"""Build replayed-prefix OPD samples from offline teacher trajectories."""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

PREFIX_REPLAY_METADATA_KEY = "prefix_replay"
PREFIX_REPLAY_MAX_TOTAL_TOKENS_FIELD = "max_total_tokens"
_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
_PREFIX_TERMINAL_ROLES = {"user", "tool"}
_TOKEN_LENGTH_FAST_PATH_MARGIN = 1024


class PrefixReplayIndexedDataset:
    """Compact prefix replay dataset backed by trajectories plus prefix indices."""

    def __init__(
        self,
        trajectories: Sequence[Mapping[str, Any]],
        indices: Sequence[Mapping[str, Any]],
    ) -> None:
        self.trajectories = [copy.deepcopy(dict(row)) for row in trajectories]
        self.indices = [dict(row) for row in indices]
        if not self.trajectories:
            raise ValueError("Prefix replay indexed dataset requires trajectories")

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.indices[index]
        trajectory_index = spec["trajectory_index"]
        message_end = spec["message_end"]
        trajectory = self.trajectories[trajectory_index]
        row = _copy_record_without_messages(trajectory)
        row["messages"] = _copy_prefix_messages(trajectory["messages"][:message_end])
        _with_prefix_metadata(
            row,
            assistant_turn_index=spec["assistant_turn_index"],
            total_assistant_turns=spec["total_assistant_turns"],
            sampling_probability=spec.get("sampling_probability"),
        )
        return row

    @classmethod
    def from_prefix_rows(
        cls, rows: Sequence[Mapping[str, Any]]
    ) -> PrefixReplayIndexedDataset:
        trajectories: list[dict[str, Any]] = []
        indices: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            trajectory = copy.deepcopy(dict(row))
            messages = _validate_messages(
                trajectory.get("messages"), context=f"prefix row {row_index} messages"
            )
            trajectory["messages"] = _stringify_tool_call_arguments(messages)
            metadata = trajectory.get(PREFIX_REPLAY_METADATA_KEY)
            if not isinstance(metadata, Mapping):
                metadata = {}
            assistant_turn_index = metadata.get("assistant_turn_index", 0)
            total_assistant_turns = metadata.get("total_assistant_turns", 1)
            if not isinstance(assistant_turn_index, int):
                assistant_turn_index = 0
            if not isinstance(total_assistant_turns, int):
                total_assistant_turns = 1
            spec: dict[str, Any] = {
                "trajectory_index": len(trajectories),
                "message_end": len(messages),
                "assistant_turn_index": assistant_turn_index,
                "total_assistant_turns": total_assistant_turns,
            }
            if "sampling_probability" in metadata:
                spec["sampling_probability"] = metadata["sampling_probability"]
            trajectories.append(trajectory)
            indices.append(spec)
        return cls(trajectories, indices)

    def to_cache_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            [copy.deepcopy(row) for row in self.trajectories],
            [dict(row) for row in self.indices],
        )

    def to_expanded_rows(self) -> list[dict[str, Any]]:
        return [self[index] for index in range(len(self))]

    def filter_by_instance_ids(
        self, instance_ids: set[str]
    ) -> PrefixReplayIndexedDataset:
        indices = [
            spec
            for spec in self.indices
            if isinstance(
                self.trajectories[spec["trajectory_index"]].get("instance_id"), str
            )
            and self.trajectories[spec["trajectory_index"]]["instance_id"]
            in instance_ids
        ]
        return PrefixReplayIndexedDataset(self.trajectories, indices)

    def first_missing_route_index(self, route_field: str) -> int | None:
        for index, spec in enumerate(self.indices):
            if route_field not in self.trajectories[spec["trajectory_index"]]:
                return index
        return None

    def route_values(self, route_field: str) -> set[str]:
        return {
            str(self.trajectories[spec["trajectory_index"]][route_field])
            for spec in self.indices
            if route_field in self.trajectories[spec["trajectory_index"]]
        }


def _copy_record_without_messages(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value) for key, value in record.items() if key != "messages"
    }


def _copy_prefix_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    # ``messages`` is already normalized and detached from the input record.
    # Prefix replay treats dataset rows as read-only, so copying only the list
    # shell avoids duplicating the same long history for every assistant turn.
    return list(messages)


def _validate_kappa(kappa: float) -> None:
    if (
        not isinstance(kappa, (int, float))
        or isinstance(kappa, bool)
        or not math.isfinite(kappa)
        or not 0.0 < kappa <= 1.0
    ):
        raise ValueError(f"kappa must be a finite number in (0, 1], got {kappa!r}")


def _validate_max_length(max_length: int | None) -> None:
    if max_length is not None and (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length <= 0
    ):
        raise ValueError(
            f"max_length must be a positive integer or None, got {max_length!r}"
        )


def _resolve_record_token_limits(
    record: Mapping[str, Any],
    *,
    max_length: int | None,
    max_total_tokens_by_instance_id: Mapping[str, int] | None,
) -> tuple[int | None, int | None]:
    if max_total_tokens_by_instance_id is None:
        return max_length, None
    if not isinstance(max_total_tokens_by_instance_id, Mapping):
        raise TypeError("max_total_tokens_by_instance_id must be a mapping or None")

    instance_id = record.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError(
            "Per-instance token limits require a non-empty string instance_id"
        )
    max_total_tokens = max_total_tokens_by_instance_id.get(instance_id)
    if (
        not isinstance(max_total_tokens, int)
        or isinstance(max_total_tokens, bool)
        or max_total_tokens <= 1
    ):
        raise ValueError(
            f"No valid max total token limit for instance_id {instance_id!r}"
        )

    prefix_max_length = max_total_tokens - 1
    if max_length is not None:
        prefix_max_length = min(prefix_max_length, max_length)
    return prefix_max_length, max_total_tokens


def _validate_messages(messages: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError(f"{context} must be a non-empty message sequence")
    normalized: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"{context}[{message_index}] must be a mapping, "
                f"got {type(message).__name__}"
            )
        role = message.get("role")
        if role not in _MESSAGE_ROLES:
            raise ValueError(
                f"{context}[{message_index}] has unsupported role {role!r}"
            )
        normalized.append(copy.deepcopy(dict(message)))
    if not normalized:
        raise ValueError(f"{context} must be a non-empty message sequence")
    return normalized


def _drop_leading_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    first_non_system = 0
    while (
        first_non_system < len(messages)
        and messages[first_non_system]["role"] == "system"
    ):
        first_non_system += 1
    return messages[first_non_system:]


def _set_route_from_metadata(
    row: dict[str, Any],
    *,
    route_field: str | None,
    route_metadata_field: str | None,
    route_default_value: str | None,
) -> None:
    if route_field is None or route_field in row:
        return
    metadata = row.get("metadata")
    if (
        route_metadata_field is not None
        and isinstance(metadata, Mapping)
        and route_metadata_field in metadata
    ):
        row[route_field] = copy.deepcopy(metadata[route_metadata_field])
    elif route_default_value is not None:
        row[route_field] = route_default_value


def _normalize_record_layout(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize flat and SWE-style ``conversations`` trajectory records."""
    row = copy.deepcopy(dict(record))
    conversations = row.pop("conversations", None)
    if not conversations:
        return row
    if not isinstance(conversations, Sequence) or isinstance(
        conversations, (str, bytes)
    ):
        raise TypeError("conversations must be a non-empty sequence")
    conversation = conversations[-1]
    if not isinstance(conversation, Mapping):
        raise TypeError("the last conversation must be a mapping")
    row["messages"] = copy.deepcopy(conversation.get("messages"))
    if "tools" in conversation:
        row["tools"] = copy.deepcopy(conversation["tools"])
    return row


def _parse_tool_call_arguments(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI JSON-string tool arguments to dictionaries for rendering.

    This mirrors SWE SFT's opt-in ``parse_tool_call_args`` preprocessing and
    is required by the Bailing V3 chat template used by the supplied model.
    Invalid JSON is preserved so the subsequent template error remains
    explicit instead of silently changing the recorded action.
    """
    patched: list[dict[str, Any]] = []
    for message in messages:
        row = copy.deepcopy(dict(message))
        tool_calls = row.get("tool_calls")
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
            patched.append(row)
            continue
        normalized_calls: list[Any] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                normalized_calls.append(copy.deepcopy(tool_call))
                continue
            normalized_call = copy.deepcopy(dict(tool_call))
            function = normalized_call.get("function", normalized_call)
            if isinstance(function, Mapping):
                normalized_function = copy.deepcopy(dict(function))
                arguments = normalized_function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        normalized_function["arguments"] = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if "function" in normalized_call:
                    normalized_call["function"] = normalized_function
                else:
                    normalized_call = normalized_function
            normalized_calls.append(normalized_call)
        row["tool_calls"] = normalized_calls
        patched.append(row)
    return patched


def _stringify_tool_call_arguments(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Store tool-call arguments in OpenAI wire format.

    Prefix replay sends cached messages through an OpenAI-compatible proxy.
    That schema requires ``tool_calls[].function.arguments`` to be a JSON
    string, while some SWE/Game data stores the same field as a mapping for
    chat-template rendering.
    """

    patched: list[dict[str, Any]] = []
    for message in messages:
        row = copy.deepcopy(dict(message))
        tool_calls = row.get("tool_calls")
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
            patched.append(row)
            continue
        normalized_calls: list[Any] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                normalized_calls.append(copy.deepcopy(tool_call))
                continue
            normalized_call = copy.deepcopy(dict(tool_call))
            function = normalized_call.get("function", normalized_call)
            if isinstance(function, Mapping):
                normalized_function = copy.deepcopy(dict(function))
                arguments = normalized_function.get("arguments")
                if not isinstance(arguments, str):
                    normalized_function["arguments"] = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                if "function" in normalized_call:
                    normalized_call["function"] = normalized_function
                else:
                    normalized_call = normalized_function
            normalized_calls.append(normalized_call)
        row["tool_calls"] = normalized_calls
        patched.append(row)
    return patched


def _with_prefix_metadata(
    row: dict[str, Any],
    *,
    assistant_turn_index: int,
    total_assistant_turns: int,
    sampling_probability: float | None,
) -> dict[str, Any]:
    existing = row.get(PREFIX_REPLAY_METADATA_KEY)
    if existing is not None and not isinstance(existing, Mapping):
        raise TypeError(f"{PREFIX_REPLAY_METADATA_KEY!r} metadata must be a mapping")
    metadata = copy.deepcopy(dict(existing or {}))
    metadata.update(
        {
            "assistant_turn_index": assistant_turn_index,
            "total_assistant_turns": total_assistant_turns,
        }
    )
    if sampling_probability is not None:
        metadata["sampling_probability"] = sampling_probability
    row[PREFIX_REPLAY_METADATA_KEY] = metadata
    return row


def _with_max_total_tokens(
    row: dict[str, Any], max_total_tokens: int | None
) -> dict[str, Any]:
    if max_total_tokens is None:
        return row
    existing = row.get(PREFIX_REPLAY_METADATA_KEY)
    if existing is not None and not isinstance(existing, Mapping):
        raise TypeError(f"{PREFIX_REPLAY_METADATA_KEY!r} metadata must be a mapping")
    metadata = copy.deepcopy(dict(existing or {}))
    metadata[PREFIX_REPLAY_MAX_TOTAL_TOKENS_FIELD] = max_total_tokens
    row[PREFIX_REPLAY_METADATA_KEY] = metadata
    return row


def _prefix_token_length(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> int:
    """Render a prefix as the rollout workflow does and return its token length."""
    from areal.utils.hf_utils import apply_chat_template

    template_kwargs = dict(chat_template_kwargs or {})
    template_kwargs.setdefault("add_generation_prompt", True)
    # Bailing V3 accepts both ``enable_thinking`` and ``thinking_option`` but
    # prioritizes the former. Do not silently override an explicit
    # ``thinking_option: on`` supplied by the rollout config.
    if "thinking_option" not in template_kwargs:
        template_kwargs.setdefault("enable_thinking", False)
    tools = row.get("tools")
    if tools is not None:
        template_kwargs.setdefault("tools", tools)
    template_messages = _parse_tool_call_arguments(row["messages"])
    token_ids = apply_chat_template(
        tokenizer,
        template_messages,
        tokenize=True,
        **template_kwargs,
    )
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise TypeError(
            "Prefix replay tokenizer must return a token-id sequence when tokenize=True"
        )
    return len(token_ids)


def _trajectory_fast_path_fits(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> bool:
    """Cheaply prove every prefix in a trajectory fits within max_length.

    Rendering the full trajectory is much cheaper than tokenizing every replay
    prefix.  The UTF-8 byte length is a conservative upper bound for normal
    byte/character tokenizer pieces; a margin covers tokenizer-added specials
    and the empty-thinking generation prompt used by Bailing V3 no-think mode.
    If this fast path cannot prove the trajectory fits, callers fall back to
    exact per-prefix tokenization.
    """
    from areal.utils.hf_utils import apply_chat_template

    template_kwargs = dict(chat_template_kwargs or {})
    if "thinking_option" not in template_kwargs:
        template_kwargs.setdefault("enable_thinking", False)
    tools = row.get("tools")
    if tools is not None:
        template_kwargs.setdefault("tools", tools)
    template_messages = _parse_tool_call_arguments(row["messages"])
    rendered = apply_chat_template(
        tokenizer,
        template_messages,
        tokenize=False,
        **template_kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError(
            "Prefix replay tokenizer must return a string when tokenize=False"
        )
    return len(rendered.encode("utf-8")) + _TOKEN_LENGTH_FAST_PATH_MARGIN <= max_length


def _replayable_assistant_positions(
    messages: Sequence[Mapping[str, Any]],
) -> list[int]:
    return [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
        and index > 0
        and messages[index - 1]["role"] in _PREFIX_TERMINAL_ROLES
    ]


def _prefix_index_fits(
    trajectory: Mapping[str, Any],
    *,
    message_end: int,
    tokenizer: Any,
    max_length: int,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> bool:
    """Return whether a prefix fits, truncating tokenization after max_length."""
    from areal.utils.hf_utils import apply_chat_template

    row = _copy_record_without_messages(trajectory)
    row["messages"] = _copy_prefix_messages(trajectory["messages"][:message_end])
    template_kwargs = dict(chat_template_kwargs or {})
    template_kwargs.setdefault("add_generation_prompt", True)
    if "thinking_option" not in template_kwargs:
        template_kwargs.setdefault("enable_thinking", False)
    tools = row.get("tools")
    if tools is not None:
        template_kwargs.setdefault("tools", tools)
    template_kwargs.setdefault("truncation", True)
    template_kwargs.setdefault("max_length", max_length + 1)
    template_messages = _parse_tool_call_arguments(row["messages"])
    token_ids = apply_chat_template(
        tokenizer,
        template_messages,
        tokenize=True,
        **template_kwargs,
    )
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise TypeError(
            "Prefix replay tokenizer must return a token-id sequence when tokenize=True"
        )
    return len(token_ids) <= max_length


def _max_fitting_replayable_turn(
    trajectory: Mapping[str, Any],
    replayable_positions: Sequence[int],
    *,
    tokenizer: Any,
    max_length: int,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> int:
    if _trajectory_fast_path_fits(
        trajectory,
        tokenizer=tokenizer,
        max_length=max_length,
        chat_template_kwargs=chat_template_kwargs,
    ):
        return len(replayable_positions) - 1

    if _prefix_index_fits(
        trajectory,
        message_end=replayable_positions[-1],
        tokenizer=tokenizer,
        max_length=max_length,
        chat_template_kwargs=chat_template_kwargs,
    ):
        return len(replayable_positions) - 1

    lower = 0
    upper = len(replayable_positions) - 2
    last_fit = -1
    while lower <= upper:
        middle = (lower + upper) // 2
        prefix_fits = _prefix_index_fits(
            trajectory,
            message_end=replayable_positions[middle],
            tokenizer=tokenizer,
            max_length=max_length,
            chat_template_kwargs=chat_template_kwargs,
        )
        if prefix_fits:
            last_fit = middle
            lower = middle + 1
        else:
            upper = middle - 1
    return last_fit


def expand_teacher_trajectory(
    record: Mapping[str, Any],
    *,
    kappa: float = 0.6,
    rng: random.Random | None = None,
    drop_system_messages: bool = False,
    route_field: str | None = None,
    route_metadata_field: str | None = "task",
    route_default_value: str | None = None,
) -> list[dict[str, Any]]:
    """Fan out one teacher trajectory and sample turns with ``p_t = kappa**t``.

    The teacher action at the selected turn is deliberately excluded. The
    resulting row contains only the recorded teacher prefix, so a rollout
    workflow can generate a fresh student action without calling the
    environment.
    """

    _validate_kappa(kappa)
    rng = rng or random.Random()
    messages = _validate_messages(
        record.get("messages"), context="teacher trajectory messages"
    )
    replayable_positions = _replayable_assistant_positions(messages)
    if not replayable_positions:
        raise ValueError(
            "Teacher trajectory must contain at least one replayable assistant turn"
        )

    rows: list[dict[str, Any]] = []
    total_turns = len(replayable_positions)
    for turn_index, message_index in enumerate(replayable_positions):
        sampling_probability = kappa**turn_index
        if turn_index > 0 and rng.random() >= sampling_probability:
            continue

        prefix = _copy_prefix_messages(messages[:message_index])
        if drop_system_messages:
            prefix = _drop_leading_system_messages(prefix)
        if not prefix:
            raise ValueError(
                f"Assistant turn {turn_index} has an empty replayable prefix"
            )

        row = _copy_record_without_messages(record)
        row["messages"] = _stringify_tool_call_arguments(prefix)
        _set_route_from_metadata(
            row,
            route_field=route_field,
            route_metadata_field=route_metadata_field,
            route_default_value=route_default_value,
        )
        rows.append(
            _with_prefix_metadata(
                row,
                assistant_turn_index=turn_index,
                total_assistant_turns=total_turns,
                sampling_probability=sampling_probability,
            )
        )
    return rows


def _normalize_trajectory_record(
    row: Mapping[str, Any],
    *,
    drop_system_messages: bool,
    route_field: str | None,
    route_metadata_field: str | None,
    route_default_value: str | None,
) -> tuple[dict[str, Any], list[int]]:
    trajectory = copy.deepcopy(dict(row))
    messages = _validate_messages(
        trajectory.get("messages"), context="teacher trajectory messages"
    )
    if drop_system_messages:
        messages = _drop_leading_system_messages(messages)
    if not messages:
        raise ValueError("Teacher trajectory is empty after dropping system messages")
    messages = _stringify_tool_call_arguments(messages)
    trajectory["messages"] = messages
    _set_route_from_metadata(
        trajectory,
        route_field=route_field,
        route_metadata_field=route_metadata_field,
        route_default_value=route_default_value,
    )
    replayable_positions = _replayable_assistant_positions(messages)
    if not replayable_positions:
        raise ValueError(
            "Teacher trajectory must contain at least one replayable assistant turn"
        )
    return trajectory, replayable_positions


def preprocess_prefix_replay_record(
    record: Mapping[str, Any],
    *,
    input_mode: Literal["auto", "trajectory", "prefix"] = "auto",
    drop_system_messages: bool = False,
    route_field: str | None = None,
    route_metadata_field: str | None = "task",
    route_default_value: str | None = None,
    parse_tool_call_args: bool = False,
    tokenizer: Any | None = None,
    max_length: int | None = None,
    max_total_tokens_by_instance_id: Mapping[str, int] | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Normalize and length-filter one record without sampling replay turns.

    Keeping the random sampling out of this function makes the expensive work
    safe to execute in any process or node order. The lightweight assembly
    phase can then consume these records in source order and preserve the
    historical global RNG stream exactly.
    """

    if not isinstance(record, Mapping):
        raise TypeError(
            f"Prefix replay record must be a mapping, got {type(record).__name__}"
        )
    _validate_max_length(max_length)
    if (
        max_length is not None or max_total_tokens_by_instance_id is not None
    ) and tokenizer is None:
        raise ValueError("tokenizer is required when token limits are provided")
    if input_mode not in ("auto", "trajectory", "prefix"):
        raise ValueError(
            f"input_mode must be 'auto', 'trajectory', or 'prefix', got {input_mode!r}"
        )

    row = _normalize_record_layout(record)
    record_max_length, max_total_tokens = _resolve_record_token_limits(
        row,
        max_length=max_length,
        max_total_tokens_by_instance_id=max_total_tokens_by_instance_id,
    )
    mode = _infer_input_mode(row) if input_mode == "auto" else input_mode
    if mode == "trajectory":
        messages = _validate_messages(row.get("messages"), context="record messages")
        if parse_tool_call_args:
            messages = _parse_tool_call_arguments(messages)
        row["messages"] = _stringify_tool_call_arguments(messages)
        trajectory, replayable_positions = _normalize_trajectory_record(
            row,
            drop_system_messages=drop_system_messages,
            route_field=route_field,
            route_metadata_field=route_metadata_field,
            route_default_value=route_default_value,
        )
        _with_max_total_tokens(trajectory, max_total_tokens)
        if record_max_length is None:
            max_fit_turn = len(replayable_positions) - 1
        else:
            assert tokenizer is not None
            max_fit_turn = _max_fitting_replayable_turn(
                trajectory,
                replayable_positions,
                tokenizer=tokenizer,
                max_length=record_max_length,
                chat_template_kwargs=chat_template_kwargs,
            )
        return {
            "mode": "trajectory",
            "trajectory": trajectory,
            "replayable_positions": replayable_positions,
            "max_fit_turn": max_fit_turn,
        }

    prefix = normalize_prefix_record(
        row,
        drop_system_messages=drop_system_messages,
        route_field=route_field,
        route_metadata_field=route_metadata_field,
        route_default_value=route_default_value,
    )
    _with_max_total_tokens(prefix, max_total_tokens)
    if record_max_length is not None:
        assert tokenizer is not None
        if (
            _prefix_token_length(
                prefix,
                tokenizer=tokenizer,
                chat_template_kwargs=chat_template_kwargs,
            )
            > record_max_length
        ):
            return None
    return {"mode": "prefix", "trajectory": prefix}


def assemble_prefix_replay_indexed_dataset(
    processed_records: Iterable[Mapping[str, Any] | None],
    *,
    kappa: float = 0.6,
    seed: int = 42,
) -> PrefixReplayIndexedDataset:
    """Assemble source-ordered preprocessed records into a compact dataset."""

    _validate_kappa(kappa)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    rng = random.Random(seed)
    trajectories: list[dict[str, Any]] = []
    indices: list[dict[str, Any]] = []
    for record_index, processed in enumerate(processed_records):
        if processed is None:
            continue
        if not isinstance(processed, Mapping):
            raise TypeError(
                f"Preprocessed prefix replay record {record_index} must be a mapping"
            )
        mode = processed.get("mode")
        trajectory = processed.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise TypeError(
                f"Preprocessed prefix replay record {record_index} has no trajectory"
            )

        if mode == "trajectory":
            replayable_positions = processed.get("replayable_positions")
            max_fit_turn = processed.get("max_fit_turn")
            if (
                not isinstance(replayable_positions, Sequence)
                or isinstance(replayable_positions, (str, bytes))
                or not replayable_positions
                or not all(
                    isinstance(position, int) for position in replayable_positions
                )
            ):
                raise TypeError(
                    f"Preprocessed prefix replay record {record_index} has invalid "
                    "replayable positions"
                )
            if not isinstance(max_fit_turn, int):
                raise TypeError(
                    f"Preprocessed prefix replay record {record_index} has invalid "
                    "max_fit_turn"
                )

            total_turns = len(replayable_positions)
            pending_indices: list[dict[str, Any]] = []
            for turn_index, message_end in enumerate(replayable_positions):
                sampling_probability = kappa**turn_index
                sampled = turn_index == 0 or rng.random() < sampling_probability
                if not sampled or turn_index > max_fit_turn:
                    continue
                pending_indices.append(
                    {
                        "message_end": message_end,
                        "assistant_turn_index": turn_index,
                        "total_assistant_turns": total_turns,
                        "sampling_probability": sampling_probability,
                    }
                )
            if not pending_indices:
                continue
            trajectory_index = len(trajectories)
            trajectories.append(dict(trajectory))
            for spec in pending_indices:
                spec["trajectory_index"] = trajectory_index
                indices.append(spec)
        elif mode == "prefix":
            metadata = trajectory.get(PREFIX_REPLAY_METADATA_KEY)
            if not isinstance(metadata, Mapping):
                metadata = {}
            assistant_turn_index = metadata.get("assistant_turn_index", 0)
            total_assistant_turns = metadata.get("total_assistant_turns", 1)
            if not isinstance(assistant_turn_index, int):
                assistant_turn_index = 0
            if not isinstance(total_assistant_turns, int):
                total_assistant_turns = 1
            messages = trajectory.get("messages")
            if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
                raise TypeError(
                    f"Preprocessed prefix replay record {record_index} has invalid messages"
                )
            spec: dict[str, Any] = {
                "trajectory_index": len(trajectories),
                "message_end": len(messages),
                "assistant_turn_index": assistant_turn_index,
                "total_assistant_turns": total_assistant_turns,
            }
            if "sampling_probability" in metadata:
                spec["sampling_probability"] = metadata["sampling_probability"]
            trajectories.append(dict(trajectory))
            indices.append(spec)
        else:
            raise ValueError(
                f"Preprocessed prefix replay record {record_index} has invalid "
                f"mode {mode!r}"
            )

    if not indices:
        raise ValueError("Prefix replay dataset contains no usable prefixes")
    return PrefixReplayIndexedDataset(trajectories, indices)


def build_prefix_replay_indexed_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    kappa: float = 0.6,
    seed: int = 42,
    input_mode: Literal["auto", "trajectory", "prefix"] = "auto",
    drop_system_messages: bool = False,
    route_field: str | None = None,
    route_metadata_field: str | None = "task",
    route_default_value: str | None = None,
    parse_tool_call_args: bool = False,
    tokenizer: Any | None = None,
    max_length: int | None = None,
    max_total_tokens_by_instance_id: Mapping[str, int] | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> PrefixReplayIndexedDataset:
    """Build a compact replay-prefix dataset.

    The cacheable representation stores each normalized trajectory once and one
    lightweight index row per selected prefix.  ``__getitem__`` slices
    ``messages[:message_end]`` lazily, so processed caches do not duplicate the
    same long SWE/Game history for every assistant turn.
    """

    _validate_kappa(kappa)
    _validate_max_length(max_length)
    if (
        max_length is not None or max_total_tokens_by_instance_id is not None
    ) and tokenizer is None:
        raise ValueError("tokenizer is required when token limits are provided")
    if input_mode not in ("auto", "trajectory", "prefix"):
        raise ValueError(
            f"input_mode must be 'auto', 'trajectory', or 'prefix', got {input_mode!r}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    processed_records: list[dict[str, Any] | None] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"Prefix replay record {record_index} must be a mapping, "
                f"got {type(record).__name__}"
            )
        try:
            processed_records.append(
                preprocess_prefix_replay_record(
                    record,
                    input_mode=input_mode,
                    drop_system_messages=drop_system_messages,
                    route_field=route_field,
                    route_metadata_field=route_metadata_field,
                    route_default_value=route_default_value,
                    parse_tool_call_args=parse_tool_call_args,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    max_total_tokens_by_instance_id=(max_total_tokens_by_instance_id),
                    chat_template_kwargs=chat_template_kwargs,
                )
            )
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"Prefix replay record {record_index}: {exc}") from exc

    return assemble_prefix_replay_indexed_dataset(
        processed_records,
        kappa=kappa,
        seed=seed,
    )


def normalize_prefix_record(
    record: Mapping[str, Any],
    *,
    drop_system_messages: bool = False,
    route_field: str | None = None,
    route_metadata_field: str | None = "task",
    route_default_value: str | None = None,
) -> dict[str, Any]:
    """Normalize an already-built prefix row, including official ReOPD pools."""

    row = copy.deepcopy(dict(record))
    prompt = row.pop("prompt", None)
    messages_value = prompt if prompt is not None else row.get("messages")
    messages = _validate_messages(messages_value, context="prefix messages")
    if drop_system_messages:
        messages = _drop_leading_system_messages(messages)
    if not messages:
        raise ValueError("Replay prefix is empty after dropping system messages")
    messages = _stringify_tool_call_arguments(messages)
    if messages[-1]["role"] not in _PREFIX_TERMINAL_ROLES:
        raise ValueError(
            "A prebuilt replay prefix must end in a user or tool message, "
            f"got {messages[-1]['role']!r}"
        )
    row["messages"] = messages
    _set_route_from_metadata(
        row,
        route_field=route_field,
        route_metadata_field=route_metadata_field,
        route_default_value=route_default_value,
    )

    paper_metadata = row.get("metadata")
    replay_metadata = row.get(PREFIX_REPLAY_METADATA_KEY)
    if replay_metadata is None and isinstance(paper_metadata, Mapping):
        turn_index = paper_metadata.get("assistant_turn_index")
        total_turns = paper_metadata.get("total_assistant_turns")
        if isinstance(turn_index, int) and isinstance(total_turns, int):
            _with_prefix_metadata(
                row,
                assistant_turn_index=turn_index,
                total_assistant_turns=total_turns,
                sampling_probability=None,
            )
    return row


def _infer_input_mode(record: Mapping[str, Any]) -> Literal["trajectory", "prefix"]:
    if isinstance(record.get("prompt"), Sequence) and not isinstance(
        record.get("prompt"), (str, bytes)
    ):
        return "prefix"
    messages = _validate_messages(record.get("messages"), context="record messages")
    return "trajectory" if messages[-1]["role"] == "assistant" else "prefix"


def build_prefix_replay_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    kappa: float = 0.6,
    seed: int = 42,
    input_mode: Literal["auto", "trajectory", "prefix"] = "auto",
    drop_system_messages: bool = False,
    route_field: str | None = None,
    route_metadata_field: str | None = "task",
    route_default_value: str | None = None,
    parse_tool_call_args: bool = False,
    tokenizer: Any | None = None,
    max_length: int | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build an in-memory replay pool from trajectories or prebuilt prefixes.

    If ``max_length`` is set, the prefix is rendered with ``tokenizer`` and the
    same chat-template options used by the rollout workflow before filtering.
    """

    _validate_kappa(kappa)
    _validate_max_length(max_length)
    if max_length is not None and tokenizer is None:
        raise ValueError("tokenizer is required when max_length is provided")
    if input_mode not in ("auto", "trajectory", "prefix"):
        raise ValueError(
            f"input_mode must be 'auto', 'trajectory', or 'prefix', got {input_mode!r}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []

    def _extend_within_max_length(rows: list[dict[str, Any]]) -> None:
        if max_length is None:
            output.extend(rows)
            return
        assert tokenizer is not None
        if not rows:
            return
        if len(rows) > 1:
            # Rows expanded from one trajectory are ordered prefixes of the same
            # conversation. Rendering the longest prefix first lets the common
            # "all prefixes fit" case avoid one tokenizer call per assistant
            # turn, which matters for long SWE/Game trajectories.
            longest_prefix_length = _prefix_token_length(
                rows[-1],
                tokenizer=tokenizer,
                chat_template_kwargs=chat_template_kwargs,
            )
            if longest_prefix_length <= max_length:
                output.extend(rows)
                return
        output.extend(
            row
            for row in rows
            if _prefix_token_length(
                row,
                tokenizer=tokenizer,
                chat_template_kwargs=chat_template_kwargs,
            )
            <= max_length
        )

    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"Prefix replay record {record_index} must be a mapping, "
                f"got {type(record).__name__}"
            )
        try:
            row = _normalize_record_layout(record)
            mode = _infer_input_mode(row) if input_mode == "auto" else input_mode
            if mode == "trajectory":
                messages = _validate_messages(
                    row.get("messages"), context="record messages"
                )
                if parse_tool_call_args:
                    messages = _parse_tool_call_arguments(messages)
                row["messages"] = _stringify_tool_call_arguments(messages)
                if max_length is not None:
                    assert tokenizer is not None
                    trajectory_fits = _trajectory_fast_path_fits(
                        row,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                else:
                    trajectory_fits = True
                rows = expand_teacher_trajectory(
                    row,
                    kappa=kappa,
                    rng=rng,
                    drop_system_messages=drop_system_messages,
                    route_field=route_field,
                    route_metadata_field=route_metadata_field,
                    route_default_value=route_default_value,
                )
                if trajectory_fits:
                    output.extend(rows)
                else:
                    _extend_within_max_length(rows)
            else:
                row = normalize_prefix_record(
                    row,
                    drop_system_messages=drop_system_messages,
                    route_field=route_field,
                    route_metadata_field=route_metadata_field,
                    route_default_value=route_default_value,
                )
                if max_length is None:
                    output.append(row)
                else:
                    assert tokenizer is not None
                    if (
                        _prefix_token_length(
                            row,
                            tokenizer=tokenizer,
                            chat_template_kwargs=chat_template_kwargs,
                        )
                        <= max_length
                    ):
                        output.append(row)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"Prefix replay record {record_index}: {exc}") from exc

    if not output:
        raise ValueError("Prefix replay dataset contains no usable prefixes")
    return output


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} must contain a JSON object")
                records.append(value)
        return records

    with path.open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        value = value["data"]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"{path} must contain a list of JSON objects")
    return value


def load_prefix_replay_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON/JSONL/Parquet records from one file or dataset directory."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Prefix replay dataset not found: {source}")
    if source.is_dir():
        files = sorted(
            candidate
            for candidate in source.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in {".jsonl", ".parquet"}
        )
        if not files:
            files = sorted(
                candidate for candidate in source.rglob("*.json") if candidate.is_file()
            )
        if not files:
            raise ValueError(
                f"Prefix replay directory contains no data files: {source}"
            )
        records: list[dict[str, Any]] = []
        for file_path in files:
            records.extend(load_prefix_replay_records(file_path))
        return records

    suffix = source.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _load_json_records(source)
    if suffix == ".parquet":
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(source), split="train")
        return [dict(record) for record in dataset]
    raise ValueError(
        f"Unsupported prefix replay dataset format {suffix!r}; "
        "expected JSON, JSONL, or Parquet"
    )


def load_prefix_replay_instance_ids(path: str | Path) -> set[str]:
    """Load and validate replay instance ids without preprocessing trajectories."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Prefix replay dataset not found: {source}")
    if source.is_file() and source.suffix.lower() == ".jsonl":
        instance_ids: set[str] = set()
        with source.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{source}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise TypeError(
                        f"{source}:{line_number} must contain a JSON object"
                    )
                instance_id = record.get("instance_id")
                if not isinstance(instance_id, str) or not instance_id.strip():
                    raise ValueError(
                        f"{source}:{line_number}: 'instance_id' must be a "
                        "non-empty string"
                    )
                instance_ids.add(instance_id)
    else:
        instance_ids = set()
        for record_index, record in enumerate(load_prefix_replay_records(source)):
            instance_id = record.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValueError(
                    f"{source}: record {record_index}: 'instance_id' must be a "
                    "non-empty string"
                )
            instance_ids.add(instance_id)
    if not instance_ids:
        raise ValueError(f"Prefix replay dataset contains no records: {source}")
    return instance_ids


def load_prefix_replay_dataset(
    path: str | Path,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Load records and build replay-prefix samples in one call."""

    return build_prefix_replay_dataset(load_prefix_replay_records(path), **kwargs)


def load_prefix_replay_indexed_dataset(
    path: str | Path,
    **kwargs: Any,
) -> PrefixReplayIndexedDataset:
    """Load records and build a compact replay-prefix dataset in one call."""

    return build_prefix_replay_indexed_dataset(
        load_prefix_replay_records(path), **kwargs
    )


__all__ = [
    "PREFIX_REPLAY_MAX_TOTAL_TOKENS_FIELD",
    "PREFIX_REPLAY_METADATA_KEY",
    "PrefixReplayIndexedDataset",
    "assemble_prefix_replay_indexed_dataset",
    "build_prefix_replay_dataset",
    "build_prefix_replay_indexed_dataset",
    "expand_teacher_trajectory",
    "load_prefix_replay_instance_ids",
    "load_prefix_replay_indexed_dataset",
    "load_prefix_replay_dataset",
    "load_prefix_replay_records",
    "normalize_prefix_record",
    "preprocess_prefix_replay_record",
]
