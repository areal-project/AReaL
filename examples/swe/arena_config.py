"""Configuration helpers for prompt-level Arena Stream mixtures."""

from __future__ import annotations

import base64
import binascii
import math
import re
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml

from examples.swe.arena_types import ArenaRewardRefConfig, ArenaStreamConfig

_STREAM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _parse_reward_ref(value: Any, *, stream_name: str) -> ArenaRewardRefConfig:
    if value in (None, ""):
        return ArenaRewardRefConfig()
    if isinstance(value, ArenaRewardRefConfig):
        reward_ref = value
    elif isinstance(value, Mapping):
        unknown = set(value) - {"key", "version"}
        if unknown:
            raise ValueError(
                f"Arena Stream {stream_name!r} expected_reward_ref has unknown "
                f"fields: {sorted(unknown)}"
            )
        reward_ref = ArenaRewardRefConfig(
            key=str(value.get("key") or "").strip(),
            version=str(value.get("version") or "").strip(),
        )
    else:
        raise ValueError(
            f"Arena Stream {stream_name!r} expected_reward_ref must be an object"
        )
    if bool(reward_ref.key) != bool(reward_ref.version):
        raise ValueError(
            f"Arena Stream {stream_name!r} expected_reward_ref requires both key "
            "and version"
        )
    return reward_ref


def parse_arena_stream_config(value: Any) -> ArenaStreamConfig:
    """Validate one inline or file-backed Stream definition."""

    if isinstance(value, ArenaStreamConfig):
        raw = {field.name: getattr(value, field.name) for field in fields(value)}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError("Each Arena Stream definition must be an object")

    known_fields = {field.name for field in fields(ArenaStreamConfig)}
    unknown = set(raw) - known_fields
    if unknown:
        raise ValueError(
            f"Arena Stream definition has unknown fields: {sorted(unknown)}"
        )

    name = str(raw.get("name") or "").strip()
    stream_id = str(raw.get("stream_id") or "").strip()
    if not name:
        raise ValueError("Arena Stream definition requires a non-empty name")
    if not _STREAM_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"Arena Stream name {name!r} may only contain letters, digits, '.', "
            "'_', and '-'"
        )
    if not stream_id:
        raise ValueError(f"Arena Stream {name!r} requires a non-empty stream_id")

    weight = raw.get("sampling_weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(f"Arena Stream {name!r} sampling_weight must be numeric")
    weight = float(weight)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(
            f"Arena Stream {name!r} sampling_weight must be finite and positive"
        )

    llm_protocol = str(raw.get("llm_protocol") or "").strip()
    if llm_protocol not in ("", "anthropic", "responses", "chat_completions"):
        raise ValueError(
            f"Arena Stream {name!r} has unsupported llm_protocol {llm_protocol!r}"
        )
    task_envs = raw.get("task_envs") or {}
    if not isinstance(task_envs, Mapping) or not all(
        isinstance(key, str) and key and isinstance(item, str)
        for key, item in task_envs.items()
    ):
        raise ValueError(f"Arena Stream {name!r} task_envs must map strings to strings")

    threshold = raw.get("reward_threshold")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                f"Arena Stream {name!r} reward_threshold must be numeric or null"
            )
        threshold = float(threshold)
        if not math.isfinite(threshold):
            raise ValueError(f"Arena Stream {name!r} reward_threshold must be finite")

    return ArenaStreamConfig(
        name=name,
        stream_id=stream_id,
        sampling_weight=weight,
        harness=str(raw.get("harness") or "").strip(),
        llm_protocol=llm_protocol,
        task_envs=dict(task_envs),
        expected_reward_ref=_parse_reward_ref(
            raw.get("expected_reward_ref"), stream_name=name
        ),
        reward_threshold=threshold,
        reward_transform_fn=str(raw.get("reward_transform_fn") or "").strip(),
    )


def load_arena_stream_configs(econfig: Any) -> list[ArenaStreamConfig]:
    """Load multi-Stream definitions or translate the legacy single Stream."""

    inline = list(_config_value(econfig, "arena_streams", []) or [])
    streams_yaml_b64 = str(
        _config_value(econfig, "arena_streams_yaml_b64", "") or ""
    ).strip()
    streams_file = str(_config_value(econfig, "arena_streams_file", "") or "").strip()
    if sum((bool(inline), bool(streams_yaml_b64), bool(streams_file))) > 1:
        raise ValueError(
            "arena_streams, arena_streams_yaml_b64, and arena_streams_file are "
            "mutually exclusive"
        )

    values: list[Any]
    if streams_yaml_b64:
        try:
            streams_yaml = base64.b64decode(streams_yaml_b64, validate=True).decode(
                "utf-8"
            )
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError(
                "arena_streams_yaml_b64 is not valid base64 UTF-8"
            ) from exc
        payload = yaml.safe_load(streams_yaml)
        if isinstance(payload, Mapping):
            unknown = set(payload) - {"streams"}
            if unknown:
                raise ValueError(
                    "Encoded inline Arena Streams YAML has unknown top-level fields: "
                    f"{sorted(unknown)}"
                )
            values = payload.get("streams")
        else:
            values = payload
        if not isinstance(values, list) or not values:
            raise ValueError(
                "Encoded inline Arena Streams YAML must contain a non-empty "
                "'streams' list"
            )
    elif streams_file:
        path = Path(streams_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Arena Streams file not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            unknown = set(payload) - {"streams"}
            if unknown:
                raise ValueError(
                    f"Arena Streams file has unknown top-level fields: {sorted(unknown)}"
                )
            values = payload.get("streams")
        else:
            values = payload
        if not isinstance(values, list) or not values:
            raise ValueError(
                "Arena Streams file must contain a non-empty 'streams' list"
            )
    elif inline:
        values = inline
    else:
        stream_id = str(_config_value(econfig, "stream_id", "") or "").strip()
        legacy_task_envs = _config_value(econfig, "arena_task_envs", {}) or {}
        if not isinstance(legacy_task_envs, Mapping) or not all(
            isinstance(key, str) and key and isinstance(item, str)
            for key, item in legacy_task_envs.items()
        ):
            raise ValueError("arena_task_envs must map strings to strings")
        legacy_threshold = _config_value(econfig, "arena_reward_threshold", None)
        if legacy_threshold is not None:
            try:
                legacy_threshold = float(legacy_threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "arena_reward_threshold must be numeric or null"
                ) from exc
            if not math.isfinite(legacy_threshold):
                raise ValueError("arena_reward_threshold must be finite")
        return [
            ArenaStreamConfig(
                name="default",
                stream_id=stream_id,
                harness=str(_config_value(econfig, "arena_harness", "") or "").strip(),
                llm_protocol=str(
                    _config_value(econfig, "arena_llm_protocol", "") or ""
                ).strip(),
                task_envs=dict(legacy_task_envs),
                reward_threshold=legacy_threshold,
                reward_transform_fn=str(
                    _config_value(econfig, "arena_reward_transform_fn", "") or ""
                ).strip(),
            )
        ]

    streams = [parse_arena_stream_config(value) for value in values]
    common_task_envs_value = _config_value(econfig, "arena_task_envs", {}) or {}
    if not isinstance(common_task_envs_value, Mapping) or not all(
        isinstance(key, str) and key and isinstance(item, str)
        for key, item in common_task_envs_value.items()
    ):
        raise ValueError("arena_task_envs must map strings to strings")
    common_task_envs = dict(common_task_envs_value)
    common_harness = str(_config_value(econfig, "arena_harness", "") or "").strip()
    streams = [
        replace(
            stream,
            harness=stream.harness or common_harness,
            task_envs={**common_task_envs, **stream.task_envs},
        )
        for stream in streams
    ]
    names = [stream.name for stream in streams]
    if len(names) != len(set(names)):
        raise ValueError("Arena Stream names must be unique")
    stream_ids = [stream.stream_id for stream in streams]
    if len(stream_ids) != len(set(stream_ids)):
        raise ValueError("Arena Stream ids must be unique")
    return streams


def build_weighted_arena_rows(
    rows_by_stream: Mapping[str, list[dict[str, str]]],
    streams: list[ArenaStreamConfig],
    epoch_size: int = 0,
    size_multiple: int = 1,
) -> list[dict[str, str]]:
    """Build a deterministic Stream-weighted epoch covering every source row."""

    if epoch_size < 0:
        raise ValueError("arena_mixture_epoch_size must be non-negative")
    if (
        isinstance(size_multiple, bool)
        or not isinstance(size_multiple, int)
        or size_multiple < 1
    ):
        raise ValueError("Arena mixture size_multiple must be a positive integer")
    for stream in streams:
        if not rows_by_stream.get(stream.name):
            raise ValueError(f"Arena Stream {stream.name!r} contains no dataset rows")

    # ``sampling_weight`` weights each source dataset rather than replacing its
    # cardinality. The default epoch is the raw union, so every source appears
    # exactly once regardless of weight. An explicit smaller epoch uses the
    # weights to choose a deterministic subset without replacement.
    capacities = [len(rows_by_stream[stream.name]) for stream in streams]
    stream_masses = [
        capacity * stream.sampling_weight
        for capacity, stream in zip(capacities, streams, strict=True)
    ]
    total_mass = sum(stream_masses)
    if not all(math.isfinite(mass) and mass > 0 for mass in stream_masses) or not (
        math.isfinite(total_mass) and total_mass > 0
    ):
        raise ValueError("Arena Stream weights overflowed the mixture mass")
    total_source_rows = sum(capacities)
    if epoch_size:
        resolved_size = epoch_size
        if resolved_size > total_source_rows:
            raise ValueError(
                f"arena_mixture_epoch_size={resolved_size} cannot exceed the "
                f"{total_source_rows} unique source rows"
            )
        if resolved_size % size_multiple:
            raise ValueError(
                f"arena_mixture_epoch_size={resolved_size} must be divisible by "
                f"the training batch size {size_multiple}"
            )
    else:
        # Do not pad the raw union to a batch multiple: padding would repeat rows.
        # The training sampler may drop a final incomplete batch instead.
        resolved_size = total_source_rows
    if resolved_size <= 0:
        raise ValueError("Arena mixture epoch must contain at least one prompt row")

    if resolved_size == total_source_rows:
        counts = capacities
    else:
        counts = [0] * len(streams)
        remaining = resolved_size
        active = set(range(len(streams)))
        while remaining and active:
            active_mass = sum(stream_masses[index] for index in active)
            capped = [
                index
                for index in active
                if remaining * stream_masses[index] / active_mass >= capacities[index]
            ]
            if capped:
                for index in capped:
                    counts[index] = capacities[index]
                    remaining -= counts[index]
                    active.remove(index)
                continue

            exact = {
                index: remaining * stream_masses[index] / active_mass
                for index in active
            }
            for index, value in exact.items():
                counts[index] = int(value)
            remainder = resolved_size - sum(counts)
            order = sorted(
                active,
                key=lambda index: (
                    -(exact[index] - counts[index]),
                    streams[index].name,
                ),
            )
            for index in order[:remainder]:
                counts[index] += 1
            remaining = 0

    mixed_rows: list[dict[str, str]] = []
    emitted = [0] * len(streams)
    while len(mixed_rows) < resolved_size:
        candidates = [
            index for index, count in enumerate(counts) if emitted[index] < count
        ]
        selected = min(
            candidates,
            key=lambda index: (
                (emitted[index] + 1) / streams[index].sampling_weight,
                streams[index].name,
            ),
        )
        stream = streams[selected]
        source_rows = rows_by_stream[stream.name]
        mixed_rows.append(dict(source_rows[emitted[selected]]))
        emitted[selected] += 1
    return mixed_rows
