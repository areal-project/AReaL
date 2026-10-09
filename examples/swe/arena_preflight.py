"""Validate a multi-Stream Arena configuration with the Arena OpenAPI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any
from urllib.parse import quote

import httpx

from examples.swe.arena_client import (
    ArenaAPIError,
    ArenaOpenAPIClient,
    infer_llm_protocol_from_harness,
)
from examples.swe.arena_config import load_arena_stream_configs
from examples.swe.arena_types import ArenaRewardRefConfig


def _arena_api_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/openapi/v1"):
        return base_url
    return f"{base_url}/openapi/v1"


def _arena_json(
    base_url: str,
    args: list[str],
    *,
    retries: int,
    client: httpx.Client | None = None,
) -> Mapping[str, Any]:
    token = os.getenv("ARENA_OPENAPI_TOKEN", "")
    if not token:
        raise ArenaAPIError("ARENA_OPENAPI_TOKEN is required for Arena preflight")

    command = tuple(args[:2])
    options = dict(zip(args[3::2], args[4::2]))
    api_base = _arena_api_base_url(base_url)
    if command == ("stream", "get") and len(args) == 3:
        method = "GET"
        url = f"{api_base}/streams/{quote(args[2], safe='')}"
        params = None
    elif command == ("harness", "version-get") and len(args) == 4:
        method = "GET"
        url = (
            f"{api_base}/harnesses/{quote(args[2], safe='')}/versions/"
            f"{quote(args[3], safe='')}"
        )
        params = None
    elif command == ("stream", "dataset") and len(args) >= 3:
        method = "POST"
        url = f"{api_base}/streams/{quote(args[2], safe='')}/dataset"
        params = {
            "limit": options.get("--limit", "1"),
            "offset": options.get("--offset", "0"),
        }
    elif command == ("stream", "tasks") and len(args) >= 3:
        method = "GET"
        url = f"{api_base}/streams/{quote(args[2], safe='')}/tasks"
        params = {
            "limit": options.get("--limit", "100"),
            "offset": options.get("--offset", "0"),
        }
    else:
        raise ValueError(f"Unsupported Arena preflight operation: {args}")

    headers = {"Authorization": f"Bearer {token}"}
    owns_client = client is None
    if client is None:
        timeout = httpx.Timeout(
            float(os.getenv("ARENA_PREFLIGHT_HTTP_TIMEOUT", "180")),
            connect=float(os.getenv("ARENA_PREFLIGHT_CONNECT_TIMEOUT", "10")),
        )
        client = httpx.Client(timeout=timeout, trust_env=False)
    try:
        for attempt in range(retries + 1):
            try:
                response = client.request(method, url, params=params, headers=headers)
            except httpx.RequestError as exc:
                if attempt == retries:
                    raise ArenaAPIError(
                        "Arena preflight request failed after "
                        f"{attempt + 1} attempts: {type(exc).__name__}"
                    ) from exc
                time.sleep(min(2**attempt, 10))
                continue
            retryable = (
                response.status_code == 429
                or response.status_code >= 500
                or (
                    response.status_code == 403
                    and "spanner-http-ant-group-watch-all" in response.text[:1000]
                )
            )
            if retryable and attempt < retries:
                response.close()
                time.sleep(min(2**attempt, 10))
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ArenaAPIError(
                    f"Arena OpenAPI returned HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                ) from exc
            try:
                payload = response.json()
            except ValueError as exc:
                raise ArenaAPIError(
                    "Arena OpenAPI returned a non-JSON response"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ArenaAPIError("Arena OpenAPI response must be a JSON object")
            return payload
    finally:
        if owns_client:
            client.close()
    raise AssertionError("unreachable")


def _payload_data(payload: Mapping[str, Any]) -> Any:
    return payload.get("data", payload)


def _reward_ref(value: Any, *, stream_name: str) -> ArenaRewardRefConfig:
    if value is None:
        return ArenaRewardRefConfig()
    if not isinstance(value, Mapping):
        raise ArenaAPIError(
            f"Arena Stream {stream_name!r} default_reward_ref must be an object"
        )
    key = value.get("key")
    version = value.get("version")
    if (
        not isinstance(key, str)
        or not key
        or not isinstance(version, str)
        or not version
    ):
        raise ArenaAPIError(
            f"Arena Stream {stream_name!r} default_reward_ref is incomplete"
        )
    return ArenaRewardRefConfig(key=key, version=version)


def _harness_ref(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    key = value.get("key")
    version = value.get("version")
    if (
        not isinstance(key, str)
        or not key
        or not isinstance(version, str)
        or not version
    ):
        return ""
    return f"{key}@{version}"


def _stream_object(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = _payload_data(payload)
    if isinstance(data, Mapping) and isinstance(data.get("stream"), Mapping):
        return data["stream"]
    if isinstance(data, Mapping):
        return data
    raise ArenaAPIError("Arena stream get response is missing the Stream object")


def _dataset_total(payload: Mapping[str, Any]) -> int:
    data = _payload_data(payload)
    total = data.get("total") if isinstance(data, Mapping) else None
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ArenaAPIError("Arena Stream dataset must contain at least one row")
    return total


def _task_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    data = _payload_data(payload)
    if isinstance(data, Mapping):
        data = data.get("items", [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, Mapping)]


def _recent_health(
    rows: list[Mapping[str, Any]],
    *,
    harness: str,
    reward_ref: ArenaRewardRefConfig,
) -> tuple[int, int]:
    harness_key, _, harness_version = harness.partition("@")
    terminal = 0
    failures = 0
    for item in rows:
        task = item.get("task", item)
        if not isinstance(task, Mapping):
            continue
        task_harness = task.get("harness_ref")
        if not isinstance(task_harness, Mapping) or (
            task_harness.get("key"),
            task_harness.get("version"),
        ) != (harness_key, harness_version):
            continue
        task_reward = task.get("reward_ref")
        if reward_ref.key and (
            not isinstance(task_reward, Mapping)
            or (
                task_reward.get("key"),
                task_reward.get("version"),
            )
            != (reward_ref.key, reward_ref.version)
        ):
            continue
        status = str(task.get("status") or "").upper()
        if status in {"DONE", "OK"}:
            terminal += 1
        elif status in ArenaOpenAPIClient.FAILED_TASK_STATUSES:
            terminal += 1
            failures += 1
    return terminal, failures


def validate_streams(
    streams_file: str = "",
    *,
    streams_yaml_b64: str = "",
    base_url: str,
    default_harness: str = "",
    retries: int = 3,
    recent_tasks: int = 100,
    min_terminal_tasks: int = 1,
    max_failure_rate: float = 0.8,
    allow_unhealthy: bool = False,
) -> list[dict[str, Any]]:
    """Resolve and validate every configured Stream without launching a task."""

    timeout = httpx.Timeout(
        float(os.getenv("ARENA_PREFLIGHT_HTTP_TIMEOUT", "180")),
        connect=float(os.getenv("ARENA_PREFLIGHT_CONNECT_TIMEOUT", "10")),
    )
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        return _validate_streams_with_client(
            streams_file,
            streams_yaml_b64=streams_yaml_b64,
            base_url=base_url,
            default_harness=default_harness,
            retries=retries,
            recent_tasks=recent_tasks,
            min_terminal_tasks=min_terminal_tasks,
            max_failure_rate=max_failure_rate,
            allow_unhealthy=allow_unhealthy,
            client=client,
        )


def _validate_streams_with_client(
    streams_file: str = "",
    *,
    streams_yaml_b64: str = "",
    base_url: str,
    default_harness: str = "",
    retries: int = 3,
    recent_tasks: int = 100,
    min_terminal_tasks: int = 1,
    max_failure_rate: float = 0.8,
    allow_unhealthy: bool = False,
    client: httpx.Client,
) -> list[dict[str, Any]]:
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if recent_tasks < 1:
        raise ValueError("recent_tasks must be positive")
    if min_terminal_tasks < 0:
        raise ValueError("min_terminal_tasks must be non-negative")
    if not 0.0 <= max_failure_rate <= 1.0:
        raise ValueError("max_failure_rate must be between 0 and 1")

    streams = load_arena_stream_configs(
        {
            "arena_streams_file": streams_file,
            "arena_streams_yaml_b64": streams_yaml_b64,
            "arena_harness": default_harness,
        }
    )
    summaries: list[dict[str, Any]] = []
    for configured in streams:
        stream_payload = _arena_json(
            base_url,
            ["stream", "get", configured.stream_id],
            retries=retries,
            client=client,
        )
        stream = _stream_object(stream_payload)
        actual_stream_id = stream.get("stream_id")
        if actual_stream_id != configured.stream_id:
            raise ArenaAPIError(
                f"Arena Stream lookup returned {actual_stream_id!r}, expected "
                f"{configured.stream_id!r}"
            )
        if stream.get("status") != "ACTIVE":
            raise ArenaAPIError(f"Arena Stream {configured.stream_id!r} is not ACTIVE")

        actual_reward_ref = _reward_ref(
            stream.get("default_reward_ref"), stream_name=configured.name
        )
        expected = configured.expected_reward_ref
        if expected.key and expected != actual_reward_ref:
            raise ArenaAPIError(
                f"Arena Stream {configured.name!r} reward_ref drifted: expected "
                f"{expected.key}@{expected.version}, got "
                f"{actual_reward_ref.key}@{actual_reward_ref.version}"
            )

        harness = configured.harness or _harness_ref(stream.get("default_harness_ref"))
        harness_key, separator, harness_version = harness.partition("@")
        if not separator or not harness_key or not harness_version:
            raise ArenaAPIError(
                f"Arena Stream {configured.name!r} requires Harness key@version"
            )
        harness_payload = _arena_json(
            base_url,
            ["harness", "version-get", harness_key, harness_version],
            retries=retries,
            client=client,
        )
        harness_data = _payload_data(harness_payload)
        harness_status = (
            harness_data.get("status") if isinstance(harness_data, Mapping) else None
        )
        if harness_status != "PUBLISHED":
            raise ArenaAPIError(f"Arena Harness {harness!r} is not PUBLISHED")

        dataset_payload = _arena_json(
            base_url,
            ["stream", "dataset", configured.stream_id, "--limit", "1"],
            retries=retries,
            client=client,
        )
        dataset_total = _dataset_total(dataset_payload)
        # The task-list endpoint may return either a paginated object or a bare
        # list under ``data``. Request the full health window directly; reusing a
        # one-row total probe would silently evaluate only one task for the latter.
        tasks_payload = _arena_json(
            base_url,
            [
                "stream",
                "tasks",
                configured.stream_id,
                "--limit",
                str(recent_tasks),
                "--offset",
                "0",
            ],
            retries=retries,
            client=client,
        )
        terminal_count, failure_count = _recent_health(
            _task_rows(tasks_payload),
            harness=harness,
            reward_ref=actual_reward_ref,
        )
        failure_rate = failure_count / terminal_count if terminal_count else 0.0
        unhealthy = (
            terminal_count < min_terminal_tasks or failure_rate >= max_failure_rate
        )
        if unhealthy and not allow_unhealthy:
            raise ArenaAPIError(
                f"Arena Stream {configured.name!r} failed health check: "
                f"terminal={terminal_count}, failures={failure_count}, "
                f"failure_rate={failure_rate:.4f}"
            )

        protocol = configured.llm_protocol or infer_llm_protocol_from_harness(harness)
        resolved = replace(
            configured,
            harness=harness,
            llm_protocol=protocol,
            expected_reward_ref=actual_reward_ref,
        )
        summary = asdict(resolved)
        # Task env values can contain operational secrets and are irrelevant to
        # the preflight report.
        summary["task_envs"] = sorted(resolved.task_envs)
        summary.update(
            {
                "dataset_total": dataset_total,
                "recent_terminal_tasks": terminal_count,
                "recent_failures": failure_count,
                "recent_failure_rate": failure_rate,
                "unhealthy_override": unhealthy,
            }
        )
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--streams-file")
    source.add_argument(
        "--streams-env-b64",
        metavar="NAME",
        help="Read base64-encoded inline Streams YAML from environment variable NAME",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--default-harness", default="")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--recent-tasks", type=int, default=100)
    parser.add_argument("--min-terminal-tasks", type=int, default=1)
    parser.add_argument("--max-failure-rate", type=float, default=0.8)
    parser.add_argument("--allow-unhealthy", action="store_true")
    args = parser.parse_args()

    streams_yaml_b64 = ""
    if args.streams_env_b64:
        streams_yaml_b64 = os.getenv(args.streams_env_b64, "")
        if not streams_yaml_b64.strip():
            parser.error(f"environment variable {args.streams_env_b64!r} is empty")

    summaries = validate_streams(
        args.streams_file or "",
        streams_yaml_b64=streams_yaml_b64,
        base_url=args.base_url,
        default_harness=args.default_harness,
        retries=args.retries,
        recent_tasks=args.recent_tasks,
        min_terminal_tasks=args.min_terminal_tasks,
        max_failure_rate=args.max_failure_rate,
        allow_unhealthy=args.allow_unhealthy,
    )
    json.dump({"streams": summaries}, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
