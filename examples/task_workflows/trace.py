# SPDX-License-Identifier: Apache-2.0
"""Small per-episode observations for the native model/tool loop."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextvars import ContextVar
from functools import lru_cache, wraps
from typing import Any

_tool_record: ContextVar[dict | None] = ContextVar("task_trace_tool", default=None)


@lru_cache(maxsize=2)
def _tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def mark_tool_execution(started: float, *, seconds: float | None = None) -> None:
    """Called inside the environment's existing lock, after any tool queue wait."""
    record = _tool_record.get()
    if record is not None:
        record.update(started=started, _clock=time.perf_counter())
        if seconds is not None:
            record["_service_seconds"] = seconds


class CallTrace:
    """Observe serial model turns and tool fan-out; never infer edges from timestamps."""

    def __init__(self, tokenizer: Any = None) -> None:
        self.tokenizer_path = os.environ.get("AREAL_TASK_TOKENIZER_PATH")
        self.tokenizer = tokenizer
        if self.tokenizer is None and self.tokenizer_path:
            self.tokenizer = _tokenizer(self.tokenizer_path)
        self.nodes: list[dict] = []
        self.requests: list[dict] = []
        self.pools: dict[str, int] = {}
        self.last_request: dict | None = None
        self.tools: list[dict] = []
        self.dependencies_complete = True
        self.lock = threading.Lock()

    def request(self, record: dict, *, messages: list | None = None) -> None:
        with self.lock:
            parents = self.tools or ([self.last_request] if self.last_request else [])
            if any(parent["finished"] is None for parent in parents):
                self.dependencies_complete = False
            record.update(
                name=f"request-{len(self.requests) + 1}",
                kind="request",
                predecessors=[parent["name"] for parent in parents],
                input_tokens=None,
                output_tokens=None,
                cached_prefix_tokens=None,
                monotonic=time.perf_counter(),
            )
            if messages is not None:
                record["tool_messages"] = [
                    {
                        "tool_call_id": message.get("tool_call_id"),
                        "content": message["content"],
                        "tokens": self._count_tokens(message["content"]),
                    }
                    for message in messages
                    if message.get("role") == "tool"
                    and isinstance(message.get("content"), str)
                ]
            self.nodes.append(record)
            self.requests.append(record)
            self.last_request, self.tools = record, []

    def bind_tools(self, runtime: Any) -> None:
        invoke = runtime.invoke
        pool = f"{runtime.kind}:{runtime.task_id}"
        self.pools[pool] = 1  # Both environments serialize execution per task.

        @wraps(invoke)
        def observed(name: str, arguments: dict) -> str:
            with self.lock:
                record = {
                    "name": f"tool-{len(self.nodes) + 1}",
                    "kind": "external",
                    "tool": name,
                    "pool": pool,
                    "predecessors": [self.last_request["name"]]
                    if self.last_request
                    else [],
                    "queued": time.time(),
                    "started": None,
                    "finished": None,
                    "seconds": None,
                    "output_tokens": None,
                }
                self.nodes.append(record)
                self.tools.append(record)
            token = _tool_record.set(record)
            result = None
            try:
                result = invoke(name, arguments)
                return result
            finally:
                record["finished"] = time.time()
                clock = record.pop("_clock", None)
                if clock is not None:
                    record["seconds"] = record.pop(
                        "_service_seconds", time.perf_counter() - clock
                    )
                _tool_record.reset(token)
                if isinstance(result, str):
                    content = result.encode("utf-8")
                    record.update(
                        returned_text=result,
                        output_tokens=self._count_tokens(result),
                        output_utf8_bytes=len(content),
                        output_sha256=hashlib.sha256(content).hexdigest(),
                    )

        runtime.invoke = observed

    def _count_tokens(self, value: str) -> int | None:
        if self.tokenizer is None:
            return None
        return len(self.tokenizer.encode(value, add_special_tokens=False))

    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "run_id": os.environ.get("AREAL_TRACE_RUN_ID"),
            "tokenizer_path": self.tokenizer_path,
            "tool_lengths_complete": all(
                row.get("output_tokens") is not None
                for row in self.nodes
                if row["kind"] == "external"
            ),
            "dependency_policy": "native_model_turns_with_tool_fanout",
            "dependencies_complete": self.dependencies_complete,
            "lengths_complete": all(
                (row["input_tokens"] or 0) > 0 and (row["output_tokens"] or 0) > 0
                for row in self.requests
            ),
            "external_pools": self.pools,
            "nodes": self.nodes,
        }


def observe_openai(record: dict, response: Any) -> None:
    value = response.model_dump() if hasattr(response, "model_dump") else response
    if not isinstance(value, dict):
        return
    usage = value.get("usage") or {}
    if usage.get("prompt_tokens") is not None:
        record["input_tokens"] = usage["prompt_tokens"]
    if usage.get("completion_tokens") is not None:
        record["output_tokens"] = usage["completion_tokens"]
    details = usage.get("prompt_tokens_details") or {}
    if details.get("cached_tokens") is not None:
        record["cached_prefix_tokens"] = details["cached_tokens"]
    if value.get("id"):
        record["completion_id"] = value["id"]
    for choice in value.get("choices") or []:
        if choice.get("finish_reason"):
            record["finish_reason"] = choice["finish_reason"]


def observe_anthropic(record: dict, value: dict) -> None:
    message = value.get("message") or value
    usage = message.get("usage") or {}
    if usage.get("input_tokens") is not None:
        record["input_tokens"] = sum(
            usage.get(key, 0) or 0
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
    if usage.get("output_tokens") is not None:
        record["output_tokens"] = usage["output_tokens"]
    # The v2 Anthropic translator can insert cache-read=0 without cache telemetry.
    if usage.get("cache_read_input_tokens"):
        record["cached_prefix_tokens"] = usage["cache_read_input_tokens"]
    if message.get("id"):
        record["completion_id"] = message["id"]
    reason = (value.get("delta") or {}).get("stop_reason") or message.get("stop_reason")
    if reason:
        record["finish_reason"] = reason


def observe_anthropic_data(record: dict, payload: bytes) -> None:
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return
    if isinstance(value, dict):
        observe_anthropic(record, value)
