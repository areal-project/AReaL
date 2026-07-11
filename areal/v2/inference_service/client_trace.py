# SPDX-License-Identifier: Apache-2.0

"""Canonical client-local observations for InfBridge generation calls.

These values commit to JSON prepared and parsed by the client.  They are not
remote-server, model-identity, or model-weight attestations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from areal.api.io_struct import ModelResponse

__all__ = [
    "GenerationAttemptTrace",
    "GenerationPhysicalTrace",
    "generation_physical_trace_bytes",
    "generation_physical_trace_sha256",
    "parsed_response_json_sha256",
    "prepared_request_json_sha256",
    "validate_generation_physical_trace_response",
]

_TRACE_KIND = "areal-generation-physical-trace-v1"
_PREPARED_REQUEST_JSON_DOMAIN = b"areal-infbridge-prepared-request-json-v1\0"
_PARSED_RESPONSE_JSON_DOMAIN = b"areal-infbridge-parsed-response-json-v1\0"
_SHA256_HEX_CHARS = frozenset("0123456789abcdef")


def _require_exact_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")


def _require_nonempty_str(name: str, value: object) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty str")


def _require_sha256(name: str, value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in _SHA256_HEX_CHARS for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _require_token_ids(name: str, value: object) -> None:
    if type(value) is not tuple or any(
        type(token_id) is not int or token_id < 0 for token_id in value
    ):
        raise ValueError(f"{name} must be a tuple of non-negative ints")


def _canonical_json_bytes(value: object) -> bytes:
    """Encode a deterministic Python-v1 JSON semantic view."""
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _canonical_json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def prepared_request_json_sha256(value: object) -> str:
    """Hash the locally prepared request JSON with an explicit domain."""
    return _canonical_json_sha256(value, domain=_PREPARED_REQUEST_JSON_DOMAIN)


def parsed_response_json_sha256(value: object) -> str:
    """Hash the locally parsed response JSON with an explicit domain."""
    return _canonical_json_sha256(value, domain=_PARSED_RESPONSE_JSON_DOMAIN)


@dataclass(frozen=True, slots=True)
class GenerationAttemptTrace:
    """Immutable client-side observations for one physical HTTP attempt.

    The hashes commit to JSON prepared and parsed inside this client; they are
    not hashes of raw HTTP bytes, and complete JSON bodies are not copied into
    the trace.  Submitted token IDs may still encode sensitive prompt content
    and require the same access and retention controls as the source request.
    """

    attempt_index: int
    client_version_before_send: int
    client_version_after_receive: int
    output_version_label: int
    remaining_new_tokens: int
    endpoint: str
    method: str
    prepared_request_json_sha256: str
    parsed_response_json_sha256: str
    submitted_input_token_ids: tuple[int, ...] | None
    raw_stop_reason: str
    output_token_ids: tuple[int, ...]
    output_logprob_count: int

    def __post_init__(self) -> None:
        _require_exact_int("attempt_index", self.attempt_index)
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        _require_exact_int(
            "client_version_before_send", self.client_version_before_send
        )
        _require_exact_int(
            "client_version_after_receive", self.client_version_after_receive
        )
        _require_exact_int("output_version_label", self.output_version_label)
        _require_exact_int("remaining_new_tokens", self.remaining_new_tokens)
        if self.remaining_new_tokens <= 0:
            raise ValueError("remaining_new_tokens must be positive")
        _require_nonempty_str("endpoint", self.endpoint)
        _require_nonempty_str("method", self.method)
        if self.method not in ("GET", "POST"):
            raise ValueError(
                "method must be the effective GET or POST transport method"
            )
        _require_sha256(
            "prepared_request_json_sha256", self.prepared_request_json_sha256
        )
        _require_sha256("parsed_response_json_sha256", self.parsed_response_json_sha256)
        if self.submitted_input_token_ids is not None:
            _require_token_ids(
                "submitted_input_token_ids", self.submitted_input_token_ids
            )
        _require_nonempty_str("raw_stop_reason", self.raw_stop_reason)
        if self.raw_stop_reason not in ("length", "stop", "tool_calls", "abort"):
            raise ValueError(
                "raw_stop_reason must be length, stop, tool_calls, or abort"
            )
        _require_token_ids("output_token_ids", self.output_token_ids)
        _require_exact_int("output_logprob_count", self.output_logprob_count)
        if self.output_logprob_count != len(self.output_token_ids):
            raise ValueError(
                "output_logprob_count must equal the number of output token IDs"
            )


@dataclass(frozen=True, slots=True)
class GenerationPhysicalTrace:
    """Immutable client-local trace for one logical generation.

    This commits to the pre/post-transport JSON views that InfBridge locally
    prepared and parsed, including abort/resubmit attempts.  It does not
    establish that the request ID was transmitted to or echoed by the server,
    and it does *not* authenticate the remote server, deployed model, or model
    weights.
    """

    schema_version: int
    request_id: str
    backend_kind: str
    backend_addr_sha256: str
    request_input_token_ids: tuple[int, ...]
    effective_max_new_tokens: int
    configured_attempt_limit: int
    initial_client_version: int
    final_client_version: int
    attempts: tuple[GenerationAttemptTrace, ...]
    final_output_token_ids: tuple[int, ...]
    final_stop_reason: str
    terminal_reason: str

    def __post_init__(self) -> None:
        _validate_generation_physical_trace(self)


def _validate_generation_physical_trace(trace: GenerationPhysicalTrace) -> None:
    _require_exact_int("schema_version", trace.schema_version)
    if trace.schema_version != 1:
        raise ValueError("schema_version must be 1")
    _require_nonempty_str("request_id", trace.request_id)
    _require_nonempty_str("backend_kind", trace.backend_kind)
    _require_sha256("backend_addr_sha256", trace.backend_addr_sha256)
    _require_token_ids("request_input_token_ids", trace.request_input_token_ids)
    _require_exact_int("effective_max_new_tokens", trace.effective_max_new_tokens)
    if trace.effective_max_new_tokens <= 0:
        raise ValueError("effective_max_new_tokens must be positive")
    _require_exact_int("configured_attempt_limit", trace.configured_attempt_limit)
    if trace.configured_attempt_limit < 0:
        raise ValueError("configured_attempt_limit must be non-negative")
    _require_exact_int("initial_client_version", trace.initial_client_version)
    _require_exact_int("final_client_version", trace.final_client_version)
    if type(trace.attempts) is not tuple or any(
        type(attempt) is not GenerationAttemptTrace for attempt in trace.attempts
    ):
        raise ValueError("attempts must be a tuple of GenerationAttemptTrace values")
    if len(trace.attempts) > trace.configured_attempt_limit:
        raise ValueError("attempts cannot exceed configured_attempt_limit")
    _require_token_ids("final_output_token_ids", trace.final_output_token_ids)

    output_count_before_attempt = 0
    for expected_index, attempt in enumerate(trace.attempts):
        # Revalidate nested frozen values in case callers bypassed dataclass
        # construction with object.__setattr__.
        attempt.__post_init__()
        if attempt.attempt_index != expected_index:
            raise ValueError("attempt indexes must be contiguous and start at zero")
        expected_remaining = (
            trace.effective_max_new_tokens - output_count_before_attempt
        )
        if attempt.remaining_new_tokens != expected_remaining:
            raise ValueError(
                "remaining_new_tokens must equal effective_max_new_tokens minus "
                "previously observed output tokens"
            )
        output_count_before_attempt += len(attempt.output_token_ids)
        if expected_index < len(trace.attempts) - 1:
            if attempt.raw_stop_reason != "abort":
                raise ValueError("only an abort may precede another physical attempt")

    observed_output = tuple(
        token_id for attempt in trace.attempts for token_id in attempt.output_token_ids
    )
    if trace.final_output_token_ids != observed_output:
        raise ValueError(
            "final_output_token_ids must equal concatenated attempt output_token_ids"
        )

    _require_nonempty_str("final_stop_reason", trace.final_stop_reason)
    if trace.final_stop_reason not in ("length", "stop", "tool_calls"):
        raise ValueError("final_stop_reason must be length, stop, or tool_calls")
    _require_nonempty_str("terminal_reason", trace.terminal_reason)
    if trace.terminal_reason not in (
        "backend_stop",
        "backend_tool_calls",
        "backend_length",
        "budget_exhausted",
        "attempt_limit",
    ):
        raise ValueError("invalid terminal_reason")

    expected_terminal: tuple[str, str | None]
    if trace.terminal_reason == "backend_stop":
        expected_terminal = ("stop", "stop")
    elif trace.terminal_reason == "backend_tool_calls":
        expected_terminal = ("tool_calls", "tool_calls")
    elif trace.terminal_reason == "backend_length":
        expected_terminal = ("length", "length")
    elif trace.terminal_reason == "budget_exhausted":
        expected_terminal = ("length", "abort")
    else:
        expected_terminal = ("length", "abort" if trace.attempts else None)

    expected_stop, expected_raw_stop = expected_terminal
    actual_raw_stop = trace.attempts[-1].raw_stop_reason if trace.attempts else None
    if trace.final_stop_reason != expected_stop or actual_raw_stop != expected_raw_stop:
        raise ValueError(
            "terminal_reason is inconsistent with the observed stop reasons"
        )
    if trace.terminal_reason == "budget_exhausted":
        if len(trace.final_output_token_ids) < trace.effective_max_new_tokens:
            raise ValueError(
                "budget_exhausted requires at least effective_max_new_tokens outputs"
            )
    if trace.terminal_reason == "attempt_limit":
        if len(trace.attempts) != trace.configured_attempt_limit:
            raise ValueError(
                "attempt_limit requires exactly configured_attempt_limit attempts"
            )
        if len(trace.final_output_token_ids) >= trace.effective_max_new_tokens:
            raise ValueError(
                "attempt_limit cannot claim an already exhausted generation budget"
            )


def _generation_physical_trace_value(trace: GenerationPhysicalTrace) -> dict[str, Any]:
    return {
        "kind": _TRACE_KIND,
        "schema_version": trace.schema_version,
        "request_id": trace.request_id,
        "backend_kind": trace.backend_kind,
        "backend_addr_sha256": trace.backend_addr_sha256,
        "request_input_token_ids": list(trace.request_input_token_ids),
        "effective_max_new_tokens": trace.effective_max_new_tokens,
        "configured_attempt_limit": trace.configured_attempt_limit,
        "initial_client_version": trace.initial_client_version,
        "final_client_version": trace.final_client_version,
        "attempts": [
            {
                "attempt_index": attempt.attempt_index,
                "client_version_before_send": attempt.client_version_before_send,
                "client_version_after_receive": attempt.client_version_after_receive,
                "output_version_label": attempt.output_version_label,
                "remaining_new_tokens": attempt.remaining_new_tokens,
                "endpoint": attempt.endpoint,
                "method": attempt.method,
                "prepared_request_json_sha256": attempt.prepared_request_json_sha256,
                "parsed_response_json_sha256": attempt.parsed_response_json_sha256,
                "submitted_input_token_ids": (
                    list(attempt.submitted_input_token_ids)
                    if attempt.submitted_input_token_ids is not None
                    else None
                ),
                "raw_stop_reason": attempt.raw_stop_reason,
                "output_token_ids": list(attempt.output_token_ids),
                "output_logprob_count": attempt.output_logprob_count,
            }
            for attempt in trace.attempts
        ],
        "final_output_token_ids": list(trace.final_output_token_ids),
        "final_stop_reason": trace.final_stop_reason,
        "terminal_reason": trace.terminal_reason,
    }


def generation_physical_trace_bytes(trace: GenerationPhysicalTrace) -> bytes:
    """Return the canonical, content-addressable JSON representation."""
    if type(trace) is not GenerationPhysicalTrace:
        raise TypeError("trace must be a GenerationPhysicalTrace")
    _validate_generation_physical_trace(trace)
    return _canonical_json_bytes(_generation_physical_trace_value(trace))


def generation_physical_trace_sha256(trace: GenerationPhysicalTrace) -> str:
    """Hash :func:`generation_physical_trace_bytes` with SHA-256."""
    return hashlib.sha256(generation_physical_trace_bytes(trace)).hexdigest()


def validate_generation_physical_trace_response(
    response: ModelResponse,
    trace: GenerationPhysicalTrace,
) -> None:
    """Validate the ``ModelResponse`` fields represented in ``trace``.

    The trace binds token IDs, stop reason, log-probability *count*, and output
    version labels.  It intentionally does not bind log-probability values,
    routed experts, tokenizer objects, or latency measurements.
    """
    generation_physical_trace_bytes(trace)
    if type(response.input_tokens) is not list:
        raise TypeError("response.input_tokens must be a list")
    response_input_tokens = tuple(response.input_tokens)
    _require_token_ids("response.input_tokens", response_input_tokens)
    if response_input_tokens != trace.request_input_token_ids:
        raise ValueError("response.input_tokens do not match request_input_token_ids")
    if type(response.output_tokens) is not list:
        raise TypeError("response.output_tokens must be a list")
    response_output_tokens = tuple(response.output_tokens)
    _require_token_ids("response.output_tokens", response_output_tokens)
    if response_output_tokens != trace.final_output_token_ids:
        raise ValueError("response.output_tokens do not match final_output_token_ids")
    if (
        type(response.stop_reason) is not str
        or response.stop_reason != trace.final_stop_reason
    ):
        raise ValueError("response.stop_reason does not match final_stop_reason")
    if type(response.output_logprobs) is not list:
        raise TypeError("response.output_logprobs must be a list")
    if len(response.output_logprobs) != len(trace.final_output_token_ids):
        raise ValueError("response.output_logprobs count does not match traced output")
    if type(response.output_versions) is not list or any(
        type(version) is not int for version in response.output_versions
    ):
        raise TypeError("response.output_versions must be a list of exact ints")
    traced_output_versions = tuple(
        attempt.output_version_label
        for attempt in trace.attempts
        for _ in attempt.output_token_ids
    )
    if tuple(response.output_versions) != traced_output_versions:
        raise ValueError(
            "response.output_versions do not match traced output_version_label values"
        )
