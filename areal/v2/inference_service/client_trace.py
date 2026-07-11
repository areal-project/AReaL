# SPDX-License-Identifier: Apache-2.0

"""Canonical client-local observations for InfBridge generation calls.

These values commit to JSON prepared and parsed by the client.  They are not
remote-server, model-identity, or model-weight attestations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from areal.api.io_struct import ModelResponse

__all__ = [
    "GenerationAttemptTrace",
    "GenerationPhysicalTrace",
    "GenerationResponseEvidence",
    "ParsedResponseJSONEvidence",
    "canonical_parsed_response_json_bytes",
    "generation_physical_trace_from_bytes",
    "generation_physical_trace_bytes",
    "generation_physical_trace_sha256",
    "generation_response_evidence_bytes",
    "generation_response_evidence_from_bytes",
    "generation_response_evidence_sha256",
    "generation_response_evidence_values",
    "parsed_response_json_sha256",
    "prepared_request_json_sha256",
    "validate_generation_response_evidence",
    "validate_generation_physical_trace_response",
]

_TRACE_KIND = "areal-generation-physical-trace-v2"
_RESPONSE_EVIDENCE_KIND = "areal-generation-response-evidence-v1"
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


def canonical_parsed_response_json_bytes(value: object) -> bytes:
    """Return the exact canonical JSON preimage used by response evidence."""

    if type(value) is not dict:
        raise TypeError("parsed response JSON must be a dict")
    return _canonical_json_bytes(value)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _parse_canonical_json_object(value: object) -> dict[str, Any]:
    if type(value) is not bytes or not value:
        raise ValueError("canonical_json_bytes must be non-empty exact bytes")
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("canonical_json_bytes must be ASCII") from error
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("canonical_json_bytes must contain strict JSON") from error
    if type(parsed) is not dict:
        raise ValueError("canonical JSON must decode to an object")
    if _canonical_json_bytes(parsed) != value:
        raise ValueError("canonical_json_bytes are not canonical JSON")
    return parsed


def _require_exact_keys(
    value: dict[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{name} has missing or unknown fields")


@dataclass(frozen=True, slots=True)
class ParsedResponseJSONEvidence:
    """A content-addressed, replayable parsed-response JSON preimage."""

    attempt_index: int
    parsed_response_json_sha256: str
    canonical_json_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_exact_int("attempt_index", self.attempt_index)
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        _require_sha256("parsed_response_json_sha256", self.parsed_response_json_sha256)
        parsed = _parse_canonical_json_object(self.canonical_json_bytes)
        if parsed_response_json_sha256(parsed) != self.parsed_response_json_sha256:
            raise ValueError(
                "canonical response JSON does not match parsed_response_json_sha256"
            )


@dataclass(frozen=True, slots=True)
class GenerationResponseEvidence:
    """Optional response preimages kept outside the compact physical trace.

    Response JSON can contain generated text and other sensitive server data.
    Callers should apply the same access, encryption, and retention controls as
    for prompts and generated token IDs.
    """

    schema_version: int
    generation_trace_sha256: str
    attempts: tuple[ParsedResponseJSONEvidence, ...]

    def __post_init__(self) -> None:
        _validate_generation_response_evidence(self)


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
    client_version_epoch_before_send: int
    client_version_epoch_after_receive: int
    output_version_epoch: int
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
        _require_exact_int(
            "client_version_epoch_before_send",
            self.client_version_epoch_before_send,
        )
        _require_exact_int(
            "client_version_epoch_after_receive",
            self.client_version_epoch_after_receive,
        )
        _require_exact_int("output_version_epoch", self.output_version_epoch)
        if (
            self.client_version_epoch_before_send < 0
            or self.client_version_epoch_after_receive
            < self.client_version_epoch_before_send
            or self.output_version_epoch < self.client_version_epoch_after_receive
        ):
            raise ValueError("client version epochs must be non-negative and monotonic")
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
    weights.  Version epochs expose calls to ``InfBridge.set_version`` between
    observations, including an A→B→A label change; they do not observe direct
    state mutation or prove which weights a remote server used.
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
    initial_client_version_epoch: int
    final_client_version_epoch: int
    attempts: tuple[GenerationAttemptTrace, ...]
    final_output_token_ids: tuple[int, ...]
    final_stop_reason: str
    terminal_reason: str

    def __post_init__(self) -> None:
        _validate_generation_physical_trace(self)


def _validate_generation_physical_trace(trace: GenerationPhysicalTrace) -> None:
    _require_exact_int("schema_version", trace.schema_version)
    if trace.schema_version != 2:
        raise ValueError("schema_version must be 2")
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
    _require_exact_int(
        "initial_client_version_epoch", trace.initial_client_version_epoch
    )
    _require_exact_int("final_client_version_epoch", trace.final_client_version_epoch)
    if (
        trace.initial_client_version_epoch < 0
        or trace.final_client_version_epoch < trace.initial_client_version_epoch
    ):
        raise ValueError(
            "trace client version epochs must be non-negative and monotonic"
        )
    if type(trace.attempts) is not tuple or any(
        type(attempt) is not GenerationAttemptTrace for attempt in trace.attempts
    ):
        raise ValueError("attempts must be a tuple of GenerationAttemptTrace values")
    if len(trace.attempts) > trace.configured_attempt_limit:
        raise ValueError("attempts cannot exceed configured_attempt_limit")
    _require_token_ids("final_output_token_ids", trace.final_output_token_ids)

    output_count_before_attempt = 0
    previous_version_epoch = trace.initial_client_version_epoch
    for expected_index, attempt in enumerate(trace.attempts):
        # Revalidate nested frozen values in case callers bypassed dataclass
        # construction with object.__setattr__.
        attempt.__post_init__()
        if attempt.attempt_index != expected_index:
            raise ValueError("attempt indexes must be contiguous and start at zero")
        if attempt.client_version_epoch_before_send < previous_version_epoch:
            raise ValueError("attempt client version epochs must be monotonic")
        expected_remaining = (
            trace.effective_max_new_tokens - output_count_before_attempt
        )
        if attempt.remaining_new_tokens != expected_remaining:
            raise ValueError(
                "remaining_new_tokens must equal effective_max_new_tokens minus "
                "previously observed output tokens"
            )
        if len(attempt.output_token_ids) > expected_remaining:
            raise ValueError(
                "attempt output_token_ids cannot exceed remaining_new_tokens"
            )
        output_count_before_attempt += len(attempt.output_token_ids)
        if expected_index < len(trace.attempts) - 1:
            if attempt.raw_stop_reason != "abort":
                raise ValueError("only an abort may precede another physical attempt")
        previous_version_epoch = attempt.output_version_epoch

    if trace.final_client_version_epoch < previous_version_epoch:
        raise ValueError("final client version epoch precedes attempt evidence")

    version_by_epoch: dict[int, int] = {}
    version_observations = [
        (trace.initial_client_version_epoch, trace.initial_client_version),
        *(
            observation
            for attempt in trace.attempts
            for observation in (
                (
                    attempt.client_version_epoch_before_send,
                    attempt.client_version_before_send,
                ),
                (
                    attempt.client_version_epoch_after_receive,
                    attempt.client_version_after_receive,
                ),
                (attempt.output_version_epoch, attempt.output_version_label),
            )
        ),
        (trace.final_client_version_epoch, trace.final_client_version),
    ]
    for version_epoch, version_label in version_observations:
        observed_label = version_by_epoch.setdefault(version_epoch, version_label)
        if observed_label != version_label:
            raise ValueError(
                "one client version epoch cannot carry different version labels"
            )

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
        "initial_client_version_epoch": trace.initial_client_version_epoch,
        "final_client_version_epoch": trace.final_client_version_epoch,
        "attempts": [
            {
                "attempt_index": attempt.attempt_index,
                "client_version_before_send": attempt.client_version_before_send,
                "client_version_after_receive": attempt.client_version_after_receive,
                "output_version_label": attempt.output_version_label,
                "client_version_epoch_before_send": (
                    attempt.client_version_epoch_before_send
                ),
                "client_version_epoch_after_receive": (
                    attempt.client_version_epoch_after_receive
                ),
                "output_version_epoch": attempt.output_version_epoch,
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


def generation_physical_trace_from_bytes(value: bytes) -> GenerationPhysicalTrace:
    """Strictly decode one canonical v2 physical-trace artifact."""

    decoded = _parse_canonical_json_object(value)
    _require_exact_keys(
        decoded,
        frozenset(
            {
                "kind",
                "schema_version",
                "request_id",
                "backend_kind",
                "backend_addr_sha256",
                "request_input_token_ids",
                "effective_max_new_tokens",
                "configured_attempt_limit",
                "initial_client_version",
                "final_client_version",
                "initial_client_version_epoch",
                "final_client_version_epoch",
                "attempts",
                "final_output_token_ids",
                "final_stop_reason",
                "terminal_reason",
            }
        ),
        "physical trace",
    )
    if decoded["kind"] != _TRACE_KIND:
        raise ValueError("physical trace kind does not match schema v2")
    request_input = decoded["request_input_token_ids"]
    attempts_value = decoded["attempts"]
    final_output = decoded["final_output_token_ids"]
    if (
        type(request_input) is not list
        or type(attempts_value) is not list
        or type(final_output) is not list
    ):
        raise ValueError("physical trace token IDs and attempts must be JSON arrays")
    attempts: list[GenerationAttemptTrace] = []
    attempt_keys = frozenset(
        {
            "attempt_index",
            "client_version_before_send",
            "client_version_after_receive",
            "output_version_label",
            "client_version_epoch_before_send",
            "client_version_epoch_after_receive",
            "output_version_epoch",
            "remaining_new_tokens",
            "endpoint",
            "method",
            "prepared_request_json_sha256",
            "parsed_response_json_sha256",
            "submitted_input_token_ids",
            "raw_stop_reason",
            "output_token_ids",
            "output_logprob_count",
        }
    )
    for attempt_value in attempts_value:
        if type(attempt_value) is not dict:
            raise ValueError("physical trace attempts must contain JSON objects")
        _require_exact_keys(attempt_value, attempt_keys, "physical trace attempt")
        submitted = attempt_value["submitted_input_token_ids"]
        output = attempt_value["output_token_ids"]
        if submitted is not None and type(submitted) is not list:
            raise ValueError("submitted_input_token_ids must be null or a JSON array")
        if type(output) is not list:
            raise ValueError("output_token_ids must be a JSON array")
        attempts.append(
            GenerationAttemptTrace(
                attempt_index=attempt_value["attempt_index"],
                client_version_before_send=attempt_value["client_version_before_send"],
                client_version_after_receive=attempt_value[
                    "client_version_after_receive"
                ],
                output_version_label=attempt_value["output_version_label"],
                client_version_epoch_before_send=attempt_value[
                    "client_version_epoch_before_send"
                ],
                client_version_epoch_after_receive=attempt_value[
                    "client_version_epoch_after_receive"
                ],
                output_version_epoch=attempt_value["output_version_epoch"],
                remaining_new_tokens=attempt_value["remaining_new_tokens"],
                endpoint=attempt_value["endpoint"],
                method=attempt_value["method"],
                prepared_request_json_sha256=attempt_value[
                    "prepared_request_json_sha256"
                ],
                parsed_response_json_sha256=attempt_value[
                    "parsed_response_json_sha256"
                ],
                submitted_input_token_ids=(
                    None if submitted is None else tuple(submitted)
                ),
                raw_stop_reason=attempt_value["raw_stop_reason"],
                output_token_ids=tuple(output),
                output_logprob_count=attempt_value["output_logprob_count"],
            )
        )
    trace = GenerationPhysicalTrace(
        schema_version=decoded["schema_version"],
        request_id=decoded["request_id"],
        backend_kind=decoded["backend_kind"],
        backend_addr_sha256=decoded["backend_addr_sha256"],
        request_input_token_ids=tuple(request_input),
        effective_max_new_tokens=decoded["effective_max_new_tokens"],
        configured_attempt_limit=decoded["configured_attempt_limit"],
        initial_client_version=decoded["initial_client_version"],
        final_client_version=decoded["final_client_version"],
        initial_client_version_epoch=decoded["initial_client_version_epoch"],
        final_client_version_epoch=decoded["final_client_version_epoch"],
        attempts=tuple(attempts),
        final_output_token_ids=tuple(final_output),
        final_stop_reason=decoded["final_stop_reason"],
        terminal_reason=decoded["terminal_reason"],
    )
    if generation_physical_trace_bytes(trace) != value:
        raise ValueError("physical trace does not round-trip canonically")
    return trace


def generation_physical_trace_sha256(trace: GenerationPhysicalTrace) -> str:
    """Hash :func:`generation_physical_trace_bytes` with SHA-256."""
    return hashlib.sha256(generation_physical_trace_bytes(trace)).hexdigest()


def _validate_generation_response_evidence(
    evidence: GenerationResponseEvidence,
) -> None:
    _require_exact_int("schema_version", evidence.schema_version)
    if evidence.schema_version != 1:
        raise ValueError("response evidence schema_version must be 1")
    _require_sha256("generation_trace_sha256", evidence.generation_trace_sha256)
    if type(evidence.attempts) is not tuple or any(
        type(attempt) is not ParsedResponseJSONEvidence for attempt in evidence.attempts
    ):
        raise ValueError(
            "response evidence attempts must be a tuple of "
            "ParsedResponseJSONEvidence values"
        )
    for expected_index, attempt in enumerate(evidence.attempts):
        attempt.__post_init__()
        if attempt.attempt_index != expected_index:
            raise ValueError(
                "response evidence attempt indexes must be contiguous and start at zero"
            )


def _generation_response_evidence_value(
    evidence: GenerationResponseEvidence,
) -> dict[str, Any]:
    return {
        "kind": _RESPONSE_EVIDENCE_KIND,
        "schema_version": evidence.schema_version,
        "generation_trace_sha256": evidence.generation_trace_sha256,
        "attempts": [
            {
                "attempt_index": attempt.attempt_index,
                "parsed_response_json_sha256": (attempt.parsed_response_json_sha256),
                "canonical_json_ascii": attempt.canonical_json_bytes.decode("ascii"),
            }
            for attempt in evidence.attempts
        ],
    }


def generation_response_evidence_bytes(
    evidence: GenerationResponseEvidence,
) -> bytes:
    """Return canonical JSON for the optional replayable response sidecar."""

    if type(evidence) is not GenerationResponseEvidence:
        raise TypeError("evidence must be GenerationResponseEvidence")
    _validate_generation_response_evidence(evidence)
    return _canonical_json_bytes(_generation_response_evidence_value(evidence))


def generation_response_evidence_from_bytes(
    value: bytes,
) -> GenerationResponseEvidence:
    """Strictly decode one canonical response-evidence sidecar."""

    decoded = _parse_canonical_json_object(value)
    _require_exact_keys(
        decoded,
        frozenset(
            {
                "kind",
                "schema_version",
                "generation_trace_sha256",
                "attempts",
            }
        ),
        "response evidence",
    )
    if decoded["kind"] != _RESPONSE_EVIDENCE_KIND:
        raise ValueError("response evidence kind does not match schema v1")
    attempts_value = decoded["attempts"]
    if type(attempts_value) is not list:
        raise ValueError("response evidence attempts must be a JSON array")
    attempts: list[ParsedResponseJSONEvidence] = []
    expected_attempt_keys = frozenset(
        {
            "attempt_index",
            "parsed_response_json_sha256",
            "canonical_json_ascii",
        }
    )
    for attempt_value in attempts_value:
        if type(attempt_value) is not dict:
            raise ValueError("response evidence attempts must contain JSON objects")
        _require_exact_keys(
            attempt_value,
            expected_attempt_keys,
            "response evidence attempt",
        )
        canonical_json_ascii = attempt_value["canonical_json_ascii"]
        if type(canonical_json_ascii) is not str:
            raise ValueError("canonical_json_ascii must be an exact str")
        try:
            canonical_json = canonical_json_ascii.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("canonical_json_ascii must contain only ASCII") from error
        attempts.append(
            ParsedResponseJSONEvidence(
                attempt_index=attempt_value["attempt_index"],
                parsed_response_json_sha256=attempt_value[
                    "parsed_response_json_sha256"
                ],
                canonical_json_bytes=canonical_json,
            )
        )
    evidence = GenerationResponseEvidence(
        schema_version=decoded["schema_version"],
        generation_trace_sha256=decoded["generation_trace_sha256"],
        attempts=tuple(attempts),
    )
    if generation_response_evidence_bytes(evidence) != value:
        raise ValueError("response evidence does not round-trip canonically")
    return evidence


def generation_response_evidence_sha256(
    evidence: GenerationResponseEvidence,
) -> str:
    return hashlib.sha256(generation_response_evidence_bytes(evidence)).hexdigest()


def generation_response_evidence_values(
    trace: GenerationPhysicalTrace,
    evidence: GenerationResponseEvidence,
) -> tuple[dict[str, Any], ...]:
    """Validate and decode response preimages bound to a physical trace."""

    generation_physical_trace_bytes(trace)
    generation_response_evidence_bytes(evidence)
    if evidence.generation_trace_sha256 != generation_physical_trace_sha256(trace):
        raise ValueError("response evidence does not bind this generation trace")
    if len(evidence.attempts) != len(trace.attempts):
        raise ValueError("response evidence must cover every physical attempt")
    values: list[dict[str, Any]] = []
    for traced_attempt, response_attempt in zip(
        trace.attempts,
        evidence.attempts,
        strict=True,
    ):
        if (
            response_attempt.attempt_index != traced_attempt.attempt_index
            or response_attempt.parsed_response_json_sha256
            != traced_attempt.parsed_response_json_sha256
        ):
            raise ValueError("response evidence attempt does not match trace")
        values.append(
            _parse_canonical_json_object(response_attempt.canonical_json_bytes)
        )
    return tuple(values)


def validate_generation_response_evidence(
    trace: GenerationPhysicalTrace,
    evidence: GenerationResponseEvidence,
) -> None:
    """Validate complete response-preimage coverage for one traced generation."""

    generation_response_evidence_values(trace, evidence)


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
