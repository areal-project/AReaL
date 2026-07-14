# SPDX-License-Identifier: Apache-2.0

"""Closed-schema V1 wire values for exact Agent session lifecycles.

The reusable ``session_key`` is not enough to address one Agent lifetime.  A
safe cross-process protocol carries the Worker-minted session incarnation and
Worker audience through an indivisible ``open`` / ``run`` / ``close``
capability.  Advertising :data:`EXACT_SESSION_LIFECYCLE_CAPABILITY_V1` means
that all three operations use these values and that stateful legacy routes are
not a fallback for that Worker pair.

This module defines JSON values only.  It registers no route, authenticates no
caller, resolves no principal, and grants no local authority.  An identity is
descriptive replay data, not a bearer credential.  A future HTTP adapter must
authenticate the DataProxy-to-Worker hop before parsing these bodies, bind the
open request to a trusted principal, and compare every returned identity
exactly before changing DataProxy state.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum

from .memory_authorization import (
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)
from .memory_session_lifecycle import MemoryWorkerSessionIdentityV1
from .protocol import QueueMode
from .session_keys import validate_session_key
from .types import AgentRequest

EXACT_SESSION_LIFECYCLE_CAPABILITY_V1 = "exact_session_lifecycle_v1"
MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1 = 16 * 1024 * 1024

_SCHEMA_VERSION = 1
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_MAX_JSON_STRING_BYTES = 8 * 1024 * 1024
_CAPABILITY_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "session_key",
        "session_incarnation_id",
        "worker_audience_id",
    }
)
_CAPABILITIES_KEYS = frozenset(
    {
        "schema_version",
        "capabilities",
        "worker_audience_id",
    }
)
_OPEN_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "session_key",
        "open_request_id",
        "expected_worker_audience_id",
    }
)
_OPEN_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "open_request_id",
        "identity",
    }
)
_TURN_KEYS = frozenset(
    {
        "message",
        "run_id",
        "history",
        "queue_mode",
        "metadata",
    }
)
_RUN_REQUEST_KEYS = frozenset({"schema_version", "identity", "turn"})
_CLOSE_REQUEST_KEYS = frozenset({"schema_version", "identity"})
_CLOSE_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "identity",
        "outcome",
    }
)


class AgentSessionLifecycleTransportError(ValueError):
    """Base class for exact Agent session lifecycle transport failures."""


class AgentSessionLifecycleWireFormatError(AgentSessionLifecycleTransportError):
    """Raised when a lifecycle value does not use the exact V1 JSON schema."""


def _exact_dict(value: object, field_name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise AgentSessionLifecycleWireFormatError(f"{field_name} must be an object")
    if any(type(key) is not str for key in value):
        raise AgentSessionLifecycleWireFormatError(f"{field_name} keys must be strings")
    return value


def _closed_keys(
    value: dict[str, object],
    expected: frozenset[str],
    field_name: str,
) -> None:
    keys = set(value)
    if keys == expected:
        return
    missing = sorted(expected - keys)
    unknown = keys - expected
    details: list[str] = []
    if missing:
        details.append(f"missing={missing!r}")
    if unknown:
        details.append(f"unknown_count={len(unknown)}")
    raise AgentSessionLifecycleWireFormatError(
        f"{field_name} must use the exact V1 schema ({', '.join(details)})"
    )


def _schema_version(value: object, field_name: str) -> None:
    if type(value) is not int or value != _SCHEMA_VERSION:
        raise AgentSessionLifecycleWireFormatError(f"{field_name} must be integer 1")


def _string(
    value: object,
    field_name: str,
    *,
    allow_blank: bool = False,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a str")
    if not allow_blank and not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    return value


def _wire_string(
    value: object,
    field_name: str,
    *,
    allow_blank: bool = False,
) -> str:
    try:
        return _string(value, field_name, allow_blank=allow_blank)
    except (TypeError, ValueError) as error:
        raise AgentSessionLifecycleWireFormatError(str(error)) from error


def _session_key(value: object) -> str:
    return validate_session_key(value)


def _wire_session_key(value: object, field_name: str) -> str:
    try:
        return validate_session_key(value)
    except (TypeError, ValueError) as error:
        raise AgentSessionLifecycleWireFormatError(
            f"{field_name} is not a canonical Agent session key"
        ) from error


def _request_id(value: object) -> str:
    value = _string(value, "open_request_id")
    suffix = value.removeprefix("aopen_")
    if (
        not value.startswith("aopen_")
        or len(suffix) != 64
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise ValueError(
            "open_request_id must be aopen_ followed by 64 lowercase hex characters"
        )
    return value


def _wire_request_id(value: object) -> str:
    try:
        return _request_id(value)
    except (TypeError, ValueError) as error:
        raise AgentSessionLifecycleWireFormatError(
            "open_request_id is not a canonical V1 idempotency key"
        ) from error


def _capability(value: object) -> str:
    value = _string(value, "capability")
    if _CAPABILITY_RE.fullmatch(value) is None:
        raise ValueError("capability must use a canonical lowercase token")
    return value


def _capabilities(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("capabilities must be a tuple")
    canonical = tuple(_capability(item) for item in value)
    if len(set(canonical)) != len(canonical):
        raise ValueError("capabilities must not contain duplicates")
    return tuple(sorted(canonical))


def _wire_capabilities(value: object) -> tuple[str, ...]:
    if type(value) is not list:
        raise AgentSessionLifecycleWireFormatError("capabilities must be an array")
    try:
        return _capabilities(tuple(value))
    except (TypeError, ValueError) as error:
        raise AgentSessionLifecycleWireFormatError(
            "capabilities contains an invalid V1 token"
        ) from error


@dataclass(slots=True)
class _JsonBudget:
    nodes: int = 0
    string_bytes: int = 0

    def add_node(self, field_name: str) -> None:
        self.nodes += 1
        if self.nodes > _MAX_JSON_NODES:
            raise ValueError(f"{field_name} exceeds the maximum JSON node count")

    def add_string(self, value: str, field_name: str) -> str:
        value = _string(value, field_name, allow_blank=True)
        self.string_bytes += len(value.encode("utf-8", "strict"))
        if self.string_bytes > _MAX_JSON_STRING_BYTES:
            raise ValueError(f"{field_name} exceeds the maximum JSON string bytes")
        return value


def _json_clone(
    value: object,
    field_name: str,
    budget: _JsonBudget,
    depth: int = 0,
) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{field_name} exceeds the maximum JSON depth")
    budget.add_node(field_name)
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain NaN or infinity")
        return value
    if type(value) is str:
        return budget.add_string(value, field_name)
    if type(value) is list:
        return [_json_clone(item, field_name, budget, depth + 1) for item in value]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in value.items():
            key = budget.add_string(key, f"{field_name} key")
            result[key] = _json_clone(item, field_name, budget, depth + 1)
        return result
    raise TypeError(f"{field_name} must contain only JSON values")


def _json_snapshot(value: object, field_name: str) -> bytes:
    encoded = _strict_json_bytes(value, field_name)
    if len(encoded) > MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1:
        raise ValueError(f"{field_name} exceeds the maximum V1 JSON bytes")
    return encoded


def _strict_json_bytes(value: object, field_name: str) -> bytes:
    cloned = _json_clone(value, field_name, _JsonBudget())
    try:
        return json.dumps(
            cloned,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise ValueError(f"{field_name} cannot be encoded as JSON") from error


def encode_agent_session_lifecycle_json_v1(value: object) -> bytes:
    """Canonically encode one complete, bounded V1 lifecycle JSON body.

    This is the paired send boundary for
    :func:`decode_agent_session_lifecycle_json_v1`.  In particular, its byte
    limit is applied after JSON escaping and includes the complete envelope,
    rather than estimating a payload from the unescaped input strings.
    """

    try:
        encoded = _strict_json_bytes(value, "lifecycle JSON body")
    except (TypeError, ValueError, RecursionError) as error:
        raise AgentSessionLifecycleWireFormatError(
            "lifecycle value is not strict bounded V1 JSON"
        ) from error
    if len(encoded) > MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1:
        raise AgentSessionLifecycleWireFormatError(
            "lifecycle JSON body exceeds the V1 byte limit"
        )
    return encoded


def _json_restore(value: bytes) -> object:
    return json.loads(value.decode("utf-8", "strict"))


def _reject_json_constant(value: str) -> object:
    del value
    raise AgentSessionLifecycleWireFormatError(
        "lifecycle JSON must not contain non-finite numbers"
    )


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AgentSessionLifecycleWireFormatError(
            "lifecycle JSON must not contain non-finite numbers"
        )
    return result


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AgentSessionLifecycleWireFormatError(
                "lifecycle JSON objects must not contain duplicate keys"
            )
        result[key] = value
    return result


def decode_agent_session_lifecycle_json_v1(value: bytes) -> object:
    """Decode one bounded UTF-8 JSON body without duplicate object names.

    A future HTTP adapter must call this only after authenticating the pair
    hop.  The byte limit applies before JSON materialization, complementing
    the turn snapshot's node, string, depth, and encoded-size budgets.
    """

    if type(value) is not bytes:
        raise TypeError("lifecycle JSON body must be bytes")
    if len(value) > MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1:
        raise AgentSessionLifecycleWireFormatError(
            "lifecycle JSON body exceeds the V1 byte limit"
        )
    try:
        text = value.decode("utf-8", "strict")
        return json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except AgentSessionLifecycleWireFormatError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as error:
        raise AgentSessionLifecycleWireFormatError(
            "lifecycle body is not strict V1 JSON"
        ) from error


def _history_snapshot(value: object) -> bytes:
    if type(value) is not list:
        raise TypeError("history must be a list")
    if any(type(item) is not dict for item in value):
        raise TypeError("history entries must be objects")
    return _json_snapshot(value, "history")


def _metadata_snapshot(value: object) -> bytes:
    if type(value) is not dict:
        raise TypeError("metadata must be a dict")
    return _json_snapshot(value, "metadata")


def _queue_mode(value: object) -> QueueMode:
    if type(value) is not QueueMode:
        raise TypeError("queue_mode must be a QueueMode")
    return value


def _wire_queue_mode(value: object) -> QueueMode:
    value = _wire_string(value, "turn.queue_mode")
    try:
        return QueueMode(value)
    except ValueError as error:
        raise AgentSessionLifecycleWireFormatError(
            "turn.queue_mode is not a supported V1 value"
        ) from error


@dataclass(frozen=True, slots=True)
class WorkerSessionIdentityWireV1:
    """Closed-schema description of one Worker-local session lifetime."""

    identity: MemoryWorkerSessionIdentityV1

    def __post_init__(self) -> None:
        if type(self.identity) is not MemoryWorkerSessionIdentityV1:
            raise TypeError("identity must be a MemoryWorkerSessionIdentityV1")
        session_key = _session_key(self.identity.session_key)
        object.__setattr__(
            self,
            "identity",
            MemoryWorkerSessionIdentityV1(
                session=MemorySessionIncarnationV1(
                    session_key=session_key,
                    incarnation_id=self.identity.session.incarnation_id,
                ),
                audience=MemoryWorkerAudienceV1(
                    self.identity.audience.audience_id,
                ),
            ),
        )

    @property
    def session_key(self) -> str:
        return self.identity.session_key

    @property
    def session_incarnation_id(self) -> str:
        return self.identity.session.incarnation_id

    @property
    def worker_audience_id(self) -> str:
        return self.identity.audience.audience_id

    @classmethod
    def from_wire(cls, value: object) -> WorkerSessionIdentityWireV1:
        wire = _exact_dict(value, "identity")
        _closed_keys(wire, _IDENTITY_KEYS, "identity")
        _schema_version(wire["schema_version"], "identity.schema_version")
        session_key = _wire_session_key(wire["session_key"], "identity.session_key")
        try:
            identity = MemoryWorkerSessionIdentityV1(
                session=MemorySessionIncarnationV1(
                    session_key=session_key,
                    incarnation_id=_wire_string(
                        wire["session_incarnation_id"],
                        "identity.session_incarnation_id",
                    ),
                ),
                audience=MemoryWorkerAudienceV1(
                    _wire_string(
                        wire["worker_audience_id"],
                        "identity.worker_audience_id",
                    )
                ),
            )
        except (TypeError, ValueError) as error:
            raise AgentSessionLifecycleWireFormatError(
                "identity contains invalid V1 values"
            ) from error
        return cls(identity)

    @classmethod
    def from_runtime_identity(
        cls,
        identity: MemoryWorkerSessionIdentityV1,
    ) -> WorkerSessionIdentityWireV1:
        return cls(identity)

    def to_runtime_identity(self) -> MemoryWorkerSessionIdentityV1:
        return MemoryWorkerSessionIdentityV1(
            session=self.identity.session,
            audience=self.identity.audience,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "session_key": self.session_key,
            "session_incarnation_id": self.session_incarnation_id,
            "worker_audience_id": self.worker_audience_id,
        }


@dataclass(frozen=True, slots=True)
class WorkerSessionCapabilitiesReceiptWireV1:
    """Worker audience plus advertised protocol capabilities.

    Presence of ``exact_session_lifecycle_v1`` is an atomic promise covering
    open, every stateful run, and close.  It must never describe a close-only
    upgrade, and an exact client must not fall back after selecting it.
    """

    audience: MemoryWorkerAudienceV1
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.audience) is not MemoryWorkerAudienceV1:
            raise TypeError("audience must be a MemoryWorkerAudienceV1")
        self.audience.canonical_bytes()
        object.__setattr__(
            self,
            "audience",
            MemoryWorkerAudienceV1(self.audience.audience_id),
        )
        object.__setattr__(self, "capabilities", _capabilities(self.capabilities))
        encode_agent_session_lifecycle_json_v1(self.to_wire())

    @property
    def supports_exact_session_lifecycle(self) -> bool:
        return EXACT_SESSION_LIFECYCLE_CAPABILITY_V1 in self.capabilities

    @classmethod
    def from_wire(cls, value: object) -> WorkerSessionCapabilitiesReceiptWireV1:
        wire = _exact_dict(value, "capabilities receipt")
        _closed_keys(wire, _CAPABILITIES_KEYS, "capabilities receipt")
        _schema_version(
            wire["schema_version"],
            "capabilities receipt.schema_version",
        )
        capabilities = _wire_capabilities(wire["capabilities"])
        try:
            audience = MemoryWorkerAudienceV1(
                _wire_string(
                    wire["worker_audience_id"],
                    "capabilities receipt.worker_audience_id",
                )
            )
        except (TypeError, ValueError) as error:
            raise AgentSessionLifecycleWireFormatError(
                "capabilities receipt contains an invalid Worker audience"
            ) from error
        return cls(audience=audience, capabilities=capabilities)

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "capabilities": list(self.capabilities),
            "worker_audience_id": self.audience.audience_id,
        }


@dataclass(frozen=True, slots=True)
class WorkerSessionOpenRequestWireV1:
    """Idempotent request to reserve one reusable Agent session key."""

    session_key: str
    open_request_id: str
    expected_audience: MemoryWorkerAudienceV1

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_key", _session_key(self.session_key))
        object.__setattr__(
            self,
            "open_request_id",
            _request_id(self.open_request_id),
        )
        if type(self.expected_audience) is not MemoryWorkerAudienceV1:
            raise TypeError("expected_audience must be a MemoryWorkerAudienceV1")
        self.expected_audience.canonical_bytes()
        object.__setattr__(
            self,
            "expected_audience",
            MemoryWorkerAudienceV1(self.expected_audience.audience_id),
        )

    @property
    def expected_worker_audience_id(self) -> str:
        return self.expected_audience.audience_id

    @classmethod
    def from_wire(cls, value: object) -> WorkerSessionOpenRequestWireV1:
        wire = _exact_dict(value, "session open request")
        _closed_keys(wire, _OPEN_REQUEST_KEYS, "session open request")
        _schema_version(wire["schema_version"], "session open request.schema_version")
        try:
            expected_audience = MemoryWorkerAudienceV1(
                _wire_string(
                    wire["expected_worker_audience_id"],
                    "session open request.expected_worker_audience_id",
                )
            )
        except (TypeError, ValueError) as error:
            raise AgentSessionLifecycleWireFormatError(
                "session open request contains an invalid expected audience"
            ) from error
        return cls(
            session_key=_wire_session_key(
                wire["session_key"],
                "session open request.session_key",
            ),
            open_request_id=_wire_request_id(wire["open_request_id"]),
            expected_audience=expected_audience,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "session_key": self.session_key,
            "open_request_id": self.open_request_id,
            "expected_worker_audience_id": self.expected_worker_audience_id,
        }


@dataclass(frozen=True, slots=True)
class WorkerSessionOpenReceiptWireV1:
    """Identity returned by an idempotent exact-session reservation."""

    open_request_id: str
    identity: WorkerSessionIdentityWireV1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "open_request_id",
            _request_id(self.open_request_id),
        )
        if type(self.identity) is not WorkerSessionIdentityWireV1:
            raise TypeError("identity must be a WorkerSessionIdentityWireV1")
        object.__setattr__(
            self,
            "identity",
            WorkerSessionIdentityWireV1.from_runtime_identity(
                self.identity.to_runtime_identity()
            ),
        )

    @classmethod
    def from_wire(cls, value: object) -> WorkerSessionOpenReceiptWireV1:
        wire = _exact_dict(value, "session open receipt")
        _closed_keys(wire, _OPEN_RECEIPT_KEYS, "session open receipt")
        _schema_version(wire["schema_version"], "session open receipt.schema_version")
        return cls(
            open_request_id=_wire_request_id(wire["open_request_id"]),
            identity=WorkerSessionIdentityWireV1.from_wire(wire["identity"]),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "open_request_id": self.open_request_id,
            "identity": self.identity.to_wire(),
        }


@dataclass(frozen=True, slots=True, init=False)
class WorkerSessionTurnWireV1:
    """Immutable snapshot of one AgentRequest payload without its key."""

    message: str = field(repr=False)
    run_id: str
    queue_mode: QueueMode
    _history_json: bytes = field(repr=False)
    _metadata_json: bytes = field(repr=False)

    def __init__(
        self,
        *,
        message: str,
        run_id: str,
        history: list[dict[str, object]],
        queue_mode: QueueMode,
        metadata: dict[str, object],
    ) -> None:
        object.__setattr__(
            self,
            "message",
            _string(message, "message", allow_blank=True),
        )
        object.__setattr__(self, "run_id", _string(run_id, "run_id"))
        object.__setattr__(self, "queue_mode", _queue_mode(queue_mode))
        history_json = _history_snapshot(history)
        metadata_json = _metadata_snapshot(metadata)
        object.__setattr__(self, "_history_json", history_json)
        object.__setattr__(self, "_metadata_json", metadata_json)
        encode_agent_session_lifecycle_json_v1(self.to_wire())

    @property
    def history(self) -> list[dict[str, object]]:
        value = _json_restore(self._history_json)
        assert type(value) is list
        return value

    @property
    def metadata(self) -> dict[str, object]:
        value = _json_restore(self._metadata_json)
        assert type(value) is dict
        return value

    @classmethod
    def from_wire(cls, value: object) -> WorkerSessionTurnWireV1:
        wire = _exact_dict(value, "turn")
        _closed_keys(wire, _TURN_KEYS, "turn")
        try:
            return cls(
                message=_wire_string(
                    wire["message"],
                    "turn.message",
                    allow_blank=True,
                ),
                run_id=_wire_string(wire["run_id"], "turn.run_id"),
                history=wire["history"],  # type: ignore[arg-type]
                queue_mode=_wire_queue_mode(wire["queue_mode"]),
                metadata=wire["metadata"],  # type: ignore[arg-type]
            )
        except AgentSessionLifecycleWireFormatError:
            raise
        except (TypeError, ValueError) as error:
            raise AgentSessionLifecycleWireFormatError(
                "turn contains invalid V1 values"
            ) from error

    @classmethod
    def from_agent_request(cls, request: AgentRequest) -> WorkerSessionTurnWireV1:
        if type(request) is not AgentRequest:
            raise TypeError("request must be an AgentRequest")
        return cls(
            message=request.message,
            run_id=request.run_id,
            history=request.history,
            queue_mode=request.queue_mode,
            metadata=request.metadata,
        )

    def to_agent_request(self, session_key: str) -> AgentRequest:
        return AgentRequest(
            message=self.message,
            session_key=_session_key(session_key),
            run_id=self.run_id,
            history=self.history,
            queue_mode=self.queue_mode,
            metadata=self.metadata,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "message": self.message,
            "run_id": self.run_id,
            "history": self.history,
            "queue_mode": self.queue_mode.value,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class WorkerSessionRunRequestWireV1:
    """One turn bound to the exact identity returned by session open."""

    identity: WorkerSessionIdentityWireV1
    turn: WorkerSessionTurnWireV1

    def __post_init__(self) -> None:
        if type(self.identity) is not WorkerSessionIdentityWireV1:
            raise TypeError("identity must be a WorkerSessionIdentityWireV1")
        if type(self.turn) is not WorkerSessionTurnWireV1:
            raise TypeError("turn must be a WorkerSessionTurnWireV1")
        object.__setattr__(
            self,
            "identity",
            WorkerSessionIdentityWireV1.from_runtime_identity(
                self.identity.to_runtime_identity()
            ),
        )
        encode_agent_session_lifecycle_json_v1(self.to_wire())

    @classmethod
    def from_wire(cls, value: object) -> WorkerSessionRunRequestWireV1:
        wire = _exact_dict(value, "session run request")
        _closed_keys(wire, _RUN_REQUEST_KEYS, "session run request")
        _schema_version(wire["schema_version"], "session run request.schema_version")
        return cls(
            identity=WorkerSessionIdentityWireV1.from_wire(wire["identity"]),
            turn=WorkerSessionTurnWireV1.from_wire(wire["turn"]),
        )

    @classmethod
    def from_agent_request(
        cls,
        identity: WorkerSessionIdentityWireV1,
        request: AgentRequest,
    ) -> WorkerSessionRunRequestWireV1:
        if type(identity) is not WorkerSessionIdentityWireV1:
            raise TypeError("identity must be a WorkerSessionIdentityWireV1")
        if type(request) is not AgentRequest:
            raise TypeError("request must be an AgentRequest")
        if validate_session_key(request.session_key) != identity.session_key:
            raise ValueError("AgentRequest session does not match its exact identity")
        return cls(
            identity=identity,
            turn=WorkerSessionTurnWireV1.from_agent_request(request),
        )

    def to_agent_request(self) -> AgentRequest:
        return self.turn.to_agent_request(self.identity.session_key)

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "identity": self.identity.to_wire(),
            "turn": self.turn.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class AgentWorkerSessionCloseRequestWireV1:
    """Compare-and-close request for one exact full-host identity."""

    identity: WorkerSessionIdentityWireV1

    def __post_init__(self) -> None:
        if type(self.identity) is not WorkerSessionIdentityWireV1:
            raise TypeError("identity must be a WorkerSessionIdentityWireV1")
        object.__setattr__(
            self,
            "identity",
            WorkerSessionIdentityWireV1.from_runtime_identity(
                self.identity.to_runtime_identity()
            ),
        )

    @classmethod
    def from_wire(cls, value: object) -> AgentWorkerSessionCloseRequestWireV1:
        wire = _exact_dict(value, "session close request")
        _closed_keys(wire, _CLOSE_REQUEST_KEYS, "session close request")
        _schema_version(
            wire["schema_version"],
            "session close request.schema_version",
        )
        return cls(WorkerSessionIdentityWireV1.from_wire(wire["identity"]))

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "identity": self.identity.to_wire(),
        }


class AgentWorkerSessionCloseOutcomeWireV1(StrEnum):
    """Wire result for a complete Agent-and-Memory session close."""

    CLOSED = "closed"
    NOT_CURRENT = "not_current"


@dataclass(frozen=True, slots=True)
class AgentWorkerSessionCloseReceiptWireV1:
    """Full-host close result bound to the exact requested identity."""

    identity: WorkerSessionIdentityWireV1
    outcome: AgentWorkerSessionCloseOutcomeWireV1

    def __post_init__(self) -> None:
        if type(self.identity) is not WorkerSessionIdentityWireV1:
            raise TypeError("identity must be a WorkerSessionIdentityWireV1")
        if type(self.outcome) is not AgentWorkerSessionCloseOutcomeWireV1:
            raise TypeError("outcome must be an AgentWorkerSessionCloseOutcomeWireV1")
        object.__setattr__(
            self,
            "identity",
            WorkerSessionIdentityWireV1.from_runtime_identity(
                self.identity.to_runtime_identity()
            ),
        )

    @classmethod
    def from_wire(cls, value: object) -> AgentWorkerSessionCloseReceiptWireV1:
        wire = _exact_dict(value, "session close receipt")
        _closed_keys(wire, _CLOSE_RECEIPT_KEYS, "session close receipt")
        _schema_version(
            wire["schema_version"],
            "session close receipt.schema_version",
        )
        outcome_value = _wire_string(
            wire["outcome"],
            "session close receipt.outcome",
        )
        try:
            outcome = AgentWorkerSessionCloseOutcomeWireV1(outcome_value)
        except ValueError as error:
            raise AgentSessionLifecycleWireFormatError(
                "session close receipt.outcome is not a V1 value"
            ) from error
        return cls(
            identity=WorkerSessionIdentityWireV1.from_wire(wire["identity"]),
            outcome=outcome,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "identity": self.identity.to_wire(),
            "outcome": self.outcome.value,
        }


__all__ = [
    "EXACT_SESSION_LIFECYCLE_CAPABILITY_V1",
    "MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1",
    "AgentWorkerSessionCloseOutcomeWireV1",
    "AgentWorkerSessionCloseReceiptWireV1",
    "AgentWorkerSessionCloseRequestWireV1",
    "AgentSessionLifecycleTransportError",
    "AgentSessionLifecycleWireFormatError",
    "WorkerSessionCapabilitiesReceiptWireV1",
    "WorkerSessionIdentityWireV1",
    "WorkerSessionOpenReceiptWireV1",
    "WorkerSessionOpenRequestWireV1",
    "WorkerSessionRunRequestWireV1",
    "WorkerSessionTurnWireV1",
    "decode_agent_session_lifecycle_json_v1",
    "encode_agent_session_lifecycle_json_v1",
]
