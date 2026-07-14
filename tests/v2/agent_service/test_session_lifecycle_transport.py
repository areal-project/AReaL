# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest

from areal.v2.agent_service import session_lifecycle_transport as transport_module
from areal.v2.agent_service.memory_authorization import (
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)
from areal.v2.agent_service.memory_session_lifecycle import (
    MemoryWorkerSessionIdentityV1,
)
from areal.v2.agent_service.protocol import QueueMode
from areal.v2.agent_service.session_lifecycle_transport import (
    EXACT_SESSION_LIFECYCLE_CAPABILITY_V1,
    AgentSessionLifecycleWireFormatError,
    AgentWorkerSessionCloseOutcomeWireV1,
    AgentWorkerSessionCloseReceiptWireV1,
    AgentWorkerSessionCloseRequestWireV1,
    WorkerSessionCapabilitiesReceiptWireV1,
    WorkerSessionIdentityWireV1,
    WorkerSessionOpenReceiptWireV1,
    WorkerSessionOpenRequestWireV1,
    WorkerSessionRunRequestWireV1,
    WorkerSessionTurnWireV1,
    decode_agent_session_lifecycle_json_v1,
    encode_agent_session_lifecycle_json_v1,
)
from areal.v2.agent_service.types import AgentRequest


def _runtime_identity(
    *,
    session_key: str = "session-1",
    incarnation_digit: str = "1",
    audience_digit: str = "2",
) -> MemoryWorkerSessionIdentityV1:
    return MemoryWorkerSessionIdentityV1(
        session=MemorySessionIncarnationV1(
            session_key=session_key,
            incarnation_id=f"msinc_{incarnation_digit * 64}",
        ),
        audience=MemoryWorkerAudienceV1(f"maud_{audience_digit * 64}"),
    )


def _identity() -> WorkerSessionIdentityWireV1:
    return WorkerSessionIdentityWireV1.from_runtime_identity(_runtime_identity())


def _open_request_id(digit: str = "3") -> str:
    return f"aopen_{digit * 64}"


def _turn_wire() -> dict[str, object]:
    return {
        "message": "remember this",
        "run_id": "run-1",
        "history": [
            {"role": "user", "content": "earlier"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "lookup", "arguments": {"x": 1}}],
            },
        ],
        "queue_mode": "followup",
        "metadata": {
            "nested": {"enabled": True, "score": 0.25},
            "items": [1, "two", None],
        },
    }


def _run_wire() -> dict[str, object]:
    return {
        "schema_version": 1,
        "identity": _identity().to_wire(),
        "turn": _turn_wire(),
    }


def test_identity_wire_round_trip_is_exact_and_detached() -> None:
    runtime = _runtime_identity()
    identity = WorkerSessionIdentityWireV1.from_runtime_identity(runtime)
    encoded = identity.to_wire()
    decoded = WorkerSessionIdentityWireV1.from_wire(encoded)

    assert decoded == identity
    assert decoded.identity == runtime
    assert decoded.identity is not runtime
    assert decoded.to_runtime_identity() == runtime
    assert decoded.to_runtime_identity() is not decoded.identity
    assert set(encoded) == {
        "schema_version",
        "session_key",
        "session_incarnation_id",
        "worker_audience_id",
    }
    assert set(encoded).isdisjoint({"principal", "authorization", "api_key", "token"})

    encoded["session_key"] = "changed"
    assert decoded.session_key == "session-1"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("session_key"),
        lambda value: value.update({"schema_version": True}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value.update({"session_key": "unsafe/key"}),
        lambda value: value.update({"session_incarnation_id": "msinc_wrong"}),
        lambda value: value.update({"session_incarnation_id": f"msinc_{'A' * 64}"}),
        lambda value: value.update({"worker_audience_id": "maud_wrong"}),
        lambda value: value.update({"worker_audience_id": True}),
    ),
)
def test_identity_wire_rejects_schema_and_value_mutants(mutate) -> None:
    value = _identity().to_wire()
    mutate(value)

    with pytest.raises(AgentSessionLifecycleWireFormatError):
        WorkerSessionIdentityWireV1.from_wire(value)


@pytest.mark.parametrize("value", (None, [], "identity", {1: "bad-key"}))
def test_identity_wire_requires_an_exact_json_object(value: object) -> None:
    with pytest.raises(AgentSessionLifecycleWireFormatError):
        WorkerSessionIdentityWireV1.from_wire(value)


def test_capabilities_receipt_is_canonical_and_binds_worker_audience() -> None:
    receipt = WorkerSessionCapabilitiesReceiptWireV1(
        audience=MemoryWorkerAudienceV1(f"maud_{'2' * 64}"),
        capabilities=("future.feature-v2", EXACT_SESSION_LIFECYCLE_CAPABILITY_V1),
    )
    encoded = receipt.to_wire()
    decoded = WorkerSessionCapabilitiesReceiptWireV1.from_wire(encoded)

    assert decoded == receipt
    assert decoded.supports_exact_session_lifecycle
    assert encoded == {
        "schema_version": 1,
        "capabilities": [
            EXACT_SESSION_LIFECYCLE_CAPABILITY_V1,
            "future.feature-v2",
        ],
        "worker_audience_id": f"maud_{'2' * 64}",
    }
    encoded["capabilities"].clear()
    assert decoded.supports_exact_session_lifecycle

    legacy = WorkerSessionCapabilitiesReceiptWireV1.from_wire(
        {
            "schema_version": 1,
            "capabilities": [],
            "worker_audience_id": f"maud_{'2' * 64}",
        }
    )
    assert not legacy.supports_exact_session_lifecycle
    assert not WorkerSessionCapabilitiesReceiptWireV1(
        MemoryWorkerAudienceV1(f"maud_{'2' * 64}")
    ).supports_exact_session_lifecycle


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("worker_audience_id"),
        lambda value: value.update({"schema_version": True}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value.update({"capabilities": "exact_session_lifecycle_v1"}),
        lambda value: value.update({"capabilities": [True]}),
        lambda value: value.update({"capabilities": ["UPPERCASE"]}),
        lambda value: value.update(
            {
                "capabilities": [
                    EXACT_SESSION_LIFECYCLE_CAPABILITY_V1,
                    EXACT_SESSION_LIFECYCLE_CAPABILITY_V1,
                ]
            }
        ),
        lambda value: value.update({"worker_audience_id": f"maud_{'z' * 64}"}),
    ),
)
def test_capabilities_receipt_rejects_schema_and_value_mutants(mutate) -> None:
    value = WorkerSessionCapabilitiesReceiptWireV1(
        MemoryWorkerAudienceV1(f"maud_{'2' * 64}"),
        (EXACT_SESSION_LIFECYCLE_CAPABILITY_V1,),
    ).to_wire()
    mutate(value)

    with pytest.raises(AgentSessionLifecycleWireFormatError):
        WorkerSessionCapabilitiesReceiptWireV1.from_wire(value)


def test_open_request_and_receipt_round_trip_without_principal_data() -> None:
    request = WorkerSessionOpenRequestWireV1(
        session_key="session-1",
        open_request_id=_open_request_id(),
        expected_audience=MemoryWorkerAudienceV1(f"maud_{'2' * 64}"),
    )
    receipt = WorkerSessionOpenReceiptWireV1(
        open_request_id=_open_request_id(),
        identity=_identity(),
    )

    assert WorkerSessionOpenRequestWireV1.from_wire(request.to_wire()) == request
    assert WorkerSessionOpenReceiptWireV1.from_wire(receipt.to_wire()) == receipt
    assert request.to_wire() == {
        "schema_version": 1,
        "session_key": "session-1",
        "open_request_id": _open_request_id(),
        "expected_worker_audience_id": f"maud_{'2' * 64}",
    }
    assert receipt.to_wire() == {
        "schema_version": 1,
        "open_request_id": _open_request_id(),
        "identity": _identity().to_wire(),
    }


@pytest.mark.parametrize(
    ("parser", "value"),
    (
        (
            WorkerSessionOpenRequestWireV1.from_wire,
            {
                "session_key": "session-1",
                "open_request_id": _open_request_id(),
                "expected_worker_audience_id": f"maud_{'2' * 64}",
            },
        ),
        (
            WorkerSessionOpenRequestWireV1.from_wire,
            {
                "schema_version": True,
                "session_key": "session-1",
                "open_request_id": _open_request_id(),
                "expected_worker_audience_id": f"maud_{'2' * 64}",
            },
        ),
        (
            WorkerSessionOpenRequestWireV1.from_wire,
            {
                "schema_version": 1,
                "session_key": "unsafe/key",
                "open_request_id": _open_request_id(),
                "expected_worker_audience_id": f"maud_{'2' * 64}",
            },
        ),
        (
            WorkerSessionOpenRequestWireV1.from_wire,
            {
                "schema_version": 1,
                "session_key": "session-1",
                "open_request_id": "aopen_wrong",
                "expected_worker_audience_id": f"maud_{'2' * 64}",
            },
        ),
        (
            WorkerSessionOpenRequestWireV1.from_wire,
            {
                "schema_version": 1,
                "session_key": "session-1",
                "open_request_id": _open_request_id(),
                "expected_worker_audience_id": f"maud_{'z' * 64}",
            },
        ),
        (
            WorkerSessionOpenReceiptWireV1.from_wire,
            {
                "schema_version": 1,
                "open_request_id": _open_request_id(),
                "identity": _identity().to_wire(),
                "extra": 1,
            },
        ),
        (
            WorkerSessionOpenReceiptWireV1.from_wire,
            {
                "schema_version": 1,
                "open_request_id": _open_request_id(),
                "identity": None,
            },
        ),
        (
            WorkerSessionOpenReceiptWireV1.from_wire,
            {
                "schema_version": 1,
                "open_request_id": True,
                "identity": _identity().to_wire(),
            },
        ),
    ),
)
def test_open_values_reject_non_v1_inputs(parser, value: object) -> None:
    with pytest.raises(AgentSessionLifecycleWireFormatError):
        parser(value)


def test_open_request_id_is_explicit_and_echoed_by_receipt() -> None:
    audience = MemoryWorkerAudienceV1(f"maud_{'2' * 64}")
    first = WorkerSessionOpenRequestWireV1(
        "session-1",
        _open_request_id("3"),
        audience,
    )
    same_retry = WorkerSessionOpenRequestWireV1.from_wire(first.to_wire())
    successor = WorkerSessionOpenRequestWireV1(
        "session-1",
        _open_request_id("4"),
        audience,
    )

    assert same_retry == first
    assert successor.open_request_id != first.open_request_id
    receipt = WorkerSessionOpenReceiptWireV1(first.open_request_id, _identity())
    assert receipt.to_wire()["open_request_id"] == first.open_request_id


def test_run_request_round_trip_snapshots_arbitrary_json_payloads() -> None:
    identity = _identity()
    history = _turn_wire()["history"]
    metadata = _turn_wire()["metadata"]
    assert type(history) is list
    assert type(metadata) is dict
    request = AgentRequest(
        message="remember this",
        session_key="session-1",
        run_id="run-1",
        history=history,
        queue_mode=QueueMode.FOLLOWUP,
        metadata=metadata,
    )
    wire_request = WorkerSessionRunRequestWireV1.from_agent_request(
        identity,
        request,
    )

    history.append({"role": "user", "content": "late mutation"})
    metadata["late"] = True
    encoded = wire_request.to_wire()
    decoded = WorkerSessionRunRequestWireV1.from_wire(encoded)
    rebuilt = decoded.to_agent_request()

    assert decoded == wire_request
    assert encoded == _run_wire()
    assert rebuilt == AgentRequest(
        message="remember this",
        session_key="session-1",
        run_id="run-1",
        history=_turn_wire()["history"],
        queue_mode=QueueMode.FOLLOWUP,
        metadata=_turn_wire()["metadata"],
    )
    assert "session_key" not in encoded["turn"]

    rebuilt.history.append({"role": "user", "content": "another mutation"})
    rebuilt.metadata["changed"] = True
    assert decoded.to_wire() == _run_wire()


@pytest.mark.parametrize(
    ("factory", "parser"),
    (
        (
            lambda: WorkerSessionCapabilitiesReceiptWireV1(
                MemoryWorkerAudienceV1(f"maud_{'2' * 64}"),
                (EXACT_SESSION_LIFECYCLE_CAPABILITY_V1,),
            ),
            WorkerSessionCapabilitiesReceiptWireV1.from_wire,
        ),
        (lambda: _identity(), WorkerSessionIdentityWireV1.from_wire),
        (
            lambda: WorkerSessionOpenRequestWireV1(
                "session-1",
                _open_request_id(),
                MemoryWorkerAudienceV1(f"maud_{'2' * 64}"),
            ),
            WorkerSessionOpenRequestWireV1.from_wire,
        ),
        (
            lambda: WorkerSessionOpenReceiptWireV1(
                _open_request_id(),
                _identity(),
            ),
            WorkerSessionOpenReceiptWireV1.from_wire,
        ),
        (
            lambda: WorkerSessionTurnWireV1.from_wire(_turn_wire()),
            WorkerSessionTurnWireV1.from_wire,
        ),
        (
            lambda: WorkerSessionRunRequestWireV1.from_wire(_run_wire()),
            WorkerSessionRunRequestWireV1.from_wire,
        ),
        (
            lambda: AgentWorkerSessionCloseRequestWireV1(_identity()),
            AgentWorkerSessionCloseRequestWireV1.from_wire,
        ),
        (
            lambda: AgentWorkerSessionCloseReceiptWireV1(
                _identity(),
                AgentWorkerSessionCloseOutcomeWireV1.CLOSED,
            ),
            AgentWorkerSessionCloseReceiptWireV1.from_wire,
        ),
    ),
)
def test_every_wire_value_is_closed_under_canonical_encode_decode(
    factory,
    parser,
) -> None:
    value = factory()

    encoded = encode_agent_session_lifecycle_json_v1(value.to_wire())
    decoded = decode_agent_session_lifecycle_json_v1(encoded)

    assert parser(decoded) == value
    assert encode_agent_session_lifecycle_json_v1(decoded) == encoded
    assert len(encoded) <= transport_module.MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1


def test_complete_run_body_limit_counts_json_escaping_and_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _run_wire()
    turn = value["turn"]
    assert type(turn) is dict
    turn["message"] = "\0" * 100
    encoded = encode_agent_session_lifecycle_json_v1(value)
    unescaped_component_bytes = len(turn["message"].encode("utf-8"))
    body_limit = len(encoded) - 1
    assert unescaped_component_bytes < body_limit

    monkeypatch.setattr(
        transport_module,
        "MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1",
        body_limit,
    )
    with pytest.raises(AgentSessionLifecycleWireFormatError, match="byte limit"):
        WorkerSessionRunRequestWireV1.from_wire(value)


def test_run_request_rejects_agent_request_for_another_identity() -> None:
    request = AgentRequest(
        message="message",
        session_key="session-2",
        run_id="run-1",
    )
    with pytest.raises(ValueError, match="does not match"):
        WorkerSessionRunRequestWireV1.from_agent_request(_identity(), request)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("identity"),
        lambda value: value.update({"schema_version": True}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value["identity"].update({"extra": "x"}),
        lambda value: value["turn"].update({"extra": "x"}),
        lambda value: value["turn"].pop("run_id"),
        lambda value: value["turn"].update({"message": 1}),
        lambda value: value["turn"].update({"message": "\ud800"}),
        lambda value: value["turn"].update({"run_id": "  "}),
        lambda value: value["turn"].update({"history": {}}),
        lambda value: value["turn"].update({"history": [[]]}),
        lambda value: value["turn"].update({"queue_mode": "unknown"}),
        lambda value: value["turn"].update({"queue_mode": QueueMode.COLLECT}),
        lambda value: value["turn"].update({"metadata": []}),
        lambda value: value["turn"].update({"metadata": {"score": math.nan}}),
        lambda value: value["turn"].update({"metadata": {1: "bad-key"}}),
        lambda value: value["turn"].update({"metadata": {"tuple": (1, 2)}}),
    ),
)
def test_run_request_rejects_schema_and_value_mutants(mutate) -> None:
    value = _run_wire()
    mutate(value)

    with pytest.raises(AgentSessionLifecycleWireFormatError):
        WorkerSessionRunRequestWireV1.from_wire(value)


def test_run_request_rejects_cyclic_and_excessively_deep_json() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    cycle_wire = _run_wire()
    cycle_wire["turn"]["metadata"] = cyclic

    with pytest.raises(AgentSessionLifecycleWireFormatError, match="invalid V1"):
        WorkerSessionRunRequestWireV1.from_wire(cycle_wire)

    nested: object = "leaf"
    for _ in range(66):
        nested = [nested]
    deep_wire = _run_wire()
    deep_wire["turn"]["metadata"] = {"nested": nested}
    with pytest.raises(AgentSessionLifecycleWireFormatError, match="invalid V1"):
        WorkerSessionRunRequestWireV1.from_wire(deep_wire)


def test_turn_snapshot_enforces_node_string_and_byte_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_wire = _run_wire()
    node_wire["turn"]["metadata"] = {"items": [1, 2, 3]}
    monkeypatch.setattr(transport_module, "_MAX_JSON_NODES", 4)
    with pytest.raises(AgentSessionLifecycleWireFormatError, match="invalid V1"):
        WorkerSessionRunRequestWireV1.from_wire(node_wire)

    monkeypatch.setattr(transport_module, "_MAX_JSON_NODES", 100_000)
    monkeypatch.setattr(transport_module, "_MAX_JSON_STRING_BYTES", 3)
    string_wire = _run_wire()
    string_wire["turn"]["metadata"] = {"key": "value"}
    with pytest.raises(AgentSessionLifecycleWireFormatError, match="invalid V1"):
        WorkerSessionRunRequestWireV1.from_wire(string_wire)

    monkeypatch.setattr(transport_module, "_MAX_JSON_STRING_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(
        transport_module, "MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1", 32
    )
    with pytest.raises(AgentSessionLifecycleWireFormatError, match="invalid V1"):
        WorkerSessionRunRequestWireV1.from_wire(_run_wire())


def test_raw_json_decoder_rejects_ambiguous_or_oversized_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert decode_agent_session_lifecycle_json_v1(
        b'{"schema_version":1,"session_key":"session-1"}'
    ) == {"schema_version": 1, "session_key": "session-1"}

    for value in (
        b'{"schema_version":1,"schema_version":1}',
        b'{"value":NaN}',
        b'{"value":1e999}',
        b'{"value":',
        b"\xff",
    ):
        with pytest.raises(AgentSessionLifecycleWireFormatError):
            decode_agent_session_lifecycle_json_v1(value)

    with pytest.raises(TypeError, match="bytes"):
        decode_agent_session_lifecycle_json_v1("{}")  # type: ignore[arg-type]

    with pytest.raises(AgentSessionLifecycleWireFormatError, match="strict V1 JSON"):
        decode_agent_session_lifecycle_json_v1(b"[" * 10_000 + b"]" * 10_000)

    monkeypatch.setattr(
        transport_module, "MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1", 4
    )
    with pytest.raises(AgentSessionLifecycleWireFormatError, match="byte limit"):
        decode_agent_session_lifecycle_json_v1(b'{"x":1}')


def test_canonical_encoder_rejects_non_json_and_over_complex_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AgentSessionLifecycleWireFormatError, match="strict bounded"):
        encode_agent_session_lifecycle_json_v1({"not_json": (1, 2)})

    monkeypatch.setattr(
        transport_module,
        "MAX_AGENT_SESSION_LIFECYCLE_BODY_BYTES_V1",
        4,
    )
    with pytest.raises(AgentSessionLifecycleWireFormatError, match="byte limit"):
        encode_agent_session_lifecycle_json_v1({"x": 1})


def test_wire_errors_do_not_echo_attacker_controlled_values() -> None:
    sentinel = "do-not-reflect-secret-sentinel"
    identity = _identity().to_wire()
    identity[sentinel] = True
    with pytest.raises(AgentSessionLifecycleWireFormatError) as unknown:
        WorkerSessionIdentityWireV1.from_wire(identity)
    assert sentinel not in str(unknown.value)

    receipt = AgentWorkerSessionCloseReceiptWireV1(
        _identity(),
        AgentWorkerSessionCloseOutcomeWireV1.CLOSED,
    ).to_wire()
    receipt["outcome"] = sentinel
    with pytest.raises(AgentSessionLifecycleWireFormatError) as invalid_outcome:
        AgentWorkerSessionCloseReceiptWireV1.from_wire(receipt)
    assert sentinel not in str(invalid_outcome.value)


@pytest.mark.parametrize(
    "outcome",
    (
        AgentWorkerSessionCloseOutcomeWireV1.CLOSED,
        AgentWorkerSessionCloseOutcomeWireV1.NOT_CURRENT,
    ),
)
def test_close_request_and_receipt_round_trip_exact_identity(outcome) -> None:
    request = AgentWorkerSessionCloseRequestWireV1(_identity())
    receipt = AgentWorkerSessionCloseReceiptWireV1(_identity(), outcome)

    assert AgentWorkerSessionCloseRequestWireV1.from_wire(request.to_wire()) == request
    assert AgentWorkerSessionCloseReceiptWireV1.from_wire(receipt.to_wire()) == receipt
    assert receipt.to_wire() == {
        "schema_version": 1,
        "identity": _identity().to_wire(),
        "outcome": outcome.value,
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("outcome"),
        lambda value: value.update({"schema_version": True}),
        lambda value: value.update({"outcome": "success"}),
        lambda value: value.update(
            {"outcome": AgentWorkerSessionCloseOutcomeWireV1.CLOSED}
        ),
        lambda value: value.update({"identity": None}),
    ),
)
def test_close_receipt_rejects_schema_and_value_mutants(mutate) -> None:
    value = AgentWorkerSessionCloseReceiptWireV1(
        _identity(),
        AgentWorkerSessionCloseOutcomeWireV1.CLOSED,
    ).to_wire()
    mutate(value)

    with pytest.raises(AgentSessionLifecycleWireFormatError):
        AgentWorkerSessionCloseReceiptWireV1.from_wire(value)


@pytest.mark.parametrize(
    "value",
    (
        None,
        [],
        {"schema_version": 1, "identity": _identity().to_wire(), "extra": 1},
        {"schema_version": True, "identity": _identity().to_wire()},
        {"schema_version": 2, "identity": _identity().to_wire()},
        {"schema_version": 1},
        {"schema_version": 1, "identity": None},
        {
            "schema_version": 1,
            "identity": {**_identity().to_wire(), "principal": "forged"},
        },
    ),
)
def test_close_request_rejects_non_v1_inputs(value: object) -> None:
    with pytest.raises(AgentSessionLifecycleWireFormatError):
        AgentWorkerSessionCloseRequestWireV1.from_wire(value)


def test_public_constructors_require_typed_values() -> None:
    with pytest.raises(TypeError, match="capabilities"):
        WorkerSessionCapabilitiesReceiptWireV1(  # type: ignore[arg-type]
            MemoryWorkerAudienceV1(f"maud_{'2' * 64}"),
            [EXACT_SESSION_LIFECYCLE_CAPABILITY_V1],
        )
    with pytest.raises(TypeError, match="identity"):
        WorkerSessionOpenReceiptWireV1(  # type: ignore[arg-type]
            _open_request_id(),
            _runtime_identity(),
        )
    with pytest.raises(TypeError, match="queue_mode"):
        WorkerSessionTurnWireV1(  # type: ignore[arg-type]
            message="message",
            run_id="run-1",
            history=[],
            queue_mode="collect",
            metadata={},
        )
    with pytest.raises(TypeError, match="outcome"):
        AgentWorkerSessionCloseReceiptWireV1(  # type: ignore[arg-type]
            _identity(),
            "closed",
        )
