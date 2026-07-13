# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from threading import Barrier

import pytest

from areal.v2.agent_service.memory import MemoryAgentSessionPinV1
from areal.v2.agent_service.memory_transport import (
    AREAL_INFERENCE_METADATA_KEY,
    AREAL_MEMORY_METADATA_KEY,
    CHAT_REQUEST_METADATA_KEY,
    MEMORY_ASSIGNMENT_PIN_FIELD,
    MEMORY_CONTROL_AUTHORIZED_FIELD,
    MemoryAgentMetadataWireV1,
    MemoryAssignmentPinWireV1,
    MemoryPinWireFormatError,
    MemorySessionPinCache,
    MemorySessionPinConflictError,
    ReservedMemoryMetadataError,
    inject_memory_assignment_pin,
    parse_memory_assignment_pin_metadata,
)
from areal.v2.memory_service.types import MemoryScope


def _hash(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _pin(suffix: str = "a") -> MemoryAgentSessionPinV1:
    assignment_hash = _hash(f"assignment-{suffix}")
    return MemoryAgentSessionPinV1(
        scope=MemoryScope(
            tenant_id="tenant-1",
            namespace="agent-long-term-memory",
            subject_id="subject-1",
        ),
        rollout_group_id="rollout-group-1",
        rollout_group_incarnation_sha256=_hash(f"incarnation-{suffix}"),
        assignment_id=f"masn_{assignment_hash[:24]}",
        assignment_content_sha256=assignment_hash,
    )


def _wire(suffix: str = "a") -> MemoryAssignmentPinWireV1:
    return MemoryAssignmentPinWireV1.from_runtime_pin(_pin(suffix))


def test_pin_wire_round_trip_is_closed_schema() -> None:
    original = _wire()
    encoded = original.to_wire()
    decoded = MemoryAssignmentPinWireV1.from_wire(encoded)

    assert decoded == original
    assert decoded.to_runtime_pin() == _pin()
    assert set(encoded) == {
        "schema_version",
        "scope",
        "rollout_group_id",
        "rollout_group_incarnation_sha256",
        "assignment_id",
        "assignment_content_sha256",
    }
    assert set(encoded["scope"]) == {"tenant_id", "namespace", "subject_id"}


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("assignment_id"),
        lambda value: value.update({"schema_version": True}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value["scope"].update({"unknown": "x"}),
        lambda value: value.update({"assignment_content_sha256": "0" * 63}),
        lambda value: value.update({"assignment_id": "masn_wrong"}),
    ),
)
def test_pin_wire_rejects_schema_and_value_mutants(mutate) -> None:
    value = _wire().to_wire()
    mutate(value)

    with pytest.raises(MemoryPinWireFormatError):
        MemoryAssignmentPinWireV1.from_wire(value)


@pytest.mark.parametrize("value", (None, [], "pin", {1: "not-a-string-key"}))
def test_pin_wire_requires_an_exact_json_object(value) -> None:
    with pytest.raises(MemoryPinWireFormatError):
        MemoryAssignmentPinWireV1.from_wire(value)


def test_reserved_metadata_round_trip_cannot_carry_exposure_claims() -> None:
    user_metadata = {"caller": "value"}
    wire = _wire()
    injected = inject_memory_assignment_pin(user_metadata, wire)

    assert user_metadata == {"caller": "value"}
    assert injected["caller"] == "value"
    assert parse_memory_assignment_pin_metadata(injected) == wire.to_runtime_pin()
    assert parse_memory_assignment_pin_metadata({"caller": "value"}) is None

    forged = dict(injected[AREAL_MEMORY_METADATA_KEY])
    forged["exposure"] = {"status": "delivered"}
    with pytest.raises(MemoryPinWireFormatError, match="unknown"):
        parse_memory_assignment_pin_metadata({AREAL_MEMORY_METADATA_KEY: forged})


def test_user_metadata_cannot_supply_or_override_reserved_memory() -> None:
    valid_envelope = MemoryAgentMetadataWireV1(_wire()).to_wire()
    with pytest.raises(ReservedMemoryMetadataError, match="reserved"):
        inject_memory_assignment_pin(
            {AREAL_MEMORY_METADATA_KEY: valid_envelope},
            _wire(),
        )
    with pytest.raises(ReservedMemoryMetadataError, match="reserved"):
        inject_memory_assignment_pin(
            {AREAL_MEMORY_METADATA_KEY: None},
            None,
        )
    with pytest.raises(ReservedMemoryMetadataError, match="reserved"):
        inject_memory_assignment_pin(
            {MEMORY_ASSIGNMENT_PIN_FIELD: _wire().to_wire()},
            None,
        )
    for key in (
        AREAL_INFERENCE_METADATA_KEY,
        CHAT_REQUEST_METADATA_KEY,
        MEMORY_CONTROL_AUTHORIZED_FIELD,
    ):
        with pytest.raises(ReservedMemoryMetadataError, match="reserved"):
            inject_memory_assignment_pin({key: {}}, None)


def test_session_cache_binds_reuses_conflicts_and_clears() -> None:
    cache = MemorySessionPinCache()
    first = _wire("first")
    replacement = _wire("replacement")

    assert cache.resolve("session-1") is None
    assert cache.resolve("session-1", first) is first
    assert cache.resolve("session-1") is first
    assert cache.resolve("session-1", _wire("first")) is first
    with pytest.raises(MemorySessionPinConflictError, match="already bound"):
        cache.resolve("session-1", replacement)

    cache.clear("session-1")
    assert cache.resolve("session-1") is None
    assert cache.resolve("session-1", replacement) is replacement
    cache.clear_all()
    assert cache.resolve("session-1") is None


def test_concurrent_first_pin_is_one_compare_and_set_winner() -> None:
    cache = MemorySessionPinCache()
    barrier = Barrier(2)

    def bind(pin: MemoryAssignmentPinWireV1):
        barrier.wait(timeout=5)
        try:
            return "bound", cache.resolve("session-1", pin)
        except MemorySessionPinConflictError as error:
            return "conflict", error

    first = _wire("first")
    second = _wire("second")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(bind, (first, second)))

    assert {status for status, _ in results} == {"bound", "conflict"}
    winner = next(value for status, value in results if status == "bound")
    assert cache.resolve("session-1") is winner
