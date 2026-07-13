# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from threading import Barrier, Event, Thread

import pytest

from areal.v2.memory_service.errors import (
    MemoryReleaseAssignmentConflictError,
    MemoryReleaseAssignmentNotFoundError,
    MemoryReleaseAttestationConflictError,
    MemoryReleaseAttestationNotFoundError,
    MemoryReleaseRevocationConflictError,
    MemoryReleaseRevocationNotFoundError,
)
from areal.v2.memory_service.history_store import InMemoryMemoryHistoryStore
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.release_control_store import (
    InMemoryMemoryReleaseControlStore,
    MemoryReleaseAssignmentPolicy,
    MemoryReleaseAttestationRevoker,
    MemoryReleaseAttestor,
    MemoryReleaseControlStore,
)
from areal.v2.memory_service.release_control_types import (
    MemoryReleaseAssignmentConsumerKind,
    MemoryReleaseRevocationReason,
)
from areal.v2.memory_service.release_store import InMemoryMemoryReleaseStore
from areal.v2.memory_service.release_types import ReleaseManifest
from areal.v2.memory_service.store import InMemoryEvidenceStore
from areal.v2.memory_service.types import EvidenceEvent, EvidenceKind, MemoryScope

_BASE = datetime(2026, 7, 13, 3, 0, tzinfo=UTC)
_ATTESTOR_VERSION = sha256(b"attestor-v1").hexdigest()
_ATTESTOR_CONFIG = sha256(b"attestor-config").hexdigest()
_REVOKER_VERSION = sha256(b"revoker-v1").hexdigest()
_REVOKER_CONFIG = sha256(b"revoker-config").hexdigest()
_ASSIGNMENT_VERSION = sha256(b"assignment-v1").hexdigest()
_ASSIGNMENT_CONFIG = sha256(b"assignment-config").hexdigest()
_GROUP_INCARNATION = sha256(b"group-incarnation").hexdigest()
_TASK_VERSION = sha256(b"task-v1").hexdigest()
_TASK_CONFIG = sha256(b"task-config").hexdigest()
_RETRIEVAL_VERSION = sha256(b"retrieval-v1").hexdigest()
_RETRIEVAL_CONFIG = sha256(b"retrieval-config").hexdigest()
_RENDERER_VERSION = sha256(b"renderer-v1").hexdigest()
_RENDERER_CONFIG = sha256(b"renderer-config").hexdigest()
_CONSUMER_VERSION = sha256(b"consumer-v1").hexdigest()
_CONSUMER_CONFIG = sha256(b"consumer-config").hexdigest()
_OTHER_HASH = sha256(b"other").hexdigest()


class _NoLookupHistory:
    def get_revision(self, scope, revision_id):
        raise AssertionError((scope, revision_id))


class _Clock:
    def __init__(self, value: datetime = _BASE) -> None:
        self.value = value
        self.queued: list[datetime] = []

    def __call__(self) -> datetime:
        if self.queued:
            self.value = self.queued.pop(0)
        return self.value


class _Attestor:
    attestor_id = "trusted-admission"
    attestor_version_sha256 = _ATTESTOR_VERSION
    attestor_config_sha256 = _ATTESTOR_CONFIG

    def __init__(self) -> None:
        self.calls = 0
        self.valid_from = _BASE - timedelta(minutes=1)
        self.valid_until = _BASE + timedelta(minutes=10)

    def attest(self, *, release, evaluated_at):
        del release, evaluated_at
        self.calls += 1
        return self.valid_from, self.valid_until


class _Revoker:
    revoker_id = "trusted-revoker"
    revoker_version_sha256 = _REVOKER_VERSION
    revoker_config_sha256 = _REVOKER_CONFIG

    def __init__(self) -> None:
        self.calls = 0
        self.reason_detail_sha256 = sha256(b"operator incident 7").hexdigest()

    def revoke(self, *, attestation, evaluated_at):
        del attestation, evaluated_at
        self.calls += 1
        return MemoryReleaseRevocationReason.OPERATOR, self.reason_detail_sha256


class _Policy:
    assignment_policy_id = "stable-group-cas"
    assignment_policy_version_sha256 = _ASSIGNMENT_VERSION
    assignment_policy_config_sha256 = _ASSIGNMENT_CONFIG

    def __init__(self) -> None:
        self.calls = 0
        self.valid_until = _BASE + timedelta(minutes=5)
        self.last_arguments = None

    def authorize(self, **arguments):
        self.calls += 1
        self.last_arguments = arguments
        return self.valid_until


def _seed(
    *,
    scope: MemoryScope | None = None,
    clock: _Clock | None = None,
    release_store=None,
    waiter_timeout_seconds: float | None = None,
):
    scope = scope or MemoryScope("tenant", "memory", "subject")
    if release_store is None:
        release_store = InMemoryMemoryReleaseStore(_NoLookupHistory())
        release = release_store.append_release(
            ReleaseManifest(scope=scope, revision_ids=()),
            idempotency_key="release-1",
        )
    else:
        release = release_store.list_releases(scope)[0]
    attestor = _Attestor()
    revoker = _Revoker()
    policy = _Policy()
    clock = clock or _Clock()
    control = InMemoryMemoryReleaseControlStore(
        release_store,
        attestor=attestor,
        revoker=revoker,
        assignment_policy=policy,
        clock=clock,
        waiter_timeout_seconds=waiter_timeout_seconds,
    )
    return scope, release_store, release, control, attestor, revoker, policy, clock


def _attest(control, scope, release, *, key="attest-1"):
    return control.attest_release(
        scope,
        release.release_id,
        release_content_sha256=release.content_hash,
        idempotency_key=key,
    )


def _assignment_arguments(attestation, **overrides):
    values = {
        "rollout_group_incarnation_sha256": _GROUP_INCARNATION,
        "attestation_id": attestation.attestation_id,
        "attestation_content_sha256": attestation.content_hash,
        "task_policy_id": "frozen-agent",
        "task_policy_version_sha256": _TASK_VERSION,
        "task_policy_config_sha256": _TASK_CONFIG,
        "retrieval_policy_id": "release-order",
        "retrieval_policy_version_sha256": _RETRIEVAL_VERSION,
        "retrieval_policy_config_sha256": _RETRIEVAL_CONFIG,
        "renderer_id": "memory-markdown",
        "renderer_version_sha256": _RENDERER_VERSION,
        "renderer_config_sha256": _RENDERER_CONFIG,
        "consumer_kind": MemoryReleaseAssignmentConsumerKind.MODEL_CALL,
        "consumer_id": "openai-compatible-model-call",
        "consumer_version_sha256": _CONSUMER_VERSION,
        "consumer_config_sha256": _CONSUMER_CONFIG,
        "max_returned_items": 4,
        "max_context_utf8_bytes": 4096,
        "idempotency_key": "assign-1",
    }
    values.update(overrides)
    return values


def _assign(control, scope, attestation, *, group="group-1", **overrides):
    return control.assign_release(
        scope,
        group,
        **_assignment_arguments(attestation, **overrides),
    )


def _revoke(control, scope, attestation, *, key="revoke-1"):
    return control.revoke_attestation(
        scope,
        attestation.attestation_id,
        attestation_content_sha256=attestation.content_hash,
        idempotency_key=key,
    )


def _resolve(control, scope, assignment):
    return control.resolve_active_assignment(
        scope,
        assignment.rollout_group_id,
        assignment.rollout_group_incarnation_sha256,
        assignment.assignment_id,
        assignment.content_hash,
    )


def test_store_creates_exact_control_records_without_mutating_release() -> None:
    scope, release_store, release, control, _, revoker, policy, _ = _seed()
    before = release_store.list_releases(scope)

    attestation = _attest(control, scope, release)
    assignment = _assign(control, scope, attestation)
    revocation = _revoke(control, scope, attestation)

    assert assignment.release_graph_sha256 == release.release_graph_sha256
    assert assignment.rollout_group_incarnation_sha256 == _GROUP_INCARNATION
    assert assignment.task_policy_config_sha256 == _TASK_CONFIG
    assert assignment.retrieval_policy_config_sha256 == _RETRIEVAL_CONFIG
    assert assignment.renderer_version_sha256 == _RENDERER_VERSION
    assert assignment.consumer_kind is MemoryReleaseAssignmentConsumerKind.MODEL_CALL
    assert assignment.assignment_valid_until == policy.valid_until
    assert assignment.assignment_valid_until <= attestation.valid_until
    assert revocation.reason_detail_sha256 == revoker.reason_detail_sha256
    assert control.get_attestation(scope, attestation.attestation_id) == attestation
    assert control.get_assignment(scope, assignment.assignment_id) == assignment
    assert control.get_revocation(scope, revocation.revocation_id) == revocation
    assert release_store.list_releases(scope) == before == (release,)


def test_constructor_selects_one_trusted_component_and_callers_cannot_override() -> None:
    signature = inspect.signature(InMemoryMemoryReleaseControlStore)
    assert {"attestor", "revoker", "assignment_policy"} <= set(signature.parameters)
    assert "attestors" not in signature.parameters
    scope, _, release, control, _, _, _, _ = _seed()
    with pytest.raises(TypeError, match="unexpected keyword"):
        control.attest_release(
            scope,
            release.release_id,
            release_content_sha256=release.content_hash,
            idempotency_key="key",
            attestor_id="caller-selected",
        )


@pytest.mark.parametrize("value", (0, -1, float("nan"), float("inf"), 10**1000))
def test_constructor_rejects_nonpositive_or_nonfinite_waiter_timeout(
    value: object,
) -> None:
    release_store = InMemoryMemoryReleaseStore(_NoLookupHistory())
    with pytest.raises(ValueError, match="positive, finite"):
        InMemoryMemoryReleaseControlStore(
            release_store,
            attestor=_Attestor(),
            revoker=_Revoker(),
            assignment_policy=_Policy(),
            waiter_timeout_seconds=value,
        )


def test_public_module_exports_control_store_contracts_by_identity() -> None:
    from areal.v2.memory_service import (
        InMemoryMemoryReleaseControlStore as PublicInMemoryStore,
    )
    from areal.v2.memory_service import (
        MemoryReleaseAssignmentPolicy as PublicAssignmentPolicy,
    )
    from areal.v2.memory_service import (
        MemoryReleaseAttestationRevoker as PublicRevoker,
    )
    from areal.v2.memory_service import MemoryReleaseAttestor as PublicAttestor
    from areal.v2.memory_service import (
        MemoryReleaseControlStore as PublicControlStore,
    )

    assert PublicInMemoryStore is InMemoryMemoryReleaseControlStore
    assert PublicAssignmentPolicy is MemoryReleaseAssignmentPolicy
    assert PublicRevoker is MemoryReleaseAttestationRevoker
    assert PublicAttestor is MemoryReleaseAttestor
    assert PublicControlStore is MemoryReleaseControlStore


@pytest.mark.parametrize(
    ("operation", "attribute"),
    (
        ("attest", "attestor_config_sha256"),
        ("revoke", "revoker_version_sha256"),
        ("assign", "assignment_policy_id"),
    ),
)
def test_component_declaration_mutation_fails_closed(
    operation: str,
    attribute: str,
) -> None:
    scope, _, release, control, attestor, revoker, policy, _ = _seed()
    component = {"attest": attestor, "revoke": revoker, "assign": policy}[operation]
    if operation != "attest":
        attestation = _attest(control, scope, release)
    setattr(component, attribute, "changed" if attribute.endswith("_id") else _OTHER_HASH)
    error = {
        "attest": MemoryReleaseAttestationConflictError,
        "revoke": MemoryReleaseRevocationConflictError,
        "assign": MemoryReleaseAssignmentConflictError,
    }[operation]
    with pytest.raises(error, match="declaration changed"):
        if operation == "attest":
            _attest(control, scope, release)
        elif operation == "revoke":
            _revoke(control, scope, attestation)
        else:
            _assign(control, scope, attestation)


def test_component_declaration_is_rechecked_after_callback() -> None:
    scope, release_store, release, _, _, revoker, policy, clock = _seed()

    class MutatingAttestor(_Attestor):
        def attest(self, **arguments):
            window = super().attest(**arguments)
            self.attestor_config_sha256 = _OTHER_HASH
            return window

    attestor = MutatingAttestor()
    control = InMemoryMemoryReleaseControlStore(
        release_store,
        attestor=attestor,
        revoker=revoker,
        assignment_policy=policy,
        clock=clock,
    )
    with pytest.raises(MemoryReleaseAttestationConflictError, match="changed"):
        _attest(control, scope, release)
    assert control.list_release_attestations(scope, release.release_id) == ()


def test_component_property_exception_is_wrapped_as_domain_conflict() -> None:
    scope, release_store, release, _, _, revoker, policy, clock = _seed()

    class RaisingAttestor(_Attestor):
        reads = 0

        @property
        def attestor_config_sha256(self):
            self.reads += 1
            if self.reads >= 2:
                raise RuntimeError("property backend unavailable")
            return _ATTESTOR_CONFIG

    control = InMemoryMemoryReleaseControlStore(
        release_store,
        attestor=RaisingAttestor(),
        revoker=revoker,
        assignment_policy=policy,
        clock=clock,
    )
    with pytest.raises(
        MemoryReleaseAttestationConflictError,
        match="declaration changed",
    ) as raised:
        _attest(control, scope, release)
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_component_property_getter_never_runs_under_store_lock() -> None:
    scope, release_store, release, _, _, revoker, policy, clock = _seed()
    child_finished = Event()
    child_errors = []

    class JoiningGetterAttestor(_Attestor):
        reads = 0
        control = None

        @property
        def attestor_config_sha256(self):
            self.reads += 1
            if self.reads == 4:
                def mutate_from_child():
                    try:
                        self.control.revoke_attestation(
                            scope,
                            f"mrat_{_OTHER_HASH[:24]}",
                            attestation_content_sha256=_OTHER_HASH,
                            idempotency_key="getter-child",
                        )
                    except Exception as error:
                        child_errors.append(error)
                    finally:
                        child_finished.set()

                thread = Thread(target=mutate_from_child, daemon=True)
                thread.start()
                assert child_finished.wait(timeout=2), (
                    "component getter ran while the store lock was held"
                )
                thread.join(timeout=2)
            return _ATTESTOR_CONFIG

    attestor = JoiningGetterAttestor()
    control = InMemoryMemoryReleaseControlStore(
        release_store,
        attestor=attestor,
        revoker=revoker,
        assignment_policy=policy,
        clock=clock,
    )
    attestor.control = control
    assert _attest(control, scope, release).release_id == release.release_id
    assert len(child_errors) == 1
    assert isinstance(child_errors[0], MemoryReleaseAttestationNotFoundError)


@pytest.mark.parametrize("operation", ("attest", "revoke", "assign", "resolve"))
def test_referenced_id_must_match_full_hash(operation: str) -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    if operation == "attest":
        with pytest.raises(MemoryReleaseAttestationConflictError, match="disagrees"):
            control.attest_release(
                scope,
                release.release_id,
                release_content_sha256=_OTHER_HASH,
                idempotency_key="bad",
            )
        return
    attestation = _attest(control, scope, release)
    if operation == "revoke":
        with pytest.raises(MemoryReleaseRevocationConflictError, match="disagrees"):
            control.revoke_attestation(
                scope,
                attestation.attestation_id,
                attestation_content_sha256=_OTHER_HASH,
                idempotency_key="bad",
            )
        return
    if operation == "assign":
        with pytest.raises(MemoryReleaseAssignmentConflictError, match="disagrees"):
            _assign(
                control,
                scope,
                attestation,
                attestation_content_sha256=_OTHER_HASH,
            )
        return
    assignment = _assign(control, scope, attestation)
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="disagrees"):
        control.resolve_active_assignment(
            scope,
            assignment.rollout_group_id,
            assignment.rollout_group_incarnation_sha256,
            assignment.assignment_id,
            _OTHER_HASH,
        )


def test_assignment_policy_receives_every_execution_binding() -> None:
    scope, _, release, control, _, _, policy, _ = _seed()
    attestation = _attest(control, scope, release)
    assignment = _assign(control, scope, attestation)

    assert policy.last_arguments is not None
    for name in (
        "rollout_group_incarnation_sha256",
        "task_policy_version_sha256",
        "task_policy_config_sha256",
        "retrieval_policy_version_sha256",
        "retrieval_policy_config_sha256",
        "renderer_version_sha256",
        "renderer_config_sha256",
        "consumer_kind",
        "consumer_version_sha256",
        "consumer_config_sha256",
        "max_returned_items",
        "max_context_utf8_bytes",
    ):
        assert policy.last_arguments[name] == getattr(assignment, name)


def test_assignment_expiry_is_bounded_by_attestation_and_must_follow_commit() -> None:
    scope, _, release, control, _, _, policy, _ = _seed()
    attestation = _attest(control, scope, release)
    policy.valid_until = attestation.valid_until + timedelta(microseconds=1)
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="expiry"):
        _assign(control, scope, attestation)
    policy.valid_until = _BASE
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="expiry"):
        _assign(control, scope, attestation, group="other", idempotency_key="other")


def test_commit_times_are_sampled_after_callbacks_and_clock_cannot_regress() -> None:
    clock = _Clock()
    scope, _, release, control, _, _, _, _ = _seed(clock=clock)
    clock.queued = [_BASE, _BASE + timedelta(microseconds=1)]
    attestation = _attest(control, scope, release)
    assert attestation.evaluated_at == _BASE
    assert attestation.attested_at == _BASE + timedelta(microseconds=1)

    clock.queued = [_BASE + timedelta(seconds=1), _BASE]
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="backwards"):
        _assign(control, scope, attestation)
    clock.queued = [_BASE + timedelta(seconds=1), _BASE]
    with pytest.raises(MemoryReleaseRevocationConflictError, match="backwards"):
        _revoke(control, scope, attestation)


def test_other_revocation_reason_requires_a_hashed_detail() -> None:
    scope, _, release, control, _, revoker, _, _ = _seed()
    attestation = _attest(control, scope, release)

    def unexplained_other(**arguments):
        del arguments
        return MemoryReleaseRevocationReason.OTHER, None

    revoker.revoke = unexplained_other
    control._revoker = control._snapshot_component(
        revoker,
        id_attribute="revoker_id",
        version_attribute="revoker_version_sha256",
        config_attribute="revoker_config_sha256",
        method_name="revoke",
        label="revoker",
    )
    with pytest.raises(MemoryReleaseRevocationConflictError, match="detail hash"):
        _revoke(control, scope, attestation)


def test_revocation_reclocks_after_final_validation_and_lock_wait() -> None:
    clock = _Clock()
    scope, _, release, control, _, _, _, _ = _seed(clock=clock)
    attestation = _attest(control, scope, release)
    final_validation = Event()
    continue_to_commit = Event()
    original_validate = control._validate_component
    revoker_validations = 0

    def observed_validation(trusted, **keywords):
        nonlocal revoker_validations
        result = original_validate(trusted, **keywords)
        if keywords["label"] == "revoker":
            revoker_validations += 1
            if revoker_validations == 3:
                final_validation.set()
                assert continue_to_commit.wait(timeout=2)
        return result

    control._validate_component = observed_validation
    committed_at = _BASE + timedelta(seconds=5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_revoke, control, scope, attestation)
        assert final_validation.wait(timeout=2)
        acquired = control._lock.acquire(timeout=2)
        if not acquired:
            continue_to_commit.set()
            pytest.fail("component validation ran while the store lock was held")
        try:
            clock.value = committed_at
            continue_to_commit.set()
        finally:
            control._lock.release()
        revocation = future.result(timeout=2)
    assert revocation.revoked_at == committed_at


def test_attestation_must_be_valid_at_its_commit_boundary() -> None:
    clock = _Clock()
    scope, _, release, control, attestor, _, _, _ = _seed(clock=clock)
    attestor.valid_until = _BASE + timedelta(microseconds=1)
    clock.queued = [_BASE, attestor.valid_until]
    with pytest.raises(MemoryReleaseAttestationConflictError, match="commit time"):
        _attest(control, scope, release)


def test_assignment_checks_attestation_at_commit_not_only_evaluation() -> None:
    clock = _Clock()
    scope, _, release, control, attestor, _, policy, _ = _seed(clock=clock)
    attestor.valid_until = _BASE + timedelta(seconds=2)
    policy.valid_until = attestor.valid_until
    attestation = _attest(control, scope, release)
    clock.queued = [_BASE + timedelta(seconds=1), attestor.valid_until]
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="commit time"):
        _assign(control, scope, attestation)


def test_idempotency_and_group_indexes_are_independent_compare_and_set() -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    attestation = _attest(control, scope, release)
    assignment = _assign(control, scope, attestation)
    assert _assign(control, scope, attestation) == assignment

    with pytest.raises(MemoryReleaseAssignmentConflictError, match="different"):
        _assign(
            control,
            scope,
            attestation,
            group="other-group",
            max_returned_items=5,
        )
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="aliases"):
        _assign(
            control,
            scope,
            attestation,
            idempotency_key="alias-key",
        )
    assert (scope, "alias-key") not in control._assignment_by_idempotency
    assert control.get_rollout_group_assignment(scope, "group-1") == assignment


def test_group_cas_binds_incarnation_and_policy_snapshot() -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    attestation = _attest(control, scope, release)
    assignment = _assign(control, scope, attestation)
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="aliases"):
        _assign(
            control,
            scope,
            attestation,
            rollout_group_incarnation_sha256=_OTHER_HASH,
            idempotency_key="new-incarnation",
        )
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="incarnation"):
        control.resolve_active_assignment(
            scope,
            assignment.rollout_group_id,
            _OTHER_HASH,
            assignment.assignment_id,
            assignment.content_hash,
        )


def test_group_and_incarnation_are_independent_one_to_one_indexes() -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    attestation = _attest(control, scope, release)
    assignment = _assign(control, scope, attestation)

    assert _assign(control, scope, attestation) == assignment
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="incarnation"):
        _assign(
            control,
            scope,
            attestation,
            group="different-group",
            idempotency_key="same-incarnation-alias",
        )
    assert (scope, "different-group") not in control._assignment_by_group
    assert (
        scope,
        "same-incarnation-alias",
    ) not in control._assignment_by_idempotency

    with pytest.raises(MemoryReleaseAssignmentConflictError, match="aliases"):
        _assign(
            control,
            scope,
            attestation,
            rollout_group_incarnation_sha256=_OTHER_HASH,
            idempotency_key="same-group-alias",
        )
    assert (scope, _OTHER_HASH) not in control._assignment_by_incarnation
    assert control._assignment_by_incarnation[
        (scope, _GROUP_INCARNATION)
    ] is assignment


def test_concurrent_group_compare_and_set_publishes_only_one_request() -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    left = _attest(control, scope, release, key="left-attestation")
    right = _attest(control, scope, release, key="right-attestation")
    barrier = Barrier(2, timeout=5)
    original_claim = control._claim

    def synchronized_claim(*arguments, **keywords):
        barrier.wait()
        return original_claim(*arguments, **keywords)

    control._claim = synchronized_claim

    def compete(item):
        key, attestation = item
        try:
            return _assign(
                control,
                scope,
                attestation,
                idempotency_key=key,
            )
        except MemoryReleaseAssignmentConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                compete,
                (("left-assignment", left), ("right-assignment", right)),
            )
        )
    successes = tuple(item for item in results if not isinstance(item, Exception))
    failures = tuple(item for item in results if isinstance(item, Exception))
    assert len(successes) == len(failures) == 1
    assert control.get_rollout_group_assignment(scope, "group-1") == successes[0]
    assert len(control._assignment_by_idempotency) == 1


def test_concurrent_groups_cannot_claim_the_same_incarnation() -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    attestation = _attest(control, scope, release)
    barrier = Barrier(2, timeout=5)
    original_claim = control._claim

    def synchronized_claim(*arguments, **keywords):
        barrier.wait()
        return original_claim(*arguments, **keywords)

    control._claim = synchronized_claim

    def compete(group: str):
        try:
            return _assign(
                control,
                scope,
                attestation,
                group=group,
                idempotency_key=f"assign-{group}",
            )
        except MemoryReleaseAssignmentConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(compete, ("group-left", "group-right")))
    successes = tuple(item for item in results if not isinstance(item, Exception))
    failures = tuple(item for item in results if isinstance(item, Exception))
    assert len(successes) == len(failures) == 1
    assert "incarnation" in str(failures[0])
    assert len(control._assignment_by_group) == 1
    assert len(control._assignment_by_incarnation) == 1
    assert len(control._assignment_by_idempotency) == 1


def test_active_resolution_fails_closed_on_incarnation_index_drift() -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    attestation = _attest(control, scope, release)
    assignment = _assign(control, scope, attestation)
    del control._assignment_by_incarnation[(scope, _GROUP_INCARNATION)]

    with pytest.raises(MemoryReleaseAssignmentConflictError, match="incarnation"):
        _resolve(control, scope, assignment)
    assert control.get_assignment(scope, assignment.assignment_id) == assignment


def test_active_resolution_is_distinct_from_historical_lookup_on_expiry() -> None:
    scope, _, release, control, _, _, policy, clock = _seed()
    attestation = _attest(control, scope, release)
    policy.valid_until = _BASE + timedelta(seconds=1)
    assignment = _assign(control, scope, attestation)
    assert _resolve(control, scope, assignment) == assignment

    clock.value = assignment.assignment_valid_until
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="expired"):
        _resolve(control, scope, assignment)
    assert control.get_assignment(scope, assignment.assignment_id) == assignment
    assert control.get_rollout_group_assignment(scope, "group-1") == assignment


def test_active_resolution_rejects_clock_rollback_before_assignment_commit() -> None:
    scope, _, release, control, _, _, _, clock = _seed()
    attestation = _attest(control, scope, release)
    clock.queued = [
        _BASE + timedelta(seconds=1),
        _BASE + timedelta(seconds=2),
    ]
    assignment = _assign(control, scope, attestation)
    clock.value = assignment.assigned_at - timedelta(microseconds=1)
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="not yet valid"):
        _resolve(control, scope, assignment)
    assert control.get_assignment(scope, assignment.assignment_id) == assignment


def test_active_resolution_reclocks_inside_final_lock_after_waiting() -> None:
    clock = _Clock()
    scope, _, release, control, _, _, policy, _ = _seed(clock=clock)
    attestation = _attest(control, scope, release)
    policy.valid_until = _BASE + timedelta(seconds=1)
    assignment = _assign(control, scope, attestation)
    graph_loaded = Event()
    allow_final_lock = Event()
    clock_called = Event()
    original_load = control._load_release_graph

    def blocking_graph_load(*arguments, **keywords):
        result = original_load(*arguments, **keywords)
        graph_loaded.set()
        assert allow_final_lock.wait(timeout=2)
        return result

    def observed_clock():
        clock_called.set()
        return clock.value

    control._load_release_graph = blocking_graph_load
    control._clock = observed_clock
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_resolve, control, scope, assignment)
        assert graph_loaded.wait(timeout=2)
        with control._lock:
            allow_final_lock.set()
            assert not clock_called.wait(timeout=0.1), (
                "active resolver sampled time before acquiring its final lock"
            )
            clock.value = assignment.assignment_valid_until
        with pytest.raises(MemoryReleaseAssignmentConflictError, match="expired"):
            future.result(timeout=2)
    assert clock_called.is_set()


def test_revocation_is_an_immediate_active_resolution_kill_switch() -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    attestation = _attest(control, scope, release)
    assignment = _assign(control, scope, attestation)
    assert _resolve(control, scope, assignment) == assignment
    revocation = _revoke(control, scope, attestation)
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="revoked"):
        _resolve(control, scope, assignment)
    assert control.get_assignment(scope, assignment.assignment_id) == assignment
    assert control.get_attestation_revocation(scope, attestation.attestation_id) == (
        revocation
    )


def test_missing_historical_records_raise_specific_not_found_errors() -> None:
    scope, _, _, control, _, _, _, _ = _seed()
    with pytest.raises(MemoryReleaseAttestationNotFoundError):
        control.get_attestation(scope, "mrat_missing")
    with pytest.raises(MemoryReleaseRevocationNotFoundError):
        control.get_revocation(scope, "mrvk_missing")
    with pytest.raises(MemoryReleaseAssignmentNotFoundError):
        control.get_assignment(scope, "masn_missing")


@pytest.mark.parametrize("operation", ("attest", "assign", "revoke"))
def test_exact_retries_are_singleflight_when_they_arrive_before_callback(
    operation: str,
) -> None:
    scope, _, release, control, attestor, _, _, _ = _seed()
    attestation = None
    if operation != "attest":
        attestation = _attest(control, scope, release)
    barrier = Barrier(8, timeout=5)
    original_claim = control._claim

    def synchronized_claim(*arguments, **keywords):
        barrier.wait()
        return original_claim(*arguments, **keywords)

    control._claim = synchronized_claim
    callback = {
        "attest": attestor,
        "assign": control._assignment_policy.component,
        "revoke": control._revoker.component,
    }[operation]

    def invoke():
        if operation == "attest":
            return _attest(control, scope, release, key="singleflight-attest")
        if operation == "assign":
            return _assign(control, scope, attestation)
        return _revoke(control, scope, attestation)

    with ThreadPoolExecutor(max_workers=8) as executor:
        values = tuple(executor.map(lambda _: invoke(), range(8)))
    assert len(set(values)) == 1
    assert callback.calls == 1


def test_cross_thread_callback_reentry_fails_immediately_without_deadlock() -> None:
    scope, release_store, release, _, _, revoker, policy, clock = _seed()
    child_finished = Event()
    child_errors = []

    class ReenteringAttestor(_Attestor):
        control = None

        def attest(self, **arguments):
            def child():
                try:
                    _attest(self.control, scope, release, key="nested")
                except Exception as error:
                    child_errors.append(error)
                finally:
                    child_finished.set()

            thread = Thread(target=child, daemon=True)
            thread.start()
            assert child_finished.wait(timeout=2), "cross-thread reentry deadlocked"
            thread.join(timeout=2)
            return super().attest(**arguments)

    attestor = ReenteringAttestor()
    control = InMemoryMemoryReleaseControlStore(
        release_store,
        attestor=attestor,
        revoker=revoker,
        assignment_policy=policy,
        clock=clock,
    )
    attestor.control = control
    with pytest.raises(MemoryReleaseAttestationConflictError, match="reentry"):
        _attest(control, scope, release)
    assert len(child_errors) == 1
    assert isinstance(child_errors[0], MemoryReleaseAttestationConflictError)
    assert control.list_release_attestations(scope, release.release_id) == ()


def test_context_guard_rejects_same_thread_cross_store_reentry() -> None:
    scope, release_store, release, _, _, revoker, policy, clock = _seed()
    _, _, other_release, other, _, _, _, _ = _seed(
        scope=MemoryScope("tenant", "memory", "other")
    )

    class CrossStoreAttestor(_Attestor):
        def attest(self, **arguments):
            try:
                _attest(other, other_release.manifest.scope, other_release)
            except MemoryReleaseAttestationConflictError:
                pass
            return super().attest(**arguments)

    control = InMemoryMemoryReleaseControlStore(
        release_store,
        attestor=CrossStoreAttestor(),
        revoker=revoker,
        assignment_policy=policy,
        clock=clock,
    )
    with pytest.raises(MemoryReleaseAttestationConflictError, match="reentry"):
        _attest(control, scope, release)
    assert other.list_release_attestations(other_release.manifest.scope, other_release.release_id) == ()


def test_global_guard_rejects_cross_thread_cross_store_reentry() -> None:
    scope, release_store, release, _, _, revoker, policy, clock = _seed()
    other_scope, _, other_release, other, _, _, _, _ = _seed(
        scope=MemoryScope("tenant", "memory", "other")
    )
    child_finished = Event()
    child_errors = []

    class CrossThreadCrossStoreAttestor(_Attestor):
        def attest(self, **arguments):
            def child():
                try:
                    _attest(other, other_scope, other_release)
                except Exception as error:
                    child_errors.append(error)
                finally:
                    child_finished.set()

            thread = Thread(target=child, daemon=True)
            thread.start()
            assert child_finished.wait(timeout=2), "global callback guard deadlocked"
            thread.join(timeout=2)
            return super().attest(**arguments)

    control = InMemoryMemoryReleaseControlStore(
        release_store,
        attestor=CrossThreadCrossStoreAttestor(),
        revoker=revoker,
        assignment_policy=policy,
        clock=clock,
    )
    with pytest.raises(MemoryReleaseAttestationConflictError, match="reentry"):
        _attest(control, scope, release)
    assert len(child_errors) == 1
    assert isinstance(child_errors[0], MemoryReleaseAttestationConflictError)
    assert control.list_release_attestations(scope, release.release_id) == ()
    assert other.list_release_attestations(
        other_scope,
        other_release.release_id,
    ) == ()


def test_condition_interrupt_cannot_leak_module_global_callback_guard() -> None:
    scope, _, release, control, attestor, _, _, _ = _seed()

    class InterruptSecondEnter:
        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.enters = 0

        def __enter__(self):
            self.enters += 1
            if self.enters == 2:
                raise KeyboardInterrupt("injected condition entry interruption")
            return self.delegate.__enter__()

        def __exit__(self, *arguments):
            return self.delegate.__exit__(*arguments)

        def wait(self, timeout=None):
            return self.delegate.wait(timeout=timeout)

        def notify_all(self):
            return self.delegate.notify_all()

    condition = InterruptSecondEnter(control._condition)
    control._condition = condition
    with pytest.raises(KeyboardInterrupt, match="condition entry"):
        _attest(control, scope, release)
    assert attestor.calls == 0
    assert control._claim_owner == {}
    assert control.list_release_attestations(scope, release.release_id) == ()

    new_scope, _, new_release, new_control, _, _, _, _ = _seed(
        scope=MemoryScope("tenant", "memory", "fresh-store")
    )
    assert _attest(new_control, new_scope, new_release).release_id == (
        new_release.release_id
    )


def test_revoke_during_assignment_callback_fails_closed_then_kills_retry() -> None:
    scope, _, release, control, _, _, policy, _ = _seed()
    attestation = _attest(control, scope, release)
    entered = Event()
    resume = Event()
    original = policy.authorize

    def blocking_authorize(**arguments):
        entered.set()
        assert resume.wait(timeout=5)
        return original(**arguments)

    policy.authorize = blocking_authorize
    control._assignment_policy = control._snapshot_component(
        policy,
        id_attribute="assignment_policy_id",
        version_attribute="assignment_policy_version_sha256",
        config_attribute="assignment_policy_config_sha256",
        method_name="authorize",
        label="assignment policy",
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_assign, control, scope, attestation)
        assert entered.wait(timeout=5)
        with pytest.raises(MemoryReleaseRevocationConflictError, match="callback"):
            _revoke(control, scope, attestation)
        resume.set()
        with pytest.raises(MemoryReleaseAssignmentConflictError, match="reentry"):
            future.result(timeout=5)
    with pytest.raises(MemoryReleaseAssignmentNotFoundError):
        control.get_rollout_group_assignment(scope, "group-1")
    _revoke(control, scope, attestation)
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="revoked"):
        _assign(control, scope, attestation)


def test_concurrent_revoke_and_assign_linearize_to_no_active_assignment() -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    attestation = _attest(control, scope, release)
    barrier = Barrier(2, timeout=5)
    original_claim = control._claim

    def synchronized_claim(*arguments, **keywords):
        barrier.wait()
        return original_claim(*arguments, **keywords)

    control._claim = synchronized_claim

    def assign():
        try:
            return _assign(control, scope, attestation)
        except MemoryReleaseAssignmentConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        assignment_future = executor.submit(assign)
        revocation_future = executor.submit(_revoke, control, scope, attestation)
        assignment_result = assignment_future.result(timeout=5)
        revocation = revocation_future.result(timeout=5)

    assert control.get_attestation_revocation(scope, attestation.attestation_id) == (
        revocation
    )
    if isinstance(assignment_result, MemoryReleaseAssignmentConflictError):
        with pytest.raises(MemoryReleaseAssignmentNotFoundError):
            control.get_rollout_group_assignment(scope, "group-1")
    else:
        with pytest.raises(MemoryReleaseAssignmentConflictError, match="revoked"):
            _resolve(control, scope, assignment_result)


class _InjectedBaseException(BaseException):
    pass


class _FailAfterWriteDict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        raise _InjectedBaseException("injected publication interruption")


@pytest.mark.parametrize(
    ("operation", "write_index"),
    (
        *(("attest", index) for index in range(3)),
        *(("revoke", index) for index in range(3)),
        *(("assign", index) for index in range(4)),
    ),
)
def test_base_exception_at_every_publication_point_rolls_back_all_indexes(
    operation: str,
    write_index: int,
) -> None:
    scope, _, release, control, _, _, _, _ = _seed()
    attestation = None
    if operation != "attest":
        attestation = _attest(control, scope, release)
    names = {
        "attest": (
            "_attestation_by_address",
            "_attestation_by_idempotency",
            "_attestations_by_release",
        ),
        "revoke": (
            "_revocation_by_address",
            "_revocation_by_attestation",
            "_revocation_by_idempotency",
        ),
        "assign": (
            "_assignment_by_address",
            "_assignment_by_group",
            "_assignment_by_incarnation",
            "_assignment_by_idempotency",
        ),
    }[operation]
    setattr(control, names[write_index], _FailAfterWriteDict())
    with pytest.raises(_InjectedBaseException):
        if operation == "attest":
            _attest(control, scope, release)
        elif operation == "revoke":
            _revoke(control, scope, attestation)
        else:
            _assign(control, scope, attestation)
    assert all(getattr(control, name) == {} for name in names)
    assert control._claim_owner == {}


def _nonempty_release_store(scope: MemoryScope):
    evidence_store = InMemoryEvidenceStore()
    evidence = evidence_store.append(
        EvidenceEvent(
            scope=scope,
            session_id="session",
            run_id="run",
            sequence_no=0,
            kind=EvidenceKind.OUTCOME,
            payload="The remembered procedure worked.",
            observed_at=_BASE,
            idempotency_key="evidence",
        )
    )
    history = InMemoryMemoryHistoryStore(evidence_store)
    candidate = history.append_candidate(
        CandidateProposal(
            scope=scope,
            content="Use the remembered procedure.",
            evidence_ids=(evidence.evidence_id,),
            idempotency_key="candidate",
        )
    )
    revision = history.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key="revision",
        )
    )
    store = InMemoryMemoryReleaseStore(history)
    store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(revision.revision_id,)),
        idempotency_key="release",
    )
    return store


class _DriftingReleaseStore:
    def __init__(self, delegate, *, drift_on_call: int) -> None:
        self.delegate = delegate
        self.drift_on_call = drift_on_call
        self.revision_calls = 0

    def get_release(self, scope, release_id):
        return self.delegate.get_release(scope, release_id)

    def get_release_revisions(self, scope, release_id):
        self.revision_calls += 1
        revisions = self.delegate.get_release_revisions(scope, release_id)
        if self.revision_calls >= self.drift_on_call:
            return (replace(revisions[0], generation=1),)
        return revisions


class _AdvanceClockOnGraphCall:
    def __init__(self, delegate, clock, *, call: int, value: datetime) -> None:
        self.delegate = delegate
        self.clock = clock
        self.call = call
        self.value = value
        self.revision_calls = 0

    def get_release(self, scope, release_id):
        return self.delegate.get_release(scope, release_id)

    def get_release_revisions(self, scope, release_id):
        self.revision_calls += 1
        revisions = self.delegate.get_release_revisions(scope, release_id)
        if self.revision_calls == self.call:
            self.clock.value = self.value
        return revisions


def test_attestation_reclocks_after_final_graph_read_before_publication() -> None:
    clock = _Clock()
    scope, release_store, release, control, attestor, _, _, _ = _seed(clock=clock)
    attestor.valid_until = _BASE + timedelta(microseconds=1)
    advancing = _AdvanceClockOnGraphCall(
        release_store,
        clock,
        call=3,
        value=attestor.valid_until,
    )
    control._release_store = advancing

    with pytest.raises(MemoryReleaseAttestationConflictError, match="commit time"):
        _attest(control, scope, release)
    assert advancing.revision_calls == 3
    assert control.list_release_attestations(scope, release.release_id) == ()


def test_assignment_reclocks_after_final_graph_read_before_publication() -> None:
    clock = _Clock()
    scope, release_store, release, control, _, _, policy, _ = _seed(clock=clock)
    attestation = _attest(control, scope, release)
    policy.valid_until = _BASE + timedelta(microseconds=1)
    advancing = _AdvanceClockOnGraphCall(
        release_store,
        clock,
        call=3,
        value=policy.valid_until,
    )
    control._release_store = advancing

    with pytest.raises(MemoryReleaseAssignmentConflictError, match="expiry"):
        _assign(control, scope, attestation)
    assert advancing.revision_calls == 3
    assert control._assignment_by_address == {}
    assert control._assignment_by_group == {}
    assert control._assignment_by_incarnation == {}
    assert control._assignment_by_idempotency == {}


@pytest.mark.parametrize("drift_on_call", (2, 3))
def test_release_graph_is_revalidated_through_attestation_commit(
    drift_on_call: int,
) -> None:
    scope = MemoryScope("tenant", "memory", "subject")
    release_store = _nonempty_release_store(scope)
    scope, _, release, control, _, _, _, _ = _seed(
        scope=scope,
        release_store=release_store,
    )
    drifting = _DriftingReleaseStore(
        release_store,
        drift_on_call=drift_on_call,
    )
    control._release_store = drifting
    with pytest.raises(MemoryReleaseAttestationConflictError, match="revision"):
        _attest(control, scope, release)
    assert drifting.revision_calls == drift_on_call
    assert control.list_release_attestations(scope, release.release_id) == ()


@pytest.mark.parametrize("drift_on_call", (2, 3))
def test_release_graph_is_revalidated_through_assignment_commit(
    drift_on_call: int,
) -> None:
    scope = MemoryScope("tenant", "memory", "subject")
    release_store = _nonempty_release_store(scope)
    scope, _, release, control, _, _, _, _ = _seed(
        scope=scope,
        release_store=release_store,
    )
    attestation = _attest(control, scope, release)
    drifting = _DriftingReleaseStore(
        release_store,
        drift_on_call=drift_on_call,
    )
    control._release_store = drifting
    with pytest.raises(MemoryReleaseAssignmentConflictError, match="revision"):
        _assign(control, scope, attestation)
    assert drifting.revision_calls == drift_on_call


def test_active_resolution_revalidates_release_graph_every_time() -> None:
    scope = MemoryScope("tenant", "memory", "subject")
    release_store = _nonempty_release_store(scope)
    scope, _, release, control, _, _, _, _ = _seed(
        scope=scope,
        release_store=release_store,
    )
    attestation = _attest(control, scope, release)
    assignment = _assign(control, scope, attestation)

    class CountingStore:
        def __init__(self, delegate):
            self.delegate = delegate
            self.revision_calls = 0

        def get_release(self, scope, release_id):
            return self.delegate.get_release(scope, release_id)

        def get_release_revisions(self, scope, release_id):
            self.revision_calls += 1
            return self.delegate.get_release_revisions(scope, release_id)

    counting = CountingStore(release_store)
    control._release_store = counting
    assert _resolve(control, scope, assignment) == assignment
    assert _resolve(control, scope, assignment) == assignment
    assert counting.revision_calls == 2


def test_claim_waiter_can_timeout_before_a_callback_starts() -> None:
    scope, _, release, control, _, _, _, _ = _seed(waiter_timeout_seconds=0.05)
    token = ("attestation-idempotency", scope, "attest-1")
    with control._condition:
        control._claim_owner[token] = -1
    with pytest.raises(MemoryReleaseAttestationConflictError, match="timed out"):
        _attest(control, scope, release)
    with control._condition:
        del control._claim_owner[token]
        control._condition.notify_all()
