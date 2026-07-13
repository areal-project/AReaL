# SPDX-License-Identifier: Apache-2.0

"""Trusted admission, revocation, and rollout assignment for Memory releases.

The reference store has exactly one trusted component for each decision.  A
caller can identify data, but cannot select a more permissive attestor,
revoker, or assignment policy.  Component identity, version, and
configuration are snapshotted at construction and checked before and after
every callback.

Callbacks deliberately run outside the store lock.  A permanently blocking
component can therefore retain its claim and exact-request waiters until their
optional timeout expires; production deployments should isolate components
behind a bounded execution boundary.  This process-local implementation
cannot safely terminate arbitrary Python callbacks.  Trusted callbacks are
serialized module-wide so a callback-spawned thread cannot mutate a different
control-store instance and escape the reentry boundary.

The injected clock is part of the trusted computing base.  It must be pure,
non-blocking, and must never access a control store: commit and active
resolution deliberately sample it while holding the store lock so expiry is
checked at the operation's true linearization boundary.

Control records are audit values, not bearer credentials.  Every operation
that grants authority resolves an ID plus its full SHA-256 in this trusted
store.  SHA-256 proves integrity, not publisher identity or usefulness.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from threading import Condition, RLock, get_ident
from time import monotonic
from typing import Protocol

from areal.v2.memory_service._atomic import _atomic_publish
from areal.v2.memory_service.errors import (
    MemoryReleaseAssignmentConflictError,
    MemoryReleaseAssignmentNotFoundError,
    MemoryReleaseAttestationConflictError,
    MemoryReleaseAttestationNotFoundError,
    MemoryReleaseRevocationConflictError,
    MemoryReleaseRevocationNotFoundError,
)
from areal.v2.memory_service.history_types import MemoryRevision, RevisionOperation
from areal.v2.memory_service.release_control_types import (
    MemoryReleaseAssignmentConsumerKind,
    MemoryReleaseAssignmentV1,
    MemoryReleaseAttestationRevocationV1,
    MemoryReleaseAttestationV1,
    MemoryReleaseRevocationReason,
    _aware_datetime,
    _digest,
    _integer,
    _scope,
    _string,
)
from areal.v2.memory_service.release_store import MemoryReleaseStore
from areal.v2.memory_service.release_types import MemoryRelease
from areal.v2.memory_service.types import MemoryScope


@dataclass(slots=True)
class _CallbackContext:
    reentry_attempted: bool = False


_CALLBACK_CONTEXT: ContextVar[_CallbackContext | None] = ContextVar(
    "areal_memory_release_control_callback_context",
    default=None,
)
_CALLBACK_GUARD_LOCK = RLock()
_CALLBACK_ACTIVE = False
_CALLBACK_REENTRY_ATTEMPTED = False


class MemoryReleaseAttestor(Protocol):
    """The deployment-selected policy that admits one exact release graph."""

    attestor_id: str
    attestor_version_sha256: str
    attestor_config_sha256: str

    def attest(
        self,
        *,
        release: MemoryRelease,
        evaluated_at: datetime,
    ) -> tuple[datetime, datetime]: ...


class MemoryReleaseAttestationRevoker(Protocol):
    """The deployment-selected policy that authorizes an irreversible kill."""

    revoker_id: str
    revoker_version_sha256: str
    revoker_config_sha256: str

    def revoke(
        self,
        *,
        attestation: MemoryReleaseAttestationV1,
        evaluated_at: datetime,
    ) -> tuple[MemoryReleaseRevocationReason, str | None]: ...


class MemoryReleaseAssignmentPolicy(Protocol):
    """The deployment-selected policy that binds an execution snapshot."""

    assignment_policy_id: str
    assignment_policy_version_sha256: str
    assignment_policy_config_sha256: str

    def authorize(
        self,
        *,
        rollout_group_id: str,
        rollout_group_incarnation_sha256: str,
        attestation: MemoryReleaseAttestationV1,
        task_policy_id: str,
        task_policy_version_sha256: str,
        task_policy_config_sha256: str,
        retrieval_policy_id: str,
        retrieval_policy_version_sha256: str,
        retrieval_policy_config_sha256: str,
        renderer_id: str,
        renderer_version_sha256: str,
        renderer_config_sha256: str,
        consumer_kind: MemoryReleaseAssignmentConsumerKind,
        consumer_id: str,
        consumer_version_sha256: str,
        consumer_config_sha256: str,
        max_returned_items: int,
        max_context_utf8_bytes: int,
        evaluated_at: datetime,
    ) -> datetime: ...


class MemoryReleaseControlStore(Protocol):
    """Backend-neutral trusted release-control contract."""

    def attest_release(
        self,
        scope: MemoryScope,
        release_id: str,
        *,
        release_content_sha256: str,
        idempotency_key: str,
    ) -> MemoryReleaseAttestationV1: ...

    def get_attestation(
        self,
        scope: MemoryScope,
        attestation_id: str,
    ) -> MemoryReleaseAttestationV1: ...

    def list_release_attestations(
        self,
        scope: MemoryScope,
        release_id: str,
    ) -> tuple[MemoryReleaseAttestationV1, ...]: ...

    def revoke_attestation(
        self,
        scope: MemoryScope,
        attestation_id: str,
        *,
        attestation_content_sha256: str,
        idempotency_key: str,
    ) -> MemoryReleaseAttestationRevocationV1: ...

    def get_revocation(
        self,
        scope: MemoryScope,
        revocation_id: str,
    ) -> MemoryReleaseAttestationRevocationV1: ...

    def get_attestation_revocation(
        self,
        scope: MemoryScope,
        attestation_id: str,
    ) -> MemoryReleaseAttestationRevocationV1: ...

    def assign_release(
        self,
        scope: MemoryScope,
        rollout_group_id: str,
        *,
        rollout_group_incarnation_sha256: str,
        attestation_id: str,
        attestation_content_sha256: str,
        task_policy_id: str,
        task_policy_version_sha256: str,
        task_policy_config_sha256: str,
        retrieval_policy_id: str,
        retrieval_policy_version_sha256: str,
        retrieval_policy_config_sha256: str,
        renderer_id: str,
        renderer_version_sha256: str,
        renderer_config_sha256: str,
        consumer_kind: MemoryReleaseAssignmentConsumerKind,
        consumer_id: str,
        consumer_version_sha256: str,
        consumer_config_sha256: str,
        max_returned_items: int,
        max_context_utf8_bytes: int,
        idempotency_key: str,
    ) -> MemoryReleaseAssignmentV1: ...

    def get_assignment(
        self,
        scope: MemoryScope,
        assignment_id: str,
    ) -> MemoryReleaseAssignmentV1: ...

    def get_rollout_group_assignment(
        self,
        scope: MemoryScope,
        rollout_group_id: str,
    ) -> MemoryReleaseAssignmentV1: ...

    def resolve_active_assignment(
        self,
        scope: MemoryScope,
        rollout_group_id: str,
        rollout_group_incarnation_sha256: str,
        assignment_id: str,
        assignment_content_sha256: str,
    ) -> MemoryReleaseAssignmentV1: ...


@dataclass(frozen=True, slots=True)
class _TrustedComponent:
    component: object
    component_id: str
    version_sha256: str
    config_sha256: str
    id_attribute: str
    version_attribute: str
    config_attribute: str
    method_name: str
    method: Callable[..., object]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical_equal(left: object, right: object) -> bool:
    if type(left) is MemoryRelease and type(right) is MemoryRelease:
        return (
            left.release_id == right.release_id
            and left.content_hash == right.content_hash
            and left.release_graph_sha256 == right.release_graph_sha256
            and left.created_at == right.created_at
            and left.commitment_bytes() == right.commitment_bytes()
        )
    return left.canonical_bytes() == right.canonical_bytes()  # type: ignore[attr-defined]


def _method_identity(value: Callable[..., object]) -> tuple[object, object]:
    return (
        getattr(value, "__self__", None),
        getattr(value, "__func__", value),
    )


def _validate_address(
    record_id: str,
    content_hash: str,
    *,
    prefix: str,
    conflict_type: type[Exception],
    label: str,
) -> None:
    if record_id != f"{prefix}{content_hash[:24]}":
        raise conflict_type(f"{label} ID disagrees with its full hash")


class InMemoryMemoryReleaseControlStore:
    """Lock-protected, process-local trusted release-control reference store."""

    def __init__(
        self,
        release_store: MemoryReleaseStore,
        *,
        attestor: MemoryReleaseAttestor,
        revoker: MemoryReleaseAttestationRevoker,
        assignment_policy: MemoryReleaseAssignmentPolicy,
        clock: Callable[[], datetime] = _utc_now,
        waiter_timeout_seconds: float | None = None,
    ) -> None:
        """Build a reference store around singular trusted components.

        ``clock`` must be a pure, non-blocking trusted callable and must not
        read or mutate any control store because it is sampled under this
        store's lock at commit and active-resolution boundaries.
        """

        if not callable(clock):
            raise TypeError("clock must be callable")
        if waiter_timeout_seconds is not None:
            valid_timeout = (
                type(waiter_timeout_seconds) in (int, float)
                and waiter_timeout_seconds > 0
            )
            if valid_timeout:
                try:
                    valid_timeout = isfinite(waiter_timeout_seconds)
                except OverflowError:
                    valid_timeout = False
            if not valid_timeout:
                raise ValueError(
                    "waiter_timeout_seconds must be positive, finite, or None"
                )
        self._release_store = release_store
        self._clock = clock
        self._waiter_timeout_seconds = waiter_timeout_seconds
        self._attestor = self._snapshot_component(
            attestor,
            id_attribute="attestor_id",
            version_attribute="attestor_version_sha256",
            config_attribute="attestor_config_sha256",
            method_name="attest",
            label="attestor",
        )
        self._revoker = self._snapshot_component(
            revoker,
            id_attribute="revoker_id",
            version_attribute="revoker_version_sha256",
            config_attribute="revoker_config_sha256",
            method_name="revoke",
            label="revoker",
        )
        self._assignment_policy = self._snapshot_component(
            assignment_policy,
            id_attribute="assignment_policy_id",
            version_attribute="assignment_policy_version_sha256",
            config_attribute="assignment_policy_config_sha256",
            method_name="authorize",
            label="assignment policy",
        )

        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._claim_owner: dict[tuple[str, MemoryScope, str], int] = {}
        self._callback_active = False
        self._callback_reentry_attempted = False

        self._attestation_by_address: dict[
            tuple[MemoryScope, str], MemoryReleaseAttestationV1
        ] = {}
        self._attestation_by_idempotency: dict[
            tuple[MemoryScope, str], MemoryReleaseAttestationV1
        ] = {}
        self._attestations_by_release: dict[
            tuple[MemoryScope, str], tuple[MemoryReleaseAttestationV1, ...]
        ] = {}
        self._revocation_by_address: dict[
            tuple[MemoryScope, str], MemoryReleaseAttestationRevocationV1
        ] = {}
        self._revocation_by_attestation: dict[
            tuple[MemoryScope, str], MemoryReleaseAttestationRevocationV1
        ] = {}
        self._revocation_by_idempotency: dict[
            tuple[MemoryScope, str], MemoryReleaseAttestationRevocationV1
        ] = {}
        self._assignment_by_address: dict[
            tuple[MemoryScope, str], MemoryReleaseAssignmentV1
        ] = {}
        self._assignment_by_group: dict[
            tuple[MemoryScope, str], MemoryReleaseAssignmentV1
        ] = {}
        self._assignment_by_incarnation: dict[
            tuple[MemoryScope, str], MemoryReleaseAssignmentV1
        ] = {}
        self._assignment_by_idempotency: dict[
            tuple[MemoryScope, str], MemoryReleaseAssignmentV1
        ] = {}

    @staticmethod
    def _snapshot_component(
        component: object,
        *,
        id_attribute: str,
        version_attribute: str,
        config_attribute: str,
        method_name: str,
        label: str,
    ) -> _TrustedComponent:
        component_id = _string(getattr(component, id_attribute, None), id_attribute)
        version = _digest(
            getattr(component, version_attribute, None),
            version_attribute,
        )
        config = _digest(
            getattr(component, config_attribute, None),
            config_attribute,
        )
        method = getattr(component, method_name, None)
        if not callable(method):
            raise TypeError(f"trusted {label} must define {method_name}")
        return _TrustedComponent(
            component=component,
            component_id=component_id,
            version_sha256=version,
            config_sha256=config,
            id_attribute=id_attribute,
            version_attribute=version_attribute,
            config_attribute=config_attribute,
            method_name=method_name,
            method=method,
        )

    @staticmethod
    def _validate_component(
        trusted: _TrustedComponent,
        *,
        conflict_type: type[Exception],
        label: str,
    ) -> None:
        try:
            component_id = _string(
                getattr(trusted.component, trusted.id_attribute, None),
                trusted.id_attribute,
            )
            version = _digest(
                getattr(trusted.component, trusted.version_attribute, None),
                trusted.version_attribute,
            )
            config = _digest(
                getattr(trusted.component, trusted.config_attribute, None),
                trusted.config_attribute,
            )
            method = getattr(trusted.component, trusted.method_name, None)
            changed = (
                component_id != trusted.component_id
                or version != trusted.version_sha256
                or config != trusted.config_sha256
                or not callable(method)
                or _method_identity(method) != _method_identity(trusted.method)
            )
        except Exception as error:
            raise conflict_type(f"trusted {label} declaration changed") from error
        if changed:
            raise conflict_type(f"trusted {label} declaration changed")

    def _mutation_entry(self, conflict_type: type[Exception]) -> None:
        global _CALLBACK_REENTRY_ATTEMPTED

        context = _CALLBACK_CONTEXT.get()
        if context is not None:
            context.reentry_attempted = True
            raise conflict_type("control-store mutation from a trusted callback")
        with _CALLBACK_GUARD_LOCK:
            if _CALLBACK_ACTIVE:
                _CALLBACK_REENTRY_ATTEMPTED = True
                raise conflict_type(
                    "control-store mutation while a global callback is active"
                )
        with self._lock:
            if self._callback_active:
                self._callback_reentry_attempted = True
                raise conflict_type("control-store mutation while a callback is active")

    def _invoke(
        self,
        trusted: _TrustedComponent,
        *,
        conflict_type: type[Exception],
        label: str,
        arguments: dict[str, object],
    ) -> object:
        global _CALLBACK_ACTIVE, _CALLBACK_REENTRY_ATTEMPTED

        self._validate_component(trusted, conflict_type=conflict_type, label=label)
        context = _CallbackContext()
        token = None
        global_acquired = False
        local_acquired = False
        global_reentry = False
        cross_thread_reentry = False
        body_failed = False
        cleanup_error: BaseException | None = None
        try:
            with _CALLBACK_GUARD_LOCK:
                if _CALLBACK_ACTIVE:
                    _CALLBACK_REENTRY_ATTEMPTED = True
                    raise conflict_type("another global control callback is active")
                # Mark acquisition before publishing the flag so every later
                # BaseException is covered by the outer finally below.
                global_acquired = True
                _CALLBACK_ACTIVE = True
                _CALLBACK_REENTRY_ATTEMPTED = False

            with self._condition:
                if self._callback_active:
                    raise conflict_type("another control callback is active")
                local_acquired = True
                self._callback_active = True
                self._callback_reentry_attempted = False

            token = _CALLBACK_CONTEXT.set(context)
            try:
                result = trusted.method(**arguments)
            except Exception as error:
                raise conflict_type(f"trusted {label} failed") from error
        except BaseException:
            body_failed = True
            raise
        finally:
            if token is not None:
                try:
                    _CALLBACK_CONTEXT.reset(token)
                except BaseException as error:
                    cleanup_error = error
                    try:
                        _CALLBACK_CONTEXT.set(None)
                    except BaseException:
                        pass
            elif local_acquired:
                # Also cover an interruption between ContextVar.set returning
                # and its token being stored in the local variable.
                try:
                    _CALLBACK_CONTEXT.set(None)
                except BaseException as error:
                    cleanup_error = error
            if local_acquired:
                try:
                    # Use the underlying lock directly so a failing custom
                    # Condition.__enter__ cannot prevent local cleanup.
                    with self._lock:
                        cross_thread_reentry = self._callback_reentry_attempted
                        self._callback_active = False
                        self._callback_reentry_attempted = False
                        self._condition.notify_all()
                except BaseException as error:
                    self._callback_active = False
                    self._callback_reentry_attempted = False
                    if cleanup_error is None:
                        cleanup_error = error
            if global_acquired:
                try:
                    with _CALLBACK_GUARD_LOCK:
                        global_reentry = _CALLBACK_REENTRY_ATTEMPTED
                        _CALLBACK_ACTIVE = False
                        _CALLBACK_REENTRY_ATTEMPTED = False
                except BaseException as error:
                    _CALLBACK_ACTIVE = False
                    _CALLBACK_REENTRY_ATTEMPTED = False
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None and not body_failed:
                raise cleanup_error
        if context.reentry_attempted or cross_thread_reentry or global_reentry:
            raise conflict_type(f"trusted {label} attempted control-store reentry")
        self._validate_component(trusted, conflict_type=conflict_type, label=label)
        return result

    def _claim(
        self,
        tokens: tuple[tuple[str, MemoryScope, str], ...],
        *,
        conflict_type: type[Exception],
    ) -> int:
        global _CALLBACK_REENTRY_ATTEMPTED

        owner = get_ident()
        deadline = (
            None
            if self._waiter_timeout_seconds is None
            else monotonic() + self._waiter_timeout_seconds
        )
        while True:
            with _CALLBACK_GUARD_LOCK:
                if _CALLBACK_ACTIVE:
                    _CALLBACK_REENTRY_ATTEMPTED = True
                    raise conflict_type(
                        "control-store mutation while a global callback is active"
                    )
            with self._condition:
                if self._callback_active:
                    self._callback_reentry_attempted = True
                    raise conflict_type(
                        "control-store mutation while a callback is active"
                    )
                holders = {
                    self._claim_owner[token]
                    for token in tokens
                    if token in self._claim_owner
                }
                if owner in holders:
                    raise conflict_type("recursive control-store claim")
                if not holders:
                    for token in tokens:
                        self._claim_owner[token] = owner
                    return owner
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise conflict_type("timed out waiting for an in-flight request")
                self._condition.wait(timeout=remaining)

    def _release_claims(
        self,
        tokens: tuple[tuple[str, MemoryScope, str], ...],
        owner: int,
    ) -> None:
        with self._condition:
            for token in tokens:
                if self._claim_owner.get(token) == owner:
                    del self._claim_owner[token]
            self._condition.notify_all()

    def _now(self, conflict_type: type[Exception]) -> datetime:
        try:
            return _aware_datetime(self._clock(), "clock result")
        except Exception as error:
            raise conflict_type("clock returned an invalid time") from error

    @staticmethod
    def _check_commit_time(
        evaluated_at: datetime,
        committed_at: datetime,
        conflict_type: type[Exception],
    ) -> None:
        if committed_at < evaluated_at:
            raise conflict_type("clock moved backwards during policy evaluation")

    @staticmethod
    def _validate_release_value(
        scope: MemoryScope,
        release_id: str,
        release_content_sha256: str,
        release: object,
    ) -> MemoryRelease:
        if type(release) is not MemoryRelease:
            raise MemoryReleaseAttestationConflictError(
                "release store returned a non-canonical release"
            )
        try:
            commitment_hash = sha256(release.commitment_bytes()).hexdigest()
        except Exception as error:
            raise MemoryReleaseAttestationConflictError(
                "release commitment could not be recomputed"
            ) from error
        if (
            release.manifest.scope != scope
            or release.release_id != release_id
            or release.content_hash != release_content_sha256
            or release.content_hash != commitment_hash
            or release.release_id != f"rel_{commitment_hash[:24]}"
            or len(release.release_graph_sha256) != 64
            or any(c not in "0123456789abcdef" for c in release.release_graph_sha256)
        ):
            raise MemoryReleaseAttestationConflictError(
                "release failed exact commitment validation"
            )
        return release

    def _load_release_graph(
        self,
        scope: MemoryScope,
        release_id: str,
        release_content_sha256: str,
        *,
        conflict_type: type[Exception],
    ) -> tuple[MemoryRelease, tuple[object, ...]]:
        try:
            first = self._release_store.get_release(scope, release_id)
            first = self._validate_release_value(
                scope,
                release_id,
                release_content_sha256,
                first,
            )
            revisions = self._release_store.get_release_revisions(scope, release_id)
            repeated = self._release_store.get_release(scope, release_id)
            repeated = self._validate_release_value(
                scope,
                release_id,
                release_content_sha256,
                repeated,
            )
        except MemoryReleaseAttestationConflictError as error:
            raise conflict_type(str(error)) from error
        except Exception as error:
            raise conflict_type("release graph could not be resolved") from error
        if type(revisions) is not tuple:
            raise conflict_type("release store returned a non-tuple revision graph")
        revision_snapshot = tuple(tuple.__iter__(revisions))
        for revision in revision_snapshot:
            if type(revision) is not MemoryRevision:
                raise conflict_type(
                    "release store returned a non-canonical revision graph"
                )
            try:
                revision_hash = sha256(revision.proposal.canonical_bytes()).hexdigest()
            except Exception as error:
                raise conflict_type(
                    "release revision commitment could not be recomputed"
                ) from error
            if (
                revision.proposal.scope != scope
                or revision.content_hash != revision_hash
                or revision.revision_id != f"rev_{revision_hash[:24]}"
                or (
                    revision.proposal.operation is RevisionOperation.ADD
                    and (
                        revision.generation != 0
                        or revision.memory_id != f"mem_{revision_hash[:24]}"
                    )
                )
                or (
                    revision.proposal.operation is not RevisionOperation.ADD
                    and revision.generation == 0
                )
            ):
                raise conflict_type(
                    "release revision failed exact commitment validation"
                )
        if (
            not _canonical_equal(first, repeated)
            or first.release_graph_sha256 != repeated.release_graph_sha256
            or tuple(getattr(item, "revision_id", None) for item in revision_snapshot)
            != first.manifest.revision_ids
        ):
            raise conflict_type("release graph changed while it was loaded")
        return first, revision_snapshot

    @staticmethod
    def _same_release_graph(
        left: tuple[MemoryRelease, tuple[object, ...]],
        right: tuple[MemoryRelease, tuple[object, ...]],
    ) -> bool:
        left_release, left_revisions = left
        right_release, right_revisions = right
        if (
            not _canonical_equal(left_release, right_release)
            or left_release.release_graph_sha256 != right_release.release_graph_sha256
            or len(left_revisions) != len(right_revisions)
        ):
            return False
        for left_revision, right_revision in zip(
            left_revisions,
            right_revisions,
            strict=True,
        ):
            if type(left_revision) is not type(right_revision):
                return False
            try:
                if (
                    left_revision.proposal.canonical_bytes()
                    != right_revision.proposal.canonical_bytes()
                    or left_revision.content_hash != right_revision.content_hash
                    or left_revision.revision_id != right_revision.revision_id
                    or left_revision.memory_id != right_revision.memory_id
                    or left_revision.generation != right_revision.generation
                ):
                    return False
            except (AttributeError, TypeError, ValueError):
                return False
        return True

    @staticmethod
    def _validate_attestation(
        scope: MemoryScope,
        value: object,
        *,
        expected_id: str | None = None,
        expected_hash: str | None = None,
    ) -> MemoryReleaseAttestationV1:
        if type(value) is not MemoryReleaseAttestationV1:
            raise MemoryReleaseAttestationConflictError(
                "stored attestation has a non-canonical type"
            )
        content_hash = sha256(value.canonical_bytes()).hexdigest()
        if (
            value.scope != scope
            or value.content_hash != content_hash
            or value.attestation_id != f"mrat_{content_hash[:24]}"
            or (expected_id is not None and value.attestation_id != expected_id)
            or (expected_hash is not None and value.content_hash != expected_hash)
        ):
            raise MemoryReleaseAttestationConflictError(
                "stored attestation failed integrity validation"
            )
        return value

    @staticmethod
    def _validate_revocation(
        scope: MemoryScope,
        value: object,
        *,
        expected_id: str | None = None,
        expected_attestation_id: str | None = None,
    ) -> MemoryReleaseAttestationRevocationV1:
        if type(value) is not MemoryReleaseAttestationRevocationV1:
            raise MemoryReleaseRevocationConflictError(
                "stored revocation has a non-canonical type"
            )
        content_hash = sha256(value.canonical_bytes()).hexdigest()
        if (
            value.scope != scope
            or value.content_hash != content_hash
            or value.revocation_id != f"mrvk_{content_hash[:24]}"
            or (expected_id is not None and value.revocation_id != expected_id)
            or (
                expected_attestation_id is not None
                and value.attestation_id != expected_attestation_id
            )
        ):
            raise MemoryReleaseRevocationConflictError(
                "stored revocation failed integrity validation"
            )
        return value

    @staticmethod
    def _validate_assignment(
        scope: MemoryScope,
        value: object,
        *,
        expected_id: str | None = None,
        expected_hash: str | None = None,
        expected_group: str | None = None,
    ) -> MemoryReleaseAssignmentV1:
        if type(value) is not MemoryReleaseAssignmentV1:
            raise MemoryReleaseAssignmentConflictError(
                "stored assignment has a non-canonical type"
            )
        content_hash = sha256(value.canonical_bytes()).hexdigest()
        if (
            value.scope != scope
            or value.content_hash != content_hash
            or value.assignment_id != f"masn_{content_hash[:24]}"
            or (expected_id is not None and value.assignment_id != expected_id)
            or (expected_hash is not None and value.content_hash != expected_hash)
            or (
                expected_group is not None
                and value.rollout_group_id != expected_group
            )
        ):
            raise MemoryReleaseAssignmentConflictError(
                "stored assignment failed integrity validation"
            )
        return value

    def attest_release(
        self,
        scope: MemoryScope,
        release_id: str,
        *,
        release_content_sha256: str,
        idempotency_key: str,
    ) -> MemoryReleaseAttestationV1:
        """Evaluate and admit one exact, fully revalidated release graph."""

        self._mutation_entry(MemoryReleaseAttestationConflictError)
        scope = _scope(scope)
        release_id = _string(release_id, "release_id")
        release_content_sha256 = _digest(
            release_content_sha256,
            "release_content_sha256",
        )
        _validate_address(
            release_id,
            release_content_sha256,
            prefix="rel_",
            conflict_type=MemoryReleaseAttestationConflictError,
            label="release",
        )
        idempotency_key = _string(idempotency_key, "idempotency_key")
        idempotency_address = (scope, idempotency_key)
        tokens = (
            ("attestation-idempotency", scope, idempotency_key),
            ("attestation-release", scope, release_id),
        )
        owner = self._claim(
            tokens,
            conflict_type=MemoryReleaseAttestationConflictError,
        )
        try:
            with self._lock:
                existing = self._attestation_by_idempotency.get(
                    idempotency_address
                )
                if existing is not None:
                    existing = self._validate_attestation(scope, existing)
                    if (
                        existing.release_id == release_id
                        and existing.release_content_sha256
                        == release_content_sha256
                    ):
                        return existing
                    raise MemoryReleaseAttestationConflictError(
                        "scoped attestation idempotency key has a different request"
                    )

            before = self._load_release_graph(
                scope,
                release_id,
                release_content_sha256,
                conflict_type=MemoryReleaseAttestationConflictError,
            )
            evaluated_at = self._now(MemoryReleaseAttestationConflictError)
            window = self._invoke(
                self._attestor,
                conflict_type=MemoryReleaseAttestationConflictError,
                label="attestor",
                arguments={"release": before[0], "evaluated_at": evaluated_at},
            )
            if type(window) is not tuple or len(window) != 2:
                raise MemoryReleaseAttestationConflictError(
                    "trusted attestor returned an invalid validity window"
                )
            try:
                valid_from = _aware_datetime(window[0], "valid_from")
                valid_until = _aware_datetime(window[1], "valid_until")
            except (TypeError, ValueError) as error:
                raise MemoryReleaseAttestationConflictError(
                    "trusted attestor returned an invalid validity window"
                ) from error
            after = self._load_release_graph(
                scope,
                release_id,
                release_content_sha256,
                conflict_type=MemoryReleaseAttestationConflictError,
            )
            if not self._same_release_graph(before, after):
                raise MemoryReleaseAttestationConflictError(
                    "release graph changed during attestation"
                )
            self._validate_component(
                self._attestor,
                conflict_type=MemoryReleaseAttestationConflictError,
                label="attestor",
            )
            final_graph = self._load_release_graph(
                scope,
                release_id,
                release_content_sha256,
                conflict_type=MemoryReleaseAttestationConflictError,
            )
            if not self._same_release_graph(after, final_graph):
                raise MemoryReleaseAttestationConflictError(
                    "release graph changed before attestation commit"
                )
            release_address = (scope, release_id)
            with self._lock:
                existing = self._attestation_by_idempotency.get(
                    idempotency_address
                )
                if existing is not None:
                    raise MemoryReleaseAttestationConflictError(
                        "attestation appeared during policy evaluation"
                    )
                attested_at = self._now(MemoryReleaseAttestationConflictError)
                self._check_commit_time(
                    evaluated_at,
                    attested_at,
                    MemoryReleaseAttestationConflictError,
                )
                if not valid_from <= attested_at < valid_until:
                    raise MemoryReleaseAttestationConflictError(
                        "attestation is not valid at commit time"
                    )
                attestation = MemoryReleaseAttestationV1.create(
                    scope=scope,
                    release_id=release_id,
                    release_content_sha256=release_content_sha256,
                    release_graph_sha256=final_graph[0].release_graph_sha256,
                    attestor_id=self._attestor.component_id,
                    attestor_version_sha256=self._attestor.version_sha256,
                    attestor_config_sha256=self._attestor.config_sha256,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    evaluated_at=evaluated_at,
                    attested_at=attested_at,
                    idempotency_key=idempotency_key,
                )
                address = (scope, attestation.attestation_id)
                collision = self._attestation_by_address.get(address)
                if collision is not None and not _canonical_equal(
                    collision,
                    attestation,
                ):
                    raise MemoryReleaseAttestationConflictError(
                        f"attestation ID collision for {attestation.attestation_id!r}"
                    )
                stored = collision if collision is not None else attestation
                release_values = self._attestations_by_release.get(
                    release_address,
                    (),
                )
                if any(item.attestation_id == stored.attestation_id for item in release_values):
                    raise MemoryReleaseAttestationConflictError(
                        "release-attestation index already contains the record"
                    )
                _atomic_publish(
                    mapping_writes=(
                        (self._attestation_by_address, address, stored),
                        (
                            self._attestation_by_idempotency,
                            idempotency_address,
                            stored,
                        ),
                        (
                            self._attestations_by_release,
                            release_address,
                            (*release_values, stored),
                        ),
                    )
                )
                return stored
        finally:
            self._release_claims(tokens, owner)

    def get_attestation(
        self,
        scope: MemoryScope,
        attestation_id: str,
    ) -> MemoryReleaseAttestationV1:
        scope = _scope(scope)
        attestation_id = _string(attestation_id, "attestation_id")
        with self._lock:
            value = self._attestation_by_address.get((scope, attestation_id))
        if value is None:
            raise MemoryReleaseAttestationNotFoundError(
                f"attestation {attestation_id!r} was not found"
            )
        return self._validate_attestation(scope, value, expected_id=attestation_id)

    def list_release_attestations(
        self,
        scope: MemoryScope,
        release_id: str,
    ) -> tuple[MemoryReleaseAttestationV1, ...]:
        scope = _scope(scope)
        release_id = _string(release_id, "release_id")
        with self._lock:
            values = self._attestations_by_release.get((scope, release_id), ())
        snapshot = tuple(values)
        for value in snapshot:
            self._validate_attestation(scope, value)
            if value.release_id != release_id:
                raise MemoryReleaseAttestationConflictError(
                    "stored release-attestation index is inconsistent"
                )
        return tuple(sorted(snapshot, key=lambda item: item.attestation_id))

    def revoke_attestation(
        self,
        scope: MemoryScope,
        attestation_id: str,
        *,
        attestation_content_sha256: str,
        idempotency_key: str,
    ) -> MemoryReleaseAttestationRevocationV1:
        """Authorize and publish an irreversible, immediately active kill."""

        self._mutation_entry(MemoryReleaseRevocationConflictError)
        scope = _scope(scope)
        attestation_id = _string(attestation_id, "attestation_id")
        attestation_content_sha256 = _digest(
            attestation_content_sha256,
            "attestation_content_sha256",
        )
        _validate_address(
            attestation_id,
            attestation_content_sha256,
            prefix="mrat_",
            conflict_type=MemoryReleaseRevocationConflictError,
            label="attestation",
        )
        idempotency_key = _string(idempotency_key, "idempotency_key")
        idempotency_address = (scope, idempotency_key)
        attestation_address = (scope, attestation_id)
        tokens = (
            ("revocation-idempotency", scope, idempotency_key),
            ("attestation-control", scope, attestation_id),
        )
        owner = self._claim(tokens, conflict_type=MemoryReleaseRevocationConflictError)
        try:
            with self._lock:
                existing = self._revocation_by_idempotency.get(idempotency_address)
                if existing is not None:
                    existing = self._validate_revocation(scope, existing)
                    if (
                        existing.attestation_id == attestation_id
                        and existing.attestation_content_sha256
                        == attestation_content_sha256
                    ):
                        return existing
                    raise MemoryReleaseRevocationConflictError(
                        "scoped revocation idempotency key has a different request"
                    )
                prior = self._revocation_by_attestation.get(attestation_address)
                if prior is not None:
                    self._validate_revocation(
                        scope,
                        prior,
                        expected_attestation_id=attestation_id,
                    )
                    raise MemoryReleaseRevocationConflictError(
                        "attestation is already revoked"
                    )
                attestation = self._attestation_by_address.get(attestation_address)
                if attestation is None:
                    raise MemoryReleaseAttestationNotFoundError(
                        f"attestation {attestation_id!r} was not found"
                    )
                attestation = self._validate_attestation(
                    scope,
                    attestation,
                    expected_id=attestation_id,
                    expected_hash=attestation_content_sha256,
                )

            evaluated_at = self._now(MemoryReleaseRevocationConflictError)
            decision = self._invoke(
                self._revoker,
                conflict_type=MemoryReleaseRevocationConflictError,
                label="revoker",
                arguments={
                    "attestation": attestation,
                    "evaluated_at": evaluated_at,
                },
            )
            if type(decision) is not tuple or len(decision) != 2:
                raise MemoryReleaseRevocationConflictError(
                    "trusted revoker returned an invalid decision"
                )
            reason, reason_detail_sha256 = decision
            if type(reason) is not MemoryReleaseRevocationReason:
                raise MemoryReleaseRevocationConflictError(
                    "trusted revoker returned an invalid reason"
                )
            if reason_detail_sha256 is not None:
                try:
                    reason_detail_sha256 = _digest(
                        reason_detail_sha256,
                        "reason_detail_sha256",
                    )
                except (TypeError, ValueError) as error:
                    raise MemoryReleaseRevocationConflictError(
                        "trusted revoker returned an invalid reason detail hash"
                    ) from error
            if (
                reason is MemoryReleaseRevocationReason.OTHER
                and reason_detail_sha256 is None
            ):
                raise MemoryReleaseRevocationConflictError(
                    "OTHER revocation reason requires a detail hash"
                )
            self._validate_component(
                self._revoker,
                conflict_type=MemoryReleaseRevocationConflictError,
                label="revoker",
            )
            with self._lock:
                current = self._attestation_by_address.get(attestation_address)
                if current is None or not _canonical_equal(current, attestation):
                    raise MemoryReleaseRevocationConflictError(
                        "attestation changed during revocation"
                    )
                if self._revocation_by_attestation.get(attestation_address) is not None:
                    raise MemoryReleaseRevocationConflictError(
                        "attestation is already revoked"
                    )
                if self._revocation_by_idempotency.get(idempotency_address) is not None:
                    raise MemoryReleaseRevocationConflictError(
                        "revocation appeared during policy evaluation"
                    )
                revoked_at = self._now(MemoryReleaseRevocationConflictError)
                self._check_commit_time(
                    evaluated_at,
                    revoked_at,
                    MemoryReleaseRevocationConflictError,
                )
                revocation = MemoryReleaseAttestationRevocationV1.create(
                    scope=scope,
                    attestation_id=attestation_id,
                    attestation_content_sha256=attestation_content_sha256,
                    revoker_id=self._revoker.component_id,
                    revoker_version_sha256=self._revoker.version_sha256,
                    revoker_config_sha256=self._revoker.config_sha256,
                    reason=reason,
                    reason_detail_sha256=reason_detail_sha256,
                    evaluated_at=evaluated_at,
                    revoked_at=revoked_at,
                    idempotency_key=idempotency_key,
                )
                address = (scope, revocation.revocation_id)
                collision = self._revocation_by_address.get(address)
                if collision is not None and not _canonical_equal(
                    collision,
                    revocation,
                ):
                    raise MemoryReleaseRevocationConflictError(
                        f"revocation ID collision for {revocation.revocation_id!r}"
                    )
                stored = collision if collision is not None else revocation
                _atomic_publish(
                    mapping_writes=(
                        (self._revocation_by_address, address, stored),
                        (
                            self._revocation_by_attestation,
                            attestation_address,
                            stored,
                        ),
                        (
                            self._revocation_by_idempotency,
                            idempotency_address,
                            stored,
                        ),
                    )
                )
                return stored
        finally:
            self._release_claims(tokens, owner)

    def get_revocation(
        self,
        scope: MemoryScope,
        revocation_id: str,
    ) -> MemoryReleaseAttestationRevocationV1:
        scope = _scope(scope)
        revocation_id = _string(revocation_id, "revocation_id")
        with self._lock:
            value = self._revocation_by_address.get((scope, revocation_id))
        if value is None:
            raise MemoryReleaseRevocationNotFoundError(
                f"revocation {revocation_id!r} was not found"
            )
        return self._validate_revocation(scope, value, expected_id=revocation_id)

    def get_attestation_revocation(
        self,
        scope: MemoryScope,
        attestation_id: str,
    ) -> MemoryReleaseAttestationRevocationV1:
        scope = _scope(scope)
        attestation_id = _string(attestation_id, "attestation_id")
        with self._lock:
            value = self._revocation_by_attestation.get((scope, attestation_id))
        if value is None:
            raise MemoryReleaseRevocationNotFoundError(
                f"attestation {attestation_id!r} has no revocation"
            )
        return self._validate_revocation(
            scope,
            value,
            expected_attestation_id=attestation_id,
        )

    @staticmethod
    def _assignment_request_matches(
        value: MemoryReleaseAssignmentV1,
        *,
        rollout_group_id: str,
        rollout_group_incarnation_sha256: str,
        attestation_id: str,
        attestation_content_sha256: str,
        task_policy_id: str,
        task_policy_version_sha256: str,
        task_policy_config_sha256: str,
        retrieval_policy_id: str,
        retrieval_policy_version_sha256: str,
        retrieval_policy_config_sha256: str,
        renderer_id: str,
        renderer_version_sha256: str,
        renderer_config_sha256: str,
        consumer_kind: MemoryReleaseAssignmentConsumerKind,
        consumer_id: str,
        consumer_version_sha256: str,
        consumer_config_sha256: str,
        max_returned_items: int,
        max_context_utf8_bytes: int,
    ) -> bool:
        return (
            value.rollout_group_id == rollout_group_id
            and value.rollout_group_incarnation_sha256
            == rollout_group_incarnation_sha256
            and value.attestation_id == attestation_id
            and value.attestation_content_sha256 == attestation_content_sha256
            and value.task_policy_id == task_policy_id
            and value.task_policy_version_sha256 == task_policy_version_sha256
            and value.task_policy_config_sha256 == task_policy_config_sha256
            and value.retrieval_policy_id == retrieval_policy_id
            and value.retrieval_policy_version_sha256
            == retrieval_policy_version_sha256
            and value.retrieval_policy_config_sha256
            == retrieval_policy_config_sha256
            and value.renderer_id == renderer_id
            and value.renderer_version_sha256 == renderer_version_sha256
            and value.renderer_config_sha256 == renderer_config_sha256
            and value.consumer_kind is consumer_kind
            and value.consumer_id == consumer_id
            and value.consumer_version_sha256 == consumer_version_sha256
            and value.consumer_config_sha256 == consumer_config_sha256
            and value.max_returned_items == max_returned_items
            and value.max_context_utf8_bytes == max_context_utf8_bytes
        )

    def assign_release(
        self,
        scope: MemoryScope,
        rollout_group_id: str,
        *,
        rollout_group_incarnation_sha256: str,
        attestation_id: str,
        attestation_content_sha256: str,
        task_policy_id: str,
        task_policy_version_sha256: str,
        task_policy_config_sha256: str,
        retrieval_policy_id: str,
        retrieval_policy_version_sha256: str,
        retrieval_policy_config_sha256: str,
        renderer_id: str,
        renderer_version_sha256: str,
        renderer_config_sha256: str,
        consumer_kind: MemoryReleaseAssignmentConsumerKind,
        consumer_id: str,
        consumer_version_sha256: str,
        consumer_config_sha256: str,
        max_returned_items: int,
        max_context_utf8_bytes: int,
        idempotency_key: str,
    ) -> MemoryReleaseAssignmentV1:
        """Bind one group ID exactly once to a complete execution snapshot."""

        self._mutation_entry(MemoryReleaseAssignmentConflictError)
        scope = _scope(scope)
        rollout_group_id = _string(rollout_group_id, "rollout_group_id")
        rollout_group_incarnation_sha256 = _digest(
            rollout_group_incarnation_sha256,
            "rollout_group_incarnation_sha256",
        )
        attestation_id = _string(attestation_id, "attestation_id")
        attestation_content_sha256 = _digest(
            attestation_content_sha256,
            "attestation_content_sha256",
        )
        _validate_address(
            attestation_id,
            attestation_content_sha256,
            prefix="mrat_",
            conflict_type=MemoryReleaseAssignmentConflictError,
            label="attestation",
        )
        task_policy_id = _string(task_policy_id, "task_policy_id")
        task_policy_version_sha256 = _digest(
            task_policy_version_sha256,
            "task_policy_version_sha256",
        )
        task_policy_config_sha256 = _digest(
            task_policy_config_sha256,
            "task_policy_config_sha256",
        )
        retrieval_policy_id = _string(retrieval_policy_id, "retrieval_policy_id")
        retrieval_policy_version_sha256 = _digest(
            retrieval_policy_version_sha256,
            "retrieval_policy_version_sha256",
        )
        retrieval_policy_config_sha256 = _digest(
            retrieval_policy_config_sha256,
            "retrieval_policy_config_sha256",
        )
        renderer_id = _string(renderer_id, "renderer_id")
        renderer_version_sha256 = _digest(
            renderer_version_sha256,
            "renderer_version_sha256",
        )
        renderer_config_sha256 = _digest(
            renderer_config_sha256,
            "renderer_config_sha256",
        )
        if type(consumer_kind) is not MemoryReleaseAssignmentConsumerKind:
            raise TypeError(
                "consumer_kind must be a MemoryReleaseAssignmentConsumerKind"
            )
        consumer_id = _string(consumer_id, "consumer_id")
        consumer_version_sha256 = _digest(
            consumer_version_sha256,
            "consumer_version_sha256",
        )
        consumer_config_sha256 = _digest(
            consumer_config_sha256,
            "consumer_config_sha256",
        )
        max_returned_items = _integer(max_returned_items, "max_returned_items")
        max_context_utf8_bytes = _integer(
            max_context_utf8_bytes,
            "max_context_utf8_bytes",
        )
        idempotency_key = _string(idempotency_key, "idempotency_key")
        idempotency_address = (scope, idempotency_key)
        group_address = (scope, rollout_group_id)
        incarnation_address = (scope, rollout_group_incarnation_sha256)
        attestation_address = (scope, attestation_id)
        tokens = (
            ("assignment-idempotency", scope, idempotency_key),
            ("assignment-group", scope, rollout_group_id),
            (
                "assignment-incarnation",
                scope,
                rollout_group_incarnation_sha256,
            ),
            ("attestation-control", scope, attestation_id),
        )
        owner = self._claim(tokens, conflict_type=MemoryReleaseAssignmentConflictError)
        try:
            request = {
                "rollout_group_id": rollout_group_id,
                "rollout_group_incarnation_sha256": rollout_group_incarnation_sha256,
                "attestation_id": attestation_id,
                "attestation_content_sha256": attestation_content_sha256,
                "task_policy_id": task_policy_id,
                "task_policy_version_sha256": task_policy_version_sha256,
                "task_policy_config_sha256": task_policy_config_sha256,
                "retrieval_policy_id": retrieval_policy_id,
                "retrieval_policy_version_sha256": retrieval_policy_version_sha256,
                "retrieval_policy_config_sha256": retrieval_policy_config_sha256,
                "renderer_id": renderer_id,
                "renderer_version_sha256": renderer_version_sha256,
                "renderer_config_sha256": renderer_config_sha256,
                "consumer_kind": consumer_kind,
                "consumer_id": consumer_id,
                "consumer_version_sha256": consumer_version_sha256,
                "consumer_config_sha256": consumer_config_sha256,
                "max_returned_items": max_returned_items,
                "max_context_utf8_bytes": max_context_utf8_bytes,
            }
            with self._lock:
                existing = self._assignment_by_idempotency.get(idempotency_address)
                if existing is not None:
                    existing = self._validate_assignment(scope, existing)
                    if self._assignment_request_matches(existing, **request):
                        return existing
                    raise MemoryReleaseAssignmentConflictError(
                        "scoped assignment idempotency key has a different request"
                    )
                group_existing = self._assignment_by_group.get(group_address)
                if group_existing is not None:
                    self._validate_assignment(
                        scope,
                        group_existing,
                        expected_group=rollout_group_id,
                    )
                    raise MemoryReleaseAssignmentConflictError(
                        "rollout group is already bound; aliases are forbidden"
                    )
                incarnation_existing = self._assignment_by_incarnation.get(
                    incarnation_address
                )
                if incarnation_existing is not None:
                    self._validate_assignment(scope, incarnation_existing)
                    raise MemoryReleaseAssignmentConflictError(
                        "rollout group incarnation is already bound"
                    )
                attestation = self._attestation_by_address.get(attestation_address)
                if attestation is None:
                    raise MemoryReleaseAttestationNotFoundError(
                        f"attestation {attestation_id!r} was not found"
                    )
                attestation = self._validate_attestation(
                    scope,
                    attestation,
                    expected_id=attestation_id,
                    expected_hash=attestation_content_sha256,
                )
                if self._revocation_by_attestation.get(attestation_address) is not None:
                    raise MemoryReleaseAssignmentConflictError(
                        "attestation is revoked"
                    )

            before = self._load_release_graph(
                scope,
                attestation.release_id,
                attestation.release_content_sha256,
                conflict_type=MemoryReleaseAssignmentConflictError,
            )
            if before[0].release_graph_sha256 != attestation.release_graph_sha256:
                raise MemoryReleaseAssignmentConflictError(
                    "attestation release graph commitment does not match"
                )
            evaluated_at = self._now(MemoryReleaseAssignmentConflictError)
            assignment_valid_until_value = self._invoke(
                self._assignment_policy,
                conflict_type=MemoryReleaseAssignmentConflictError,
                label="assignment policy",
                arguments={
                    "rollout_group_id": rollout_group_id,
                    "rollout_group_incarnation_sha256": (
                        rollout_group_incarnation_sha256
                    ),
                    "attestation": attestation,
                    "task_policy_id": task_policy_id,
                    "task_policy_version_sha256": task_policy_version_sha256,
                    "task_policy_config_sha256": task_policy_config_sha256,
                    "retrieval_policy_id": retrieval_policy_id,
                    "retrieval_policy_version_sha256": (
                        retrieval_policy_version_sha256
                    ),
                    "retrieval_policy_config_sha256": (
                        retrieval_policy_config_sha256
                    ),
                    "renderer_id": renderer_id,
                    "renderer_version_sha256": renderer_version_sha256,
                    "renderer_config_sha256": renderer_config_sha256,
                    "consumer_kind": consumer_kind,
                    "consumer_id": consumer_id,
                    "consumer_version_sha256": consumer_version_sha256,
                    "consumer_config_sha256": consumer_config_sha256,
                    "max_returned_items": max_returned_items,
                    "max_context_utf8_bytes": max_context_utf8_bytes,
                    "evaluated_at": evaluated_at,
                },
            )
            try:
                assignment_valid_until = _aware_datetime(
                    assignment_valid_until_value,
                    "assignment_valid_until",
                )
            except (TypeError, ValueError) as error:
                raise MemoryReleaseAssignmentConflictError(
                    "trusted assignment policy returned an invalid expiry"
                ) from error
            after = self._load_release_graph(
                scope,
                attestation.release_id,
                attestation.release_content_sha256,
                conflict_type=MemoryReleaseAssignmentConflictError,
            )
            if not self._same_release_graph(before, after):
                raise MemoryReleaseAssignmentConflictError(
                    "release graph changed during assignment"
                )
            self._validate_component(
                self._assignment_policy,
                conflict_type=MemoryReleaseAssignmentConflictError,
                label="assignment policy",
            )
            final_graph = self._load_release_graph(
                scope,
                attestation.release_id,
                attestation.release_content_sha256,
                conflict_type=MemoryReleaseAssignmentConflictError,
            )
            if not self._same_release_graph(after, final_graph):
                raise MemoryReleaseAssignmentConflictError(
                    "release graph changed before assignment commit"
                )
            with self._lock:
                current_attestation = self._attestation_by_address.get(
                    attestation_address
                )
                if current_attestation is None or not _canonical_equal(
                    current_attestation,
                    attestation,
                ):
                    raise MemoryReleaseAssignmentConflictError(
                        "attestation changed during assignment"
                    )
                if self._revocation_by_attestation.get(attestation_address) is not None:
                    raise MemoryReleaseAssignmentConflictError(
                        "attestation was revoked during assignment"
                    )
                if self._assignment_by_group.get(group_address) is not None:
                    raise MemoryReleaseAssignmentConflictError(
                        "rollout group was bound during assignment"
                    )
                if (
                    self._assignment_by_incarnation.get(incarnation_address)
                    is not None
                ):
                    raise MemoryReleaseAssignmentConflictError(
                        "rollout group incarnation was bound during assignment"
                    )
                if self._assignment_by_idempotency.get(idempotency_address) is not None:
                    raise MemoryReleaseAssignmentConflictError(
                        "assignment appeared during policy evaluation"
                    )
                assigned_at = self._now(MemoryReleaseAssignmentConflictError)
                self._check_commit_time(
                    evaluated_at,
                    assigned_at,
                    MemoryReleaseAssignmentConflictError,
                )
                if not (
                    attestation.valid_from
                    <= assigned_at
                    < attestation.valid_until
                ):
                    raise MemoryReleaseAssignmentConflictError(
                        "attestation is not valid at assignment commit time"
                    )
                if not (
                    assigned_at
                    < assignment_valid_until
                    <= attestation.valid_until
                ):
                    raise MemoryReleaseAssignmentConflictError(
                        "assignment expiry must follow commit and not exceed attestation"
                    )
                assignment = MemoryReleaseAssignmentV1.create(
                    scope=scope,
                    rollout_group_id=rollout_group_id,
                    rollout_group_incarnation_sha256=(
                        rollout_group_incarnation_sha256
                    ),
                    attestation_id=attestation_id,
                    attestation_content_sha256=attestation_content_sha256,
                    release_id=attestation.release_id,
                    release_content_sha256=attestation.release_content_sha256,
                    release_graph_sha256=attestation.release_graph_sha256,
                    assignment_policy_id=self._assignment_policy.component_id,
                    assignment_policy_version_sha256=(
                        self._assignment_policy.version_sha256
                    ),
                    assignment_policy_config_sha256=(
                        self._assignment_policy.config_sha256
                    ),
                    task_policy_id=task_policy_id,
                    task_policy_version_sha256=task_policy_version_sha256,
                    task_policy_config_sha256=task_policy_config_sha256,
                    retrieval_policy_id=retrieval_policy_id,
                    retrieval_policy_version_sha256=(
                        retrieval_policy_version_sha256
                    ),
                    retrieval_policy_config_sha256=(
                        retrieval_policy_config_sha256
                    ),
                    renderer_id=renderer_id,
                    renderer_version_sha256=renderer_version_sha256,
                    renderer_config_sha256=renderer_config_sha256,
                    consumer_kind=consumer_kind,
                    consumer_id=consumer_id,
                    consumer_version_sha256=consumer_version_sha256,
                    consumer_config_sha256=consumer_config_sha256,
                    max_returned_items=max_returned_items,
                    max_context_utf8_bytes=max_context_utf8_bytes,
                    evaluated_at=evaluated_at,
                    assigned_at=assigned_at,
                    assignment_valid_until=assignment_valid_until,
                    idempotency_key=idempotency_key,
                )
                address = (scope, assignment.assignment_id)
                collision = self._assignment_by_address.get(address)
                if collision is not None and not _canonical_equal(
                    collision,
                    assignment,
                ):
                    raise MemoryReleaseAssignmentConflictError(
                        f"assignment ID collision for {assignment.assignment_id!r}"
                    )
                stored = collision if collision is not None else assignment
                _atomic_publish(
                    mapping_writes=(
                        (self._assignment_by_address, address, stored),
                        (self._assignment_by_group, group_address, stored),
                        (
                            self._assignment_by_incarnation,
                            incarnation_address,
                            stored,
                        ),
                        (
                            self._assignment_by_idempotency,
                            idempotency_address,
                            stored,
                        ),
                    )
                )
                return stored
        finally:
            self._release_claims(tokens, owner)

    def get_assignment(
        self,
        scope: MemoryScope,
        assignment_id: str,
    ) -> MemoryReleaseAssignmentV1:
        """Return historical audit data without asserting current authority."""

        scope = _scope(scope)
        assignment_id = _string(assignment_id, "assignment_id")
        with self._lock:
            value = self._assignment_by_address.get((scope, assignment_id))
        if value is None:
            raise MemoryReleaseAssignmentNotFoundError(
                f"assignment {assignment_id!r} was not found"
            )
        return self._validate_assignment(scope, value, expected_id=assignment_id)

    def get_rollout_group_assignment(
        self,
        scope: MemoryScope,
        rollout_group_id: str,
    ) -> MemoryReleaseAssignmentV1:
        """Return a group's historical binding without checking liveness."""

        scope = _scope(scope)
        rollout_group_id = _string(rollout_group_id, "rollout_group_id")
        with self._lock:
            value = self._assignment_by_group.get((scope, rollout_group_id))
        if value is None:
            raise MemoryReleaseAssignmentNotFoundError(
                f"rollout group {rollout_group_id!r} has no assignment"
            )
        return self._validate_assignment(
            scope,
            value,
            expected_group=rollout_group_id,
        )

    def resolve_active_assignment(
        self,
        scope: MemoryScope,
        rollout_group_id: str,
        rollout_group_incarnation_sha256: str,
        assignment_id: str,
        assignment_content_sha256: str,
    ) -> MemoryReleaseAssignmentV1:
        """Resolve live authority with immediate kill-switch semantics.

        Runtime integrations must call this independently immediately before
        the query, render, and configured consumer boundaries.  A successful
        earlier lookup or the historical assignment value is not a lease.
        """

        scope = _scope(scope)
        rollout_group_id = _string(rollout_group_id, "rollout_group_id")
        rollout_group_incarnation_sha256 = _digest(
            rollout_group_incarnation_sha256,
            "rollout_group_incarnation_sha256",
        )
        assignment_id = _string(assignment_id, "assignment_id")
        assignment_content_sha256 = _digest(
            assignment_content_sha256,
            "assignment_content_sha256",
        )
        _validate_address(
            assignment_id,
            assignment_content_sha256,
            prefix="masn_",
            conflict_type=MemoryReleaseAssignmentConflictError,
            label="assignment",
        )
        group_address = (scope, rollout_group_id)
        incarnation_address = (scope, rollout_group_incarnation_sha256)
        assignment_address = (scope, assignment_id)
        with self._lock:
            assignment = self._assignment_by_address.get(assignment_address)
            if assignment is None:
                raise MemoryReleaseAssignmentNotFoundError(
                    f"assignment {assignment_id!r} was not found"
                )
            assignment = self._validate_assignment(
                scope,
                assignment,
                expected_id=assignment_id,
                expected_hash=assignment_content_sha256,
                expected_group=rollout_group_id,
            )
            if (
                assignment.rollout_group_incarnation_sha256
                != rollout_group_incarnation_sha256
                or self._assignment_by_group.get(group_address) is not assignment
                or self._assignment_by_incarnation.get(incarnation_address)
                is not assignment
            ):
                raise MemoryReleaseAssignmentConflictError(
                    "assignment does not bind the requested group incarnation"
                )
            attestation_address = (scope, assignment.attestation_id)
            attestation = self._attestation_by_address.get(attestation_address)
            if attestation is None:
                raise MemoryReleaseAssignmentConflictError(
                    "assignment attestation is unavailable"
                )
            attestation = self._validate_attestation(
                scope,
                attestation,
                expected_id=assignment.attestation_id,
                expected_hash=assignment.attestation_content_sha256,
            )
            if (
                assignment.release_id != attestation.release_id
                or assignment.release_content_sha256
                != attestation.release_content_sha256
                or assignment.release_graph_sha256
                != attestation.release_graph_sha256
            ):
                raise MemoryReleaseAssignmentConflictError(
                    "assignment release binding disagrees with its attestation"
                )
            if self._revocation_by_attestation.get(attestation_address) is not None:
                raise MemoryReleaseAssignmentConflictError(
                    "assignment attestation is revoked"
                )

        graph = self._load_release_graph(
            scope,
            assignment.release_id,
            assignment.release_content_sha256,
            conflict_type=MemoryReleaseAssignmentConflictError,
        )
        if graph[0].release_graph_sha256 != assignment.release_graph_sha256:
            raise MemoryReleaseAssignmentConflictError(
                "assignment release graph commitment does not match"
            )
        with self._lock:
            if self._assignment_by_address.get(assignment_address) is not assignment:
                raise MemoryReleaseAssignmentConflictError(
                    "assignment changed during active resolution"
                )
            if self._assignment_by_group.get(group_address) is not assignment:
                raise MemoryReleaseAssignmentConflictError(
                    "group binding changed during active resolution"
                )
            if (
                self._assignment_by_incarnation.get(incarnation_address)
                is not assignment
            ):
                raise MemoryReleaseAssignmentConflictError(
                    "group incarnation binding changed during active resolution"
                )
            current_attestation = self._attestation_by_address.get(
                attestation_address
            )
            if current_attestation is not attestation:
                raise MemoryReleaseAssignmentConflictError(
                    "attestation changed during active resolution"
                )
            if self._revocation_by_attestation.get(attestation_address) is not None:
                raise MemoryReleaseAssignmentConflictError(
                    "assignment attestation is revoked"
                )
            now = self._now(MemoryReleaseAssignmentConflictError)
            if not (
                attestation.valid_from
                <= now
                and assignment.assigned_at <= now
                < assignment.assignment_valid_until
                <= attestation.valid_until
            ):
                raise MemoryReleaseAssignmentConflictError(
                    "assignment or attestation is expired or not yet valid"
                )
            return assignment
