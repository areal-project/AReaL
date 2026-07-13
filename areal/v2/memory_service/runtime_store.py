# SPDX-License-Identifier: Apache-2.0

"""Reference runtime ledger for source-read query provenance and exposure.

The in-memory implementation owns every transition.  Registered retrievers,
renderers, and consumers produce selection, exact rendered bytes, and actual
consumer-call receipts; callers never submit those records themselves.  The
final join reloads the complete stored chain and derives injection from
acknowledged byte spans.

Source-read receipts contain only object IDs and full content hashes.  They
show what this honest runtime read and validated for one attempt; they do not
show that a model used the memory or that the memory improved task utility.

This is an honest-runtime integrity boundary.  Python object privacy and
content hashes do not authenticate a malicious in-process caller.  A remote
consumer boundary will additionally require service authentication or signed
receipts.
"""

from __future__ import annotations

import json
import secrets
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from threading import Condition, RLock, get_ident
from typing import Protocol

from areal.v2.memory_service.errors import (
    MemoryBoundaryMismatchError,
    MemoryConsumerAckConflictError,
    MemoryConsumerAckNotFoundError,
    MemoryDeliveryConflictError,
    MemoryDeliveryNotFoundError,
    MemoryExposureConflictError,
    MemoryExposureNotFoundError,
    MemoryQueryConflictError,
    MemoryQueryNotFoundError,
)
from areal.v2.memory_service.history_store import MemoryHistoryStore
from areal.v2.memory_service.history_types import (
    MemoryCandidate,
    MemoryRevision,
    RevisionOperation,
)
from areal.v2.memory_service.release_store import MemoryReleaseStore
from areal.v2.memory_service.release_types import MemoryRelease, ReleaseManifest
from areal.v2.memory_service.runtime_types import (
    MemoryConsumerAckV1,
    MemoryConsumerKind,
    MemoryDeliveryV1,
    MemoryEvidenceRefV1,
    MemoryExposureV1,
    MemoryQueryAttemptV1,
    MemoryQueryItemV1,
    MemoryQueryResultV1,
    MemoryQuerySpecV1,
    MemoryRenderedRevisionRangeV1,
    MemoryRenderedRevisionSpanV1,
    MemoryRevisionRefV1,
    MemorySourceObjectKind,
    MemorySourceObjectRefV1,
    MemorySourceReadEventV1,
    MemorySourceReadOperation,
    MemorySourceReadReceiptV1,
)
from areal.v2.memory_service.types import EvidenceRecord, MemoryScope

_RUNTIME_COMPONENT_CALL_ACTIVE: ContextVar[bool] = ContextVar(
    "areal_memory_runtime_component_call_active",
    default=False,
)


class _SourceReadSession:
    """Private per-attempt collector; callers cannot submit read provenance."""

    __slots__ = ("_events",)

    def __init__(self) -> None:
        self._events: list[MemorySourceReadEventV1] = []

    def record(
        self,
        operation: MemorySourceReadOperation,
        returned_objects: tuple[MemorySourceObjectRefV1, ...],
    ) -> None:
        self._events.append(
            MemorySourceReadEventV1(
                sequence_no=len(self._events),
                operation=operation,
                returned_objects=returned_objects,
            )
        )

    def snapshot(self) -> tuple[MemorySourceReadEventV1, ...]:
        return tuple(self._events)


def _source_ref(
    kind: MemorySourceObjectKind,
    value: MemoryRelease | MemoryRevision | MemoryCandidate | EvidenceRecord,
) -> MemorySourceObjectRefV1:
    if kind is MemorySourceObjectKind.RELEASE:
        if type(value) is not MemoryRelease:
            raise MemoryQueryConflictError("release read returned a non-release value")
        object_id = value.release_id
    elif kind is MemorySourceObjectKind.REVISION:
        if type(value) is not MemoryRevision:
            raise MemoryQueryConflictError(
                "revision read returned a non-revision value"
            )
        object_id = value.revision_id
    elif kind is MemorySourceObjectKind.CANDIDATE:
        if type(value) is not MemoryCandidate:
            raise MemoryQueryConflictError(
                "candidate read returned a non-candidate value"
            )
        object_id = value.candidate_id
    else:
        if type(value) is not EvidenceRecord:
            raise MemoryQueryConflictError(
                "evidence read returned a non-evidence value"
            )
        object_id = value.evidence_id
    try:
        return MemorySourceObjectRefV1(
            kind=kind,
            object_id=object_id,
            object_content_sha256=value.content_hash,
        )
    except (TypeError, ValueError) as error:
        raise MemoryQueryConflictError(
            "source read returned invalid address or content commitments"
        ) from error


def _new_runtime_nonce() -> str:
    """Return a fresh 256-bit attempt or delivery nonce."""

    return secrets.token_hex(32)


def _string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a str")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    return value


def _digest(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _scope(value: object) -> MemoryScope:
    if type(value) is not MemoryScope:
        raise TypeError("scope must be a MemoryScope")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple")
    values = tuple(tuple.__iter__(value))
    if any(type(item) is not str or not item.strip() for item in values):
        raise TypeError(f"{field_name} must contain non-blank str values")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


def _range_tuple(
    value: object,
) -> tuple[MemoryRenderedRevisionRangeV1, ...]:
    if type(value) is not tuple:
        raise TypeError("rendered_ranges must be a tuple")
    values = tuple(tuple.__iter__(value))
    if any(type(item) is not MemoryRenderedRevisionRangeV1 for item in values):
        raise TypeError(
            "rendered_ranges must contain MemoryRenderedRevisionRangeV1 values"
        )
    return values


def _history_sha256(history: tuple[bytes, ...]) -> str:
    digest = sha256(b"areal-memory-runtime-history-v1\0")
    digest.update(len(history).to_bytes(8, "big"))
    for item in history:
        digest.update(len(item).to_bytes(8, "big"))
        digest.update(item)
    return digest.hexdigest()


def _revision_ref(revision: MemoryRevision) -> MemoryRevisionRefV1:
    if type(revision) is not MemoryRevision:
        raise TypeError("release store returned a non-MemoryRevision value")
    return MemoryRevisionRefV1(
        revision_id=revision.revision_id,
        revision_content_sha256=revision.content_hash,
    )


def _canonical_equal(left: object, right: object) -> bool:
    return left.canonical_bytes() == right.canonical_bytes()  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class MemoryRenderOutputV1:
    """Ephemeral output returned only by a store-registered trusted renderer."""

    rendered_context: bytes
    rendered_ranges: tuple[MemoryRenderedRevisionRangeV1, ...]

    def __post_init__(self) -> None:
        if type(self.rendered_context) is not bytes:
            raise TypeError("rendered_context must be bytes")
        _range_tuple(self.rendered_ranges)


@dataclass(frozen=True, slots=True)
class MemoryRetrievalOutputV1:
    """Ephemeral ordered selection from a store-registered retriever."""

    retrieved_revision_ids: tuple[str, ...]
    returned_revision_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _string_tuple(self.retrieved_revision_ids, "retrieved_revision_ids")
        _string_tuple(self.returned_revision_ids, "returned_revision_ids")


class MemoryRuntimeRetriever(Protocol):
    """Deterministic retrieval policy registered by exact ID and version."""

    retrieval_policy_id: str
    retrieval_policy_version_sha256: str

    def retrieve(
        self,
        *,
        attempt: MemoryQueryAttemptV1,
        query: bytes,
        eligible_items: tuple[MemoryQueryItemV1, ...],
    ) -> MemoryRetrievalOutputV1: ...


class MemoryRuntimeRenderer(Protocol):
    """Trusted renderer registered by immutable ID and implementation hash."""

    renderer_id: str
    renderer_version_sha256: str

    def render(self, query_result: MemoryQueryResultV1) -> MemoryRenderOutputV1: ...


@dataclass(frozen=True, slots=True)
class MemoryConsumerCallV1:
    """Ephemeral actual call bytes plus an unpersisted consumer output."""

    delivery_id: str
    delivery_content_sha256: str
    call_id: str
    submitted_prompt: bytes
    context_start: int
    context_end: int
    observed_query_sha256: str
    observed_history_sha256: str
    observed_history_length: int
    input_token_ids: tuple[int, ...] | None
    output: object

    def __post_init__(self) -> None:
        _string(self.delivery_id, "delivery_id")
        _digest(self.delivery_content_sha256, "delivery_content_sha256")
        _string(self.call_id, "call_id")
        if type(self.submitted_prompt) is not bytes:
            raise TypeError("submitted_prompt must be bytes")
        if type(self.context_start) is not int or type(self.context_end) is not int:
            raise TypeError("context offsets must be ints")
        if (
            not 0
            <= self.context_start
            <= self.context_end
            <= len(self.submitted_prompt)
        ):
            raise ValueError("context offsets must select a submitted prompt slice")
        if self.input_token_ids is not None:
            if type(self.input_token_ids) is not tuple or any(
                type(item) is not int or item < 0
                for item in tuple.__iter__(self.input_token_ids)
            ):
                raise TypeError(
                    "input_token_ids must be a tuple of non-negative ints or None"
                )
        _digest(self.observed_query_sha256, "observed_query_sha256")
        _digest(self.observed_history_sha256, "observed_history_sha256")
        if (
            type(self.observed_history_length) is not int
            or self.observed_history_length < 0
        ):
            raise TypeError("observed_history_length must be a non-negative int")


class MemoryRuntimeConsumer(Protocol):
    """Trusted component that owns the actual consumer/model invocation.

    A consumer that crosses an external boundary MUST use
    ``(delivery.scope, delivery.trajectory_id, call_id)`` as its durable
    idempotency key.  Its stored value must bind a request fingerprint over the
    exact delivery, context, query, and history.  The same key and fingerprint
    replays the cached result; the same key with a different fingerprint fails
    closed.  A cached result must return its original receipt so the store can
    also reject a request mismatch.  The in-memory ledger can make its local
    acknowledgement and exposure atomic, but it cannot infer whether an
    exception happened before or after a remote side effect.
    """

    consumer_kind: MemoryConsumerKind
    consumer_id: str
    consumer_version_sha256: str

    def submit(
        self,
        *,
        delivery: MemoryDeliveryV1,
        rendered_context: bytes,
        query: bytes,
        history: tuple[bytes, ...],
        call_id: str,
    ) -> MemoryConsumerCallV1: ...


class MemoryRuntimeStore(Protocol):
    """Backend-neutral phase and audit-read API for actual exposure."""

    def begin_query(self, spec: MemoryQuerySpecV1) -> MemoryQueryAttemptV1: ...

    def get_query_attempt(
        self,
        scope: MemoryScope,
        attempt_id: str,
    ) -> MemoryQueryAttemptV1: ...

    def resolve_query(
        self,
        scope: MemoryScope,
        attempt_id: str,
        *,
        query: bytes,
    ) -> MemoryQueryResultV1: ...

    def get_query_result(
        self,
        scope: MemoryScope,
        query_result_id: str,
    ) -> MemoryQueryResultV1: ...

    def get_source_read_receipt(
        self,
        scope: MemoryScope,
        source_read_receipt_id: str,
    ) -> MemorySourceReadReceiptV1: ...

    def prepare_delivery(
        self,
        scope: MemoryScope,
        query_result_id: str,
        *,
        renderer_id: str,
        renderer_version_sha256: str,
    ) -> MemoryDeliveryV1: ...

    def get_delivery(
        self,
        scope: MemoryScope,
        delivery_id: str,
    ) -> MemoryDeliveryV1: ...

    def submit_delivery(
        self,
        scope: MemoryScope,
        delivery_id: str,
        *,
        consumer_id: str,
        consumer_version_sha256: str,
        call_id: str,
        query: bytes,
        history: tuple[bytes, ...],
    ) -> tuple[MemoryExposureV1, object]: ...

    def get_consumer_ack(
        self,
        scope: MemoryScope,
        consumer_ack_id: str,
    ) -> MemoryConsumerAckV1: ...

    def get_exposure(
        self,
        scope: MemoryScope,
        exposure_id: str,
    ) -> MemoryExposureV1: ...

    def list_exposures(self, scope: MemoryScope) -> tuple[MemoryExposureV1, ...]: ...


class InMemoryMemoryRuntimeStore:
    """Lock-protected reference phase ledger over immutable Memory stores."""

    def __init__(
        self,
        history_store: MemoryHistoryStore,
        release_store: MemoryReleaseStore,
        *,
        retrievers: tuple[MemoryRuntimeRetriever, ...] = (),
        renderers: tuple[MemoryRuntimeRenderer, ...] = (),
        consumers: tuple[MemoryRuntimeConsumer, ...] = (),
    ) -> None:
        if type(retrievers) is not tuple:
            raise TypeError("retrievers must be a tuple")
        if type(renderers) is not tuple:
            raise TypeError("renderers must be a tuple")
        if type(consumers) is not tuple:
            raise TypeError("consumers must be a tuple")
        self._history_store = history_store
        self._release_store = release_store
        self._lock = RLock()
        self._submission_condition = Condition(self._lock)
        self._active_component_thread_ids: set[int] = set()
        self._retriever_by_version: dict[tuple[str, str], MemoryRuntimeRetriever] = {}
        for retriever in tuple.__iter__(retrievers):
            retrieval_policy_id = _string(
                getattr(retriever, "retrieval_policy_id", None),
                "retrieval_policy_id",
            )
            retrieval_policy_version = _digest(
                getattr(retriever, "retrieval_policy_version_sha256", None),
                "retrieval_policy_version_sha256",
            )
            key = (retrieval_policy_id, retrieval_policy_version)
            if key in self._retriever_by_version:
                raise ValueError("retriever registry must not contain duplicates")
            if not callable(getattr(retriever, "retrieve", None)):
                raise TypeError("registered retriever must define retrieve")
            self._retriever_by_version[key] = retriever
        self._renderer_by_version: dict[tuple[str, str], MemoryRuntimeRenderer] = {}
        for renderer in tuple.__iter__(renderers):
            renderer_id = _string(
                getattr(renderer, "renderer_id", None),
                "renderer_id",
            )
            renderer_version = _digest(
                getattr(renderer, "renderer_version_sha256", None),
                "renderer_version_sha256",
            )
            key = (renderer_id, renderer_version)
            if key in self._renderer_by_version:
                raise ValueError("renderer registry must not contain duplicates")
            if not callable(getattr(renderer, "render", None)):
                raise TypeError("registered renderer must define render")
            self._renderer_by_version[key] = renderer
        self._consumer_by_version: dict[
            tuple[str, str], tuple[MemoryRuntimeConsumer, MemoryConsumerKind]
        ] = {}
        for consumer in tuple.__iter__(consumers):
            consumer_id = _string(
                getattr(consumer, "consumer_id", None),
                "consumer_id",
            )
            consumer_version = _digest(
                getattr(consumer, "consumer_version_sha256", None),
                "consumer_version_sha256",
            )
            consumer_kind = getattr(consumer, "consumer_kind", None)
            if type(consumer_kind) is not MemoryConsumerKind:
                raise TypeError(
                    "registered consumer must define an exact MemoryConsumerKind"
                )
            key = (consumer_id, consumer_version)
            if key in self._consumer_by_version:
                raise ValueError("consumer registry must not contain duplicates")
            if not callable(getattr(consumer, "submit", None)):
                raise TypeError("registered consumer must define submit")
            self._consumer_by_version[key] = (consumer, consumer_kind)
        self._attempt_by_address: dict[
            tuple[MemoryScope, str], MemoryQueryAttemptV1
        ] = {}
        self._attempt_by_idempotency: dict[
            tuple[MemoryScope, str], MemoryQueryAttemptV1
        ] = {}
        self._attempt_by_trajectory_slot: dict[
            tuple[MemoryScope, str, int], MemoryQueryAttemptV1
        ] = {}
        self._trajectory_binding: dict[
            tuple[MemoryScope, str], tuple[str, str, str, str, str]
        ] = {}
        self._rollout_group_binding: dict[
            tuple[MemoryScope, str], tuple[str, str, str, str]
        ] = {}
        self._result_by_address: dict[tuple[MemoryScope, str], MemoryQueryResultV1] = {}
        self._result_by_attempt: dict[tuple[MemoryScope, str], MemoryQueryResultV1] = {}
        self._source_read_receipt_by_address: dict[
            tuple[MemoryScope, str], MemorySourceReadReceiptV1
        ] = {}
        self._source_read_receipt_by_attempt: dict[
            tuple[MemoryScope, str], MemorySourceReadReceiptV1
        ] = {}
        self._resolution_claim_by_attempt: dict[tuple[MemoryScope, str], int] = {}
        self._delivery_by_address: dict[tuple[MemoryScope, str], MemoryDeliveryV1] = {}
        self._delivery_by_result: dict[tuple[MemoryScope, str], MemoryDeliveryV1] = {}
        self._delivery_claim_by_result: dict[tuple[MemoryScope, str], int] = {}
        self._context_by_delivery: dict[tuple[MemoryScope, str], bytes] = {}
        self._ack_by_address: dict[tuple[MemoryScope, str], MemoryConsumerAckV1] = {}
        self._ack_by_delivery: dict[tuple[MemoryScope, str], MemoryConsumerAckV1] = {}
        self._ack_by_call: dict[tuple[MemoryScope, str, str], MemoryConsumerAckV1] = {}
        self._consumer_output_by_ack: dict[tuple[MemoryScope, str], object] = {}
        self._submission_claim_by_delivery: dict[
            tuple[MemoryScope, str], tuple[str, str, str, str, str, int]
        ] = {}
        self._submission_owner_by_delivery: dict[tuple[MemoryScope, str], int] = {}
        self._submission_delivery_by_call: dict[
            tuple[MemoryScope, str, str], tuple[MemoryScope, str]
        ] = {}
        self._submission_call_by_delivery: dict[
            tuple[MemoryScope, str], tuple[MemoryScope, str, str]
        ] = {}
        self._exposure_by_address: dict[tuple[MemoryScope, str], MemoryExposureV1] = {}
        self._exposure_by_attempt: dict[tuple[MemoryScope, str], MemoryExposureV1] = {}
        self._exposure_by_ack: dict[tuple[MemoryScope, str], MemoryExposureV1] = {}

    @staticmethod
    def _atomic_writes(writes: tuple[tuple[dict, tuple, object], ...]) -> None:
        """Apply in-memory index writes with best-effort BaseException rollback."""

        missing = object()
        previous: list[tuple[dict, tuple, object]] = []
        try:
            for mapping, key, value in writes:
                previous.append((mapping, key, mapping.get(key, missing)))
                mapping[key] = value
        except BaseException:
            for mapping, key, old_value in reversed(previous):
                if old_value is missing:
                    mapping.pop(key, None)
                else:
                    mapping[key] = old_value
            raise

    def _read_release(
        self,
        scope: MemoryScope,
        release_id: str,
        read_session: _SourceReadSession | None,
    ) -> MemoryRelease:
        value = self._release_store.get_release(scope, release_id)
        if read_session is not None:
            read_session.record(
                MemorySourceReadOperation.GET_RELEASE,
                (_source_ref(MemorySourceObjectKind.RELEASE, value),),
            )
        return value

    def _read_release_revisions(
        self,
        scope: MemoryScope,
        release_id: str,
        read_session: _SourceReadSession | None,
    ) -> tuple[MemoryRevision, ...]:
        values = self._release_store.get_release_revisions(scope, release_id)
        if type(values) is not tuple:
            raise MemoryQueryConflictError(
                "release revision read returned a non-tuple value"
            )
        if read_session is not None:
            read_session.record(
                MemorySourceReadOperation.GET_RELEASE_REVISIONS,
                tuple(
                    _source_ref(MemorySourceObjectKind.REVISION, item)
                    for item in values
                ),
            )
        return values

    def _read_revision(
        self,
        scope: MemoryScope,
        revision_id: str,
        read_session: _SourceReadSession | None,
    ) -> MemoryRevision:
        value = self._history_store.get_revision(scope, revision_id)
        if read_session is not None:
            read_session.record(
                MemorySourceReadOperation.GET_REVISION,
                (_source_ref(MemorySourceObjectKind.REVISION, value),),
            )
        return value

    def _read_candidate(
        self,
        scope: MemoryScope,
        candidate_id: str,
        read_session: _SourceReadSession | None,
    ) -> MemoryCandidate:
        value = self._history_store.get_candidate(scope, candidate_id)
        if read_session is not None:
            read_session.record(
                MemorySourceReadOperation.GET_CANDIDATE,
                (_source_ref(MemorySourceObjectKind.CANDIDATE, value),),
            )
        return value

    def _read_candidate_evidence(
        self,
        scope: MemoryScope,
        candidate_id: str,
        read_session: _SourceReadSession | None,
    ) -> tuple[EvidenceRecord, ...]:
        values = self._history_store.get_candidate_evidence(scope, candidate_id)
        if type(values) is not tuple:
            raise MemoryQueryConflictError(
                "candidate evidence read returned a non-tuple value"
            )
        if read_session is not None:
            read_session.record(
                MemorySourceReadOperation.GET_CANDIDATE_EVIDENCE,
                tuple(
                    _source_ref(MemorySourceObjectKind.EVIDENCE, item)
                    for item in values
                ),
            )
        return values

    @staticmethod
    def _item_read_signatures(
        item: MemoryQueryItemV1,
    ) -> tuple[
        tuple[MemorySourceReadOperation, tuple[MemorySourceObjectRefV1, ...]],
        tuple[MemorySourceReadOperation, tuple[MemorySourceObjectRefV1, ...]],
    ]:
        return (
            (
                MemorySourceReadOperation.GET_CANDIDATE,
                (
                    MemorySourceObjectRefV1(
                        kind=MemorySourceObjectKind.CANDIDATE,
                        object_id=item.candidate_id,
                        object_content_sha256=item.candidate_content_sha256,
                    ),
                ),
            ),
            (
                MemorySourceReadOperation.GET_CANDIDATE_EVIDENCE,
                tuple(
                    MemorySourceObjectRefV1(
                        kind=MemorySourceObjectKind.EVIDENCE,
                        object_id=evidence.evidence_id,
                        object_content_sha256=evidence.evidence_content_sha256,
                    )
                    for evidence in item.evidence
                ),
            ),
        )

    @classmethod
    def _validate_resolution_read_events(
        cls,
        *,
        attempt: MemoryQueryAttemptV1,
        events: tuple[MemorySourceReadEventV1, ...],
        graph_event_count: int,
        eligible_event_count: int,
        eligible_items: tuple[MemoryQueryItemV1, ...],
        returned_items: tuple[MemoryQueryItemV1, ...],
    ) -> None:
        """Fail closed if private capture missed or reordered a runtime getter."""

        signatures = tuple(
            (event.operation, event.returned_objects) for event in events
        )
        expected_release = (
            MemorySourceReadOperation.GET_RELEASE,
            (
                MemorySourceObjectRefV1(
                    kind=MemorySourceObjectKind.RELEASE,
                    object_id=attempt.spec.release_id,
                    object_content_sha256=attempt.release_content_sha256,
                ),
            ),
        )
        expected_revisions = (
            MemorySourceReadOperation.GET_RELEASE_REVISIONS,
            tuple(
                MemorySourceObjectRefV1(
                    kind=MemorySourceObjectKind.REVISION,
                    object_id=item.revision_id,
                    object_content_sha256=item.revision_content_sha256,
                )
                for item in attempt.release_revisions
            ),
        )
        eligible_signatures = tuple(
            signature
            for item in eligible_items
            for signature in cls._item_read_signatures(item)
        )
        returned_signatures = tuple(
            signature
            for item in returned_items
            for signature in cls._item_read_signatures(item)
        )
        graph_signatures = signatures[:graph_event_count]
        graph_body = graph_signatures[2:]
        if (
            len(signatures) < 2
            or signatures[:2] != (expected_release, expected_revisions)
            or signatures[graph_event_count:eligible_event_count] != eligible_signatures
            or signatures[eligible_event_count:] != returned_signatures
            or any(
                not any(
                    graph_body[index : index + 2] == cls._item_read_signatures(item)
                    for index in range(max(0, len(graph_body) - 1))
                )
                for item in eligible_items
            )
        ):
            raise MemoryQueryConflictError(
                "runtime source-read capture is incomplete or out of order"
            )

    def _validate_evidence(
        self,
        scope: MemoryScope,
        record: EvidenceRecord,
    ) -> None:
        if type(record) is not EvidenceRecord or record.event.scope != scope:
            raise MemoryQueryConflictError("history store returned non-scoped evidence")
        expected_hash = sha256(record.event.canonical_bytes()).hexdigest()
        if (
            record.content_hash != expected_hash
            or record.evidence_id != f"evd_{expected_hash[:24]}"
        ):
            raise MemoryQueryConflictError(
                "history store returned evidence with invalid commitments"
            )

    def _validate_candidate(
        self,
        scope: MemoryScope,
        candidate: MemoryCandidate,
        *,
        expected_candidate_id: str | None = None,
        read_session: _SourceReadSession | None = None,
    ) -> tuple[EvidenceRecord, ...]:
        if type(candidate) is not MemoryCandidate or candidate.proposal.scope != scope:
            raise MemoryQueryConflictError(
                "history store returned a non-scoped candidate"
            )
        expected_hash = sha256(candidate.proposal.canonical_bytes()).hexdigest()
        if (
            candidate.content_hash != expected_hash
            or candidate.candidate_id != f"cand_{expected_hash[:24]}"
            or (
                expected_candidate_id is not None
                and candidate.candidate_id != expected_candidate_id
            )
        ):
            raise MemoryQueryConflictError(
                "history store returned a candidate with invalid commitments"
            )
        evidence = self._read_candidate_evidence(
            scope,
            candidate.candidate_id,
            read_session,
        )
        if (
            type(evidence) is not tuple
            or any(type(item) is not EvidenceRecord for item in evidence)
            or tuple(item.evidence_id for item in evidence)
            != candidate.proposal.evidence_ids
        ):
            raise MemoryQueryConflictError(
                "candidate evidence does not match its immutable proposal"
            )
        for record in evidence:
            self._validate_evidence(scope, record)
        repeated_hash = sha256(candidate.proposal.canonical_bytes()).hexdigest()
        if (
            repeated_hash != expected_hash
            or candidate.content_hash != expected_hash
            or candidate.candidate_id != f"cand_{expected_hash[:24]}"
            or (
                expected_candidate_id is not None
                and candidate.candidate_id != expected_candidate_id
            )
        ):
            raise MemoryQueryConflictError(
                "candidate changed while its evidence was being validated"
            )
        return evidence

    def _validate_revision(
        self,
        scope: MemoryScope,
        revision: MemoryRevision,
        *,
        visiting: set[str],
        validated: dict[str, MemoryRevision],
        read_session: _SourceReadSession | None = None,
    ) -> None:
        path: list[tuple[MemoryRevision, str]] = []
        visited_revision_ids: list[str] = []
        try:
            current = revision
            while True:
                if (
                    type(current) is not MemoryRevision
                    or current.proposal.scope != scope
                ):
                    raise MemoryQueryConflictError(
                        "history store returned a non-scoped revision"
                    )
                existing = validated.get(current.revision_id)
                if existing is not None:
                    if existing != current:
                        raise MemoryQueryConflictError(
                            "one revision ID resolved to different immutable values"
                        )
                    break
                if current.revision_id in visiting:
                    raise MemoryQueryConflictError("revision ancestry contains a cycle")
                visiting.add(current.revision_id)
                visited_revision_ids.append(current.revision_id)

                expected_hash = sha256(current.proposal.canonical_bytes()).hexdigest()
                path.append((current, expected_hash))
                if (
                    current.content_hash != expected_hash
                    or current.revision_id != f"rev_{expected_hash[:24]}"
                ):
                    raise MemoryQueryConflictError(
                        "history store returned a revision with invalid commitments"
                    )
                candidate = self._read_candidate(
                    scope,
                    current.proposal.candidate_id,
                    read_session,
                )
                self._validate_candidate(
                    scope,
                    candidate,
                    expected_candidate_id=current.proposal.candidate_id,
                    read_session=read_session,
                )
                if current.proposal.operation is RevisionOperation.ADD:
                    if (
                        current.proposal.parent_revision_id is not None
                        or current.generation != 0
                        or current.memory_id != f"mem_{expected_hash[:24]}"
                    ):
                        raise MemoryQueryConflictError(
                            "ADD revision derived fields are invalid"
                        )
                    break

                parent_id = current.proposal.parent_revision_id
                if type(parent_id) is not str:
                    raise MemoryQueryConflictError(
                        "non-ADD revision has no exact parent"
                    )
                parent = self._read_revision(scope, parent_id, read_session)
                if parent.revision_id != parent_id:
                    raise MemoryQueryConflictError(
                        "history store returned a different parent address"
                    )
                current = parent

            for current, expected_hash in reversed(path):
                if current.proposal.operation is not RevisionOperation.ADD:
                    parent_id = current.proposal.parent_revision_id
                    parent = validated.get(parent_id)
                    if parent is None:
                        raise MemoryQueryConflictError(
                            "parent revision was not validated"
                        )
                    if (
                        current.memory_id != parent.memory_id
                        or current.generation != parent.generation + 1
                    ):
                        raise MemoryQueryConflictError(
                            "revision lineage derived fields are invalid"
                        )
                repeated_hash = sha256(current.proposal.canonical_bytes()).hexdigest()
                if (
                    repeated_hash != expected_hash
                    or current.content_hash != expected_hash
                    or current.revision_id != f"rev_{expected_hash[:24]}"
                ):
                    raise MemoryQueryConflictError(
                        "revision changed while its ancestry was being validated"
                    )
                validated[current.revision_id] = current
        except MemoryQueryConflictError:
            raise
        except Exception as error:
            raise MemoryQueryConflictError(
                "revision graph failed integrity validation"
            ) from error
        finally:
            for revision_id in visited_revision_ids:
                visiting.discard(revision_id)

    def _load_release(
        self,
        scope: MemoryScope,
        release_id: str,
        read_session: _SourceReadSession | None = None,
    ) -> tuple[MemoryRelease, tuple[MemoryRevision, ...]]:
        try:
            release = self._read_release(scope, release_id, read_session)
            revisions = self._read_release_revisions(
                scope,
                release_id,
                read_session,
            )
        except Exception as error:
            raise MemoryQueryConflictError(
                "release store failed to resolve the committed source graph"
            ) from error
        if (
            type(release) is not MemoryRelease
            or type(release.manifest) is not ReleaseManifest
            or type(revisions) is not tuple
            or any(type(item) is not MemoryRevision for item in revisions)
            or release.manifest.scope != scope
            or release.release_id != release_id
            or tuple(item.revision_id for item in revisions)
            != release.manifest.revision_ids
        ):
            raise MemoryQueryConflictError(
                "release store returned a non-canonical scoped release graph"
            )
        try:
            _digest(release.release_graph_sha256, "release_graph_sha256")
            expected_release_hash = sha256(release.commitment_bytes()).hexdigest()
        except (TypeError, ValueError) as error:
            raise MemoryQueryConflictError(
                "release store returned a release with invalid commitments"
            ) from error
        if (
            release.content_hash != expected_release_hash
            or release.release_id != f"rel_{expected_release_hash[:24]}"
        ):
            raise MemoryQueryConflictError(
                "release store returned a release with invalid commitments"
            )
        validated: dict[str, MemoryRevision] = {}
        for revision in revisions:
            self._validate_revision(
                scope,
                revision,
                visiting=set(),
                validated=validated,
                read_session=read_session,
            )
        repeated_release_hash = sha256(release.commitment_bytes()).hexdigest()
        if (
            repeated_release_hash != expected_release_hash
            or release.content_hash != expected_release_hash
            or release.release_id != release_id
            or release.release_id != f"rel_{expected_release_hash[:24]}"
            or tuple(item.revision_id for item in revisions)
            != release.manifest.revision_ids
        ):
            raise MemoryQueryConflictError(
                "release changed while its graph was being validated"
            )
        release_refs = tuple(_revision_ref(item) for item in revisions)
        if len({item.revision_id for item in release_refs}) != len(release_refs):
            raise MemoryQueryConflictError("release graph contains duplicate revisions")
        return release, revisions

    def begin_query(self, spec: MemoryQuerySpecV1) -> MemoryQueryAttemptV1:
        """Pin an explicit release before any retrieval is reported."""

        with self._lock:
            if _RUNTIME_COMPONENT_CALL_ACTIVE.get():
                raise MemoryQueryConflictError(
                    "registered runtime components cannot mutate the runtime store"
                )
        if type(spec) is not MemoryQuerySpecV1:
            raise TypeError("spec must be a MemoryQuerySpecV1")
        idempotency_address = (spec.scope, spec.idempotency_key)
        with self._lock:
            existing = self._attempt_by_idempotency.get(idempotency_address)
            if existing is not None:
                if existing.spec == spec:
                    return existing
                raise MemoryQueryConflictError(
                    "scoped query idempotency key has different content"
                )

        release, revisions = self._load_release(spec.scope, spec.release_id)
        slot_address = (
            spec.scope,
            spec.trajectory_id,
            spec.query_sequence_no,
        )
        trajectory_address = (spec.scope, spec.trajectory_id)
        group_address = (spec.scope, spec.rollout_group_id)
        trajectory_binding = (
            spec.rollout_group_id,
            release.release_id,
            release.content_hash,
            spec.task_policy_id,
            spec.task_policy_version_sha256,
        )
        group_binding = (
            release.release_id,
            release.content_hash,
            spec.task_policy_id,
            spec.task_policy_version_sha256,
        )
        with self._lock:
            claimed_slot = self._attempt_by_trajectory_slot.get(slot_address)
            if claimed_slot is not None:
                if claimed_slot.spec == spec:
                    return claimed_slot
                raise MemoryQueryConflictError(
                    "trajectory query sequence is already bound"
                )
            if (
                self._trajectory_binding.get(
                    trajectory_address,
                    trajectory_binding,
                )
                != trajectory_binding
            ):
                raise MemoryQueryConflictError(
                    "trajectory execution snapshot cannot change"
                )
            if (
                self._rollout_group_binding.get(
                    group_address,
                    group_binding,
                )
                != group_binding
            ):
                raise MemoryQueryConflictError(
                    "rollout group execution snapshot cannot change"
                )
        attempt = MemoryQueryAttemptV1.create(
            spec=spec,
            release_content_sha256=release.content_hash,
            release_revisions=tuple(_revision_ref(item) for item in revisions),
            attempt_nonce=_new_runtime_nonce(),
        )
        address = (spec.scope, attempt.attempt_id)
        with self._lock:
            existing = self._attempt_by_idempotency.get(idempotency_address)
            if existing is not None:
                if existing.spec == spec:
                    return existing
                raise MemoryQueryConflictError(
                    "scoped query idempotency key has different content"
                )
            claimed_slot = self._attempt_by_trajectory_slot.get(slot_address)
            if claimed_slot is not None:
                if claimed_slot.spec == spec:
                    return claimed_slot
                raise MemoryQueryConflictError(
                    "trajectory query sequence is already bound"
                )
            if (
                self._trajectory_binding.get(
                    trajectory_address,
                    trajectory_binding,
                )
                != trajectory_binding
            ):
                raise MemoryQueryConflictError(
                    "trajectory execution snapshot cannot change"
                )
            if (
                self._rollout_group_binding.get(
                    group_address,
                    group_binding,
                )
                != group_binding
            ):
                raise MemoryQueryConflictError(
                    "rollout group execution snapshot cannot change"
                )
            collision = self._attempt_by_address.get(address)
            if collision is not None:
                if _canonical_equal(collision, attempt):
                    self._attempt_by_idempotency[idempotency_address] = collision
                    return collision
                raise MemoryQueryConflictError(
                    f"query attempt ID collision for {attempt.attempt_id!r}"
                )
            self._atomic_writes(
                (
                    (self._attempt_by_address, address, attempt),
                    (
                        self._attempt_by_idempotency,
                        idempotency_address,
                        attempt,
                    ),
                    (self._attempt_by_trajectory_slot, slot_address, attempt),
                    (
                        self._trajectory_binding,
                        trajectory_address,
                        trajectory_binding,
                    ),
                    (self._rollout_group_binding, group_address, group_binding),
                )
            )
            return attempt

    def get_query_attempt(
        self,
        scope: MemoryScope,
        attempt_id: str,
    ) -> MemoryQueryAttemptV1:
        _scope(scope)
        attempt_id = _string(attempt_id, "attempt_id")
        with self._lock:
            value = self._attempt_by_address.get((scope, attempt_id))
            if value is None:
                raise MemoryQueryNotFoundError(
                    f"query attempt {attempt_id!r} was not found"
                )
            return value

    def _query_item(
        self,
        scope: MemoryScope,
        revision: MemoryRevision,
        release_position: int,
        read_session: _SourceReadSession,
    ) -> MemoryQueryItemV1:
        revision_id = revision.revision_id
        revision_content_sha256 = revision.content_hash
        revision_proposal_sha256 = sha256(
            revision.proposal.canonical_bytes()
        ).hexdigest()
        candidate_address = revision.proposal.candidate_id
        memory_id = revision.memory_id
        generation = revision.generation
        candidate = self._read_candidate(
            scope,
            candidate_address,
            read_session,
        )
        candidate_id = candidate.candidate_id
        candidate_content_sha256 = candidate.content_hash
        candidate_content = candidate.proposal.content
        evidence = self._validate_candidate(
            scope,
            candidate,
            expected_candidate_id=candidate_address,
            read_session=read_session,
        )
        if (
            sha256(revision.proposal.canonical_bytes()).hexdigest()
            != revision_proposal_sha256
            or revision.content_hash != revision_content_sha256
            or revision.content_hash != revision_proposal_sha256
            or revision.revision_id != revision_id
            or revision.memory_id != memory_id
            or revision.generation != generation
            or revision.proposal.candidate_id != candidate_address
            or candidate.proposal.scope != scope
            or candidate.candidate_id != candidate_id
            or candidate.content_hash != candidate_content_sha256
            or candidate.proposal.content != candidate_content
            or candidate_id != candidate_address
            or tuple(item.evidence_id for item in evidence)
            != candidate.proposal.evidence_ids
            or any(item.event.scope != scope for item in evidence)
        ):
            raise MemoryQueryConflictError(
                "history store returned a non-canonical scoped memory graph"
            )
        return MemoryQueryItemV1(
            release_position=release_position,
            revision=MemoryRevisionRefV1(
                revision_id=revision_id,
                revision_content_sha256=revision_content_sha256,
            ),
            memory_id=memory_id,
            generation=generation,
            candidate_id=candidate_id,
            candidate_content_sha256=candidate_content_sha256,
            evidence=tuple(
                MemoryEvidenceRefV1(
                    evidence_id=item.evidence_id,
                    evidence_content_sha256=item.content_hash,
                )
                for item in evidence
            ),
            content=candidate_content,
        )

    def resolve_query(
        self,
        scope: MemoryScope,
        attempt_id: str,
        *,
        query: bytes,
    ) -> MemoryQueryResultV1:
        """Single-flight one registered retrieval for a query attempt."""

        scope = _scope(scope)
        attempt_id = _string(attempt_id, "attempt_id")
        if type(query) is not bytes:
            raise TypeError("query must be bytes")
        address = (scope, attempt_id)
        owner_thread_id = get_ident()
        with self._submission_condition:
            if _RUNTIME_COMPONENT_CALL_ACTIVE.get():
                raise MemoryQueryConflictError(
                    "registered runtime components cannot mutate the runtime store"
                )
            if owner_thread_id in self._resolution_claim_by_attempt.values():
                raise MemoryQueryConflictError("nested query resolution is not allowed")
            while address in self._resolution_claim_by_attempt:
                self._submission_condition.wait()
            self._resolution_claim_by_attempt[address] = owner_thread_id
        try:
            return self._resolve_query_unclaimed(
                scope,
                attempt_id,
                query=query,
            )
        finally:
            with self._submission_condition:
                if self._resolution_claim_by_attempt.get(address) == owner_thread_id:
                    self._resolution_claim_by_attempt.pop(address, None)
                    self._submission_condition.notify_all()

    def _resolve_query_unclaimed(
        self,
        scope: MemoryScope,
        attempt_id: str,
        *,
        query: bytes,
    ) -> MemoryQueryResultV1:
        """Run the attempt's registered retriever over its pinned release."""

        scope = _scope(scope)
        attempt_id = _string(attempt_id, "attempt_id")
        if type(query) is not bytes:
            raise TypeError("query must be bytes")
        attempt = self.get_query_attempt(scope, attempt_id)
        try:
            attempt.canonical_bytes()
        except (TypeError, ValueError) as error:
            raise MemoryQueryConflictError(
                "query attempt failed integrity validation"
            ) from error
        if sha256(query).hexdigest() != attempt.spec.query_sha256:
            raise MemoryBoundaryMismatchError(
                "retrieval query does not match the frozen query commitment"
            )
        retriever_key = (
            attempt.spec.retrieval_policy_id,
            attempt.spec.retrieval_policy_version_sha256,
        )
        retriever = self._retriever_by_version.get(retriever_key)
        if retriever is None:
            raise MemoryQueryConflictError("query retrieval policy is not registered")
        with self._lock:
            existing = self._result_by_attempt.get((scope, attempt_id))
            if existing is not None:
                validated, _ = self._load_validated_query_chain(
                    scope,
                    existing.query_result_id,
                )
                return validated
        read_session = _SourceReadSession()
        release, revisions = self._load_release(
            scope,
            attempt.spec.release_id,
            read_session,
        )
        refs = tuple(_revision_ref(item) for item in revisions)
        if (
            release.content_hash != attempt.release_content_sha256
            or refs != attempt.release_revisions
        ):
            raise MemoryQueryConflictError(
                "query attempt no longer matches its pinned release graph"
            )
        graph_event_count = len(read_session.snapshot())
        eligible_items = tuple(
            self._query_item(scope, revision, position, read_session)
            for position, revision in enumerate(revisions)
        )
        eligible_event_count = len(read_session.snapshot())
        repeated_release_hash = sha256(release.commitment_bytes()).hexdigest()
        if (
            repeated_release_hash != release.content_hash
            or release.release_id != attempt.spec.release_id
            or tuple(_revision_ref(item) for item in revisions) != refs
            or tuple(item.revision_id for item in revisions)
            != release.manifest.revision_ids
        ):
            raise MemoryQueryConflictError(
                "release graph changed while query items were being resolved"
            )
        try:
            registered_id = _string(
                getattr(retriever, "retrieval_policy_id", None),
                "retrieval_policy_id",
            )
            registered_version = _digest(
                getattr(retriever, "retrieval_policy_version_sha256", None),
                "retrieval_policy_version_sha256",
            )
        except (TypeError, ValueError) as error:
            raise MemoryQueryConflictError(
                "registered retriever identity changed"
            ) from error
        if (registered_id, registered_version) != retriever_key or not callable(
            getattr(retriever, "retrieve", None)
        ):
            raise MemoryQueryConflictError("registered retriever identity changed")
        component_thread_id = get_ident()
        component_token = _RUNTIME_COMPONENT_CALL_ACTIVE.set(True)
        with self._lock:
            self._active_component_thread_ids.add(component_thread_id)
        try:
            try:
                selected = retriever.retrieve(
                    attempt=attempt,
                    query=query,
                    eligible_items=eligible_items,
                )
            except Exception as error:
                raise MemoryQueryConflictError("registered retriever failed") from error
        finally:
            _RUNTIME_COMPONENT_CALL_ACTIVE.reset(component_token)
            with self._lock:
                self._active_component_thread_ids.discard(component_thread_id)
        if type(selected) is not MemoryRetrievalOutputV1:
            raise MemoryQueryConflictError(
                "registered retriever returned a non-canonical selection"
            )
        try:
            retrieved_ids = _string_tuple(
                selected.retrieved_revision_ids,
                "retrieved_revision_ids",
            )
            returned_ids = _string_tuple(
                selected.returned_revision_ids,
                "returned_revision_ids",
            )
        except (TypeError, ValueError) as error:
            raise MemoryQueryConflictError(
                "retrieval stages must contain unique revision IDs"
            ) from error
        revision_by_id = {item.revision_id: item for item in revisions}
        try:
            retrieved = tuple(
                _revision_ref(revision_by_id[item]) for item in retrieved_ids
            )
            returned_revisions = tuple(revision_by_id[item] for item in returned_ids)
        except KeyError as error:
            raise MemoryQueryConflictError(
                "retrieval references a revision outside the pinned release"
            ) from error
        position_by_revision_id = {
            revision.revision_id: position
            for position, revision in enumerate(revisions)
        }
        returned = tuple(
            self._query_item(
                scope,
                revision,
                position_by_revision_id[revision.revision_id],
                read_session,
            )
            for revision in returned_revisions
        )
        if len(returned) > attempt.spec.max_returned_items:
            raise MemoryQueryConflictError(
                "returned revisions exceed the query's frozen item budget"
            )
        try:
            read_events = read_session.snapshot()
            self._validate_resolution_read_events(
                attempt=attempt,
                events=read_events,
                graph_event_count=graph_event_count,
                eligible_event_count=eligible_event_count,
                eligible_items=eligible_items,
                returned_items=returned,
            )
            source_read_receipt = MemorySourceReadReceiptV1.create(
                attempt=attempt,
                read_events=read_events,
            )
            result = MemoryQueryResultV1.create(
                attempt=attempt,
                source_read_receipt=source_read_receipt,
                retrieved_revisions=retrieved,
                returned_items=returned,
            )
        except (TypeError, ValueError) as error:
            raise MemoryQueryConflictError(
                "retrieval stages do not form valid ranked subsets"
            ) from error

        attempt_address = (scope, attempt_id)
        receipt_address = (scope, source_read_receipt.source_read_receipt_id)
        result_address = (scope, result.query_result_id)
        with self._lock:
            existing = self._result_by_attempt.get(attempt_address)
            if existing is not None:
                if _canonical_equal(existing, result):
                    return existing
                raise MemoryQueryConflictError(
                    "query attempt already has a different immutable result"
                )
            collision = self._result_by_address.get(result_address)
            if collision is not None and not _canonical_equal(collision, result):
                raise MemoryQueryConflictError(
                    f"query result ID collision for {result.query_result_id!r}"
                )
            receipt_collision = self._source_read_receipt_by_address.get(
                receipt_address
            )
            if receipt_collision is not None and not _canonical_equal(
                receipt_collision,
                source_read_receipt,
            ):
                raise MemoryQueryConflictError(
                    "source-read receipt ID collision for "
                    f"{source_read_receipt.source_read_receipt_id!r}"
                )
            stored = collision if collision is not None else result
            stored_receipt = (
                receipt_collision
                if receipt_collision is not None
                else source_read_receipt
            )
            self._atomic_writes(
                (
                    (
                        self._source_read_receipt_by_address,
                        receipt_address,
                        stored_receipt,
                    ),
                    (
                        self._source_read_receipt_by_attempt,
                        attempt_address,
                        stored_receipt,
                    ),
                    (self._result_by_address, result_address, stored),
                    (self._result_by_attempt, attempt_address, stored),
                )
            )
            return stored

    def get_query_result(
        self,
        scope: MemoryScope,
        query_result_id: str,
    ) -> MemoryQueryResultV1:
        _scope(scope)
        query_result_id = _string(query_result_id, "query_result_id")
        with self._lock:
            value = self._result_by_address.get((scope, query_result_id))
            if value is None:
                raise MemoryQueryNotFoundError(
                    f"query result {query_result_id!r} was not found"
                )
            return value

    def get_source_read_receipt(
        self,
        scope: MemoryScope,
        source_read_receipt_id: str,
    ) -> MemorySourceReadReceiptV1:
        """Return only a runtime-published immutable source-read receipt."""

        scope = _scope(scope)
        source_read_receipt_id = _string(
            source_read_receipt_id,
            "source_read_receipt_id",
        )
        with self._lock:
            value = self._source_read_receipt_by_address.get(
                (scope, source_read_receipt_id)
            )
            if value is None:
                raise MemoryQueryNotFoundError(
                    f"source-read receipt {source_read_receipt_id!r} was not found"
                )
            return value

    def _load_validated_query_chain(
        self,
        scope: MemoryScope,
        query_result_id: str,
    ) -> tuple[MemoryQueryResultV1, MemoryQueryAttemptV1]:
        """Reload and bind one stored result to its exact query attempt."""

        with self._lock:
            result = self._result_by_address.get((scope, query_result_id))
            attempt = (
                None
                if result is None
                else self._attempt_by_address.get((scope, result.attempt_id))
            )
            reverse = (
                None
                if attempt is None
                else self._result_by_attempt.get((scope, attempt.attempt_id))
            )
            receipt = (
                None
                if result is None
                else self._source_read_receipt_by_address.get(
                    (scope, result.source_read_receipt_id)
                )
            )
            receipt_reverse = (
                None
                if attempt is None
                else self._source_read_receipt_by_attempt.get(
                    (scope, attempt.attempt_id)
                )
            )
        if (
            result is None
            or attempt is None
            or reverse is None
            or receipt is None
            or receipt_reverse is None
        ):
            raise MemoryQueryConflictError("stored query chain is incomplete")
        try:
            result.canonical_bytes()
            attempt.canonical_bytes()
            reverse.canonical_bytes()
            receipt.canonical_bytes()
            receipt_reverse.canonical_bytes()
            expected_release_ref = MemorySourceObjectRefV1(
                kind=MemorySourceObjectKind.RELEASE,
                object_id=attempt.spec.release_id,
                object_content_sha256=attempt.release_content_sha256,
            )
            expected_revision_refs = tuple(
                MemorySourceObjectRefV1(
                    kind=MemorySourceObjectKind.REVISION,
                    object_id=item.revision_id,
                    object_content_sha256=item.revision_content_sha256,
                )
                for item in attempt.release_revisions
            )
        except (TypeError, ValueError) as error:
            raise MemoryQueryConflictError(
                "stored query chain failed integrity validation"
            ) from error
        events = receipt.read_events
        if (
            result.query_result_id != query_result_id
            or reverse.query_result_id != result.query_result_id
            or reverse.content_hash != result.content_hash
            or result.attempt_id != attempt.attempt_id
            or result.attempt_content_sha256 != attempt.content_hash
            or result.source_read_receipt_id != receipt.source_read_receipt_id
            or result.source_read_receipt_content_sha256 != receipt.content_hash
            or receipt_reverse.source_read_receipt_id != receipt.source_read_receipt_id
            or receipt_reverse.content_hash != receipt.content_hash
            or receipt.scope != scope
            or receipt.attempt_id != attempt.attempt_id
            or receipt.attempt_content_sha256 != attempt.content_hash
            or result.scope != scope
            or attempt.spec.scope != scope
            or result.release_id != attempt.spec.release_id
            or result.release_content_sha256 != attempt.release_content_sha256
            or result.trajectory_id != attempt.spec.trajectory_id
            or result.rollout_group_id != attempt.spec.rollout_group_id
            or result.eligible_revisions != attempt.release_revisions
            or len(events) < 2
            or events[0].operation is not MemorySourceReadOperation.GET_RELEASE
            or events[0].returned_objects != (expected_release_ref,)
            or events[1].operation
            is not MemorySourceReadOperation.GET_RELEASE_REVISIONS
            or events[1].returned_objects != expected_revision_refs
        ):
            raise MemoryQueryConflictError(
                "stored query result is not bound to its exact attempt"
            )
        return result, attempt

    def prepare_delivery(
        self,
        scope: MemoryScope,
        query_result_id: str,
        *,
        renderer_id: str,
        renderer_version_sha256: str,
    ) -> MemoryDeliveryV1:
        """Single-flight one registered render for a query result."""

        scope = _scope(scope)
        query_result_id = _string(query_result_id, "query_result_id")
        renderer_id = _string(renderer_id, "renderer_id")
        renderer_version_sha256 = _digest(
            renderer_version_sha256,
            "renderer_version_sha256",
        )
        address = (scope, query_result_id)
        owner_thread_id = get_ident()
        with self._submission_condition:
            if _RUNTIME_COMPONENT_CALL_ACTIVE.get():
                raise MemoryDeliveryConflictError(
                    "registered runtime components cannot mutate the runtime store"
                )
            if owner_thread_id in self._delivery_claim_by_result.values():
                raise MemoryDeliveryConflictError(
                    "nested delivery rendering is not allowed"
                )
            while address in self._delivery_claim_by_result:
                self._submission_condition.wait()
            self._delivery_claim_by_result[address] = owner_thread_id
        try:
            return self._prepare_delivery_unclaimed(
                scope,
                query_result_id,
                renderer_id=renderer_id,
                renderer_version_sha256=renderer_version_sha256,
            )
        finally:
            with self._submission_condition:
                if self._delivery_claim_by_result.get(address) == owner_thread_id:
                    self._delivery_claim_by_result.pop(address, None)
                    self._submission_condition.notify_all()

    def _prepare_delivery_unclaimed(
        self,
        scope: MemoryScope,
        query_result_id: str,
        *,
        renderer_id: str,
        renderer_version_sha256: str,
    ) -> MemoryDeliveryV1:
        """Invoke a registered renderer and freeze its exact output bytes."""

        scope = _scope(scope)
        query_result_id = _string(query_result_id, "query_result_id")
        renderer_id = _string(renderer_id, "renderer_id")
        renderer_version_sha256 = _digest(
            renderer_version_sha256,
            "renderer_version_sha256",
        )
        renderer = self._renderer_by_version.get((renderer_id, renderer_version_sha256))
        if renderer is None:
            raise MemoryDeliveryConflictError(
                "requested renderer ID and version are not registered"
            )
        result_address = (scope, query_result_id)
        with self._lock:
            existing = self._delivery_by_result.get(result_address)
            if existing is not None:
                if (
                    existing.renderer_id == renderer_id
                    and existing.renderer_version_sha256 == renderer_version_sha256
                ):
                    return existing
                raise MemoryDeliveryConflictError(
                    "query result already has a different immutable delivery"
                )
        try:
            result, attempt = self._load_validated_query_chain(
                scope,
                query_result_id,
            )
        except MemoryQueryConflictError as error:
            raise MemoryDeliveryConflictError(
                "query result chain failed integrity validation"
            ) from error
        try:
            registered_id = _string(
                getattr(renderer, "renderer_id", None),
                "renderer_id",
            )
            registered_version = _digest(
                getattr(renderer, "renderer_version_sha256", None),
                "renderer_version_sha256",
            )
        except (TypeError, ValueError) as error:
            raise MemoryDeliveryConflictError(
                "registered renderer identity changed"
            ) from error
        if (registered_id, registered_version) != (
            renderer_id,
            renderer_version_sha256,
        ) or not callable(getattr(renderer, "render", None)):
            raise MemoryDeliveryConflictError("registered renderer identity changed")
        component_thread_id = get_ident()
        component_token = _RUNTIME_COMPONENT_CALL_ACTIVE.set(True)
        with self._lock:
            self._active_component_thread_ids.add(component_thread_id)
        try:
            try:
                rendered = renderer.render(result)
            except Exception as error:
                raise MemoryDeliveryConflictError(
                    "registered renderer failed"
                ) from error
        finally:
            _RUNTIME_COMPONENT_CALL_ACTIVE.reset(component_token)
            with self._lock:
                self._active_component_thread_ids.discard(component_thread_id)
        if type(rendered) is not MemoryRenderOutputV1:
            raise MemoryDeliveryConflictError(
                "registered renderer returned a non-canonical output"
            )
        rendered_context = rendered.rendered_context
        ranges = _range_tuple(rendered.rendered_ranges)
        try:
            rendered_context.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise ValueError("rendered_context must be valid UTF-8") from error
        if len(rendered_context) > attempt.spec.max_context_utf8_bytes:
            raise MemoryDeliveryConflictError(
                "rendered context exceeds the query's frozen byte budget"
            )
        expected_ids = tuple(
            item.revision.revision_id for item in result.returned_items
        )
        range_ids = tuple(item.revision_id for item in ranges)
        if range_ids != expected_ids:
            raise MemoryDeliveryConflictError(
                "V1 renderer must cover every returned revision exactly once and in order"
            )
        if not expected_ids and rendered_context:
            raise MemoryDeliveryConflictError(
                "an empty query result must render an empty context"
            )
        if expected_ids and not rendered_context:
            raise MemoryDeliveryConflictError(
                "a non-empty query result must render a non-empty context"
            )
        returned_by_id = {
            item.revision.revision_id: item.revision for item in result.returned_items
        }
        spans: list[MemoryRenderedRevisionSpanV1] = []
        last_end = 0
        for item in ranges:
            if item.rendered_start < last_end or item.rendered_end > len(
                rendered_context
            ):
                raise MemoryDeliveryConflictError(
                    "rendered ranges must be ordered, disjoint, and in bounds"
                )
            fragment = rendered_context[item.rendered_start : item.rendered_end]
            try:
                fragment.decode("utf-8", "strict")
            except UnicodeDecodeError as error:
                raise MemoryDeliveryConflictError(
                    "rendered ranges must align to UTF-8 boundaries"
                ) from error
            spans.append(
                MemoryRenderedRevisionSpanV1(
                    revision=returned_by_id[item.revision_id],
                    rendered_start=item.rendered_start,
                    rendered_end=item.rendered_end,
                    rendered_fragment_sha256=sha256(fragment).hexdigest(),
                )
            )
            last_end = item.rendered_end

        delivery = MemoryDeliveryV1.create(
            query_result=result,
            renderer_id=renderer_id,
            renderer_version_sha256=renderer_version_sha256,
            rendered_context_sha256=sha256(rendered_context).hexdigest(),
            rendered_context_utf8_bytes=len(rendered_context),
            rendered_spans=tuple(spans),
            delivery_nonce=_new_runtime_nonce(),
        )
        address = (scope, delivery.delivery_id)
        with self._lock:
            existing = self._delivery_by_result.get(result_address)
            if existing is not None:
                existing_context = self._context_by_delivery[
                    (scope, existing.delivery_id)
                ]
                if (
                    existing.renderer_id == renderer_id
                    and existing.renderer_version_sha256 == renderer_version_sha256
                    and existing_context == rendered_context
                    and existing.rendered_spans == tuple(spans)
                ):
                    return existing
                raise MemoryDeliveryConflictError(
                    "query result already has a different immutable delivery"
                )
            collision = self._delivery_by_address.get(address)
            if collision is not None and not _canonical_equal(collision, delivery):
                raise MemoryDeliveryConflictError(
                    f"delivery ID collision for {delivery.delivery_id!r}"
                )
            stored = collision if collision is not None else delivery
            self._atomic_writes(
                (
                    (self._delivery_by_address, address, stored),
                    (self._delivery_by_result, result_address, stored),
                    (
                        self._context_by_delivery,
                        (scope, stored.delivery_id),
                        bytes(rendered_context),
                    ),
                )
            )
            return stored

    def get_delivery(
        self,
        scope: MemoryScope,
        delivery_id: str,
    ) -> MemoryDeliveryV1:
        _scope(scope)
        delivery_id = _string(delivery_id, "delivery_id")
        with self._lock:
            value = self._delivery_by_address.get((scope, delivery_id))
            if value is None:
                raise MemoryDeliveryNotFoundError(
                    f"delivery {delivery_id!r} was not found"
                )
            return value

    def _load_validated_delivery_chain(
        self,
        scope: MemoryScope,
        delivery_id: str,
        *,
        query: bytes,
    ) -> tuple[
        MemoryDeliveryV1,
        MemoryQueryResultV1,
        MemoryQueryAttemptV1,
        bytes,
    ]:
        """Validate the complete immutable chain before consumer side effects."""

        with self._lock:
            delivery = self._delivery_by_address.get((scope, delivery_id))
            context = self._context_by_delivery.get((scope, delivery_id))
            result = (
                None
                if delivery is None
                else self._result_by_address.get((scope, delivery.query_result_id))
            )
            attempt = (
                None
                if result is None
                else self._attempt_by_address.get((scope, result.attempt_id))
            )
            reverse_delivery = (
                None
                if result is None
                else self._delivery_by_result.get((scope, result.query_result_id))
            )
            reverse_result = (
                None
                if attempt is None
                else self._result_by_attempt.get((scope, attempt.attempt_id))
            )
        if (
            delivery is None
            or result is None
            or attempt is None
            or reverse_delivery is None
            or reverse_result is None
            or type(context) is not bytes
        ):
            raise MemoryBoundaryMismatchError("consumer delivery chain is incomplete")
        try:
            delivery.canonical_bytes()
            result.canonical_bytes()
            attempt.canonical_bytes()
            reverse_delivery.canonical_bytes()
            reverse_result.canonical_bytes()
        except (TypeError, ValueError) as error:
            raise MemoryBoundaryMismatchError(
                "consumer delivery chain failed integrity validation"
            ) from error
        if (
            delivery.delivery_id != delivery_id
            or reverse_delivery.delivery_id != delivery.delivery_id
            or reverse_delivery.content_hash != delivery.content_hash
            or delivery.query_result_id != result.query_result_id
            or delivery.query_result_content_sha256 != result.content_hash
            or reverse_result.query_result_id != result.query_result_id
            or reverse_result.content_hash != result.content_hash
            or result.attempt_id != attempt.attempt_id
            or result.attempt_content_sha256 != attempt.content_hash
            or delivery.scope != scope
            or result.scope != scope
            or attempt.spec.scope != scope
            or delivery.release_id != result.release_id
            or result.release_id != attempt.spec.release_id
            or delivery.release_content_sha256 != result.release_content_sha256
            or result.release_content_sha256 != attempt.release_content_sha256
            or delivery.trajectory_id != result.trajectory_id
            or result.trajectory_id != attempt.spec.trajectory_id
            or result.rollout_group_id != attempt.spec.rollout_group_id
            or result.eligible_revisions != attempt.release_revisions
        ):
            raise MemoryBoundaryMismatchError(
                "consumer delivery chain has mismatched addresses"
            )
        if (
            len(context) != delivery.rendered_context_utf8_bytes
            or sha256(context).hexdigest() != delivery.rendered_context_sha256
            or tuple(item.revision for item in delivery.rendered_spans)
            != result.returned_revisions
            or any(
                item.rendered_end > len(context)
                or sha256(context[item.rendered_start : item.rendered_end]).hexdigest()
                != item.rendered_fragment_sha256
                for item in delivery.rendered_spans
            )
        ):
            raise MemoryBoundaryMismatchError(
                "consumer rendered context failed integrity validation"
            )
        if sha256(query).hexdigest() != attempt.spec.query_sha256:
            raise MemoryBoundaryMismatchError(
                "consumer query does not match the frozen query commitment"
            )
        return delivery, result, attempt, context

    def _clear_submission_claim(
        self,
        delivery_address: tuple[MemoryScope, str],
        *,
        owner_thread_id: int,
    ) -> None:
        with self._submission_condition:
            if (
                self._submission_owner_by_delivery.get(delivery_address)
                != owner_thread_id
            ):
                return
            self._submission_claim_by_delivery.pop(delivery_address, None)
            self._submission_owner_by_delivery.pop(delivery_address, None)
            call_address = self._submission_call_by_delivery.pop(
                delivery_address,
                None,
            )
            if call_address is not None:
                self._submission_delivery_by_call.pop(call_address, None)
            self._submission_condition.notify_all()

    def _commit_consumer_exposure(
        self,
        *,
        scope: MemoryScope,
        delivery_address: tuple[MemoryScope, str],
        call_address: tuple[MemoryScope, str, str],
        ack: MemoryConsumerAckV1,
        exposure: MemoryExposureV1,
        output: object,
    ) -> tuple[MemoryExposureV1, object]:
        """Publish all local ack/exposure indexes or roll every write back."""

        ack_address = (scope, ack.consumer_ack_id)
        attempt_address = (scope, exposure.attempt_id)
        exposure_address = (scope, exposure.exposure_id)
        with self._submission_condition:
            existing = self._ack_by_delivery.get(delivery_address)
            if existing is not None:
                if not _canonical_equal(existing, ack):
                    raise MemoryConsumerAckConflictError(
                        "delivery already has a different consumer acknowledgement"
                    )
                existing_exposure = self._exposure_by_ack.get(
                    (scope, existing.consumer_ack_id)
                )
                output_address = (scope, existing.consumer_ack_id)
                if (
                    existing_exposure is None
                    or output_address not in self._consumer_output_by_ack
                ):
                    raise MemoryExposureConflictError(
                        "acknowledgement has no atomic exposure"
                    )
                return (
                    existing_exposure,
                    self._consumer_output_by_ack[output_address],
                )
            if self._ack_by_call.get(call_address) is not None:
                raise MemoryConsumerAckConflictError(
                    "consumer call ID is already bound to another acknowledgement"
                )
            collision = self._ack_by_address.get(ack_address)
            if collision is not None and not _canonical_equal(collision, ack):
                raise MemoryConsumerAckConflictError(
                    f"consumer acknowledgement ID collision for {ack.consumer_ack_id!r}"
                )
            existing_exposure = self._exposure_by_attempt.get(attempt_address)
            if existing_exposure is not None and not _canonical_equal(
                existing_exposure,
                exposure,
            ):
                raise MemoryExposureConflictError(
                    "query attempt already has a different exposure"
                )
            exposure_collision = self._exposure_by_address.get(exposure_address)
            if exposure_collision is not None and not _canonical_equal(
                exposure_collision,
                exposure,
            ):
                raise MemoryExposureConflictError(
                    f"exposure ID collision for {exposure.exposure_id!r}"
                )
            stored = collision if collision is not None else ack
            stored_exposure = (
                exposure_collision if exposure_collision is not None else exposure
            )
            output_address = (scope, stored.consumer_ack_id)
            missing = object()
            previous: list[tuple[dict, tuple, object]] = []
            writes = (
                (self._ack_by_address, ack_address, stored),
                (self._ack_by_delivery, delivery_address, stored),
                (self._ack_by_call, call_address, stored),
                (self._consumer_output_by_ack, output_address, output),
                (
                    self._exposure_by_address,
                    exposure_address,
                    stored_exposure,
                ),
                (self._exposure_by_attempt, attempt_address, stored_exposure),
                (self._exposure_by_ack, ack_address, stored_exposure),
            )
            try:
                for mapping, key, value in writes:
                    previous.append((mapping, key, mapping.get(key, missing)))
                    mapping[key] = value
            except BaseException:
                for mapping, key, old_value in reversed(previous):
                    if old_value is missing:
                        mapping.pop(key, None)
                    else:
                        mapping[key] = old_value
                raise
            return stored_exposure, output

    def submit_delivery(
        self,
        scope: MemoryScope,
        delivery_id: str,
        *,
        consumer_id: str,
        consumer_version_sha256: str,
        call_id: str,
        query: bytes,
        history: tuple[bytes, ...],
    ) -> tuple[MemoryExposureV1, object]:
        """Invoke a consumer and atomically persist its ack plus exposure."""

        with self._lock:
            if _RUNTIME_COMPONENT_CALL_ACTIVE.get():
                raise MemoryConsumerAckConflictError(
                    "registered runtime components cannot mutate the runtime store"
                )
        scope = _scope(scope)
        delivery_id = _string(delivery_id, "delivery_id")
        consumer_id = _string(consumer_id, "consumer_id")
        consumer_version_sha256 = _digest(
            consumer_version_sha256,
            "consumer_version_sha256",
        )
        call_id = _string(call_id, "call_id")
        if type(query) is not bytes:
            raise TypeError("query must be bytes")
        if type(history) is not tuple or any(
            type(item) is not bytes for item in tuple.__iter__(history)
        ):
            raise TypeError("history must be a tuple of bytes")
        registered_consumer = self._consumer_by_version.get(
            (consumer_id, consumer_version_sha256)
        )
        if registered_consumer is None:
            raise MemoryConsumerAckConflictError(
                "requested consumer ID and version are not registered"
            )
        consumer, registered_consumer_kind = registered_consumer
        try:
            registered_id = _string(
                getattr(consumer, "consumer_id", None),
                "consumer_id",
            )
            registered_version = _digest(
                getattr(consumer, "consumer_version_sha256", None),
                "consumer_version_sha256",
            )
            consumer_kind = getattr(consumer, "consumer_kind", None)
        except (TypeError, ValueError) as error:
            raise MemoryConsumerAckConflictError(
                "registered consumer identity changed"
            ) from error
        if (
            (registered_id, registered_version)
            != (consumer_id, consumer_version_sha256)
            or type(consumer_kind) is not MemoryConsumerKind
            or consumer_kind is not registered_consumer_kind
            or not callable(getattr(consumer, "submit", None))
        ):
            raise MemoryConsumerAckConflictError("registered consumer identity changed")
        delivery, result, attempt, expected_context = (
            self._load_validated_delivery_chain(
                scope,
                delivery_id,
                query=query,
            )
        )
        claim = (
            consumer_id,
            consumer_version_sha256,
            call_id,
            sha256(query).hexdigest(),
            _history_sha256(history),
            len(history),
        )
        delivery_address = (scope, delivery_id)
        call_address = (scope, delivery.trajectory_id, call_id)
        owner_thread_id = get_ident()
        with self._submission_condition:
            if owner_thread_id in self._submission_owner_by_delivery.values():
                raise MemoryConsumerAckConflictError(
                    "nested consumer submission is not allowed"
                )
            while delivery_address in self._submission_claim_by_delivery:
                if self._submission_claim_by_delivery[delivery_address] != claim:
                    raise MemoryConsumerAckConflictError(
                        "delivery has a conflicting in-flight consumer call"
                    )
                if (
                    self._submission_owner_by_delivery.get(delivery_address)
                    == owner_thread_id
                ):
                    raise MemoryConsumerAckConflictError(
                        "reentrant consumer submission is not allowed"
                    )
                self._submission_condition.wait()
            existing = self._ack_by_delivery.get(delivery_address)
            if existing is not None:
                exposure = self._exposure_by_ack.get((scope, existing.consumer_ack_id))
                try:
                    existing.canonical_bytes()
                    if exposure is None:
                        raise ValueError("missing exposure")
                    exposure.canonical_bytes()
                except (TypeError, ValueError) as error:
                    raise MemoryExposureConflictError(
                        "stored acknowledgement chain failed integrity validation"
                    ) from error
                if (
                    existing.consumer_id == consumer_id
                    and existing.consumer_version_sha256 == consumer_version_sha256
                    and existing.call_id == call_id
                    and existing.observed_query_sha256 == sha256(query).hexdigest()
                    and existing.observed_history_sha256 == _history_sha256(history)
                    and existing.observed_history_length == len(history)
                    and existing.delivery_id == delivery.delivery_id
                    and existing.delivery_content_sha256 == delivery.content_hash
                    and exposure.consumer_ack_id == existing.consumer_ack_id
                    and exposure.consumer_ack_content_sha256 == existing.content_hash
                    and exposure.delivery_id == delivery.delivery_id
                    and exposure.delivery_content_sha256 == delivery.content_hash
                    and exposure.attempt_id == attempt.attempt_id
                    and exposure.attempt_content_sha256 == attempt.content_hash
                ):
                    output_address = (scope, existing.consumer_ack_id)
                    if output_address not in self._consumer_output_by_ack:
                        raise MemoryExposureConflictError(
                            "acknowledgement has no stored consumer output"
                        )
                    return (
                        exposure,
                        self._consumer_output_by_ack[output_address],
                    )
                raise MemoryConsumerAckConflictError(
                    "delivery already has a different consumer acknowledgement"
                )
            existing_call = self._ack_by_call.get(call_address)
            if existing_call is not None:
                raise MemoryConsumerAckConflictError(
                    "consumer call ID is already bound to another acknowledgement"
                )
            claimed_delivery = self._submission_delivery_by_call.get(call_address)
            if claimed_delivery is not None and claimed_delivery != delivery_address:
                raise MemoryConsumerAckConflictError(
                    "consumer call ID has a conflicting in-flight delivery"
                )
            self._submission_claim_by_delivery[delivery_address] = claim
            self._submission_owner_by_delivery[delivery_address] = owner_thread_id
            self._submission_delivery_by_call[call_address] = delivery_address
            self._submission_call_by_delivery[delivery_address] = call_address

        component_thread_id = get_ident()
        component_token = _RUNTIME_COMPONENT_CALL_ACTIVE.set(True)
        with self._lock:
            self._active_component_thread_ids.add(component_thread_id)
        try:
            submitted = consumer.submit(
                delivery=delivery,
                rendered_context=expected_context,
                query=query,
                history=history,
                call_id=call_id,
            )
            if type(submitted) is not MemoryConsumerCallV1:
                raise MemoryBoundaryMismatchError(
                    "registered consumer returned a non-canonical call result"
                )
            if (
                submitted.delivery_id != delivery.delivery_id
                or submitted.delivery_content_sha256 != delivery.content_hash
                or submitted.call_id != call_id
                or submitted.observed_query_sha256 != sha256(query).hexdigest()
                or submitted.observed_history_sha256 != _history_sha256(history)
                or submitted.observed_history_length != len(history)
            ):
                raise MemoryBoundaryMismatchError(
                    "consumer receipt does not bind the exact delivery request"
                )
            submitted_prompt = submitted.submitted_prompt
            context_start = submitted.context_start
            context_end = submitted.context_end
            input_token_ids = submitted.input_token_ids
            actual_context = submitted_prompt[context_start:context_end]
            if actual_context != expected_context:
                raise MemoryBoundaryMismatchError(
                    "consumer call did not contain the exact rendered context slice"
                )
        except BaseException:
            self._clear_submission_claim(
                delivery_address,
                owner_thread_id=owner_thread_id,
            )
            raise
        finally:
            _RUNTIME_COMPONENT_CALL_ACTIVE.reset(component_token)
            with self._lock:
                self._active_component_thread_ids.discard(component_thread_id)
        try:
            token_hash: str | None
            token_count: int | None
            if consumer_kind is MemoryConsumerKind.CONTEXT:
                if input_token_ids is not None:
                    raise ValueError(
                        "context consumers must not report input token IDs"
                    )
                token_hash = None
                token_count = None
            else:
                if type(input_token_ids) is not tuple:
                    raise TypeError("model-call input_token_ids must be a tuple")
                tokens = tuple(tuple.__iter__(input_token_ids))
                if not tokens or any(
                    type(item) is not int or item < 0 for item in tokens
                ):
                    raise ValueError(
                        "model-call input_token_ids must be non-empty non-negative ints"
                    )
                token_hash = sha256(
                    json.dumps(list(tokens), separators=(",", ":")).encode("ascii")
                ).hexdigest()
                token_count = len(tokens)
            ack = MemoryConsumerAckV1.create(
                delivery=delivery,
                consumer_kind=consumer_kind,
                consumer_id=consumer_id,
                consumer_version_sha256=consumer_version_sha256,
                call_id=call_id,
                submitted_prompt_sha256=sha256(submitted_prompt).hexdigest(),
                submitted_prompt_context_start=context_start,
                submitted_prompt_context_end=context_end,
                submitted_prompt_context_sha256=sha256(actual_context).hexdigest(),
                submitted_prompt_context_utf8_bytes=len(actual_context),
                observed_query_sha256=submitted.observed_query_sha256,
                observed_history_sha256=submitted.observed_history_sha256,
                observed_history_length=submitted.observed_history_length,
                submitted_input_token_ids_sha256=token_hash,
                submitted_input_token_count=token_count,
            )
            exposure = MemoryExposureV1.create(
                attempt=attempt,
                query_result=result,
                delivery=delivery,
                consumer_ack=ack,
            )
        except BaseException:
            self._clear_submission_claim(
                delivery_address,
                owner_thread_id=owner_thread_id,
            )
            raise
        try:
            return self._commit_consumer_exposure(
                scope=scope,
                delivery_address=delivery_address,
                call_address=call_address,
                ack=ack,
                exposure=exposure,
                output=submitted.output,
            )
        finally:
            self._clear_submission_claim(
                delivery_address,
                owner_thread_id=owner_thread_id,
            )

    def get_consumer_ack(
        self,
        scope: MemoryScope,
        consumer_ack_id: str,
    ) -> MemoryConsumerAckV1:
        _scope(scope)
        consumer_ack_id = _string(consumer_ack_id, "consumer_ack_id")
        with self._lock:
            value = self._ack_by_address.get((scope, consumer_ack_id))
            if value is None:
                raise MemoryConsumerAckNotFoundError(
                    f"consumer acknowledgement {consumer_ack_id!r} was not found"
                )
            return value

    def get_exposure(
        self,
        scope: MemoryScope,
        exposure_id: str,
    ) -> MemoryExposureV1:
        _scope(scope)
        exposure_id = _string(exposure_id, "exposure_id")
        with self._lock:
            value = self._exposure_by_address.get((scope, exposure_id))
            if value is None:
                raise MemoryExposureNotFoundError(
                    f"exposure {exposure_id!r} was not found"
                )
            return value

    def list_exposures(self, scope: MemoryScope) -> tuple[MemoryExposureV1, ...]:
        """Return a stable scoped exposure snapshot."""

        _scope(scope)
        with self._lock:
            return tuple(
                sorted(
                    (
                        value
                        for (record_scope, _record_id), value in (
                            self._exposure_by_address.items()
                        )
                        if record_scope == scope
                    ),
                    key=lambda item: item.exposure_id,
                )
            )
