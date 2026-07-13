# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import pytest

from areal.v2.memory_service import (
    CandidateProposal,
    EvidenceEvent,
    EvidenceKind,
    InMemoryEvidenceStore,
    InMemoryMemoryHistoryStore,
    InMemoryMemoryReleaseStore,
    InMemoryMemoryRuntimeStore,
    MemoryBoundaryMismatchError,
    MemoryConsumerAckConflictError,
    MemoryConsumerCallV1,
    MemoryConsumerKind,
    MemoryDeliveryConflictError,
    MemoryExposureStatus,
    MemoryQueryConflictError,
    MemoryQueryNotFoundError,
    MemoryQuerySpecV1,
    MemoryRenderedRevisionRangeV1,
    MemoryRenderOutputV1,
    MemoryRetrievalOutputV1,
    MemoryScope,
    ReleaseManifest,
    RevisionOperation,
    RevisionProposal,
    runtime_store,
)

_BASE = datetime(2026, 7, 12, tzinfo=UTC)
_HASH_A = hashlib.sha256(b"task-policy-v1").hexdigest()
_HASH_B = hashlib.sha256(b"retrieval-policy-v1").hexdigest()
_HASH_C = hashlib.sha256(b"renderer-v1").hexdigest()
_HASH_D = hashlib.sha256(b"consumer-v1").hexdigest()
_GOLDEN_RUNTIME_HASHES = (
    "b97ab95704e8d7c2af7a2d3b429ea1c7557d70c792aece1410e60dba3ec10d1e",
    "b6c63a6ead1cb751414b301e0c4879549fed3225d1775a0113c702c957458cce",
    "03b099ff17689ae5b3742893cead3cfb9ffa960458f38f2bc51eb99fdda71357",
    "c0fa033b21f0f9bf1b730ab676a1b0e806a509118c1901ec9dab3380fe253238",
    "10b0d471914af9d901615ded59402696fada7b7e1680490b83629469fd347b11",
)


class _ReleaseOrderRetriever:
    retrieval_policy_id = "release-order-v1"
    retrieval_policy_version_sha256 = _HASH_B

    def __init__(self, *, returned: bool = True, reverse: bool = False) -> None:
        self.returned = returned
        self.reverse = reverse
        self.calls = 0

    def retrieve(self, *, attempt, query, eligible_items):
        del attempt, query
        self.calls += 1
        ids = tuple(item.revision.revision_id for item in eligible_items)
        if self.reverse:
            ids = tuple(reversed(ids))
        return MemoryRetrievalOutputV1(
            retrieved_revision_ids=ids if self.returned else (),
            returned_revision_ids=ids if self.returned else (),
        )


class _FixedRetriever(_ReleaseOrderRetriever):
    def __init__(self, retrieved, returned) -> None:
        super().__init__()
        self.retrieved = retrieved
        self.returned = returned

    def retrieve(self, *, attempt, query, eligible_items):
        del attempt, query
        self.calls += 1
        ids = tuple(item.revision.revision_id for item in eligible_items)

        def select(indexes):
            return tuple(
                ids[index] if index < len(ids) else "rev_foreign" for index in indexes
            )

        return MemoryRetrievalOutputV1(
            retrieved_revision_ids=select(self.retrieved),
            returned_revision_ids=select(self.returned),
        )


class _LineRenderer:
    renderer_id = "json-lines-v1"
    renderer_version_sha256 = _HASH_C

    def __init__(self) -> None:
        self.calls = 0

    def render(self, query_result):
        self.calls += 1
        context = bytearray()
        ranges = []
        for item in query_result.returned_items:
            start = len(context)
            context.extend(item.content.encode())
            context.extend(b"\n")
            ranges.append(
                MemoryRenderedRevisionRangeV1(
                    revision_id=item.revision.revision_id,
                    rendered_start=start,
                    rendered_end=len(context),
                )
            )
        return MemoryRenderOutputV1(bytes(context), tuple(ranges))


class _ContextConsumer:
    consumer_kind = MemoryConsumerKind.CONTEXT
    consumer_id = "test-context-boundary"
    consumer_version_sha256 = _HASH_D

    def __init__(self) -> None:
        self.calls = 0

    def submit(
        self,
        *,
        delivery,
        rendered_context,
        query,
        history,
        call_id,
    ):
        self.calls += 1
        prefix = b"system\n"
        prompt = prefix + rendered_context + b"query\n" + query
        return MemoryConsumerCallV1(
            delivery_id=delivery.delivery_id,
            delivery_content_sha256=delivery.content_hash,
            call_id=call_id,
            submitted_prompt=prompt,
            context_start=len(prefix),
            context_end=len(prefix) + len(rendered_context),
            observed_query_sha256=hashlib.sha256(query).hexdigest(),
            observed_history_sha256=runtime_store._history_sha256(history),
            observed_history_length=len(history),
            input_token_ids=None,
            output="context-consumer-output",
        )


class _ModelConsumer(_ContextConsumer):
    consumer_kind = MemoryConsumerKind.MODEL_CALL
    consumer_id = "token-submit-boundary"

    def submit(self, **kwargs):
        call = super().submit(**kwargs)
        return replace(call, input_token_ids=(101, 202, 303), output="model-output")


class _BadRenderer(_LineRenderer):
    renderer_id = "bad-renderer"

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def render(self, query_result):
        rendered = super().render(query_result)
        if self.mode == "drop":
            return replace(rendered, rendered_ranges=rendered.rendered_ranges[:-1])
        if self.mode == "reverse":
            return replace(
                rendered,
                rendered_ranges=tuple(reversed(rendered.rendered_ranges)),
            )
        if self.mode == "utf8":
            context = "é\nx\n".encode()
            return MemoryRenderOutputV1(
                rendered_context=context,
                rendered_ranges=(
                    replace(
                        rendered.rendered_ranges[0],
                        rendered_start=0,
                        rendered_end=1,
                    ),
                    replace(
                        rendered.rendered_ranges[1],
                        rendered_start=3,
                        rendered_end=len(context),
                    ),
                ),
            )
        raise AssertionError("unknown bad renderer mode")


class _WrongContextConsumer(_ContextConsumer):
    consumer_id = "wrong-context-boundary"

    def submit(self, **kwargs):
        call = super().submit(**kwargs)
        prompt = bytearray(call.submitted_prompt)
        prompt[call.context_start] ^= 1
        return replace(call, submitted_prompt=bytes(prompt))


class _PreSubmitFailOnceConsumer(_ContextConsumer):
    consumer_id = "fail-once-boundary"

    def submit(self, **kwargs):
        if self.calls == 0:
            self.calls += 1
            raise RuntimeError("injected consumer failure")
        return super().submit(**kwargs)


class _DurablyIdempotentConsumer(_ContextConsumer):
    consumer_id = "durably-idempotent-boundary"

    def __init__(self) -> None:
        super().__init__()
        self.side_effects = 0
        self.receipts = {}
        self.fingerprints = {}
        self.lose_first_response = True

    def submit(self, *, delivery, rendered_context, query, history, call_id):
        key = (
            delivery.scope,
            delivery.trajectory_id,
            call_id,
        )
        fingerprint = (
            delivery.delivery_id,
            delivery.content_hash,
            hashlib.sha256(rendered_context).hexdigest(),
            hashlib.sha256(query).hexdigest(),
            runtime_store._history_sha256(history),
        )
        self.calls += 1
        if key in self.fingerprints and self.fingerprints[key] != fingerprint:
            raise RuntimeError("idempotency key has a different request")
        if key not in self.receipts:
            self.side_effects += 1
            self.fingerprints[key] = fingerprint
            prefix = b"system\n"
            self.receipts[key] = MemoryConsumerCallV1(
                delivery_id=delivery.delivery_id,
                delivery_content_sha256=delivery.content_hash,
                call_id=call_id,
                submitted_prompt=(prefix + rendered_context + b"query\n" + query),
                context_start=len(prefix),
                context_end=len(prefix) + len(rendered_context),
                observed_query_sha256=hashlib.sha256(query).hexdigest(),
                observed_history_sha256=runtime_store._history_sha256(history),
                observed_history_length=len(history),
                input_token_ids=None,
                output="durably-cached-output",
            )
        if self.lose_first_response:
            self.lose_first_response = False
            raise RuntimeError("response lost after external side effect")
        return self.receipts[key]


class _ReentrantConsumer(_ContextConsumer):
    consumer_id = "reentrant-boundary"

    def __init__(self) -> None:
        super().__init__()
        self.store = None
        self.inner_error = None

    def submit(self, *, delivery, rendered_context, query, history, call_id):
        try:
            self.store.submit_delivery(
                delivery.scope,
                delivery.delivery_id,
                consumer_id=self.consumer_id,
                consumer_version_sha256=self.consumer_version_sha256,
                call_id=call_id,
                query=query,
                history=history,
            )
        except MemoryConsumerAckConflictError as error:
            self.inner_error = error
        return super().submit(
            delivery=delivery,
            rendered_context=rendered_context,
            query=query,
            history=history,
            call_id=call_id,
        )


class _UnsafeCrossScopeCacheConsumer(_ContextConsumer):
    consumer_id = "unsafe-cross-scope-cache"

    def __init__(self) -> None:
        super().__init__()
        self.cached_call = None

    def submit(self, *, delivery, rendered_context, query, history, call_id):
        self.calls += 1
        if self.cached_call is None:
            prefix = b"system\n"
            self.cached_call = MemoryConsumerCallV1(
                delivery_id=delivery.delivery_id,
                delivery_content_sha256=delivery.content_hash,
                call_id=call_id,
                submitted_prompt=(prefix + rendered_context + b"query\n" + query),
                context_start=len(prefix),
                context_end=len(prefix) + len(rendered_context),
                observed_query_sha256=hashlib.sha256(query).hexdigest(),
                observed_history_sha256=runtime_store._history_sha256(history),
                observed_history_length=len(history),
                input_token_ids=None,
                output=history[0].decode(),
            )
        return self.cached_call


class _FailFirstWriteDict(dict):
    def __init__(self, values):
        super().__init__(values)
        self.failed = False

    def __setitem__(self, key, value):
        if not self.failed:
            self.failed = True
            raise KeyboardInterrupt("injected local commit interruption")
        return super().__setitem__(key, value)


def _graph(
    *,
    empty: bool = False,
    subject_id: str = "subject-1",
    retriever=None,
    renderer=None,
    consumers=None,
):
    evidence_store = InMemoryEvidenceStore()
    history_store = InMemoryMemoryHistoryStore(evidence_store)
    release_store = InMemoryMemoryReleaseStore(history_store)
    scope = MemoryScope(
        tenant_id="tenant-1",
        namespace="agent-long-term-memory",
        subject_id=subject_id,
    )
    revision_ids: list[str] = []
    for index, content in enumerate(
        () if empty else ("timezone=Asia/Shanghai", "language=zh-CN")
    ):
        evidence = evidence_store.append(
            EvidenceEvent(
                scope=scope,
                session_id="capture-session",
                run_id="capture-run",
                sequence_no=index,
                kind=EvidenceKind.USER_MESSAGE,
                payload=content,
                observed_at=_BASE + timedelta(seconds=index),
                idempotency_key=f"evidence-{index}",
            )
        )
        candidate = history_store.append_candidate(
            CandidateProposal(
                scope=scope,
                content=content,
                evidence_ids=(evidence.evidence_id,),
                idempotency_key=f"candidate-{index}",
            )
        )
        revision = history_store.append_revision(
            RevisionProposal(
                scope=scope,
                candidate_id=candidate.candidate_id,
                operation=RevisionOperation.ADD,
                parent_revision_id=None,
                idempotency_key=f"revision-{index}",
            )
        )
        revision_ids.append(revision.revision_id)
    release = release_store.append_release(
        ReleaseManifest(scope=scope, revision_ids=tuple(revision_ids)),
        idempotency_key="release",
    )
    if renderer is None:
        renderer = _LineRenderer()
    if retriever is None:
        retriever = _ReleaseOrderRetriever()
    if consumers is None:
        consumers = (_ContextConsumer(), _ModelConsumer())
    return (
        scope,
        history_store,
        release_store,
        release,
        InMemoryMemoryRuntimeStore(
            history_store,
            release_store,
            retrievers=(retriever,),
            renderers=(renderer,),
            consumers=consumers,
        ),
    )


def _spec(
    scope: MemoryScope,
    release_id: str,
    *,
    suffix: str = "a",
    max_returned_items: int = 2,
    max_context_utf8_bytes: int = 4096,
) -> MemoryQuerySpecV1:
    return MemoryQuerySpecV1(
        scope=scope,
        release_id=release_id,
        trajectory_id=f"trajectory-{suffix}",
        rollout_group_id="rollout-group-1",
        query_sequence_no=0,
        query_sha256=hashlib.sha256(f"query-{suffix}".encode()).hexdigest(),
        task_policy_id="frozen-task-agent",
        task_policy_version_sha256=_HASH_A,
        retrieval_policy_id="release-order-v1",
        retrieval_policy_version_sha256=_HASH_B,
        max_returned_items=max_returned_items,
        max_context_utf8_bytes=max_context_utf8_bytes,
        idempotency_key=f"query-{suffix}",
    )


def _query_all(store, scope, release, spec):
    attempt = store.begin_query(spec)
    del release
    result = store.resolve_query(
        scope,
        attempt.attempt_id,
        query=spec.idempotency_key.encode(),
    )
    return attempt, result


def _deliver(store, scope, result):
    delivery = store.prepare_delivery(
        scope,
        result.query_result_id,
        renderer_id="json-lines-v1",
        renderer_version_sha256=_HASH_C,
    )
    context = b"".join(item.content.encode() + b"\n" for item in result.returned_items)
    return context, delivery


def _ack_context(
    store,
    scope,
    delivery,
    context,
    *,
    call_id="call-1",
    observed_query=b"query-a",
):
    del context
    exposure, _output = store.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id="test-context-boundary",
        consumer_version_sha256=_HASH_D,
        call_id=call_id,
        query=observed_query,
        history=(),
    )
    ack = store.get_consumer_ack(scope, exposure.consumer_ack_id)
    return exposure, ack


def test_query_to_actual_exposure_derives_all_runtime_stages(monkeypatch) -> None:
    nonces = iter(("01" * 32, "02" * 32))
    monkeypatch.setattr(runtime_store, "_new_runtime_nonce", lambda: next(nonces))
    scope, _history, _releases, release, store = _graph()
    attempt, result = _query_all(
        store, scope, release, _spec(scope, release.release_id)
    )
    context, delivery = _deliver(store, scope, result)
    exposure, ack = _ack_context(store, scope, delivery, context)

    assert attempt.release_content_sha256 == release.content_hash
    assert tuple(item.revision_id for item in attempt.release_revisions) == (
        release.manifest.revision_ids
    )
    assert result.eligible_revisions == attempt.release_revisions
    assert result.retrieved_revisions == attempt.release_revisions
    assert result.returned_revisions == attempt.release_revisions
    assert tuple(item.revision for item in delivery.rendered_spans) == (
        result.returned_revisions
    )
    assert ack.submitted_prompt_context_sha256 == hashlib.sha256(context).hexdigest()
    assert exposure.injected_revisions == result.returned_revisions
    assert exposure.status is MemoryExposureStatus.DELIVERED
    assert exposure.release_content_sha256 == release.content_hash
    assert all(item.evidence for item in result.returned_items)
    assert store.get_exposure(scope, exposure.exposure_id) == exposure
    assert store.list_exposures(scope) == (exposure,)
    replay, replay_output = store.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id="test-context-boundary",
        consumer_version_sha256=_HASH_D,
        call_id="call-1",
        query=b"query-a",
        history=(),
    )
    assert replay == exposure
    assert replay_output == "context-consumer-output"
    assert (
        tuple(
            record.content_hash for record in (attempt, result, delivery, ack, exposure)
        )
        == _GOLDEN_RUNTIME_HASHES
    )

    for record, prefix, id_field in (
        (attempt, "mqat_", "attempt_id"),
        (result, "mqres_", "query_result_id"),
        (delivery, "mdel_", "delivery_id"),
        (ack, "mack_", "consumer_ack_id"),
        (exposure, "mexp_", "exposure_id"),
    ):
        assert hashlib.sha256(record.canonical_bytes()).hexdigest() == (
            record.content_hash
        )
        public_id = getattr(record, id_field)
        assert public_id == prefix + record.content_hash[:24]


def test_model_call_ack_hashes_actual_prompt_slice_and_token_ids() -> None:
    scope, _history, _releases, release, store = _graph()
    attempt, result = _query_all(
        store, scope, release, _spec(scope, release.release_id)
    )
    context, delivery = _deliver(store, scope, result)
    tokens = (101, 202, 303)
    exposure, output = store.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id="token-submit-boundary",
        consumer_version_sha256=_HASH_D,
        call_id="model-call-1",
        query=b"query-a",
        history=(),
    )
    prompt = b"system\n" + context + b"query\nquery-a"
    expected_token_hash = hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":")).encode()
    ).hexdigest()

    ack = store.get_consumer_ack(scope, exposure.consumer_ack_id)
    assert ack.submitted_prompt_sha256 == hashlib.sha256(prompt).hexdigest()
    assert ack.submitted_input_token_ids_sha256 == expected_token_hash
    assert ack.submitted_input_token_count == len(tokens)
    assert output == "model-output"
    assert exposure.status is MemoryExposureStatus.DELIVERED


def test_empty_release_is_an_explicit_acknowledged_memory_off_treatment() -> None:
    scope, _history, _releases, release, store = _graph(empty=True)
    spec = _spec(
        scope,
        release.release_id,
        max_returned_items=0,
        max_context_utf8_bytes=0,
    )
    attempt, result = _query_all(store, scope, release, spec)
    context, delivery = _deliver(store, scope, result)
    assert context == b""
    assert delivery.rendered_spans == ()
    exposure, ack = _ack_context(store, scope, delivery, context)

    assert result.eligible_revisions == ()
    assert result.retrieved_revisions == ()
    assert result.returned_revisions == ()
    assert exposure.injected_revisions == ()
    assert exposure.status is MemoryExposureStatus.MEMORY_OFF
    assert ack.submitted_prompt_context_sha256 == hashlib.sha256(b"").hexdigest()


def test_nonempty_release_with_no_return_is_not_memory_off() -> None:
    scope, _history, _releases, release, store = _graph(
        retriever=_ReleaseOrderRetriever(returned=False)
    )
    attempt = store.begin_query(_spec(scope, release.release_id))
    result = store.resolve_query(
        scope,
        attempt.attempt_id,
        query=b"query-a",
    )
    context, delivery = _deliver(store, scope, result)
    exposure, _ack = _ack_context(store, scope, delivery, context)

    assert exposure.eligible_revisions
    assert exposure.returned_revisions == exposure.injected_revisions == ()
    assert exposure.status is MemoryExposureStatus.NO_MEMORY_RETURNED


@pytest.mark.parametrize(
    ("retrieved", "returned"),
    (
        ((0, 0), (0,)),
        ((0,), (1,)),
        ((99,), ()),
    ),
)
def test_registered_retriever_rejects_duplicate_and_non_subset(
    retrieved,
    returned,
) -> None:
    scope, _history, _releases, release, store = _graph(
        retriever=_FixedRetriever(retrieved, returned)
    )
    attempt = store.begin_query(_spec(scope, release.release_id))

    with pytest.raises(MemoryQueryConflictError):
        store.resolve_query(
            scope,
            attempt.attempt_id,
            query=b"query-a",
        )


def test_registered_retriever_can_rerank_release_items() -> None:
    scope, _history, _releases, release, store = _graph(
        retriever=_ReleaseOrderRetriever(reverse=True)
    )
    _attempt, result = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id),
    )

    assert tuple(item.revision_id for item in result.retrieved_revisions) == tuple(
        reversed(release.manifest.revision_ids)
    )
    assert tuple(item.release_position for item in result.returned_items) == (1, 0)


def test_renderer_cannot_drop_reorder_or_exceed_budget() -> None:
    for mode, reason in (
        ("drop", "cover every returned"),
        ("reverse", "cover every returned"),
        ("utf8", "UTF-8 boundaries"),
    ):
        scope, _history, _releases, release, store = _graph(renderer=_BadRenderer(mode))
        attempt, result = _query_all(
            store,
            scope,
            release,
            _spec(scope, release.release_id),
        )
        with pytest.raises(MemoryDeliveryConflictError, match=reason):
            store.prepare_delivery(
                scope,
                result.query_result_id,
                renderer_id="bad-renderer",
                renderer_version_sha256=_HASH_C,
            )
        assert store.list_exposures(scope) == ()

    scope, _history, _releases, release, small = _graph()
    _small_attempt, small_result = _query_all(
        small,
        scope,
        release,
        _spec(scope, release.release_id, suffix="small", max_context_utf8_bytes=1),
    )
    with pytest.raises(MemoryDeliveryConflictError, match="byte budget"):
        small.prepare_delivery(
            scope,
            small_result.query_result_id,
            renderer_id="json-lines-v1",
            renderer_version_sha256=_HASH_C,
        )

    with pytest.raises(MemoryDeliveryConflictError, match="not registered"):
        small.prepare_delivery(
            scope,
            small_result.query_result_id,
            renderer_id="unregistered",
            renderer_version_sha256=_HASH_C,
        )


def test_only_registered_consumer_can_ack_exact_context_and_query() -> None:
    wrong = _WrongContextConsumer()
    scope, _history, _releases, release, store = _graph(consumers=(wrong,))
    attempt, result = _query_all(
        store, scope, release, _spec(scope, release.release_id)
    )
    _context, delivery = _deliver(store, scope, result)

    assert not hasattr(store, "acknowledge_delivery")
    with pytest.raises(MemoryBoundaryMismatchError, match="exact rendered"):
        store.submit_delivery(
            scope,
            delivery.delivery_id,
            consumer_id=wrong.consumer_id,
            consumer_version_sha256=_HASH_D,
            call_id="bad-context",
            query=b"query-a",
            history=(),
        )
    with pytest.raises(MemoryBoundaryMismatchError, match="query commitment"):
        store.submit_delivery(
            scope,
            delivery.delivery_id,
            consumer_id=wrong.consumer_id,
            consumer_version_sha256=_HASH_D,
            call_id="bad-query",
            query=b"different-query-with-same-memory",
            history=(),
        )
    with pytest.raises(MemoryConsumerAckConflictError, match="not registered"):
        store.submit_delivery(
            scope,
            delivery.delivery_id,
            consumer_id="adapter-self-report",
            consumer_version_sha256=_HASH_D,
            call_id="forged",
            query=b"query-a",
            history=(),
        )
    assert store.list_exposures(scope) == ()


def test_pre_submit_consumer_failure_releases_claim_for_safe_retry() -> None:
    consumer = _PreSubmitFailOnceConsumer()
    scope, _history, _releases, release, store = _graph(consumers=(consumer,))
    attempt, result = _query_all(
        store, scope, release, _spec(scope, release.release_id)
    )
    _context, delivery = _deliver(store, scope, result)

    with pytest.raises(RuntimeError, match="injected consumer failure"):
        store.submit_delivery(
            scope,
            delivery.delivery_id,
            consumer_id=consumer.consumer_id,
            consumer_version_sha256=_HASH_D,
            call_id="retry-call",
            query=b"query-a",
            history=(),
        )
    exposure, output = store.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id=consumer.consumer_id,
        consumer_version_sha256=_HASH_D,
        call_id="retry-call",
        query=b"query-a",
        history=(),
    )

    assert consumer.calls == 2
    assert output == "context-consumer-output"
    assert exposure.status is MemoryExposureStatus.DELIVERED


def test_external_retry_relies_on_durable_consumer_call_id_idempotency() -> None:
    consumer = _DurablyIdempotentConsumer()
    scope, _history, _releases, release, store = _graph(consumers=(consumer,))
    _attempt, result = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id),
    )
    _context, delivery = _deliver(store, scope, result)

    with pytest.raises(RuntimeError, match="after external side effect"):
        store.submit_delivery(
            scope,
            delivery.delivery_id,
            consumer_id=consumer.consumer_id,
            consumer_version_sha256=_HASH_D,
            call_id="durable-call",
            query=b"query-a",
            history=(),
        )
    exposure, output = store.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id=consumer.consumer_id,
        consumer_version_sha256=_HASH_D,
        call_id="durable-call",
        query=b"query-a",
        history=(),
    )

    assert consumer.calls == 2
    assert consumer.side_effects == 1
    assert output == "durably-cached-output"
    assert exposure.status is MemoryExposureStatus.DELIVERED


def test_cached_consumer_receipt_cannot_cross_scope_or_request_boundaries() -> None:
    consumer = _UnsafeCrossScopeCacheConsumer()
    alice_scope, _history, _releases, alice_release, alice = _graph(
        subject_id="alice",
        consumers=(consumer,),
    )
    bob_scope, _history, _releases, bob_release, bob = _graph(
        subject_id="bob",
        consumers=(consumer,),
    )
    _attempt, alice_result = _query_all(
        alice,
        alice_scope,
        alice_release,
        _spec(alice_scope, alice_release.release_id),
    )
    _context, alice_delivery = _deliver(alice, alice_scope, alice_result)
    alice_exposure, alice_output = alice.submit_delivery(
        alice_scope,
        alice_delivery.delivery_id,
        consumer_id=consumer.consumer_id,
        consumer_version_sha256=_HASH_D,
        call_id="shared-call",
        query=b"query-a",
        history=(b"ALICE-SECRET",),
    )
    assert alice_output == "ALICE-SECRET"
    assert alice_exposure.status is MemoryExposureStatus.DELIVERED

    _attempt, bob_result = _query_all(
        bob,
        bob_scope,
        bob_release,
        _spec(bob_scope, bob_release.release_id),
    )
    _context, bob_delivery = _deliver(bob, bob_scope, bob_result)
    with pytest.raises(MemoryBoundaryMismatchError, match="exact delivery request"):
        bob.submit_delivery(
            bob_scope,
            bob_delivery.delivery_id,
            consumer_id=consumer.consumer_id,
            consumer_version_sha256=_HASH_D,
            call_id="shared-call",
            query=b"query-a",
            history=(b"BOB-SECRET",),
        )

    assert consumer.calls == 2
    assert bob.list_exposures(bob_scope) == ()
    assert bob._submission_claim_by_delivery == {}


def test_same_thread_reentrant_consumer_submission_fails_without_deadlock() -> None:
    consumer = _ReentrantConsumer()
    scope, _history, _releases, release, store = _graph(consumers=(consumer,))
    consumer.store = store
    _attempt, result = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id),
    )
    _context, delivery = _deliver(store, scope, result)

    exposure, _output = store.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id=consumer.consumer_id,
        consumer_version_sha256=_HASH_D,
        call_id="reentrant-call",
        query=b"query-a",
        history=(),
    )

    assert isinstance(consumer.inner_error, MemoryConsumerAckConflictError)
    assert "cannot mutate" in str(consumer.inner_error)
    assert exposure.status is MemoryExposureStatus.DELIVERED


def test_local_commit_failure_rolls_back_every_index_and_releases_claim() -> None:
    consumer = _DurablyIdempotentConsumer()
    consumer.lose_first_response = False
    scope, _history, _releases, release, store = _graph(consumers=(consumer,))
    _attempt, result = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id),
    )
    _context, delivery = _deliver(store, scope, result)
    store._ack_by_call = _FailFirstWriteDict(store._ack_by_call)

    with pytest.raises(KeyboardInterrupt, match="commit interruption"):
        store.submit_delivery(
            scope,
            delivery.delivery_id,
            consumer_id=consumer.consumer_id,
            consumer_version_sha256=_HASH_D,
            call_id="atomic-call",
            query=b"query-a",
            history=(),
        )

    assert store._ack_by_address == {}
    assert store._ack_by_delivery == {}
    assert store._ack_by_call == {}
    assert store._consumer_output_by_ack == {}
    assert store._exposure_by_address == {}
    assert store._exposure_by_attempt == {}
    assert store._exposure_by_ack == {}
    assert store._submission_claim_by_delivery == {}
    assert store._submission_owner_by_delivery == {}
    assert store._active_component_thread_ids == set()

    exposure, output = store.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id=consumer.consumer_id,
        consumer_version_sha256=_HASH_D,
        call_id="atomic-call",
        query=b"query-a",
        history=(),
    )
    assert consumer.calls == 2
    assert consumer.side_effects == 1
    assert output == "durably-cached-output"
    assert exposure.status is MemoryExposureStatus.DELIVERED


def test_attempt_result_and_delivery_publication_are_rollback_safe() -> None:
    retriever = _ReleaseOrderRetriever()
    renderer = _LineRenderer()
    scope, _history, _releases, release, store = _graph(
        retriever=retriever,
        renderer=renderer,
    )
    spec = _spec(scope, release.release_id)
    store._attempt_by_trajectory_slot = _FailFirstWriteDict(
        store._attempt_by_trajectory_slot
    )
    with pytest.raises(KeyboardInterrupt, match="commit interruption"):
        store.begin_query(spec)
    assert store._attempt_by_address == {}
    assert store._attempt_by_idempotency == {}
    assert store._attempt_by_trajectory_slot == {}
    assert store._trajectory_binding == {}
    assert store._rollout_group_binding == {}
    attempt = store.begin_query(spec)

    store._result_by_attempt = _FailFirstWriteDict(store._result_by_attempt)
    with pytest.raises(KeyboardInterrupt, match="commit interruption"):
        store.resolve_query(scope, attempt.attempt_id, query=b"query-a")
    assert store._result_by_address == {}
    assert store._result_by_attempt == {}
    assert store._resolution_claim_by_attempt == {}
    result = store.resolve_query(scope, attempt.attempt_id, query=b"query-a")

    store._context_by_delivery = _FailFirstWriteDict(store._context_by_delivery)
    with pytest.raises(KeyboardInterrupt, match="commit interruption"):
        store.prepare_delivery(
            scope,
            result.query_result_id,
            renderer_id=renderer.renderer_id,
            renderer_version_sha256=renderer.renderer_version_sha256,
        )
    assert store._delivery_by_address == {}
    assert store._delivery_by_result == {}
    assert store._context_by_delivery == {}
    assert store._delivery_claim_by_result == {}
    delivery = store.prepare_delivery(
        scope,
        result.query_result_id,
        renderer_id=renderer.renderer_id,
        renderer_version_sha256=renderer.renderer_version_sha256,
    )

    assert delivery.query_result_id == result.query_result_id
    assert retriever.calls == 2
    assert renderer.calls == 2


def test_consumer_retry_binds_exact_history_bytes_not_only_length() -> None:
    consumer = _ContextConsumer()
    scope, _history, _releases, release, store = _graph(consumers=(consumer,))
    _attempt, result = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id),
    )
    _context, delivery = _deliver(store, scope, result)
    first, first_output = store.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id=consumer.consumer_id,
        consumer_version_sha256=_HASH_D,
        call_id="history-bound-call",
        query=b"query-a",
        history=(b"AAA",),
    )
    replay, replay_output = store.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id=consumer.consumer_id,
        consumer_version_sha256=_HASH_D,
        call_id="history-bound-call",
        query=b"query-a",
        history=(b"AAA",),
    )

    assert replay == first
    assert replay_output == first_output
    assert consumer.calls == 1
    with pytest.raises(MemoryConsumerAckConflictError, match="different consumer"):
        store.submit_delivery(
            scope,
            delivery.delivery_id,
            consumer_id=consumer.consumer_id,
            consumer_version_sha256=_HASH_D,
            call_id="history-bound-call",
            query=b"query-a",
            history=(b"BBB",),
        )
    assert consumer.calls == 1


def test_ack_is_bound_to_attempt_delivery_and_call() -> None:
    scope, _history, _releases, release, store = _graph()
    attempt_a, result_a = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id, suffix="a"),
    )
    context_a, delivery_a = _deliver(store, scope, result_a)
    exposure_a, ack_a = _ack_context(
        store,
        scope,
        delivery_a,
        context_a,
        call_id="shared-call",
    )

    attempt_b, result_b = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id, suffix="b"),
    )
    context_b, delivery_b = _deliver(store, scope, result_b)
    assert context_b == context_a
    assert exposure_a.attempt_id == attempt_a.attempt_id
    assert exposure_a.attempt_id != attempt_b.attempt_id

    _exposure_b, ack_b = _ack_context(
        store,
        scope,
        delivery_b,
        context_b,
        call_id="shared-call",
        observed_query=b"query-b",
    )
    spec_c = replace(
        _spec(scope, release.release_id, suffix="b"),
        query_sequence_no=1,
        query_sha256=hashlib.sha256(b"query-b-second").hexdigest(),
        idempotency_key="query-b-second",
    )
    _attempt_c, result_c = _query_all(store, scope, release, spec_c)
    context_c, delivery_c = _deliver(store, scope, result_c)
    del context_c
    with pytest.raises(MemoryConsumerAckConflictError, match="call ID"):
        # A different delivery cannot reuse the same trajectory/call pair.
        store.submit_delivery(
            scope,
            delivery_c.delivery_id,
            consumer_id="test-context-boundary",
            consumer_version_sha256=_HASH_D,
            call_id=ack_b.call_id,
            query=b"query-b-second",
            history=(),
        )


def test_scope_isolation_hides_foreign_runtime_records() -> None:
    scope, _history, _releases, release, store = _graph()
    foreign = MemoryScope(
        tenant_id="tenant-2",
        namespace=scope.namespace,
        subject_id=scope.subject_id,
    )
    attempt = store.begin_query(_spec(scope, release.release_id))

    with pytest.raises(MemoryQueryNotFoundError):
        store.get_query_attempt(foreign, attempt.attempt_id)
    with pytest.raises(MemoryQueryNotFoundError):
        store.resolve_query(
            foreign,
            attempt.attempt_id,
            query=b"query-a",
        )
    assert store.list_exposures(foreign) == ()


def test_query_slot_trajectory_and_rollout_group_are_immutably_pinned() -> None:
    scope, _history, releases, release, store = _graph()
    first = _spec(scope, release.release_id)
    store.begin_query(first)

    with pytest.raises(MemoryQueryConflictError, match="query sequence"):
        store.begin_query(replace(first, idempotency_key="different-retry-key"))

    empty = releases.append_release(
        ReleaseManifest(scope=scope, revision_ids=()),
        idempotency_key="empty-release",
    )
    with pytest.raises(MemoryQueryConflictError, match="trajectory execution"):
        store.begin_query(
            replace(
                first,
                release_id=empty.release_id,
                query_sequence_no=1,
                query_sha256=hashlib.sha256(b"next-query").hexdigest(),
                idempotency_key="next-query",
            )
        )
    with pytest.raises(MemoryQueryConflictError, match="rollout group"):
        store.begin_query(
            replace(
                _spec(scope, empty.release_id, suffix="other-trajectory"),
                idempotency_key="other-trajectory-empty",
            )
        )
    with pytest.raises(MemoryQueryConflictError, match="rollout group"):
        store.begin_query(
            replace(
                _spec(scope, release.release_id, suffix="policy-drift"),
                task_policy_version_sha256=hashlib.sha256(
                    b"different-task-policy"
                ).hexdigest(),
            )
        )


@pytest.mark.parametrize(
    "mutation",
    ("release_hash", "revision_memory_id", "candidate_content", "evidence_payload"),
)
def test_begin_query_reloads_and_rejects_mutated_source_graph(mutation: str) -> None:
    scope, history, _releases, release, store = _graph()
    revision = history.get_revision(scope, release.manifest.revision_ids[0])
    candidate = history.get_candidate(scope, revision.proposal.candidate_id)
    evidence = history.get_candidate_evidence(scope, candidate.candidate_id)[0]
    if mutation == "release_hash":
        object.__setattr__(release, "content_hash", "0" * 64)
    elif mutation == "revision_memory_id":
        object.__setattr__(revision, "memory_id", "mem_forged")
    elif mutation == "candidate_content":
        object.__setattr__(candidate.proposal, "content", "forged-content")
    else:
        object.__setattr__(evidence.event, "payload", "forged-evidence")

    with pytest.raises(MemoryQueryConflictError):
        store.begin_query(_spec(scope, release.release_id))


def test_source_backends_must_return_the_exact_requested_addresses() -> None:
    scope, history, releases, release, _store = _graph()
    empty = releases.append_release(
        ReleaseManifest(scope=scope, revision_ids=()),
        idempotency_key="wrong-address-empty",
    )

    class WrongReleaseStore:
        def get_release(self, requested_scope, requested_release_id):
            del requested_release_id
            return releases.get_release(requested_scope, empty.release_id)

        def get_release_revisions(self, requested_scope, requested_release_id):
            del requested_release_id
            return releases.get_release_revisions(
                requested_scope,
                empty.release_id,
            )

    wrong_release_runtime = InMemoryMemoryRuntimeStore(
        history,
        WrongReleaseStore(),
        retrievers=(_ReleaseOrderRetriever(),),
    )
    with pytest.raises(MemoryQueryConflictError, match="release graph"):
        wrong_release_runtime.begin_query(_spec(scope, release.release_id))

    first_revision = history.get_revision(
        scope,
        release.manifest.revision_ids[0],
    )
    second_revision = history.get_revision(
        scope,
        release.manifest.revision_ids[1],
    )

    class WrongCandidateHistory:
        def __getattr__(self, name):
            return getattr(history, name)

        def get_candidate(self, requested_scope, candidate_id):
            del candidate_id
            return history.get_candidate(
                requested_scope,
                second_revision.proposal.candidate_id,
            )

    assert first_revision.proposal.candidate_id != second_revision.proposal.candidate_id
    wrong_candidate_runtime = InMemoryMemoryRuntimeStore(
        WrongCandidateHistory(),
        releases,
        retrievers=(_ReleaseOrderRetriever(),),
    )
    with pytest.raises(MemoryQueryConflictError, match="invalid commitments"):
        wrong_candidate_runtime.begin_query(_spec(scope, release.release_id))


def test_candidate_cannot_change_while_evidence_is_loaded() -> None:
    scope, history, releases, release, _store = _graph()

    class MutatingHistory:
        def __getattr__(self, name):
            return getattr(history, name)

        def get_candidate_evidence(self, requested_scope, candidate_id):
            candidate = history.get_candidate(requested_scope, candidate_id)
            evidence = history.get_candidate_evidence(
                requested_scope,
                candidate_id,
            )
            object.__setattr__(
                candidate.proposal,
                "content",
                "forged-after-first-validation",
            )
            return evidence

    runtime = InMemoryMemoryRuntimeStore(
        MutatingHistory(),
        releases,
        retrievers=(_ReleaseOrderRetriever(),),
    )
    with pytest.raises(MemoryQueryConflictError, match="changed"):
        runtime.begin_query(_spec(scope, release.release_id))


def test_revision_cannot_change_while_query_item_is_materialized() -> None:
    scope, history, releases, release, _store = _graph()
    first_revision = history.get_revision(
        scope,
        release.manifest.revision_ids[0],
    )
    second_revision = history.get_revision(
        scope,
        release.manifest.revision_ids[1],
    )

    class MutatingRevisionHistory:
        def __init__(self) -> None:
            self.first_candidate_reads = 0

        def __getattr__(self, name):
            return getattr(history, name)

        def get_candidate(self, requested_scope, candidate_id):
            if candidate_id == first_revision.proposal.candidate_id:
                self.first_candidate_reads += 1
                if self.first_candidate_reads == 3:
                    object.__setattr__(
                        first_revision.proposal,
                        "candidate_id",
                        second_revision.proposal.candidate_id,
                    )
                    return history.get_candidate(
                        requested_scope,
                        second_revision.proposal.candidate_id,
                    )
            return history.get_candidate(requested_scope, candidate_id)

    mutating_history = MutatingRevisionHistory()
    retriever = _ReleaseOrderRetriever()
    runtime = InMemoryMemoryRuntimeStore(
        mutating_history,
        releases,
        retrievers=(retriever,),
    )
    attempt = runtime.begin_query(_spec(scope, release.release_id))

    with pytest.raises(MemoryQueryConflictError, match="invalid commitments"):
        runtime.resolve_query(scope, attempt.attempt_id, query=b"query-a")
    assert retriever.calls == 0


def test_runtime_values_are_frozen_strict_and_fail_closed_on_hash_drift() -> None:
    scope, _history, _releases, release, store = _graph()
    attempt = store.begin_query(_spec(scope, release.release_id))

    with pytest.raises(FrozenInstanceError):
        attempt.attempt_id = "mqat_changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        replace(attempt.spec, query_sequence_no=True)
    with pytest.raises(ValueError, match="canonical attempt"):
        replace(attempt, content_hash="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(attempt.spec, query_sha256="A" * 64)


@pytest.mark.parametrize(
    "mutation",
    ("attempt_query", "result_id", "delivery_id", "context_bytes"),
)
def test_complete_chain_is_revalidated_before_consumer_side_effects(
    mutation: str,
) -> None:
    consumer = _ContextConsumer()
    scope, _history, _releases, release, store = _graph(consumers=(consumer,))
    attempt, result = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id),
    )
    _context, delivery = _deliver(store, scope, result)
    delivery_id = delivery.delivery_id
    submitted_query = b"query-a"
    if mutation == "attempt_query":
        submitted_query = b"forged-query"
        object.__setattr__(
            attempt.spec,
            "query_sha256",
            hashlib.sha256(submitted_query).hexdigest(),
        )
    elif mutation == "result_id":
        object.__setattr__(result, "query_result_id", "mqres_" + "0" * 24)
    elif mutation == "delivery_id":
        object.__setattr__(delivery, "delivery_id", "mdel_" + "0" * 24)
    else:
        store._context_by_delivery[(scope, delivery_id)] = b"forged-context"

    with pytest.raises(MemoryBoundaryMismatchError):
        store.submit_delivery(
            scope,
            delivery_id,
            consumer_id=consumer.consumer_id,
            consumer_version_sha256=_HASH_D,
            call_id="must-not-run",
            query=submitted_query,
            history=(),
        )

    assert consumer.calls == 0
    assert store._submission_claim_by_delivery == {}
    assert store._submission_owner_by_delivery == {}
    assert store._submission_delivery_by_call == {}
    assert store._submission_call_by_delivery == {}
    assert store.list_exposures(scope) == ()


def test_registered_component_identity_is_revalidated_before_each_call() -> None:
    retriever = _ReleaseOrderRetriever()
    scope, _history, _releases, release, store = _graph(retriever=retriever)
    attempt = store.begin_query(_spec(scope, release.release_id))
    retriever.retrieval_policy_id = "mutated-retriever"
    with pytest.raises(MemoryQueryConflictError, match="identity changed"):
        store.resolve_query(scope, attempt.attempt_id, query=b"query-a")
    assert retriever.calls == 0

    renderer = _LineRenderer()
    scope, _history, _releases, release, store = _graph(renderer=renderer)
    _attempt, result = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id),
    )
    renderer.renderer_id = "mutated-renderer"
    with pytest.raises(MemoryDeliveryConflictError, match="identity changed"):
        store.prepare_delivery(
            scope,
            result.query_result_id,
            renderer_id="json-lines-v1",
            renderer_version_sha256=_HASH_C,
        )
    assert renderer.calls == 0

    consumer = _ContextConsumer()
    scope, _history, _releases, release, store = _graph(consumers=(consumer,))
    _attempt, result = _query_all(
        store,
        scope,
        release,
        _spec(scope, release.release_id),
    )
    _context, delivery = _deliver(store, scope, result)
    consumer.consumer_id = "mutated-consumer"
    with pytest.raises(MemoryConsumerAckConflictError, match="identity changed"):
        store.submit_delivery(
            scope,
            delivery.delivery_id,
            consumer_id="test-context-boundary",
            consumer_version_sha256=_HASH_D,
            call_id="must-not-run",
            query=b"query-a",
            history=(),
        )
    assert consumer.calls == 0


def test_concurrent_exact_retries_converge_at_every_stage() -> None:
    retriever = _ReleaseOrderRetriever()
    renderer = _LineRenderer()
    consumer = _ContextConsumer()
    scope, _history, _releases, release, store = _graph(
        retriever=retriever,
        renderer=renderer,
        consumers=(consumer,),
    )
    spec = _spec(scope, release.release_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = tuple(pool.map(lambda _: store.begin_query(spec), range(32)))
    assert len({item.attempt_id for item in attempts}) == 1
    attempt = attempts[0]

    def resolve(_):
        return store.resolve_query(
            scope,
            attempt.attempt_id,
            query=b"query-a",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(resolve, range(32)))
    assert len({item.query_result_id for item in results}) == 1
    assert retriever.calls == 1
    result = results[0]

    def deliver(_):
        return store.prepare_delivery(
            scope,
            result.query_result_id,
            renderer_id="json-lines-v1",
            renderer_version_sha256=_HASH_C,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        deliveries = tuple(pool.map(deliver, range(32)))
    assert len({item.delivery_id for item in deliveries}) == 1
    assert renderer.calls == 1
    delivery = deliveries[0]

    def acknowledge(_):
        exposure, _output = store.submit_delivery(
            scope,
            delivery.delivery_id,
            consumer_id="test-context-boundary",
            consumer_version_sha256=_HASH_D,
            call_id="call",
            query=b"query-a",
            history=(),
        )
        return exposure

    with ThreadPoolExecutor(max_workers=8) as pool:
        exposures = tuple(pool.map(acknowledge, range(32)))
    assert len({item.exposure_id for item in exposures}) == 1
    assert consumer.calls == 1
    assert store.list_exposures(scope) == (exposures[0],)


def test_revision_validation_is_iterative_for_deep_lineages() -> None:
    evidence_store = InMemoryEvidenceStore()
    history_store = InMemoryMemoryHistoryStore(evidence_store)
    release_store = InMemoryMemoryReleaseStore(history_store)
    scope = MemoryScope(
        tenant_id="tenant-1",
        namespace="agent-long-term-memory",
        subject_id="deep-lineage",
    )
    evidence = evidence_store.append(
        EvidenceEvent(
            scope=scope,
            session_id="capture-session",
            run_id="capture-run",
            sequence_no=0,
            kind=EvidenceKind.USER_MESSAGE,
            payload="stable evidence",
            observed_at=_BASE,
            idempotency_key="deep-evidence",
        )
    )
    parent_revision_id = None
    # This is deliberately above CPython's default recursion limit.  A valid
    # lineage must not become unreadable merely because it has many updates.
    for generation in range(1100):
        candidate = history_store.append_candidate(
            CandidateProposal(
                scope=scope,
                content=f"preference-version-{generation}",
                evidence_ids=(evidence.evidence_id,),
                idempotency_key=f"deep-candidate-{generation}",
            )
        )
        revision = history_store.append_revision(
            RevisionProposal(
                scope=scope,
                candidate_id=candidate.candidate_id,
                operation=(
                    RevisionOperation.ADD
                    if parent_revision_id is None
                    else RevisionOperation.REFINE
                ),
                parent_revision_id=parent_revision_id,
                idempotency_key=f"deep-revision-{generation}",
            )
        )
        parent_revision_id = revision.revision_id

    assert parent_revision_id is not None
    release = release_store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(parent_revision_id,)),
        idempotency_key="deep-release",
    )
    runtime = InMemoryMemoryRuntimeStore(
        history_store,
        release_store,
        retrievers=(_ReleaseOrderRetriever(),),
    )

    attempt = runtime.begin_query(_spec(scope, release.release_id))

    assert attempt.release_revisions[0].revision_id == parent_revision_id
