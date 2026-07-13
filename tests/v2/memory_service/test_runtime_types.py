# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest

from areal.v2.memory_service import (
    MemoryConsumerAckV1,
    MemoryConsumerKind,
    MemoryDeliveryV1,
    MemoryEvidenceRefV1,
    MemoryExposureStatus,
    MemoryExposureV1,
    MemoryQueryAttemptV1,
    MemoryQueryItemV1,
    MemoryQueryResultV1,
    MemoryQuerySpecV1,
    MemoryRenderedRevisionSpanV1,
    MemoryRevisionRefV1,
    MemoryScope,
    MemorySourceObjectKind,
    MemorySourceObjectRefV1,
    MemorySourceReadEventV1,
    MemorySourceReadOperation,
    MemorySourceReadPhase,
    MemorySourceReadReceiptV1,
    MemorySourceReadTranscriptV1,
)

_CREATED_AT = datetime(2026, 7, 12, tzinfo=UTC)
_GOLDEN_CHAIN_HASHES = (
    "b7f0db3f75f467da569c759cddf05eca4f850ef99ae80ddb414b1939781af745",
    "fbca92ff928821b766dfd5e295da386b78515e7ec2708f80300f543bbfe18fb4",
    "bde480ed20735b398a0af6bac6ce9799176b93341cc11f4774d23746b92c112f",
    "8953da421b7f6f16c67bad5a927f7bc22b0baac8674a618f019bae91413a2c97",
    "ce83a58acd7efd7a3a8959d8e0416cd501250715dceb7caaa34cb3eef515d48c",
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _empty_history_hash() -> str:
    return hashlib.sha256(
        b"areal-memory-runtime-history-v1\0" + (0).to_bytes(8, "big")
    ).hexdigest()


def _spec(*, suffix: str = "a") -> MemoryQuerySpecV1:
    return MemoryQuerySpecV1(
        scope=MemoryScope(
            tenant_id="tenant-1",
            namespace="agent-memory",
            subject_id="subject-1",
        ),
        release_id="rel_1234567890abcdef12345678",
        trajectory_id=f"trajectory-{suffix}",
        rollout_group_id="rollout-group-1",
        query_sequence_no=0,
        query_sha256=_hash(f"query-{suffix}"),
        task_policy_id="frozen-agent",
        task_policy_version_sha256=_hash("task-policy-v1"),
        retrieval_policy_id="release-order",
        retrieval_policy_version_sha256=_hash("retrieval-v1"),
        max_returned_items=2,
        max_context_utf8_bytes=1024,
        idempotency_key=f"query-{suffix}",
    )


def _source_receipt(
    attempt: MemoryQueryAttemptV1,
    returned_items: tuple[MemoryQueryItemV1, ...] = (),
) -> MemorySourceReadReceiptV1:
    events = [
        MemorySourceReadEventV1(
            sequence_no=0,
            operation=MemorySourceReadOperation.GET_RELEASE_REVISIONS,
            requested_ids=(attempt.spec.release_id,),
            returned_objects=(),
        )
    ]
    for item in returned_items:
        events.extend(
            (
                MemorySourceReadEventV1(
                    sequence_no=len(events),
                    operation=MemorySourceReadOperation.GET_CANDIDATE,
                    requested_ids=(item.candidate_id,),
                    returned_objects=(
                        MemorySourceObjectRefV1(
                            kind=MemorySourceObjectKind.CANDIDATE,
                            object_id=item.candidate_id,
                            object_content_sha256=item.candidate_content_sha256,
                        ),
                    ),
                ),
                MemorySourceReadEventV1(
                    sequence_no=len(events) + 1,
                    operation=MemorySourceReadOperation.GET_CANDIDATE_EVIDENCE,
                    requested_ids=(item.candidate_id,),
                    returned_objects=tuple(
                        MemorySourceObjectRefV1(
                            kind=MemorySourceObjectKind.EVIDENCE,
                            object_id=evidence.evidence_id,
                            object_content_sha256=evidence.evidence_content_sha256,
                        )
                        for evidence in item.evidence
                    ),
                ),
            )
        )
    return MemorySourceReadReceiptV1.create(
        attempt=attempt,
        read_events=tuple(events),
        created_at=_CREATED_AT,
    )


def _pin_transcript(spec: MemoryQuerySpecV1) -> MemorySourceReadTranscriptV1:
    return MemorySourceReadTranscriptV1.create(
        scope=spec.scope,
        phase=MemorySourceReadPhase.ATTEMPT_PIN,
        read_events=(
            MemorySourceReadEventV1(
                sequence_no=0,
                operation=MemorySourceReadOperation.GET_RELEASE_REVISIONS,
                requested_ids=(spec.release_id,),
                returned_objects=(),
            ),
        ),
        created_at=_CREATED_AT,
    )


def _address(kind: MemorySourceObjectKind, label: str) -> MemorySourceObjectRefV1:
    content_hash = _hash(label)
    prefix = {
        MemorySourceObjectKind.RELEASE: "rel_",
        MemorySourceObjectKind.REVISION: "rev_",
        MemorySourceObjectKind.CANDIDATE: "cand_",
        MemorySourceObjectKind.EVIDENCE: "evd_",
    }[kind]
    return MemorySourceObjectRefV1(
        kind=kind,
        object_id=f"{prefix}{content_hash[:24]}",
        object_content_sha256=content_hash,
    )


def _canonical_source_receipt() -> tuple[
    MemoryQueryAttemptV1,
    MemorySourceReadReceiptV1,
]:
    release = _address(MemorySourceObjectKind.RELEASE, "receipt-release")
    revision = _address(MemorySourceObjectKind.REVISION, "receipt-revision")
    spec = replace(_spec(), release_id=release.object_id)
    attempt = MemoryQueryAttemptV1.create(
        spec=spec,
        release_content_sha256=release.object_content_sha256,
        release_revisions=(
            MemoryRevisionRefV1(
                revision_id=revision.object_id,
                revision_content_sha256=revision.object_content_sha256,
            ),
        ),
        pin_read_transcript=_pin_transcript(spec),
        attempt_nonce="03" * 32,
        created_at=_CREATED_AT,
    )
    receipt = MemorySourceReadReceiptV1.create(
        attempt=attempt,
        read_events=(
            MemorySourceReadEventV1(
                sequence_no=0,
                operation=MemorySourceReadOperation.GET_RELEASE,
                requested_ids=(release.object_id,),
                returned_objects=(release,),
            ),
            MemorySourceReadEventV1(
                sequence_no=1,
                operation=MemorySourceReadOperation.GET_RELEASE_REVISIONS,
                requested_ids=(release.object_id,),
                returned_objects=(revision,),
            ),
        ),
        created_at=_CREATED_AT,
    )
    return attempt, receipt


def _chain(*, empty: bool = False):
    refs = ()
    items = ()
    if not empty:
        refs = (
            MemoryRevisionRefV1("rev_one", _hash("revision-one")),
            MemoryRevisionRefV1("rev_two", _hash("revision-two")),
        )
        items = tuple(
            MemoryQueryItemV1(
                release_position=index,
                revision=revision,
                memory_id=f"mem_{index}",
                generation=0,
                candidate_id=("cand_" + _hash(f"candidate-{index}")[:24]),
                candidate_content_sha256=_hash(f"candidate-{index}"),
                evidence=(
                    MemoryEvidenceRefV1(
                        evidence_id=("evd_" + _hash(f"evidence-{index}")[:24]),
                        evidence_content_sha256=_hash(f"evidence-{index}"),
                    ),
                ),
                content=f"fact-{index}",
            )
            for index, revision in enumerate(refs)
        )
    spec = _spec()
    attempt = MemoryQueryAttemptV1.create(
        spec=spec,
        release_content_sha256=_hash("release"),
        release_revisions=refs,
        pin_read_transcript=_pin_transcript(spec),
        attempt_nonce="01" * 32,
        created_at=_CREATED_AT,
    )
    receipt = _source_receipt(attempt, items)
    result = MemoryQueryResultV1.create(
        attempt=attempt,
        source_read_receipt=receipt,
        retrieved_revisions=refs,
        returned_items=items,
        created_at=_CREATED_AT,
    )
    context = b"" if empty else b"fact-0\nfact-1\n"
    spans = ()
    if not empty:
        spans = (
            MemoryRenderedRevisionSpanV1(
                revision=refs[0],
                rendered_start=0,
                rendered_end=7,
                rendered_fragment_sha256=hashlib.sha256(context[0:7]).hexdigest(),
            ),
            MemoryRenderedRevisionSpanV1(
                revision=refs[1],
                rendered_start=7,
                rendered_end=14,
                rendered_fragment_sha256=hashlib.sha256(context[7:14]).hexdigest(),
            ),
        )
    delivery = MemoryDeliveryV1.create(
        query_result=result,
        renderer_id="json-lines-v1",
        renderer_version_sha256=_hash("renderer-v1"),
        rendered_context_sha256=hashlib.sha256(context).hexdigest(),
        rendered_context_utf8_bytes=len(context),
        rendered_spans=spans,
        delivery_nonce="02" * 32,
        created_at=_CREATED_AT,
    )
    prompt = b"system\n" + context + b"query\n"
    token_ids = (101, 102, 103)
    ack = MemoryConsumerAckV1.create(
        delivery=delivery,
        consumer_kind=MemoryConsumerKind.MODEL_CALL,
        consumer_id="model-submit-boundary",
        consumer_version_sha256=_hash("boundary-v1"),
        call_id="call-1",
        submitted_prompt_sha256=hashlib.sha256(prompt).hexdigest(),
        submitted_prompt_context_start=len(b"system\n"),
        submitted_prompt_context_end=len(b"system\n") + len(context),
        submitted_prompt_context_sha256=hashlib.sha256(context).hexdigest(),
        submitted_prompt_context_utf8_bytes=len(context),
        observed_query_sha256=spec.query_sha256,
        observed_history_sha256=_empty_history_hash(),
        observed_history_length=0,
        submitted_input_token_ids_sha256=hashlib.sha256(
            json.dumps(list(token_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        submitted_input_token_count=len(token_ids),
        created_at=_CREATED_AT,
    )
    exposure = MemoryExposureV1.create(
        attempt=attempt,
        query_result=result,
        delivery=delivery,
        consumer_ack=ack,
        created_at=_CREATED_AT,
    )
    return attempt, result, delivery, ack, exposure


def test_source_read_receipt_has_golden_bytes_hash_and_id() -> None:
    _attempt, receipt = _canonical_source_receipt()

    assert receipt.canonical_bytes() == (
        b'{"attempt_content_sha256":"05ceeb4171014af2a6cb29c7213b4b286a838119cf174dfb'
        b'd82748f1b223eeb1","attempt_id":"mqat_05ceeb4171014af2a6cb29c7","read_event'
        b's":[{"operation":"get_release","requested_ids":["rel_2c177320e7d3e84d689b4ae'
        b'0"],"returned_objects":[{"kind":"release","object_content_sha256":"2c177320e'
        b'7d3e84d689b4ae0a3100c19b524f437d440eb480adea9b3b67f7b90","object_id":"re'
        b'l_2c177320e7d3e84d689b4ae0"}],"sequence_no":0},{"operation":"get_release_revi'
        b'sions","requested_ids":["rel_2c177320e7d3e84d689b4ae0"],"returned_objects"'
        b':[{"kind":"revision","object_content_sha256":"9cfc4ce35446d35f9c356550260f772'
        b'41e629f25cc3e3f1d088940a8a4e473d6","object_id":"rev_9cfc4ce35446d35f9c356'
        b'550"}],"sequence_no":1}],"record_kind":"memory_source_read_receipt","schema_v'
        b'ersion":1,"scope":{"namespace":"agent-memory","subject_id":"subject-1","tenan'
        b't_id":"tenant-1"}}'
    )
    assert (
        receipt.content_hash
        == "eacedf74057c8dc2aaaa5f517c0c8225c4b33a143d3b570620a196e57d50dcb7"
    )
    assert receipt.source_read_receipt_id == "msrr_eacedf74057c8dc2aaaa5f51"
    assert b"receipt-release" not in receipt.canonical_bytes()


def test_source_read_values_reject_address_kind_order_and_duplicate_mutants() -> None:
    attempt, receipt = _canonical_source_receipt()
    release_ref = receipt.read_events[0].returned_objects[0]
    revision_ref = receipt.read_events[1].returned_objects[0]

    with pytest.raises(ValueError, match="object_id"):
        replace(release_ref, object_id="rel_" + "0" * 24)
    with pytest.raises(ValueError, match="kind"):
        replace(release_ref, kind=MemorySourceObjectKind.REVISION)
    with pytest.raises(ValueError, match="kind"):
        replace(
            receipt.read_events[0],
            returned_objects=(revision_ref,),
        )
    with pytest.raises(ValueError, match="request and return"):
        replace(
            receipt.read_events[0],
            requested_ids=("rel_" + "0" * 24,),
        )
    with pytest.raises(TypeError):
        replace(receipt.read_events[0], requested_ids=[release_ref.object_id])
    with pytest.raises(ValueError, match="source address"):
        replace(
            receipt.read_events[1],
            requested_ids=(revision_ref.object_id,),
        )
    with pytest.raises(ValueError, match="duplicate"):
        MemorySourceReadEventV1(
            sequence_no=0,
            operation=MemorySourceReadOperation.GET_RELEASE_REVISIONS,
            requested_ids=(attempt.spec.release_id,),
            returned_objects=(revision_ref, revision_ref),
        )
    with pytest.raises(ValueError, match="contiguous"):
        MemorySourceReadReceiptV1.create(
            attempt=attempt,
            read_events=(
                replace(receipt.read_events[0], sequence_no=1),
                replace(receipt.read_events[1], sequence_no=0),
            ),
        )


def test_attempt_commits_to_acyclic_pin_read_transcript() -> None:
    attempt, _receipt = _canonical_source_receipt()
    transcript = _pin_transcript(attempt.spec)
    rebound = MemoryQueryAttemptV1.create(
        spec=attempt.spec,
        release_content_sha256=attempt.release_content_sha256,
        release_revisions=attempt.release_revisions,
        pin_read_transcript=transcript,
        attempt_nonce="05" * 32,
    )

    assert rebound.pin_read_transcript_id == transcript.source_read_transcript_id
    assert rebound.pin_read_transcript_content_sha256 == transcript.content_hash
    assert b"attempt_id" not in transcript.canonical_bytes()
    assert transcript.phase is MemorySourceReadPhase.ATTEMPT_PIN


def test_query_result_rejects_cross_attempt_or_forged_receipt_reference() -> None:
    attempt, receipt = _canonical_source_receipt()
    other_attempt = MemoryQueryAttemptV1.create(
        spec=(other_spec := replace(attempt.spec, trajectory_id="other-trajectory")),
        release_content_sha256=attempt.release_content_sha256,
        release_revisions=attempt.release_revisions,
        pin_read_transcript=_pin_transcript(other_spec),
        attempt_nonce="04" * 32,
    )
    with pytest.raises(ValueError, match="exact attempt"):
        MemoryQueryResultV1.create(
            attempt=other_attempt,
            source_read_receipt=receipt,
            retrieved_revisions=(),
            returned_items=(),
        )
    with pytest.raises(ValueError, match="canonical source-read"):
        replace(receipt, content_hash="0" * 64)


def test_runtime_chain_is_content_addressed_and_injection_is_derived() -> None:
    attempt, result, delivery, ack, exposure = _chain()

    assert exposure.injected_revisions == result.returned_revisions
    assert exposure.injected_revisions == tuple(
        item.revision for item in delivery.rendered_spans
    )
    assert exposure.status is MemoryExposureStatus.DELIVERED
    assert (
        tuple(
            record.content_hash for record in (attempt, result, delivery, ack, exposure)
        )
        == _GOLDEN_CHAIN_HASHES
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
        assert getattr(record, id_field) == prefix + record.content_hash[:24]
    with pytest.raises(ValueError, match="evidence_id"):
        replace(
            result.returned_items[0].evidence[0],
            evidence_content_sha256=_hash("different-evidence"),
        )


def test_empty_release_has_a_first_class_memory_off_exposure() -> None:
    attempt, result, delivery, ack, exposure = _chain(empty=True)

    assert attempt.release_revisions == ()
    assert result.eligible_revisions == result.returned_revisions == ()
    assert delivery.rendered_spans == ()
    assert ack.submitted_prompt_context_utf8_bytes == 0
    assert exposure.injected_revisions == ()
    assert exposure.status is MemoryExposureStatus.MEMORY_OFF


def test_nonempty_release_with_no_return_is_not_memory_off() -> None:
    attempt, _result, _delivery, _ack, _exposure = _chain()
    result = MemoryQueryResultV1.create(
        attempt=attempt,
        source_read_receipt=_source_receipt(attempt),
        retrieved_revisions=(),
        returned_items=(),
    )
    empty_hash = hashlib.sha256(b"").hexdigest()
    delivery = MemoryDeliveryV1.create(
        query_result=result,
        renderer_id="json-lines-v1",
        renderer_version_sha256=_hash("renderer-v1"),
        rendered_context_sha256=empty_hash,
        rendered_context_utf8_bytes=0,
        rendered_spans=(),
        delivery_nonce="05" * 32,
    )
    ack = MemoryConsumerAckV1.create(
        delivery=delivery,
        consumer_kind=MemoryConsumerKind.CONTEXT,
        consumer_id="context-boundary",
        consumer_version_sha256=_hash("boundary-v1"),
        call_id="empty-result-call",
        submitted_prompt_sha256=empty_hash,
        submitted_prompt_context_start=0,
        submitted_prompt_context_end=0,
        submitted_prompt_context_sha256=empty_hash,
        submitted_prompt_context_utf8_bytes=0,
        observed_query_sha256=attempt.spec.query_sha256,
        observed_history_sha256=_empty_history_hash(),
        observed_history_length=0,
        submitted_input_token_ids_sha256=None,
        submitted_input_token_count=None,
    )
    exposure = MemoryExposureV1.create(
        attempt=attempt,
        query_result=result,
        delivery=delivery,
        consumer_ack=ack,
    )

    assert exposure.eligible_revisions == attempt.release_revisions
    assert exposure.returned_revisions == exposure.injected_revisions == ()
    assert exposure.status is MemoryExposureStatus.NO_MEMORY_RETURNED


def test_query_result_preserves_policy_rank_and_release_positions() -> None:
    attempt, result, _delivery, _ack, _exposure = _chain()
    refs = attempt.release_revisions

    reranked = MemoryQueryResultV1.create(
        attempt=attempt,
        source_read_receipt=_source_receipt(
            attempt,
            tuple(reversed(result.returned_items)),
        ),
        retrieved_revisions=tuple(reversed(refs)),
        returned_items=tuple(reversed(result.returned_items)),
    )
    assert reranked.retrieved_revisions == tuple(reversed(refs))
    assert tuple(item.release_position for item in reranked.returned_items) == (1, 0)
    with pytest.raises(ValueError, match="source-read receipt"):
        MemoryQueryResultV1.create(
            attempt=attempt,
            source_read_receipt=_source_receipt(attempt, result.returned_items),
            retrieved_revisions=tuple(reversed(refs)),
            returned_items=tuple(reversed(result.returned_items)),
        )
    with pytest.raises(ValueError, match="ordered retrieved"):
        MemoryQueryResultV1.create(
            attempt=attempt,
            source_read_receipt=_source_receipt(
                attempt,
                (result.returned_items[1],),
            ),
            retrieved_revisions=(refs[0],),
            returned_items=(result.returned_items[1],),
        )
    with pytest.raises(ValueError, match="duplicate"):
        replace(result, eligible_revisions=(refs[0], refs[0]))
    same_public_id = replace(
        refs[0],
        revision_content_sha256=_hash("different-full-hash"),
    )
    with pytest.raises(ValueError, match="duplicate revision IDs"):
        MemoryQueryAttemptV1.create(
            spec=attempt.spec,
            release_content_sha256=attempt.release_content_sha256,
            release_revisions=(refs[0], same_public_id),
            pin_read_transcript=_pin_transcript(attempt.spec),
            attempt_nonce="04" * 32,
        )
    first_evidence = result.returned_items[0].evidence[0]
    with pytest.raises(ValueError, match="duplicate evidence IDs"):
        replace(
            result.returned_items[0],
            evidence=(
                first_evidence,
                first_evidence,
            ),
        )


def test_exposure_rejects_partial_render_and_cross_chain_ack() -> None:
    attempt, result, delivery, ack, _exposure = _chain()
    partial = MemoryDeliveryV1.create(
        query_result=result,
        renderer_id=delivery.renderer_id,
        renderer_version_sha256=delivery.renderer_version_sha256,
        rendered_context_sha256=delivery.rendered_context_sha256,
        rendered_context_utf8_bytes=delivery.rendered_context_utf8_bytes,
        rendered_spans=delivery.rendered_spans[:1],
        delivery_nonce="03" * 32,
    )
    with pytest.raises(ValueError, match="acknowledged chain"):
        MemoryExposureV1.create(
            attempt=attempt,
            query_result=result,
            delivery=partial,
            consumer_ack=ack,
        )


def test_runtime_values_are_frozen_and_reject_type_or_hash_drift() -> None:
    attempt, _result, _delivery, _ack, exposure = _chain()

    with pytest.raises(FrozenInstanceError):
        exposure.status = MemoryExposureStatus.MEMORY_OFF  # type: ignore[misc]
    with pytest.raises(TypeError):
        replace(attempt.spec, query_sequence_no=True)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(attempt.spec, query_sha256="A" * 64)
    with pytest.raises(ValueError, match="canonical exposure"):
        replace(exposure, content_hash="0" * 64)


@pytest.mark.parametrize(
    ("record_index", "id_field", "prefix"),
    (
        (0, "attempt_id", "mqat_"),
        (1, "query_result_id", "mqres_"),
        (2, "delivery_id", "mdel_"),
        (3, "consumer_ack_id", "mack_"),
        (4, "exposure_id", "mexp_"),
    ),
)
def test_canonical_bytes_revalidates_public_record_identity(
    record_index: int,
    id_field: str,
    prefix: str,
) -> None:
    records = _chain()
    record = records[record_index]
    object.__setattr__(record, id_field, prefix + "0" * 24)

    with pytest.raises(ValueError, match=id_field):
        record.canonical_bytes()
