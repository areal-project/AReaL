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
)

_CREATED_AT = datetime(2026, 7, 12, tzinfo=UTC)
_GOLDEN_CHAIN_HASHES = (
    "a25e87f46b9b1168ad9839ec1d7ad375d09cfd8d481125bbbd8b77a66282f22e",
    "e5c42edaac3149d737c0b49448fda1c28494ef8288398392f5b5c9dda2e3b7f2",
    "997767d46b0582865b2ebeecd51346eb63f50ad1d8287ae9e230cd7e6018a6cb",
    "2d18f9fe736ca83be703e1eabf8c2f8fc3b1aafe5c35ca84ff57242d0c1a3ae0",
    "b545cd8792ec93d6b99f7f5a95a6b99cca061569f296e789652ed0fe0e4b69ed",
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
                candidate_id=f"cand_{index}",
                candidate_content_sha256=_hash(f"candidate-{index}"),
                evidence=(
                    MemoryEvidenceRefV1(
                        evidence_id=f"evd_{index}",
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
        attempt_nonce="01" * 32,
        created_at=_CREATED_AT,
    )
    result = MemoryQueryResultV1.create(
        attempt=attempt,
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
        retrieved_revisions=tuple(reversed(refs)),
        returned_items=tuple(reversed(result.returned_items)),
    )
    assert reranked.retrieved_revisions == tuple(reversed(refs))
    assert tuple(item.release_position for item in reranked.returned_items) == (1, 0)
    with pytest.raises(ValueError, match="ordered retrieved"):
        MemoryQueryResultV1.create(
            attempt=attempt,
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
            attempt_nonce="04" * 32,
        )
    first_evidence = result.returned_items[0].evidence[0]
    with pytest.raises(ValueError, match="duplicate evidence IDs"):
        replace(
            result.returned_items[0],
            evidence=(
                first_evidence,
                replace(
                    first_evidence,
                    evidence_content_sha256=_hash("other-evidence-hash"),
                ),
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
