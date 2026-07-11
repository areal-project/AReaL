"""Tests for auditable InfBridge client-local call traces."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.v2.inference_service.test_inf_bridge import (
    _make_bridge,
    _make_request,
    _make_sglang_response,
    _make_vllm_response,
)

from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    ModelRequest,
    ModelResponse,
)
from areal.v2.inference_service.backend import TraceableInfBridgeBackend
from areal.v2.inference_service.client_trace import (
    GenerationAttemptTrace,
    GenerationPhysicalTrace,
    GenerationResponseEvidence,
    ParsedResponseJSONEvidence,
    generation_physical_trace_bytes,
    generation_physical_trace_from_bytes,
    generation_physical_trace_sha256,
    generation_response_evidence_bytes,
    generation_response_evidence_from_bytes,
    generation_response_evidence_sha256,
    generation_response_evidence_values,
    validate_generation_physical_trace_response,
    validate_generation_response_evidence,
)
from areal.v2.inference_service.inf_bridge import GenerationTraceValidationError
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend
from areal.v2.inference_service.vllm.bridge import VLLMBridgeBackend


def _domain_separated_json_sha256(value: Any, *, domain: bytes) -> str:
    """Hash canonical JSON with an independently supplied semantic domain."""
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + payload).hexdigest()


class _ThirdPartyBackendWithoutTrace:
    """SGLang-compatible backend intentionally lacking the optional trace hook."""

    def __init__(self) -> None:
        self._delegate = SGLangBridgeBackend()

    def build_generation_request(
        self,
        req: ModelRequest,
        with_lora: bool,
        version: int = -1,
    ) -> HttpRequest:
        return self._delegate.build_generation_request(req, with_lora, version)

    def parse_generation_response(
        self,
        response: dict[str, Any],
    ) -> HttpGenerationResult:
        return self._delegate.parse_generation_response(response)

    def get_pause_request(self) -> HttpRequest:
        return self._delegate.get_pause_request()

    def get_resume_request(self) -> HttpRequest:
        return self._delegate.get_resume_request()

    def get_offload_request(self) -> HttpRequest:
        return self._delegate.get_offload_request()

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        return self._delegate.get_onload_request(tags)

    def get_generation_max_new_tokens(self, http_req: HttpRequest) -> int:
        return self._delegate.get_generation_max_new_tokens(http_req)

    def patch_generation_request(
        self,
        http_req: HttpRequest,
        req: ModelRequest,
        accumulated_tokens: list[int],
        remaining_tokens: int,
    ) -> None:
        self._delegate.patch_generation_request(
            http_req,
            req,
            accumulated_tokens,
            remaining_tokens,
        )


class _OpaqueMetadata:
    """Metadata value accepted by the backend but intentionally not copyable."""

    def __deepcopy__(self, _memo: dict[int, Any]) -> _OpaqueMetadata:
        raise TypeError("opaque metadata cannot be copied")


class TestInfBridgePhysicalTrace:
    """Auditable, immutable client-local observations around HTTP attempts."""

    @pytest.mark.asyncio
    async def test_sglang_normal_stop_records_canonical_physical_trace(self):
        sent_payloads: list[dict[str, Any]] = []
        raw_response = _make_sglang_response([(-0.5, 100), (-0.3, 101)], "stop")

        async def mock_send(http_req, **kwargs):
            sent_payloads.append(deepcopy(http_req.payload))
            return deepcopy(raw_response)

        bridge = _make_bridge(version=7, backend_addr="http://mock/")
        bridge._send_request = mock_send
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=5)
        req.rid = "trace-normal"

        resp, trace = await bridge.agenerate_with_trace(req)

        assert isinstance(resp, ModelResponse)
        assert isinstance(trace, GenerationPhysicalTrace)
        assert resp.output_tokens == [100, 101]
        assert trace.schema_version == 2
        assert trace.request_id == "trace-normal"
        assert trace.backend_kind == "SGLangBridgeBackend"
        assert trace.backend_addr_sha256 == hashlib.sha256(b"http://mock").hexdigest()
        assert trace.request_input_token_ids == (1, 2, 3)
        assert trace.initial_client_version == 7
        assert trace.final_client_version == 7
        assert trace.initial_client_version_epoch == 0
        assert trace.final_client_version_epoch == 0
        assert trace.effective_max_new_tokens == 5
        assert trace.configured_attempt_limit == 20
        assert trace.final_output_token_ids == (100, 101)
        assert trace.final_stop_reason == "stop"
        assert trace.terminal_reason == "backend_stop"

        assert len(trace.attempts) == 1
        attempt = trace.attempts[0]
        assert isinstance(attempt, GenerationAttemptTrace)
        assert attempt.attempt_index == 0
        assert attempt.client_version_before_send == 7
        assert attempt.client_version_after_receive == 7
        assert attempt.output_version_label == 7
        assert attempt.client_version_epoch_before_send == 0
        assert attempt.client_version_epoch_after_receive == 0
        assert attempt.output_version_epoch == 0
        assert attempt.remaining_new_tokens == 5
        assert attempt.endpoint == "/generate"
        assert attempt.method == "POST"
        assert attempt.prepared_request_json_sha256 == _domain_separated_json_sha256(
            sent_payloads[0],
            domain=b"areal-infbridge-prepared-request-json-v1\0",
        )
        assert attempt.parsed_response_json_sha256 == _domain_separated_json_sha256(
            raw_response,
            domain=b"areal-infbridge-parsed-response-json-v1\0",
        )
        assert attempt.submitted_input_token_ids == (1, 2, 3)
        assert attempt.raw_stop_reason == "stop"
        assert attempt.output_token_ids == (100, 101)
        assert attempt.output_logprob_count == 2

        encoded = generation_physical_trace_bytes(trace)
        decoded = json.loads(encoded)
        assert set(decoded) == {
            "attempts",
            "backend_addr_sha256",
            "backend_kind",
            "configured_attempt_limit",
            "effective_max_new_tokens",
            "final_client_version",
            "final_client_version_epoch",
            "final_output_token_ids",
            "final_stop_reason",
            "initial_client_version",
            "initial_client_version_epoch",
            "kind",
            "request_id",
            "request_input_token_ids",
            "schema_version",
            "terminal_reason",
        }
        assert set(decoded["attempts"][0]) == {
            "attempt_index",
            "client_version_after_receive",
            "client_version_before_send",
            "client_version_epoch_after_receive",
            "client_version_epoch_before_send",
            "endpoint",
            "method",
            "output_logprob_count",
            "output_token_ids",
            "output_version_label",
            "output_version_epoch",
            "parsed_response_json_sha256",
            "prepared_request_json_sha256",
            "raw_stop_reason",
            "remaining_new_tokens",
            "submitted_input_token_ids",
        }
        assert decoded["kind"] == "areal-generation-physical-trace-v2"
        expected = (
            b'{"attempts":[{"attempt_index":0,"client_version_after_receive":7,'
            b'"client_version_before_send":7,"client_version_epoch_after_receive":0,'
            b'"client_version_epoch_before_send":0,"endpoint":"/generate",'
            b'"method":"POST",'
            b'"output_logprob_count":2,"output_token_ids":[100,101],'
            b'"output_version_epoch":0,"output_version_label":7,'
            b'"parsed_response_json_sha256":'
            b'"a6139c5961208dbfa553d0c330516f9c234517dc1776cae35627aedbe308fb0d",'
            b'"prepared_request_json_sha256":'
            b'"edef4683dba2caa3a86090ef4a3a61876908d3a1307d348107df9f5fa97acab5",'
            b'"raw_stop_reason":"stop","remaining_new_tokens":5,'
            b'"submitted_input_token_ids":[1,2,3]}],"backend_addr_sha256":'
            b'"76f5495ef9aa27156aca83370226a757446a93e03858b06a5b58f9a0e75edfaf",'
            b'"backend_kind":"SGLangBridgeBackend","configured_attempt_limit":20,'
            b'"effective_max_new_tokens":5,"final_client_version":7,'
            b'"final_client_version_epoch":0,"final_output_token_ids":[100,101],'
            b'"final_stop_reason":"stop","initial_client_version":7,'
            b'"initial_client_version_epoch":0,'
            b'"kind":"areal-generation-physical-trace-v2",'
            b'"request_id":"trace-normal","request_input_token_ids":[1,2,3],'
            b'"schema_version":2,"terminal_reason":"backend_stop"}'
        )
        assert encoded == expected
        restored = generation_physical_trace_from_bytes(encoded)
        assert restored == trace
        assert generation_physical_trace_bytes(restored) == encoded
        assert generation_physical_trace_sha256(trace) == (
            "2b6971d8eebfb2ec6b3251f76dced0dc019083608ee399d9a693e514ed6f329d"
        )

    @pytest.mark.asyncio
    async def test_abort_resubmit_records_each_patched_payload_immutably(self):
        sent_requests = []

        async def mock_send(http_req, **kwargs):
            sent_requests.append(http_req)
            if len(sent_requests) == 1:
                return _make_sglang_response([(-0.5, 100), (-0.3, 101)], "abort")
            return _make_sglang_response([(-0.2, 200)], "stop")

        bridge = _make_bridge()
        bridge._send_request = mock_send
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=5)
        req.rid = "trace-resubmit"

        resp, trace = await bridge.agenerate_with_trace(req)

        assert resp.output_tokens == [100, 101, 200]
        assert [attempt.attempt_index for attempt in trace.attempts] == [0, 1]
        assert [attempt.remaining_new_tokens for attempt in trace.attempts] == [5, 3]
        assert [attempt.submitted_input_token_ids for attempt in trace.attempts] == [
            (1, 2, 3),
            (1, 2, 3, 100, 101),
        ]
        assert [attempt.raw_stop_reason for attempt in trace.attempts] == [
            "abort",
            "stop",
        ]
        assert [attempt.output_token_ids for attempt in trace.attempts] == [
            (100, 101),
            (200,),
        ]
        assert trace.final_output_token_ids == (100, 101, 200)
        assert trace.final_stop_reason == "stop"
        assert trace.terminal_reason == "backend_stop"
        assert (
            trace.attempts[0].prepared_request_json_sha256
            != trace.attempts[1].prepared_request_json_sha256
        )

        # The backend mutates and reuses one HttpRequest.  Later request/user
        # mutation must not rewrite already-recorded evidence.
        sent_requests[-1].payload["input_ids"][:] = [999]
        req.input_ids[:] = [888]
        assert trace.request_input_token_ids == (1, 2, 3)
        assert trace.attempts[0].submitted_input_token_ids == (1, 2, 3)
        assert trace.attempts[1].submitted_input_token_ids == (1, 2, 3, 100, 101)

    @pytest.mark.asyncio
    async def test_optional_response_evidence_replays_every_parsed_json_attempt(self):
        raw_responses = (
            _make_sglang_response([(-0.5, 100)], "abort"),
            _make_sglang_response([(-0.2, 200)], "stop"),
        )
        bridge = _make_bridge()
        bridge._send_request = AsyncMock(side_effect=deepcopy(raw_responses))
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=5)
        req.rid = "trace-response-evidence"

        (
            response,
            trace,
            evidence,
        ) = await bridge.agenerate_with_trace_and_response_evidence(req)

        assert response.output_tokens == [100, 200]
        assert type(evidence) is GenerationResponseEvidence
        assert evidence.schema_version == 1
        assert evidence.generation_trace_sha256 == generation_physical_trace_sha256(
            trace
        )
        assert all(
            type(attempt) is ParsedResponseJSONEvidence for attempt in evidence.attempts
        )
        assert tuple(attempt.attempt_index for attempt in evidence.attempts) == (0, 1)
        expected_values = tuple(
            json.loads(json.dumps(value, allow_nan=False)) for value in raw_responses
        )
        assert generation_response_evidence_values(trace, evidence) == expected_values
        assert validate_generation_response_evidence(trace, evidence) is None
        encoded = generation_response_evidence_bytes(evidence)
        assert encoded == json.dumps(
            json.loads(encoded),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        assert len(encoded) == 648
        assert generation_response_evidence_sha256(evidence) == (
            "553b0ab264d1439189f90bea891fc329232f2b8d9a4ebea35894a4f25d898e0e"
        )
        assert (
            generation_response_evidence_sha256(evidence)
            == hashlib.sha256(encoded).hexdigest()
        )
        restored = generation_response_evidence_from_bytes(encoded)
        assert restored == evidence
        assert generation_response_evidence_bytes(restored) == encoded

        changed_preimage = ParsedResponseJSONEvidence(
            attempt_index=0,
            parsed_response_json_sha256=_domain_separated_json_sha256(
                raw_responses[1],
                domain=b"areal-infbridge-parsed-response-json-v1\0",
            ),
            canonical_json_bytes=json.dumps(
                raw_responses[1],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii"),
        )
        forged = replace(
            evidence,
            attempts=(changed_preimage, evidence.attempts[1]),
        )
        with pytest.raises(ValueError, match="attempt does not match trace"):
            validate_generation_response_evidence(trace, forged)
        assert "canonical_json_bytes" not in repr(evidence.attempts[0])
        assert "meta_info" not in repr(evidence.attempts[0])

    @pytest.mark.asyncio
    async def test_compact_trace_api_does_not_collect_response_preimages(self):
        bridge = _make_bridge()
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.5, 100)], "stop")
        )
        req = _make_request(input_ids=[1], max_new_tokens=2)

        result = await bridge.agenerate_with_trace(req)

        assert type(result) is tuple
        assert len(result) == 2
        assert all(
            not hasattr(attempt, "canonical_json_bytes")
            for attempt in result[1].attempts
        )

    @pytest.mark.parametrize(
        "canonical_json_bytes",
        (
            b'{ "a":1}',
            b'{"a":1,"a":1}',
            b'{"value":NaN}',
            b"[]",
            bytearray(b'{"a":1}'),
        ),
        ids=("whitespace", "duplicate-key", "nan", "non-object", "wrong-type"),
    )
    def test_response_evidence_rejects_noncanonical_or_ambiguous_json(
        self,
        canonical_json_bytes: object,
    ):
        with pytest.raises(ValueError):
            ParsedResponseJSONEvidence(
                attempt_index=0,
                parsed_response_json_sha256="0" * 64,
                canonical_json_bytes=canonical_json_bytes,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_persisted_trace_and_sidecar_loaders_fail_closed(self):
        bridge = _make_bridge()
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.5, 100)], "stop")
        )
        req = _make_request(input_ids=[1], max_new_tokens=2)
        req.rid = "trace-loader"
        _, trace, evidence = await bridge.agenerate_with_trace_and_response_evidence(
            req
        )
        trace_bytes = generation_physical_trace_bytes(trace)
        evidence_bytes = generation_response_evidence_bytes(evidence)

        def canonical(value: object) -> bytes:
            return json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")

        trace_value = json.loads(trace_bytes)
        wrong_trace_kind = deepcopy(trace_value)
        wrong_trace_kind["kind"] = "areal-generation-physical-trace-v1"
        wrong_trace_schema = deepcopy(trace_value)
        wrong_trace_schema["schema_version"] = True
        extra_trace_field = deepcopy(trace_value)
        extra_trace_field["unknown"] = 0
        duplicate_trace_key = trace_bytes.replace(
            b'{"attempts":',
            b'{"attempts":[],"attempts":',
            1,
        )
        for malformed in (
            b" " + trace_bytes,
            canonical(wrong_trace_kind),
            canonical(wrong_trace_schema),
            canonical(extra_trace_field),
            duplicate_trace_key,
            bytearray(trace_bytes),
        ):
            with pytest.raises((TypeError, ValueError)):
                generation_physical_trace_from_bytes(malformed)  # type: ignore[arg-type]

        evidence_value = json.loads(evidence_bytes)
        wrong_evidence_kind = deepcopy(evidence_value)
        wrong_evidence_kind["kind"] = "wrong-kind"
        wrong_evidence_schema = deepcopy(evidence_value)
        wrong_evidence_schema["schema_version"] = True
        extra_evidence_field = deepcopy(evidence_value)
        extra_evidence_field["unknown"] = 0
        duplicate_evidence_key = evidence_bytes.replace(
            b'{"attempts":',
            b'{"attempts":[],"attempts":',
            1,
        )
        for malformed in (
            b" " + evidence_bytes,
            canonical(wrong_evidence_kind),
            canonical(wrong_evidence_schema),
            canonical(extra_evidence_field),
            duplicate_evidence_key,
            bytearray(evidence_bytes),
        ):
            with pytest.raises((TypeError, ValueError)):
                generation_response_evidence_from_bytes(malformed)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_attempt_limit_preserves_raw_abort_evidence(self):
        bridge = _make_bridge(max_resubmit_retries=3)
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.1, 10)], "abort")
        )
        req = _make_request(input_ids=[1, 2], max_new_tokens=100)
        req.rid = "trace-attempt-limit"

        resp, trace = await bridge.agenerate_with_trace(req)

        assert resp.stop_reason == "length"
        assert [attempt.attempt_index for attempt in trace.attempts] == [0, 1, 2]
        assert [attempt.remaining_new_tokens for attempt in trace.attempts] == [
            100,
            99,
            98,
        ]
        assert [attempt.raw_stop_reason for attempt in trace.attempts] == [
            "abort",
            "abort",
            "abort",
        ]
        assert trace.final_output_token_ids == (10, 10, 10)
        assert trace.effective_max_new_tokens == 100
        assert trace.configured_attempt_limit == 3
        assert trace.final_stop_reason == "length"
        assert trace.terminal_reason == "attempt_limit"

    @pytest.mark.asyncio
    async def test_budget_exhaustion_is_distinct_from_attempt_limit(self):
        bridge = _make_bridge(max_resubmit_retries=10)
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.1, 10), (-0.2, 11)], "abort")
        )
        req = _make_request(input_ids=[1, 2], max_new_tokens=4)
        req.rid = "trace-budget"

        resp, trace = await bridge.agenerate_with_trace(req)

        assert resp.stop_reason == "length"
        assert [attempt.remaining_new_tokens for attempt in trace.attempts] == [4, 2]
        assert [attempt.raw_stop_reason for attempt in trace.attempts] == [
            "abort",
            "abort",
        ]
        assert trace.final_output_token_ids == (10, 11, 10, 11)
        assert trace.effective_max_new_tokens == 4
        assert trace.configured_attempt_limit == 10
        assert trace.final_stop_reason == "length"
        assert trace.terminal_reason == "budget_exhausted"

    @pytest.mark.asyncio
    async def test_traced_api_rejects_backend_output_beyond_remaining_budget(self):
        bridge = _make_bridge()
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response(
                [(-0.1, token_id) for token_id in range(9)],
                "stop",
            )
        )
        req = _make_request(input_ids=[1, 2], max_new_tokens=8)
        req.rid = "trace-over-budget"

        with pytest.raises(
            GenerationTraceValidationError,
            match="cannot exceed remaining_new_tokens",
        ):
            await bridge.agenerate_with_trace(req)

        legacy = await bridge.agenerate(req)
        assert legacy.output_tokens == list(range(9))

    @pytest.mark.asyncio
    async def test_trace_snapshots_versions_before_send_and_after_receive(self):
        call_count = 0
        bridge = _make_bridge(version=1)

        async def mock_send(http_req, **kwargs):
            nonlocal call_count
            call_count += 1
            bridge.set_version(call_count + 1)
            if call_count == 1:
                return _make_sglang_response([(-0.5, 100)], "abort")
            return _make_sglang_response([(-0.2, 200)], "stop")

        bridge._send_request = mock_send
        req = _make_request(input_ids=[1, 2], max_new_tokens=5)
        req.rid = "trace-versions"

        resp, trace = await bridge.agenerate_with_trace(req)

        assert resp.output_versions == [2, 3]
        assert trace.initial_client_version == 1
        assert trace.final_client_version == 3
        assert [attempt.client_version_before_send for attempt in trace.attempts] == [
            1,
            2,
        ]
        assert [attempt.client_version_after_receive for attempt in trace.attempts] == [
            2,
            3,
        ]
        assert [attempt.output_version_label for attempt in trace.attempts] == [2, 3]

    @pytest.mark.asyncio
    async def test_version_epoch_exposes_aba_changes_hidden_by_equal_labels(self):
        bridge = _make_bridge(version=17)

        async def change_and_restore_version(http_req, **kwargs):
            bridge.set_version(18)
            bridge.set_version(17)
            return _make_sglang_response([(-0.5, 100)], "stop")

        bridge._send_request = change_and_restore_version
        req = _make_request(input_ids=[1], max_new_tokens=2)
        req.rid = "trace-version-aba"

        response, trace = await bridge.agenerate_with_trace(req)

        assert response.output_versions == [17]
        assert trace.initial_client_version == trace.final_client_version == 17
        assert trace.initial_client_version_epoch == 0
        assert trace.final_client_version_epoch == 2
        attempt = trace.attempts[0]
        assert attempt.client_version_before_send == 17
        assert attempt.client_version_after_receive == 17
        assert attempt.output_version_label == 17
        assert attempt.client_version_epoch_before_send == 0
        assert attempt.client_version_epoch_after_receive == 2
        assert attempt.output_version_epoch == 2

        forged = deepcopy(trace)
        object.__setattr__(forged, "final_client_version", 18)
        with pytest.raises(ValueError, match="one client version epoch"):
            generation_physical_trace_bytes(forged)

    @pytest.mark.asyncio
    async def test_output_version_label_is_observed_after_backend_parsing(self):
        backend = _ThirdPartyBackendWithoutTrace()
        bridge = _make_bridge(backend=backend, version=4)
        original_parse = backend.parse_generation_response

        async def mock_send(http_req, **kwargs):
            bridge.set_version(5)
            return _make_sglang_response([(-0.5, 100), (-0.3, 101)], "stop")

        def parse_and_advance_version(response):
            bridge.set_version(6)
            return original_parse(response)

        backend.parse_generation_response = parse_and_advance_version
        bridge._send_request = mock_send
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=3)
        req.rid = "trace-parser-version"

        resp, trace = await bridge.agenerate_with_trace(req)

        attempt = trace.attempts[0]
        assert attempt.client_version_before_send == 4
        assert attempt.client_version_after_receive == 5
        assert attempt.output_version_label == 6
        assert trace.final_client_version == 6
        reconstructed_versions = [
            attempt.output_version_label
            for attempt in trace.attempts
            for _ in attempt.output_token_ids
        ]
        assert reconstructed_versions == [6, 6]
        assert reconstructed_versions == resp.output_versions
        validate_generation_physical_trace_response(resp, trace)

        resp.output_versions = [5, 5]
        with pytest.raises(ValueError, match="response.output_versions"):
            validate_generation_physical_trace_response(resp, trace)

    @pytest.mark.asyncio
    async def test_output_version_label_preserves_legacy_post_extend_read_point(self):
        class VersionChangingList(list):
            def __init__(self, values, callback):
                super().__init__(values)
                self._callback = callback

            def __iter__(self):
                self._callback()
                return super().__iter__()

        async def run(*, traced: bool):
            backend = _ThirdPartyBackendWithoutTrace()
            bridge = _make_bridge(backend=backend, version=1)

            def parse_with_version_changes(response):
                return HttpGenerationResult(
                    output_tokens=VersionChangingList(
                        [100], lambda: bridge.set_version(8)
                    ),
                    output_logprobs=VersionChangingList(
                        [-0.5], lambda: bridge.set_version(9)
                    ),
                    stop_reason="stop",
                )

            backend.parse_generation_response = parse_with_version_changes
            bridge._send_request = AsyncMock(
                return_value=_make_sglang_response([(-0.5, 100)], "stop")
            )
            req = _make_request(input_ids=[1, 2, 3], max_new_tokens=3)
            req.rid = f"trace-post-extend-{traced}"
            if traced:
                return await bridge.agenerate_with_trace(req)
            return await bridge.agenerate(req), None

        legacy_resp, _ = await run(traced=False)
        traced_resp, trace = await run(traced=True)

        assert legacy_resp.output_versions == [9]
        assert traced_resp.output_versions == [9]
        assert trace is not None
        assert trace.attempts[0].output_version_label == 9

    @pytest.mark.parametrize(
        "field_name,error_type",
        [
            ("input_tokens", ValueError),
            ("output_tokens", ValueError),
            ("output_versions", TypeError),
        ],
    )
    @pytest.mark.parametrize("forged_value", [True, 1.0], ids=["bool", "float"])
    @pytest.mark.asyncio
    async def test_response_binding_rejects_equal_but_wrongly_typed_integers(
        self,
        field_name: str,
        error_type: type[Exception],
        forged_value: object,
    ):
        bridge = _make_bridge(version=1)
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.5, 1)], "stop")
        )
        req = _make_request(input_ids=[1], max_new_tokens=1)
        req.rid = f"trace-wrong-type-{field_name}"
        resp, trace = await bridge.agenerate_with_trace(req)
        validate_generation_physical_trace_response(resp, trace)

        setattr(resp, field_name, [forged_value])

        with pytest.raises(error_type, match=rf"response\.{field_name}"):
            validate_generation_physical_trace_response(resp, trace)

    @pytest.mark.asyncio
    async def test_vllm_text_trace_snapshots_submitted_prompt_token_ids(self):
        bridge = _make_bridge(backend=VLLMBridgeBackend())
        bridge._send_request = AsyncMock(
            return_value=_make_vllm_response([100], [-0.5], "stop")
        )
        req = _make_request(input_ids=[11, 12], max_new_tokens=3)
        req.rid = "trace-vllm-text"

        _, trace = await bridge.agenerate_with_trace(req)

        assert trace.backend_kind == "VLLMBridgeBackend"
        assert trace.attempts[0].endpoint == "/v1/completions"
        assert trace.attempts[0].submitted_input_token_ids == (11, 12)

    @pytest.mark.asyncio
    async def test_vllm_vision_trace_marks_physical_token_ids_unobservable(self):
        bridge = _make_bridge(backend=VLLMBridgeBackend())
        bridge._send_request = AsyncMock(
            return_value=_make_vllm_response([100], [-0.5], "stop")
        )
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=3)
        req.rid = "trace-vllm-vision"
        req.vision_msg_vllm = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "placeholder"},
                        },
                    ],
                }
            ]
        ]
        req.image_data = ["iVBOR-test-data"]
        original_vision_messages = deepcopy(req.vision_msg_vllm)
        original_image_data = deepcopy(req.image_data)

        _, trace = await bridge.agenerate_with_trace(req)

        assert trace.attempts[0].endpoint == "/v1/chat/completions"
        assert trace.request_input_token_ids == (1, 2, 3)
        assert trace.attempts[0].submitted_input_token_ids is None
        assert req.vision_msg_vllm == original_vision_messages
        assert req.image_data == original_image_data

    @pytest.mark.asyncio
    async def test_third_party_backend_without_optional_trace_hook_remains_compatible(
        self,
    ):
        backend = _ThirdPartyBackendWithoutTrace()
        assert not isinstance(backend, TraceableInfBridgeBackend)

        bridge = _make_bridge(backend=backend)
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.5, 100)], "stop")
        )
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=3)
        req.rid = "trace-third-party"

        resp, trace = await bridge.agenerate_with_trace(req)

        assert resp.output_tokens == [100]
        assert trace.backend_kind == "_ThirdPartyBackendWithoutTrace"
        assert trace.attempts[0].submitted_input_token_ids is None
        assert trace.effective_max_new_tokens == 3
        assert trace.configured_attempt_limit == 20

    @pytest.mark.asyncio
    async def test_trace_preserves_opaque_metadata_accepted_by_legacy_path(self):
        raw_response = _make_sglang_response([(-0.5, 100)], "stop")
        bridge = _make_bridge()
        bridge._send_request = AsyncMock(
            side_effect=[deepcopy(raw_response), deepcopy(raw_response)]
        )
        req = _make_request(
            input_ids=[1, 2, 3],
            max_new_tokens=3,
            metadata={"opaque": _OpaqueMetadata()},
        )
        req.rid = "trace-opaque-metadata"

        legacy_response = await bridge.agenerate(req)
        traced_response, trace = await bridge.agenerate_with_trace(req)

        assert legacy_response.output_tokens == [100]
        assert traced_response.output_tokens == [100]
        assert trace.request_id == "trace-opaque-metadata"
        assert bridge._send_request.await_count == 2

    @pytest.mark.asyncio
    async def test_trace_records_effective_post_method_for_non_get_request_label(self):
        backend = _ThirdPartyBackendWithoutTrace()
        original_build = backend.build_generation_request

        def build_with_put_method(req, with_lora, version=-1):
            http_req = original_build(req, with_lora, version)
            http_req.method = "PUT"
            return http_req

        backend.build_generation_request = build_with_put_method
        bridge = _make_bridge(backend=backend)
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.5, 100)], "stop")
        )

        _, trace = await bridge.agenerate_with_trace(
            _make_request(input_ids=[1, 2, 3], max_new_tokens=3)
        )

        assert bridge._send_request.await_args.args[0].method == "PUT"
        assert trace.attempts[0].method == "POST"

    @pytest.mark.asyncio
    async def test_traced_api_rejects_corrupt_prepared_budget_before_transport(self):
        backend = _ThirdPartyBackendWithoutTrace()
        original_patch = backend.patch_generation_request

        def patch_with_wrong_budget(
            http_req,
            req,
            accumulated_tokens,
            remaining_tokens,
        ):
            original_patch(http_req, req, accumulated_tokens, remaining_tokens)
            http_req.payload["sampling_params"]["max_new_tokens"] = remaining_tokens - 1

        backend.patch_generation_request = patch_with_wrong_budget
        bridge = _make_bridge(backend=backend)
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.5, 100)], "stop")
        )
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=3)

        with pytest.raises(ValueError, match="patched backend max_new_tokens"):
            await bridge.agenerate_with_trace(req)
        bridge._send_request.assert_not_awaited()

        legacy_resp = await bridge.agenerate(req)
        assert legacy_resp.output_tokens == [100]
        assert legacy_resp.stop_reason == "stop"
        bridge._send_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_traced_api_propagates_transport_runtime_error_unchanged(self):
        transport_error = RuntimeError("transport failed")
        bridge = _make_bridge()
        bridge._send_request = AsyncMock(side_effect=transport_error)

        with pytest.raises(RuntimeError, match="transport failed") as caught:
            await bridge.agenerate_with_trace(_make_request(max_new_tokens=3))

        assert caught.value is transport_error
        bridge._send_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_traced_api_propagates_backend_parse_error_unchanged(self):
        parse_error = ValueError("backend parse failed")
        backend = _ThirdPartyBackendWithoutTrace()

        def fail_parse(response):
            raise parse_error

        backend.parse_generation_response = fail_parse
        bridge = _make_bridge(backend=backend)
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.5, 100)], "stop")
        )

        with pytest.raises(ValueError, match="backend parse failed") as caught:
            await bridge.agenerate_with_trace(_make_request(max_new_tokens=3))

        assert caught.value is parse_error
        bridge._send_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_traced_api_propagates_cancellation_unchanged(self):
        cancellation = asyncio.CancelledError("generation cancelled")
        bridge = _make_bridge()
        bridge._send_request = AsyncMock(side_effect=cancellation)

        with pytest.raises(asyncio.CancelledError) as caught:
            await bridge.agenerate_with_trace(_make_request(max_new_tokens=3))

        assert caught.value is cancellation
        bridge._send_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_trace_snapshots_request_and_attempt_config_before_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        call_count = 0
        bridge = _make_bridge(max_resubmit_retries=2)
        original_pause_state = bridge.pause_state
        replacement_pause_state = type(original_pause_state)()
        req = _make_request(input_ids=[1, 2], max_new_tokens=5)
        req.rid = "trace-original-request"
        observed_timeouts: list[float] = []
        observed_sleeps: list[float] = []
        pause_check_count = 0

        async def original_is_paused():
            nonlocal pause_check_count
            pause_check_count += 1
            return pause_check_count == 2

        replacement_is_paused = AsyncMock(
            side_effect=AssertionError("replacement pause state must not be observed")
        )

        async def record_sleep(delay: float):
            observed_sleeps.append(delay)

        monkeypatch.setattr(original_pause_state, "is_paused", original_is_paused)
        monkeypatch.setattr(
            replacement_pause_state,
            "is_paused",
            replacement_is_paused,
        )
        monkeypatch.setattr(asyncio, "sleep", record_sleep)

        async def mutate_original_state_during_send(http_req, **kwargs):
            nonlocal call_count
            call_count += 1
            observed_timeouts.append(kwargs["timeout"])
            if call_count == 1:
                req.rid = "trace-mutated-request"
                req.input_ids[:] = [9, 9, 9]
                bridge.max_resubmit_retries = 99
                bridge.request_timeout = 1e-6
                bridge.resubmit_wait = 999.0
                bridge.pause_state = replacement_pause_state
                return _make_sglang_response([(-0.5, 100)], "abort")
            return _make_sglang_response([(-0.2, 200)], "stop")

        bridge._send_request = mutate_original_state_during_send

        resp, trace = await bridge.agenerate_with_trace(req)

        assert call_count == 2
        assert req.rid == "trace-mutated-request"
        assert req.input_ids == [9, 9, 9]
        assert bridge.max_resubmit_retries == 99
        assert observed_timeouts == [120.0, 120.0]
        assert observed_sleeps == [0.01]
        assert replacement_is_paused.await_count == 0
        assert resp.input_tokens == [1, 2]
        assert resp.output_tokens == [100, 200]
        assert trace.request_id == "trace-original-request"
        assert trace.request_input_token_ids == (1, 2)
        assert trace.configured_attempt_limit == 2
        assert trace.effective_max_new_tokens == 5
        assert [attempt.submitted_input_token_ids for attempt in trace.attempts] == [
            (1, 2),
            (1, 2, 100),
        ]

    @pytest.mark.parametrize(
        "backend,payload_key",
        [
            (SGLangBridgeBackend(), "input_ids"),
            (VLLMBridgeBackend(), "prompt"),
        ],
        ids=["sglang", "vllm"],
    )
    @pytest.mark.parametrize("bad_token", [True, -1], ids=["bool", "negative"])
    def test_builtin_backend_snapshot_rejects_invalid_token_ids(
        self,
        backend: SGLangBridgeBackend | VLLMBridgeBackend,
        payload_key: str,
        bad_token: object,
    ):
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=3)
        http_req = backend.build_generation_request(req, with_lora=False, version=0)
        http_req.payload[payload_key] = [bad_token]

        with pytest.raises(ValueError, match="non-negative ints"):
            backend.snapshot_generation_input_ids(http_req)

    @pytest.mark.asyncio
    async def test_zero_attempt_limit_produces_valid_zero_attempt_trace(self):
        bridge = _make_bridge(max_resubmit_retries=0, version=9)
        bridge._send_request = AsyncMock()
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=5)
        req.rid = "trace-zero-attempt-limit"

        resp, trace = await bridge.agenerate_with_trace(req)

        bridge._send_request.assert_not_awaited()
        assert resp.output_tokens == []
        assert resp.output_logprobs == []
        assert resp.stop_reason == "length"
        assert trace.effective_max_new_tokens == 5
        assert trace.configured_attempt_limit == 0
        assert trace.attempts == ()
        assert trace.final_output_token_ids == ()
        assert trace.initial_client_version == 9
        assert trace.final_client_version == 9
        assert trace.final_stop_reason == "length"
        assert trace.terminal_reason == "attempt_limit"
        assert json.loads(generation_physical_trace_bytes(trace))["attempts"] == []

    @pytest.mark.asyncio
    async def test_trace_validation_rejects_forged_or_wrongly_typed_evidence(self):
        bridge = _make_bridge()
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.5, 100)], "stop")
        )
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=3)
        req.rid = "trace-validation"
        _, trace = await bridge.agenerate_with_trace(req)

        with pytest.raises(ValueError, match="schema_version must be 2"):
            replace(trace, schema_version=1)
        with pytest.raises(ValueError, match="final_output_token_ids"):
            generation_physical_trace_bytes(
                replace(trace, final_output_token_ids=(999,))
            )
        with pytest.raises(TypeError, match="initial_client_version"):
            generation_physical_trace_bytes(replace(trace, initial_client_version=True))
        with pytest.raises(ValueError, match="prepared_request_json_sha256"):
            bad_attempt = replace(
                trace.attempts[0], prepared_request_json_sha256="not-a-sha256"
            )
            generation_physical_trace_bytes(replace(trace, attempts=(bad_attempt,)))
        with pytest.raises(ValueError, match="remaining_new_tokens"):
            generation_physical_trace_bytes(replace(trace, effective_max_new_tokens=4))
        with pytest.raises(ValueError, match="configured_attempt_limit"):
            generation_physical_trace_bytes(replace(trace, configured_attempt_limit=0))
        with pytest.raises(ValueError, match="remaining_new_tokens"):
            bad_attempt = replace(trace.attempts[0], remaining_new_tokens=2)
            generation_physical_trace_bytes(replace(trace, attempts=(bad_attempt,)))
        with pytest.raises(ValueError, match="configured_attempt_limit"):
            generation_physical_trace_bytes(
                replace(
                    trace,
                    attempts=(),
                    final_client_version=trace.initial_client_version,
                    final_output_token_ids=(),
                    final_stop_reason="length",
                    terminal_reason="attempt_limit",
                )
            )

        with pytest.raises(FrozenInstanceError):
            trace.final_stop_reason = "length"
        with pytest.raises(FrozenInstanceError):
            trace.attempts[0].raw_stop_reason = "abort"

    @pytest.mark.asyncio
    async def test_legacy_agenerate_still_returns_plain_model_response(self):
        bridge = _make_bridge()
        bridge._send_request = AsyncMock(
            return_value=_make_sglang_response([(-0.5, 100)], "stop")
        )

        resp = await bridge.agenerate(_make_request(input_ids=[1, 2, 3]))

        assert isinstance(resp, ModelResponse)
        assert not isinstance(resp, tuple)
        assert not hasattr(resp, "generation_trace")
