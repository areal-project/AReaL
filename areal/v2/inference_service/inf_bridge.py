# SPDX-License-Identifier: Apache-2.0

"""InfBridge -- HTTP client implementing _AsyncGenerateEngine protocol.

Supports pluggable backends (SGLang, vLLM, etc.) via the InfBridgeBackend protocol.
InfBridge owns the HTTP transport and pause/abort/resubmit loop; the backend
translates between ModelRequest / raw JSON and endpoint-specific payloads.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import time
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
import numpy as np

from areal.api.io_struct import HttpRequest, ModelRequest, ModelResponse
from areal.utils import logging
from areal.v2.inference_service.backend import (
    InfBridgeBackend,
    TraceableInfBridgeBackend,
)
from areal.v2.inference_service.client_trace import (
    GenerationAttemptTrace,
    GenerationPhysicalTrace,
    GenerationResponseEvidence,
    ParsedResponseJSONEvidence,
    canonical_parsed_response_json_bytes,
    generation_physical_trace_bytes,
    generation_physical_trace_sha256,
    parsed_response_json_sha256,
    prepared_request_json_sha256,
    validate_generation_physical_trace_response,
)

__all__ = [
    "GenerationAttemptTrace",
    "GenerationPhysicalTrace",
    "GenerationResponseEvidence",
    "InfBridge",
    "generation_physical_trace_bytes",
    "generation_physical_trace_sha256",
    "validate_generation_physical_trace_response",
]

if TYPE_CHECKING:
    from areal.v2.inference_service.data_proxy.pause import PauseState

_StopReason = Literal["length", "stop", "tool_calls", "abort"]
_TerminalReason = Literal[
    "backend_stop",
    "backend_tool_calls",
    "backend_length",
    "budget_exhausted",
    "attempt_limit",
]

logger = logging.getLogger("InferenceInfBridge")


class InfBridge:
    """Backend-agnostic HTTP client implementing ``_AsyncGenerateEngine`` protocol.

    All inference-server specifics are delegated to *backend*
    (:class:`InfBridgeBackend`).  InfBridge owns:

    * HTTP transport (send / receive)
    * Pause / resume coordination via :class:`PauseState`
    * Abort → resubmit loop with token accumulation
    * Version tracking

    Parameters
    ----------
    backend:
        An object satisfying :class:`InfBridgeBackend`.
    backend_addr:
        Base URL of the inference server (e.g. ``http://localhost:30000``).
    pause_state:
        Shared pause flag (set by the weight-update path).
    request_timeout:
        HTTP timeout per generation call (seconds).
    max_resubmit_retries:
        Maximum number of abort → resubmit cycles.
    resubmit_wait:
        Sleep duration (seconds) between pause-state polls.
    version:
        Initial weight version.
    """

    def __init__(
        self,
        backend: InfBridgeBackend,
        backend_addr: str,
        pause_state: PauseState,
        request_timeout: float = 120.0,
        max_resubmit_retries: int = 20,
        resubmit_wait: float = 0.5,
        version: int = 0,
    ) -> None:
        self.backend = backend
        self.backend_addr = backend_addr.rstrip("/")
        self.pause_state = pause_state
        self.request_timeout = request_timeout
        self.max_resubmit_retries = max_resubmit_retries
        self.resubmit_wait = resubmit_wait
        self._version = version
        self._version_epoch = 0
        self._client = httpx.AsyncClient(timeout=request_timeout)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # -- version tracking ---------------------------------------------------

    def set_version(self, version: int) -> None:
        self._version = version
        self._version_epoch += 1

    def get_version(self) -> int:
        return self._version

    def get_version_epoch(self) -> int:
        """Return a monotonic count of local :meth:`set_version` calls."""

        return self._version_epoch

    # -- pause / resume -----------------------------------------------------

    async def pause(self) -> None:
        """Pause generation by setting pause_state and calling the backend."""
        await self.pause_state.set_paused(True)
        http_req = self.backend.get_pause_request()
        await self._send_request(http_req, timeout=10.0)
        logger.info("Pause request sent to %s", self.backend_addr)

    async def resume(self) -> None:
        """Resume generation by calling the backend and clearing pause_state."""
        http_req = self.backend.get_resume_request()
        await self._send_request(http_req, timeout=10.0)
        await self.pause_state.set_paused(False)
        logger.info("Resume request sent to %s", self.backend_addr)

    async def offload(self) -> None:
        """Offload model memory on the backend inference server."""
        http_req = self.backend.get_offload_request()
        await self._send_request(http_req, timeout=30.0)
        logger.info("Offload request sent to %s", self.backend_addr)

    async def onload(self, tags: list[str] | None = None) -> None:
        """Reload model memory on the backend inference server."""
        http_req = self.backend.get_onload_request(tags=tags)
        await self._send_request(http_req, timeout=30.0)
        logger.info("Onload request sent to %s", self.backend_addr)

    # -- HTTP transport (shared across all backends) -------------------------

    async def _send_request(
        self,
        http_req: HttpRequest,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send an :class:`HttpRequest` and return the parsed JSON body.

        Parameters
        ----------
        http_req:
            The endpoint + payload to send.
        timeout:
            Per-request timeout override.  Falls back to
            ``self.request_timeout``.

        Returns
        -------
        dict
            Parsed JSON response.

        Raises
        ------
        httpx.HTTPStatusError
            On non-2xx responses.
        """
        _timeout = timeout if timeout is not None else self.request_timeout
        url = f"{self.backend_addr}{http_req.endpoint}"
        if http_req.method == "GET":
            resp = await self._client.get(url, timeout=_timeout)
        else:
            resp = await self._client.post(url, json=http_req.payload, timeout=_timeout)
        if resp.status_code >= 400:
            body = resp.text[:500]
            logger.error("Backend returned %d for %s: %s", resp.status_code, url, body)
        resp.raise_for_status()
        return resp.json()

    # -- main generation with pause/abort/resubmit --------------------------

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Generate a response for *req* via the configured backend.

        Implements the ``_AsyncGenerateEngine`` protocol.
        Handles the pause → abort → resubmit loop transparently.
        """
        result = await self._agenerate(
            req,
            collect_trace=False,
            collect_response_evidence=False,
        )
        return cast(ModelResponse, result)

    async def agenerate_with_trace(
        self,
        req: ModelRequest,
    ) -> tuple[ModelResponse, GenerationPhysicalTrace]:
        """Generate and return immutable client-side physical-call evidence.

        Unlike :meth:`agenerate`, this method content-addresses each patched
        request and parsed response and records every abort/resubmit attempt.
        It does not claim to attest the remote model or its weights.

        A trace is returned only after successful transport, parsing, and
        validation.  Transport errors, parser errors, cancellation, and
        fail-closed trace checks propagate without being converted to a
        ``length`` response.
        """
        result = await self._agenerate(
            req,
            collect_trace=True,
            collect_response_evidence=False,
        )
        return cast(tuple[ModelResponse, GenerationPhysicalTrace], result)

    async def agenerate_with_trace_and_response_evidence(
        self,
        req: ModelRequest,
    ) -> tuple[ModelResponse, GenerationPhysicalTrace, GenerationResponseEvidence]:
        """Generate with a compact trace plus replayable response JSON sidecar.

        The evidence sidecar can contain generated text and server metadata, so
        callers should persist it only under prompt-equivalent data controls.
        It supports offline parser replay but is still client-local evidence,
        not remote-server or model attestation.
        """

        result = await self._agenerate(
            req,
            collect_trace=True,
            collect_response_evidence=True,
        )
        return cast(
            tuple[
                ModelResponse,
                GenerationPhysicalTrace,
                GenerationResponseEvidence,
            ],
            result,
        )

    async def _agenerate(
        self,
        req: ModelRequest,
        *,
        collect_trace: bool,
        collect_response_evidence: bool,
    ) -> (
        ModelResponse
        | tuple[ModelResponse, GenerationPhysicalTrace]
        | tuple[
            ModelResponse,
            GenerationPhysicalTrace,
            GenerationResponseEvidence,
        ]
    ):
        """Shared generation loop; tracing is opt-in to preserve legacy behavior."""
        if collect_response_evidence and not collect_trace:
            raise ValueError("response evidence requires physical tracing")
        # The traced API snapshots mutable request/configuration state.  The
        # legacy API intentionally retains its existing object-sharing behavior.
        if collect_trace:
            generation_req = req.copy()
            # ModelRequest.copy() intentionally keeps several nested objects
            # shallow.  Trace mode isolates mutable JSON-facing state while
            # retaining heavyweight tokenizer/processor references.
            generation_req.metadata = copy.deepcopy(req.metadata)
            generation_req.image_data = copy.deepcopy(req.image_data)
            generation_req.vision_msg_vllm = copy.deepcopy(req.vision_msg_vllm)
        else:
            generation_req = req
        backend = self.backend
        backend_addr = self.backend_addr
        configured_attempt_limit = self.max_resubmit_retries
        initial_client_version = self._version
        initial_client_version_epoch = self._version_epoch

        if generation_req.gconfig.n_samples != 1:
            raise ValueError(
                "InfBridge only supports n_samples=1, got "
                f"{generation_req.gconfig.n_samples}"
            )
        if collect_trace and (
            type(configured_attempt_limit) is not int or configured_attempt_limit < 0
        ):
            raise ValueError(
                "traced generation requires a non-negative integer max_resubmit_retries"
            )
        if collect_trace:
            if type(generation_req.rid) is not str or not generation_req.rid:
                raise ValueError("request_id must be a non-empty str")
            if any(
                type(token_id) is not int or token_id < 0
                for token_id in generation_req.input_ids
            ):
                raise ValueError(
                    "request_input_token_ids must contain non-negative ints"
                )
            if type(initial_client_version) is not int:
                raise TypeError("initial_client_version must be an int")

        # Avoid hashing/copying trace-only evidence on the legacy path.
        request_id = generation_req.rid if collect_trace else ""
        request_input_token_ids = (
            tuple(generation_req.input_ids) if collect_trace else ()
        )
        backend_kind = type(backend).__qualname__ if collect_trace else ""
        backend_addr_sha256 = (
            hashlib.sha256(backend_addr.encode("utf-8")).hexdigest()
            if collect_trace
            else ""
        )

        # Build the initial HTTP request via the snapshotted backend/version.
        http_req = backend.build_generation_request(
            generation_req,
            with_lora=False,
            version=initial_client_version,
        )

        ori_max_new_tokens = backend.get_generation_max_new_tokens(http_req)
        if ori_max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be > 0, got {ori_max_new_tokens} "
                f"(max_tokens={generation_req.gconfig.max_tokens}, "
                f"input_len={len(generation_req.input_ids)}, "
                f"max_new_tokens={generation_req.gconfig.max_new_tokens})"
            )
        if collect_trace:
            if type(ori_max_new_tokens) is not int:
                raise TypeError("effective_max_new_tokens must be an int")

        accumulated_tokens: list[int] = []
        accumulated_logprobs: list[float] = []
        accumulated_versions: list[int] = []
        stop_reason: _StopReason | None = None
        final_routed_experts: np.ndarray | None = None
        attempt_traces: list[GenerationAttemptTrace] = []
        response_evidence_attempts: list[ParsedResponseJSONEvidence] = []
        terminal_reason: _TerminalReason | None = None

        t0 = time.monotonic()

        for _attempt in range(configured_attempt_limit):
            while await self.pause_state.is_paused():
                await asyncio.sleep(self.resubmit_wait)

            remaining = ori_max_new_tokens - len(accumulated_tokens)
            if remaining <= 0:
                stop_reason = "length"
                terminal_reason = "budget_exhausted"
                break

            backend.patch_generation_request(
                http_req,
                generation_req,
                accumulated_tokens,
                remaining,
            )

            if collect_trace:
                if self.backend_addr != backend_addr:
                    raise RuntimeError("backend_addr changed during traced generation")
                prepared_budget = backend.get_generation_max_new_tokens(http_req)
                if type(prepared_budget) is not int or prepared_budget != remaining:
                    raise ValueError(
                        "patched backend max_new_tokens does not match remaining budget"
                    )
                submitted_input_token_ids = (
                    backend.snapshot_generation_input_ids(http_req)
                    if isinstance(backend, TraceableInfBridgeBackend)
                    else None
                )
                endpoint = http_req.endpoint
                method = "GET" if http_req.method == "GET" else "POST"
                prepared_request_hash = prepared_request_json_sha256(http_req.payload)
                # This must remain the final synchronous observation before
                # entering the transport await.
                client_version_before_send = self._version
                client_version_epoch_before_send = self._version_epoch

            data = await self._send_request(http_req)
            if collect_trace:
                # Bracket the transport return before hashing/parsing locally.
                client_version_after_receive = self._version
                client_version_epoch_after_receive = self._version_epoch
                parsed_response_hash = parsed_response_json_sha256(data)
                if collect_response_evidence:
                    parsed_response_bytes = canonical_parsed_response_json_bytes(data)
            result = backend.parse_generation_response(data)

            accumulated_tokens.extend(result.output_tokens)
            accumulated_logprobs.extend(result.output_logprobs)
            # Preserve the exact legacy read point used to label output tokens.
            output_version_label = self._version
            output_version_epoch = self._version_epoch
            accumulated_versions.extend(
                [output_version_label] * len(result.output_tokens)
            )
            stop_reason = cast(_StopReason, result.stop_reason)

            if collect_trace:
                attempt_traces.append(
                    GenerationAttemptTrace(
                        attempt_index=_attempt,
                        client_version_before_send=client_version_before_send,
                        client_version_after_receive=client_version_after_receive,
                        output_version_label=output_version_label,
                        client_version_epoch_before_send=(
                            client_version_epoch_before_send
                        ),
                        client_version_epoch_after_receive=(
                            client_version_epoch_after_receive
                        ),
                        output_version_epoch=output_version_epoch,
                        remaining_new_tokens=remaining,
                        endpoint=endpoint,
                        method=method,
                        prepared_request_json_sha256=prepared_request_hash,
                        parsed_response_json_sha256=parsed_response_hash,
                        submitted_input_token_ids=submitted_input_token_ids,
                        raw_stop_reason=result.stop_reason,
                        output_token_ids=tuple(result.output_tokens),
                        output_logprob_count=len(result.output_logprobs),
                    )
                )
                if collect_response_evidence:
                    response_evidence_attempts.append(
                        ParsedResponseJSONEvidence(
                            attempt_index=_attempt,
                            parsed_response_json_sha256=parsed_response_hash,
                            canonical_json_bytes=parsed_response_bytes,
                        )
                    )

            if result.routed_experts is not None:
                if final_routed_experts is None:
                    final_routed_experts = result.routed_experts
                else:
                    final_routed_experts = np.concatenate(
                        [final_routed_experts, result.routed_experts], axis=0
                    )

            if stop_reason in ("stop", "tool_calls", "length"):
                terminal_reason = {
                    "stop": "backend_stop",
                    "tool_calls": "backend_tool_calls",
                    "length": "backend_length",
                }[stop_reason]
                break

            if len(accumulated_tokens) >= ori_max_new_tokens:
                stop_reason = "length"
                terminal_reason = "budget_exhausted"
                break

            logger.debug(
                "Abort detected, resubmit attempt %d, accumulated %d tokens",
                _attempt + 1,
                len(accumulated_tokens),
            )

        if stop_reason == "abort" or stop_reason is None:
            stop_reason = "length"
            terminal_reason = "attempt_limit"

        latency = time.monotonic() - t0
        response = ModelResponse(
            input_tokens=list(generation_req.input_ids),
            output_tokens=accumulated_tokens,
            output_logprobs=accumulated_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            tokenizer=generation_req.tokenizer,
            latency=latency,
            routed_experts=final_routed_experts,
        )

        if not collect_trace:
            return response
        if terminal_reason is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("generation completed without a terminal reason")

        trace = GenerationPhysicalTrace(
            schema_version=2,
            request_id=request_id,
            backend_kind=backend_kind,
            backend_addr_sha256=backend_addr_sha256,
            request_input_token_ids=request_input_token_ids,
            effective_max_new_tokens=ori_max_new_tokens,
            configured_attempt_limit=configured_attempt_limit,
            initial_client_version=initial_client_version,
            final_client_version=self._version,
            initial_client_version_epoch=initial_client_version_epoch,
            final_client_version_epoch=self._version_epoch,
            attempts=tuple(attempt_traces),
            final_output_token_ids=tuple(accumulated_tokens),
            final_stop_reason=stop_reason,
            terminal_reason=terminal_reason,
        )
        validate_generation_physical_trace_response(response, trace)
        if not collect_response_evidence:
            return response, trace
        response_evidence = GenerationResponseEvidence(
            schema_version=1,
            generation_trace_sha256=generation_physical_trace_sha256(trace),
            attempts=tuple(response_evidence_attempts),
        )
        return response, trace, response_evidence
