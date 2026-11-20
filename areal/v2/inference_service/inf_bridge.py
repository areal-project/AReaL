# SPDX-License-Identifier: Apache-2.0

"""InfBridge -- HTTP client implementing _AsyncGenerateEngine protocol.

Supports pluggable backends (SGLang, vLLM, etc.) via the InfBridgeBackend protocol.
InfBridge owns the HTTP transport and pause/abort/resubmit loop; the backend
translates between ModelRequest / raw JSON and endpoint-specific payloads.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
import numpy as np

from areal.api.io_struct import HttpRequest
from areal.utils import logging, stats_tracker
from areal.v2.inference_service.backend import InfBridgeBackend

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest, ModelResponse
    from areal.v2.inference_service.data_proxy.pause import PauseState

_StopReason = Literal["length", "stop", "tool_calls", "abort"]

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
        return_routed_experts: bool = False,
    ) -> None:
        self.backend = backend
        self.backend_addr = backend_addr.rstrip("/")
        self.pause_state = pause_state
        self.request_timeout = request_timeout
        self.max_resubmit_retries = max_resubmit_retries
        self.resubmit_wait = resubmit_wait
        self._version = version
        self.return_routed_experts = return_routed_experts
        self._client = httpx.AsyncClient(timeout=request_timeout)
        self._control_client = httpx.AsyncClient(timeout=request_timeout)
        self._segments = asyncio.Condition()
        self._generation_held = False
        self._active_segments = 0

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
        await self._control_client.aclose()

    async def hold_generation(self) -> None:
        """Prevent new backend segments while retaining logical requests."""
        async with self._segments:
            self._generation_held = True
            await self.pause_state.set_paused(True)

    async def wait_generation_idle(self) -> None:
        """Wait until in-flight segment responses have been accumulated."""
        async with self._segments:
            if not self._generation_held:
                raise RuntimeError("Hold generation before waiting for segments")
            await self._segments.wait_for(lambda: self._active_segments == 0)

    async def rebind(self, backend_addr: str) -> None:
        """Change the backend after held segments have finished unwinding."""
        backend_addr = backend_addr.rstrip("/")
        if not backend_addr.startswith(("http://", "https://")):
            raise ValueError("backend_addr must be an HTTP(S) URL")
        async with self._segments:
            if not self._generation_held or self._active_segments:
                raise RuntimeError("Backend rebinding requires held, idle generation")
            self.backend_addr = backend_addr

    async def release_generation(self) -> None:
        """Release logical requests after the backend has been prepared."""
        async with self._segments:
            await self.pause_state.set_paused(False)
            self._generation_held = False
            self._segments.notify_all()

    async def _begin_segment(self) -> None:
        async with self._segments:
            await self._segments.wait_for(lambda: not self._generation_held)
            self._active_segments += 1

    async def _end_segment(self) -> None:
        async with self._segments:
            self._active_segments -= 1
            self._segments.notify_all()

    # -- version tracking ---------------------------------------------------

    def set_version(self, version: int) -> None:
        self._version = version

    def get_version(self) -> int:
        return self._version

    # -- pause / resume -----------------------------------------------------

    async def pause(self) -> None:
        """Pause generation by setting pause_state and calling the backend."""
        await self.pause_state.set_paused(True)
        http_req = self.backend.get_pause_request()
        await self._send_request(http_req, timeout=10.0, control=True)
        logger.info("Pause request sent to %s", self.backend_addr)

    async def resume(self) -> None:
        """Resume generation by calling the backend and clearing pause_state."""
        http_req = self.backend.get_resume_request()
        await self._send_request(http_req, timeout=10.0, control=True)
        await self.pause_state.set_paused(False)
        logger.info("Resume request sent to %s", self.backend_addr)

    async def offload(self) -> None:
        """Offload model memory on the backend inference server."""
        http_req = self.backend.get_offload_request()
        await self._send_request(http_req, timeout=30.0, control=True)
        logger.info("Offload request sent to %s", self.backend_addr)

    async def onload(self, tags: list[str] | None = None) -> None:
        """Reload model memory on the backend inference server."""
        http_req = self.backend.get_onload_request(tags=tags)
        await self._send_request(http_req, timeout=30.0, control=True)
        logger.info("Onload request sent to %s", self.backend_addr)

    # -- HTTP transport (shared across all backends) -------------------------

    async def _send_request(
        self,
        http_req: HttpRequest,
        *,
        timeout: float | None = None,
        control: bool = False,
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
        client = self._control_client if control else self._client
        if http_req.method == "GET":
            resp = await client.get(url, timeout=_timeout)
        else:
            resp = await client.post(url, json=http_req.payload, timeout=_timeout)
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
        from areal.api.io_struct import ModelResponse

        if req.gconfig.n_samples != 1:
            raise ValueError(
                f"InfBridge only supports n_samples=1, got {req.gconfig.n_samples}"
            )

        req = req.copy()
        if self.return_routed_experts:
            req.metadata["return_routed_experts"] = True
        requested_routing = bool(req.metadata.get("return_routed_experts", False))

        # Build the initial HTTP request via the backend
        http_req = self.backend.build_generation_request(
            req,
            with_lora=False,
            version=self._version,
        )

        # Extract effective max_new_tokens from the payload the backend built
        ori_max_new_tokens = self.backend.get_generation_max_new_tokens(http_req)
        if ori_max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be > 0, got {ori_max_new_tokens} "
                f"(max_tokens={req.gconfig.max_tokens}, "
                f"input_len={len(req.input_ids)}, "
                f"max_new_tokens={req.gconfig.max_new_tokens})"
            )

        accumulated_tokens: list[int] = []
        accumulated_logprobs: list[float] = []
        accumulated_versions: list[int] = []
        stop_reason: _StopReason | None = None
        final_routed_experts: np.ndarray | None = None

        ready_at = time.time()
        t0 = time.monotonic()
        first_token: float | None = None
        last_finish: float | None = None

        for _attempt in range(self.max_resubmit_retries):
            # Wait while paused (weight update in progress)
            while await self.pause_state.is_paused():
                await asyncio.sleep(self.resubmit_wait)

            await self._begin_segment()
            try:
                # Adjust max_new_tokens for already-generated tokens
                remaining = ori_max_new_tokens - len(accumulated_tokens)
                if remaining <= 0:
                    stop_reason = "length"
                    break

                # Patch the payload for this iteration (extend input, shrink budget)
                self.backend.patch_generation_request(
                    http_req,
                    req,
                    accumulated_tokens,
                    remaining,
                )

                # A weight update may complete before this segment's response arrives.
                request_version = self._version
                sent = time.monotonic() - t0
                data = await self._send_request(http_req)
                result = self.backend.parse_generation_response(data)

                if requested_routing and result.routed_experts is None:
                    if result.stop_reason != "abort":
                        raise RuntimeError(
                            "Requested return_routed_experts=True but the inference "
                            "backend response did not include routed_experts. "
                            "This usually means the SGLang server was not launched with "
                            "enable_return_routed_experts=True or the model is not a MoE "
                            "model."
                        )

                if result.output_tokens:
                    # Anchor each backend interval to its local dispatch, including
                    # waits between segments without comparing remote clocks.
                    if not accumulated_tokens and result.first_token_delay is not None:
                        first_token = sent + result.first_token_delay
                    last_finish = (
                        None
                        if result.finish_delay is None
                        else sent + result.finish_delay
                    )
                accumulated_tokens.extend(result.output_tokens)
                accumulated_logprobs.extend(result.output_logprobs)
                accumulated_versions.extend(
                    [request_version] * len(result.output_tokens)
                )
                stop_reason = cast(_StopReason, result.stop_reason)

                if result.routed_experts is not None:
                    if final_routed_experts is None:
                        final_routed_experts = result.routed_experts
                    else:
                        consumed_rows = final_routed_experts.shape[0]
                        if result.routed_experts.shape[0] < consumed_rows:
                            raise RuntimeError(
                                "SGLang routed_experts for a resubmitted request is "
                                "shorter than the previously accumulated routing rows: "
                                f"new_shape={result.routed_experts.shape}, "
                                f"accumulated_shape={final_routed_experts.shape}."
                            )
                        final_routed_experts = np.concatenate(
                            [
                                final_routed_experts,
                                result.routed_experts[consumed_rows:],
                            ],
                            axis=0,
                        )

                # Record speculative-decoding acceptance metrics from SGLang
                # meta_info. Recorded per generation segment (partial rollout may
                # issue multiple /generate calls); exported as a segment-level mean.
                if result.spec_accept_rate is not None:
                    stats_tracker.get("rollout").scalar(
                        spec_accept_rate=result.spec_accept_rate
                    )
                if result.spec_accept_length is not None:
                    stats_tracker.get("rollout").scalar(
                        spec_accept_length=result.spec_accept_length
                    )

                if stop_reason in ("stop", "tool_calls", "length"):
                    break

                if len(accumulated_tokens) >= ori_max_new_tokens:
                    stop_reason = "length"
                    break

                # stop_reason == "abort" → continue loop (resubmit)
                logger.debug(
                    "Abort detected, resubmit attempt %d, accumulated %d tokens",
                    _attempt + 1,
                    len(accumulated_tokens),
                )
            finally:
                await self._end_segment()

        # Final abort at max retries → treat as length
        if stop_reason == "abort" or stop_reason is None:
            stop_reason = "length"

        if requested_routing and accumulated_tokens and final_routed_experts is None:
            raise RuntimeError(
                "Requested return_routed_experts=True but no routed_experts were "
                "received after abort/resubmit. This usually means the SGLang "
                "server was not launched with enable_return_routed_experts=True "
                "or the model is not a MoE model."
            )

        latency = time.monotonic() - t0
        output_len = len(accumulated_tokens)
        tpot = None
        if output_len > 1 and first_token is not None and last_finish is not None:
            if last_finish >= first_token:
                tpot = (last_finish - first_token) / (output_len - 1)
        logger.info(
            "request_timing %s",
            json.dumps(
                dict(
                    request_id=req.rid,
                    session_id=req.metadata.get("areal_session_id"),
                    ready_at=ready_at,
                    completed_at=ready_at + latency,
                    input_tokens=len(req.input_ids),
                    output_tokens=output_len,
                    ttft_seconds=first_token,
                    tpot_seconds=tpot,
                    stop_reason=stop_reason,
                )
            ),
        )

        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=accumulated_tokens,
            output_logprobs=accumulated_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            tokenizer=req.tokenizer,
            latency=latency,
            ttft=float("inf") if first_token is None else first_token,
            tpot=tpot,
            routed_experts=final_routed_experts,
        )
