# SPDX-License-Identifier: Apache-2.0

"""InfBridge -- HTTP client implementing _AsyncGenerateEngine protocol.

Supports pluggable backends (SGLang, vLLM, etc.) via the InfBridgeBackend protocol.
InfBridge owns the HTTP transport and pause/abort/resubmit loop; the backend
translates between ModelRequest / raw JSON and endpoint-specific payloads.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
import numpy as np

from areal.api.io_struct import HttpRequest
from areal.utils import logging
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
    ) -> None:
        self.backend = backend
        self.backend_addr = backend_addr.rstrip("/")
        self.pause_state = pause_state
        self.request_timeout = request_timeout
        self.max_resubmit_retries = max_resubmit_retries
        self.resubmit_wait = resubmit_wait
        self._version = version
        self._client = httpx.AsyncClient(timeout=request_timeout)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

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
        from areal.api.io_struct import ModelResponse

        if req.gconfig.n_samples != 1:
            raise ValueError(
                f"InfBridge only supports n_samples=1, got {req.gconfig.n_samples}"
            )

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

        t0 = time.monotonic()

        for _attempt in range(self.max_resubmit_retries):
            # Wait while paused (weight update in progress)
            while await self.pause_state.is_paused():
                await asyncio.sleep(self.resubmit_wait)

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

            data = await self._send_request(http_req)
            result = self.backend.parse_generation_response(data)

            accumulated_tokens.extend(result.output_tokens)
            accumulated_logprobs.extend(result.output_logprobs)
            accumulated_versions.extend([self._version] * len(result.output_tokens))
            stop_reason = cast(_StopReason, result.stop_reason)

            if result.routed_experts is not None:
                if final_routed_experts is None:
                    final_routed_experts = result.routed_experts
                else:
                    final_routed_experts = np.concatenate(
                        [final_routed_experts, result.routed_experts], axis=0
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

        # Final abort at max retries → treat as length
        if stop_reason == "abort" or stop_reason is None:
            stop_reason = "length"

        latency = time.monotonic() - t0

        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=accumulated_tokens,
            output_logprobs=accumulated_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            tokenizer=req.tokenizer,
            latency=latency,
            routed_experts=final_routed_experts,
        )
