# SPDX-License-Identifier: Apache-2.0

"""Inference bridge backend protocol.

Defines the :class:`InfBridgeBackend` protocol that concrete backends
(SGLang, vLLM, …) must satisfy.  See ``sglang/bridge.py`` and
``vllm/bridge.py`` for implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from areal.api.io_struct import HttpGenerationResult, HttpRequest

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InfBridgeBackend(Protocol):
    """Protocol for inference-server backends used by :class:`InfBridge`.

    Each method converts between AReaL domain objects and HTTP payloads
    specific to a particular inference server (SGLang, vLLM, …).
    """

    def build_generation_request(
        self,
        req: ModelRequest,
        with_lora: bool,
        version: int = -1,
    ) -> HttpRequest:
        """Translate a :class:`ModelRequest` into a backend-specific HTTP request.

        Parameters
        ----------
        req:
            The model-level generation request.
        with_lora:
            Whether to include LoRA adapter info in the payload.
        version:
            Current weight version (used for LoRA versioning).

        Returns
        -------
        HttpRequest
            An endpoint + JSON payload ready for :pymethod:`InfBridge._send_request`.
        """
        ...

    def parse_generation_response(
        self,
        response: dict[str, Any],
    ) -> HttpGenerationResult:
        """Parse a raw JSON response from the backend into an
        :class:`HttpGenerationResult`.
        """
        ...

    def get_pause_request(self) -> HttpRequest:
        """Return the HTTP request that pauses generation on the backend."""
        ...

    def get_resume_request(self) -> HttpRequest:
        """Return the HTTP request that resumes generation on the backend."""
        ...

    def get_offload_request(self) -> HttpRequest:
        """Return the HTTP request that offloads model memory on the backend."""
        ...

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Return the HTTP request that reloads model memory on the backend."""
        ...

    def get_generation_max_new_tokens(self, http_req: HttpRequest) -> int:
        """Return the current generation budget encoded in ``http_req``."""
        ...

    def patch_generation_request(
        self,
        http_req: HttpRequest,
        req: ModelRequest,
        accumulated_tokens: list[int],
        remaining_tokens: int,
    ) -> None:
        """Mutate ``http_req`` for an abort/resubmit iteration."""
        ...
