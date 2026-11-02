# SPDX-License-Identifier: Apache-2.0

"""One backend service iteration and its KV occupancy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...core import integer


@dataclass(frozen=True)
class ServiceSlice:
    request_id: str
    context_tokens: int
    query_tokens: int
    kind: Literal["prefill", "decode"]
    emit_token: bool

    def __post_init__(self) -> None:
        if not self.request_id or self.kind not in {"prefill", "decode"}:
            raise ValueError("Request identity and service kind are required")
        integer("context_tokens", self.context_tokens)
        integer("query_tokens", self.query_tokens, minimum=1)
        if self.kind == "decode" and (self.query_tokens != 1 or not self.emit_token):
            raise ValueError("A decode slice emits exactly one token")
        if type(self.emit_token) is not bool:
            raise ValueError("emit_token must be boolean")


@dataclass(frozen=True)
class ServiceState:
    """Occupied token slots, including explicitly retained prefix-cache entries."""

    kv_tokens: int

    def __post_init__(self) -> None:
        integer("kv_tokens", self.kv_tokens)
