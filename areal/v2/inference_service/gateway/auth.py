# SPDX-License-Identifier: Apache-2.0

"""Authentication helpers for the inference gateway."""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass
class AuthResult:
    """Result of API key authentication."""

    key_type: str  # "admin" | "session"
    api_key: str


class AuthError(Exception):
    """Raised when auth fails."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


def extract_bearer_token(request: Request) -> str:
    """Extract API token from Authorization header.

    Raises HTTPException(401) if missing or malformed.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header. Expected 'Bearer <token>'.",
    )


def require_admin_key(request: Request, admin_api_key: str) -> str:
    """Validate that the request carries the admin API key.

    Returns the bearer token on success. Raises HTTPException(403) on failure.
    """
    token = extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_api_key):
        raise HTTPException(status_code=403, detail="Admin API key required.")
    return token
