# SPDX-License-Identifier: Apache-2.0
"""Generation-only session capabilities compatible with the legacy proxy."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping


def derive_session_gateway_api_key(admin_api_key: str) -> str:
    digest = hmac.new(
        admin_api_key.encode(), b"areal-session-gateway-key-v1", hashlib.sha256
    ).hexdigest()
    return f"areal-gateway-{digest}"


def derive_session_gateway_token(admin_api_key: str, session_id: str) -> str:
    digest = hmac.new(
        derive_session_gateway_api_key(admin_api_key).encode(),
        b"areal-session-gateway-token-v1:" + session_id.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"areal-session-{digest}"


def gateway_session_id(headers: Mapping[str, str], admin_api_key: str) -> str | None:
    """Return a verified session ID, or None for ordinary session credentials.

    The caller must additionally check the session is active. Only generation
    handlers call this helper; control operations retain their original auth.
    """
    headers = {k.lower(): v for k, v in headers.items()}
    auth = headers.get("authorization", "")
    tokens = [headers.get("x-api-key", "")]
    if auth.lower().startswith("bearer "):
        tokens.append(auth[7:].strip())
    key = derive_session_gateway_api_key(admin_api_key)
    if not any(hmac.compare_digest(token, key) for token in tokens):
        return None
    session_id = headers.get("x-session-id", "").strip()
    token = headers.get("x-session-token", "").strip()
    if (
        not session_id
        or not token
        or not hmac.compare_digest(
            token, derive_session_gateway_token(admin_api_key, session_id)
        )
    ):
        raise ValueError("Missing or invalid session routing capability.")
    return session_id
