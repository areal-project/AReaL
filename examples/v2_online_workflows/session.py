# SPDX-License-Identifier: Apache-2.0
"""Client side of v2's generation-only session gateway capability protocol.

Keep this client usable without importing the training stack. The server verifies
the same HMAC protocol in areal.v2.inference_service.session_auth.
"""

import hashlib
import hmac


def session_gateway_key(admin_key: str) -> str:
    digest = hmac.new(
        admin_key.encode(), b"areal-session-gateway-key-v1", hashlib.sha256
    ).hexdigest()
    return "areal-gateway-" + digest


def session_gateway_headers(admin_key: str, session_id: str) -> dict[str, str]:
    digest = hmac.new(
        session_gateway_key(admin_key).encode(),
        b"areal-session-gateway-token-v1:" + session_id.encode(),
        hashlib.sha256,
    ).hexdigest()
    return {"X-Session-Id": session_id, "X-Session-Token": "areal-session-" + digest}
