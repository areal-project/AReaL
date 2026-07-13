# SPDX-License-Identifier: Apache-2.0

"""Canonical session identifiers shared by every Agent Service hop.

Session keys currently occupy one HTTP path segment on turn and history
endpoints.  Treating arbitrary text as a key is unsafe: every proxy/ASGI hop
may decode percent escapes again, while ``/``, ``?``, and ``#`` change URL
structure.  The allowlist below makes a validated key byte-for-byte stable at
Gateway, Router, DataProxy, and Worker boundaries.

Business fields used to *derive* a key (for example OpenAI ``model`` and
``user``) remain arbitrary UTF-8 strings.  Readable legacy keys are retained
when every component is already path-safe and unambiguous; otherwise a
domain-separated digest produces a safe deterministic identifier.
"""

from __future__ import annotations

import hashlib
import re

MAX_SESSION_KEY_LENGTH = 256

_SESSION_KEY_RE = re.compile(r"[A-Za-z0-9._~:-]+\Z")
_DERIVATION_COMPONENT_RE = re.compile(r"[A-Za-z0-9._~-]+\Z")
_DERIVATION_DOMAIN = b"areal-agent-service-session-key-v1\x00"


def validate_session_key(value: object) -> str:
    """Return a canonical path-safe session key or reject it.

    Validation never strips, normalizes, decodes, or rewrites caller input.
    That fail-closed behavior is important: silently changing a key could make
    a turn and its later close operation address different incarnations.
    """

    if type(value) is not str:
        raise TypeError("session_key must be a string")
    if not value:
        raise ValueError("session_key must not be empty")
    if len(value) > MAX_SESSION_KEY_LENGTH:
        raise ValueError(
            f"session_key must be at most {MAX_SESSION_KEY_LENGTH} ASCII characters"
        )
    if value in {".", ".."}:
        raise ValueError("session_key must not be a URL dot segment")
    if _SESSION_KEY_RE.fullmatch(value) is None:
        raise ValueError(
            "session_key may contain only ASCII letters, digits, '.', '_', '~', "
            "':', and '-'"
        )
    return value


def derive_session_key(namespace: str, *components: str) -> str:
    """Derive a stable safe key without constraining the source fields.

    The readable ``namespace:component:...`` form is used only when each
    component is independently safe and contains no colon.  This excludes the
    ambiguous pair ``('a:b', 'c')`` versus ``('a', 'b:c')``.  All other inputs
    use length-prefixed UTF-8 bytes under a versioned domain before hashing.
    """

    namespace = validate_session_key(namespace)
    if ":" in namespace:
        raise ValueError("session key namespace must not contain ':'")
    encoded_components: list[bytes] = []
    readable = True
    for component in components:
        if type(component) is not str:
            raise TypeError("session key derivation components must be strings")
        try:
            encoded = component.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(
                "session key derivation components must be valid UTF-8 strings"
            ) from error
        encoded_components.append(encoded)
        if _DERIVATION_COMPONENT_RE.fullmatch(component) is None:
            readable = False

    candidate = ":".join((namespace, *components))
    if readable and len(candidate) <= MAX_SESSION_KEY_LENGTH:
        return validate_session_key(candidate)

    digest = hashlib.sha256()
    digest.update(_DERIVATION_DOMAIN)
    for component in (namespace.encode("ascii"), *encoded_components):
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return validate_session_key(f"{namespace}:sha256:{digest.hexdigest()}")


def session_key_sha256(session_key: object) -> str:
    """Return the digest used by exact-identity lifecycle receipts."""

    return hashlib.sha256(validate_session_key(session_key).encode("ascii")).hexdigest()
