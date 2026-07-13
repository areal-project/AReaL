# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest

from areal.v2.agent_service.session_keys import (
    MAX_SESSION_KEY_LENGTH,
    derive_session_key,
    session_key_sha256,
    validate_session_key,
)


@pytest.mark.parametrize(
    "session_key",
    (
        "a",
        "s1",
        "chat:model:user",
        "agent:default:user-1",
        "A._~-:z",
        "_safe",
        "-safe",
        ".safe",
        ":safe",
        "a" * MAX_SESSION_KEY_LENGTH,
    ),
)
def test_validate_session_key_accepts_canonical_ascii_identity(
    session_key: str,
) -> None:
    assert validate_session_key(session_key) == session_key


@pytest.mark.parametrize(
    "session_key",
    (
        "",
        " ",
        " leading",
        "trailing ",
        ".",
        "..",
        "s/b",
        "s\\b",
        "s?query",
        "s#fragment",
        "s%2Fb",
        "s%25252Fb",
        "s\x00b",
        "s\tb",
        "s\nb",
        "s\rb",
        "s\x1fb",
        "s\x7fb",
        "会话",
        "emoji-😀",
        "a" * (MAX_SESSION_KEY_LENGTH + 1),
    ),
)
def test_validate_session_key_rejects_ambiguous_or_noncanonical_identity(
    session_key: str,
) -> None:
    with pytest.raises(ValueError):
        validate_session_key(session_key)


@pytest.mark.parametrize(
    "session_key",
    (None, True, 1, b"s1", ["s1"], {"session_key": "s1"}),
)
def test_validate_session_key_rejects_non_string_identity(session_key: object) -> None:
    with pytest.raises(TypeError):
        validate_session_key(session_key)


def test_derive_session_key_preserves_safe_unambiguous_legacy_form() -> None:
    assert derive_session_key("chat", "model", "user-1") == "chat:model:user-1"
    assert derive_session_key("agent", "default", "u1") == "agent:default:u1"
    assert derive_session_key("chat", "_model", "-user") == "chat:_model:-user"


@pytest.mark.parametrize(
    ("model", "user"),
    (
        ("org/model", "user"),
        ("model", "用户"),
        ("model:variant", "user"),
        ("model", "user:group"),
        ("m" * MAX_SESSION_KEY_LENGTH, "user"),
    ),
)
def test_derive_session_key_hashes_arbitrary_business_fields_safely(
    model: str,
    user: str,
) -> None:
    first = derive_session_key("chat", model, user)
    second = derive_session_key("chat", model, user)

    assert first == second
    assert re.fullmatch(r"chat:sha256:[0-9a-f]{64}", first)
    assert validate_session_key(first) == first


def test_derive_session_key_separates_components_and_protocol_domains() -> None:
    left = derive_session_key("chat", "a:b", "c")
    right = derive_session_key("chat", "a", "b:c")
    agent = derive_session_key("agent", "a:b", "c")

    assert len({left, right, agent}) == 3


def test_derive_session_key_rejects_non_utf8_surrogate_cleanly() -> None:
    with pytest.raises(ValueError, match="valid UTF-8"):
        derive_session_key("chat", "model", "\ud800")


def test_session_key_digest_is_stable_and_requires_a_valid_identity() -> None:
    assert session_key_sha256("chat:model:user") == session_key_sha256(
        "chat:model:user"
    )
    assert session_key_sha256("chat:model:user") != session_key_sha256(
        "chat:model:other"
    )
    with pytest.raises(ValueError):
        session_key_sha256("s%2Fb")
