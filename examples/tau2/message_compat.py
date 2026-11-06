# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for the pinned tau2-bench fork."""

from __future__ import annotations

from typing import Any


def validate_participant_message(message: Any) -> None:
    """Validate a tau2 participant message with its concrete class in errors.

    The pinned tau2-bench fork defines this method on
    ``ParticipantMessageBase`` but hardcodes ``AssistantMessage`` in its
    validation error. Installing this method on that base preserves the
    fork's validation rules while making errors from ``UserMessage`` accurate.
    """
    if not (message.has_text_content() or message.is_tool_call()):
        raise ValueError(
            f"{type(message).__name__} must have either content or tool calls. "
            f"Got {message}"
        )


def install_participant_message_validation_patch(
    participant_message_base: type[Any],
) -> None:
    """Patch the pinned fork's participant-message validator in place."""
    participant_message_base.validate = validate_participant_message
