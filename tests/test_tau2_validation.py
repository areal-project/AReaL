# SPDX-License-Identifier: Apache-2.0

import pytest

from examples.tau2.message_compat import (
    install_participant_message_validation_patch,
)


class _ParticipantMessage:
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls

    def has_text_content(self) -> bool:
        return self.content is not None and bool(self.content.strip())

    def is_tool_call(self) -> bool:
        return self.tool_calls is not None

    def __str__(self) -> str:
        return f"{type(self).__name__}(content={self.content!r})"


class AssistantMessage(_ParticipantMessage):
    pass


class UserMessage(_ParticipantMessage):
    pass


def test_compatibility_validator_names_concrete_participant_class():
    install_participant_message_validation_patch(_ParticipantMessage)

    with pytest.raises(ValueError, match="^UserMessage must have either content"):
        UserMessage().validate()

    with pytest.raises(ValueError, match="^AssistantMessage must have either content"):
        AssistantMessage().validate()


def test_compatibility_validator_keeps_existing_validity_rules():
    install_participant_message_validation_patch(_ParticipantMessage)

    assert UserMessage(content="Hello").validate() is None
    assert AssistantMessage(tool_calls=[]).validate() is None


def test_tau2_validation_error_names_the_concrete_participant():
    pytest.importorskip("tau2")

    from tau2.data_model.message import (
        AssistantMessage,
        ParticipantMessageBase,
        UserMessage,
    )

    install_participant_message_validation_patch(ParticipantMessageBase)

    with pytest.raises(ValueError, match="^UserMessage must have either content"):
        UserMessage(role="user").validate()

    with pytest.raises(ValueError, match="^AssistantMessage must have either content"):
        AssistantMessage(role="assistant").validate()
