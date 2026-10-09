from __future__ import annotations

from examples.swe.preprocessors import MergeSystemMessages


def test_merge_system_messages_coalesces_and_moves_them_to_front() -> None:
    """Qwen should receive one leading system message from Anthropic blocks."""
    # Arrange
    messages = [
        {"role": "system", "content": "base"},
        {"role": "system", "content": "tools"},
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "working"},
        {"role": "system", "content": ""},
        {"role": "user", "content": "continue"},
    ]

    # Act
    result = MergeSystemMessages()(messages)

    # Assert
    assert result == [
        {"role": "system", "content": "base\n\ntools"},
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "continue"},
    ]


def test_merge_system_messages_leaves_conversation_without_system_unchanged() -> None:
    """The preprocessor should be a no-op when no system message is present."""
    # Arrange
    messages = [{"role": "user", "content": "hello"}]

    # Act
    result = MergeSystemMessages()(messages)

    # Assert
    assert result is messages
