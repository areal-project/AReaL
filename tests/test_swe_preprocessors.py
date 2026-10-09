from __future__ import annotations

from examples.swe.preprocessors import (
    MergeSystemMessages,
    RestoreGrayboxOutputInstructions,
)


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


def test_restore_graybox_output_instructions_is_stable_across_requests() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": "Read public/TASK.md and output-contract.json. Build stages/02-graybox.blend.",
        },
        {"role": "assistant", "content": "working"},
    ]

    preprocessor = RestoreGrayboxOutputInstructions()
    result = preprocessor(messages)
    assert result is messages
    assert result[1]["content"].endswith(
        "Write deliverables to output/ using the task-required structure."
    )
    assert preprocessor(messages) == result
    assert result[1]["content"].count("Write deliverables to output/") == 1
    assert result[0]["content"] == "system"
    assert result[2]["content"] == "working"


def test_restore_graybox_output_instructions_handles_multimodal_task() -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Rebuild this scene."}, image],
        },
        {"role": "user", "content": "Follow up."},
    ]

    preprocessor = RestoreGrayboxOutputInstructions()
    result = preprocessor(messages)

    assert result is messages
    assert result[0]["content"][:2] == [
        {"type": "text", "text": "Rebuild this scene."},
        image,
    ]
    assert result[0]["content"][2]["text"].endswith(
        "Write deliverables to output/ using the task-required structure."
    )
    assert result[1]["content"] == "Follow up."
    assert len(preprocessor(messages)[0]["content"]) == 3


def test_restore_graybox_output_instructions_keeps_existing_hint() -> None:
    content = (
        "Build the scene. Graybox public inputs are unpacked at public/. "
        "Run Blender as blender. Write deliverables to output/ using the "
        "task-required structure."
    )
    messages = [{"role": "user", "content": content}]

    RestoreGrayboxOutputInstructions()(messages)

    assert messages[0]["content"] == content
