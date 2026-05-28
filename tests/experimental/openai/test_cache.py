from unittest.mock import MagicMock

import pytest
from openai.types.responses.response import Response
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

from tests.utils import get_model_path

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils.hf_utils import load_hf_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer once for all tests in this module."""
    return load_hf_tokenizer(
        get_model_path("/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B")
    )


@pytest.fixture
def mock_interaction(tokenizer):
    """Returns a function to create a mock InteractionWithTokenLogpReward."""

    def _create_mock_interaction(
        id: str,
        messages: list[dict] | None = None,
        response_text: str = "",
        is_completion: bool = True,
        created: int = 0,
        reward: float | None = None,
        chat_template_type: str = "concat",
        messages_delimiter_start: str = "<|im_start|>",
        messages_delimiter_end: str = "<|im_end|>",
    ):
        messages = messages or []
        mock_model_response = MagicMock()

        # Tokenize prompt messages to get input_tokens
        start, end = messages_delimiter_start, messages_delimiter_end
        prompt_strs = []
        for msg in messages:
            prompt_strs.append(f"{start}{msg['role']}\n{msg['content']}{end}\n")
        prompt_strs.append(f"{start}assistant\n")
        input_tokens = tokenizer.encode("".join(prompt_strs))

        # Tokenize response_text to get output_tokens
        output_tokens = tokenizer.encode(response_text)

        mock_model_response.input_tokens = input_tokens
        mock_model_response.output_tokens = output_tokens

        output_message = ResponseOutputMessage(
            id=id,
            role="assistant",
            status="completed",
            type="message",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text=response_text,
                    type="output_text",
                )
            ],
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=mock_model_response,
            reward=reward,
            chat_template_type=chat_template_type,
            output_message_list=[output_message.model_dump(exclude_none=True)],
        )

        if is_completion:
            completion_mock = MagicMock()
            completion_mock.id = id
            completion_mock.created = created
            interaction.completion = completion_mock
            interaction.messages = messages
            interaction.response = Response(
                id=id,
                created_at=created,
                model="None",
                object="response",
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
                output=[output_message],
            )
        else:
            assert False, "Non-completion interactions are not implemented in unittest."
        return interaction

    return _create_mock_interaction


def test_set_reward(mock_interaction):
    cache = InteractionCache()
    interaction = mock_interaction(id="1")
    cache["1"] = interaction
    cache.set_reward("1", 10.0)
    assert cache["1"].reward == 10.0


def test_set_last_reward(mock_interaction):
    cache = InteractionCache()
    cache["1"] = mock_interaction(id="1")
    cache["2"] = mock_interaction(id="2")
    cache.set_last_reward(20.0)
    assert cache["1"].reward is None
    assert cache["2"].reward == 20.0


def test_export_with_reward_discount(mock_interaction):
    cache = InteractionCache()
    cache["1"] = mock_interaction(id="1", created=1)
    cache["2"] = mock_interaction(id="2", created=2)
    cache["3"] = mock_interaction(id="3", created=3, reward=10.0)

    # Sort items for predictable order before exporting
    ordered_items = sorted(
        cache.items(),
        key=lambda item: item[1].completion.created if item[1].completion else 0,
    )
    ordered_cache = InteractionCache(ordered_items)

    # Export should trigger apply_reward_discount internally
    ordered_cache.export_interactions(style="individual", reward_discount=0.9)

    assert ordered_cache["3"].reward == pytest.approx(10.0)
    assert ordered_cache["2"].reward == pytest.approx(9.0)
    assert ordered_cache["1"].reward == pytest.approx(8.1)


def test_export_triggers_reward_discount_once(mock_interaction):
    cache = InteractionCache()
    cache["1"] = mock_interaction(id="1", reward=10.0)
    # First call triggers it
    cache.apply_reward_discount(turn_discount=0.9)
    # Second call should not trigger it again
    with pytest.raises(
        RuntimeError, match="apply_reward_discount should only be called once."
    ):
        cache.apply_reward_discount(turn_discount=0.9)


def test_export_interactions_individual_style(mock_interaction):
    cache = InteractionCache()
    cache["1"] = mock_interaction(id="1")
    cache["2"] = mock_interaction(id="2")

    exported = cache.export_interactions(style="individual")
    assert len(exported) == 2
    assert "1" in exported
    assert "2" in exported
    # Ensure parent is not built for individual style
    assert exported["1"].parent is None
    assert exported["2"].parent is None


def test_export_interactions_concat_style(mock_interaction):
    cache = InteractionCache()
    # Tree: root -> i1 -> i2 -> i4
    #             \
    #              -> i3
    # Leaves: i3, i4
    i1_messages = [{"role": "user", "content": "A"}]
    i1 = mock_interaction(
        id="1",
        messages=[{"role": "user", "content": "A"}],
        response_text="B",
        created=1,
    )
    i2_messages = (
        i1_messages
        + [o.model_dump(exclude_none=True) for o in i1.response.output]
        + [{"role": "user", "content": "C"}]
    )
    i2 = mock_interaction(
        id="2",
        messages=i2_messages,
        response_text="D",
        created=2,
    )
    i3 = mock_interaction(
        id="3",
        messages=[{"role": "user", "content": "A"}],
        response_text="E",  # Different response from i1
        created=3,
    )
    i4_messages = (
        i2_messages
        + [o.model_dump(exclude_none=True) for o in i2.response.output]
        + [{"role": "user", "content": "F"}]
    )
    i4 = mock_interaction(
        id="4",
        messages=i4_messages,
        response_text="G",
        created=4,
    )

    cache[i1.completion.id] = i1
    cache[i2.completion.id] = i2
    cache[i3.completion.id] = i3
    cache[i4.completion.id] = i4

    exported = cache.export_interactions(style="concat")

    assert len(exported) == 2
    assert i3.completion.id in exported
    assert i4.completion.id in exported
    assert exported[i4.completion.id].parent == i2
    assert exported[i3.completion.id].parent is None  # Branches from root
    assert i2.parent == i1
    assert i1.parent is None


def test_export_interactions_concat_style_output_be_refactored(mock_interaction):
    """
    Tests that if a parent's response is refactored (e.g. 'think' tokens removed),
    the child still correctly identifies the parent based on token matching.
    """
    cache = InteractionCache()
    # Parent's actual response has extra "think" tokens
    i1 = mock_interaction(
        id="1",
        messages=[{"role": "user", "content": "A"}],
        response_text="think: 123, response: B",
        created=1,
    )
    # Child's prompt uses the "clean" version of the parent's response
    i2 = mock_interaction(
        id="2",
        messages=[
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ],
        response_text="D",
        created=2,
    )

    cache[i1.completion.id] = i1
    cache[i2.completion.id] = i2

    # The token logic should not match because "think: 123, response: B" is not a prefix of the prompt for i2
    exported = cache.export_interactions(style="concat")

    assert len(exported) == 2  # Both are considered leaves
    assert i1.parent is None
    assert i2.parent is None


def test_multiple_exports_after_build(mock_interaction):
    cache = InteractionCache()
    i1 = mock_interaction(
        id="1",
        messages=[{"role": "user", "content": "A"}],
        response_text="B",
        created=1,
    )
    i2 = mock_interaction(
        id="2",
        messages=[
            {"role": "user", "content": "A"},
            i1.response.output[0].model_dump(exclude_none=True),
        ],
        response_text="C",
        created=2,
    )
    cache[i1.completion.id] = i1
    cache[i2.completion.id] = i2

    # First export: concat
    exported_concat = cache.export_interactions(style="concat")
    assert len(exported_concat) == 1
    assert i2.completion.id in exported_concat

    # Second export: individual
    exported_individual = cache.export_interactions(style="individual")
    assert len(exported_individual) == 2
    assert i1.completion.id in exported_individual
    assert i2.completion.id in exported_individual

    # Third export: concat again
    exported_concat_2 = cache.export_interactions(style="concat")
    assert len(exported_concat_2) == 1
    assert i2.completion.id in exported_concat_2


def test_export_interactions_empty_cache(mock_interaction):
    cache = InteractionCache()
    exported = cache.export_interactions(style="concat")
    assert len(exported) == 0


def test_export_interactions_invalid_style(mock_interaction):
    cache = InteractionCache()
    cache["1"] = mock_interaction(id="1")
    with pytest.raises(ValueError, match="Invalid export interactions style"):
        cache.export_interactions(style="invalid_style")


def test_export_concat_wrong_template_type(mock_interaction):
    cache = InteractionCache()
    cache["1"] = mock_interaction(id="1", chat_template_type="hf")
    with pytest.raises(
        ValueError, match="Cannot export interactions in 'concat' style"
    ):
        cache.export_interactions(style="concat")


if __name__ == "__main__":
    pytest.main([__file__])
