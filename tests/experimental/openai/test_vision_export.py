# SPDX-License-Identifier: Apache-2.0

"""CPU tests for multimodal support on the agent/proxy path.

Covers the three defects the vision export closes: the prompt must be
vision-expanded, the exported tensor dict must carry ``multi_modal_input`` and
``mm_token_type_ids``, and the payload must survive the proxy's HTTP
serialization without shipping one copy of ``pixel_values`` per turn.
"""

import json

import pytest
import torch

from tests.experimental.openai.vision_stubs import (
    EOS_ID,
    IMAGE_PAD_ID,
    PATCH_DIM,
    PATCHES_PER_IMAGE,
    VIDEO_PAD_ID,
    StubProcessor,
    StubTokenizer,
    image_data_uri,
    make_image,
    user_message_with_image,
)

from areal.api import ModelResponse
from areal.experimental.openai.client import (
    _check_collapsed_matches_expanded,
    _extract_images_from_messages,
    _process_vision_prompt,
    _require_processor,
    concat_vision_prompt_with_parent,
    drop_vision_pad_tokens,
    vision_pad_token_ids,
)
from areal.experimental.openai.proxy.server import (
    deserialize_interactions,
    serialize_interactions,
)
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    interactions_to_trajectory,
)
from areal.utils.hf_utils import (
    VISION_PAD_TOKENS,
    _is_multimodal_processor,
    apply_chat_template,
    media_marker_token_ids,
    vision_pad_tokens,
)
from areal.utils.image import base642image, image2base64


@pytest.fixture
def tokenizer():
    return StubTokenizer()


@pytest.fixture
def processor():
    return StubProcessor()


# ---------------------------------------------------------------------------
# Image codec
# ---------------------------------------------------------------------------


def test_base642image_round_trips_image2base64():
    """Test that base642image inverts image2base64 for RGB images."""
    original = make_image(42)

    decoded = base642image(image2base64(original))

    assert len(decoded) == 1
    assert decoded[0].mode == "RGB"
    assert decoded[0].size == original.size
    assert decoded[0].getpixel((0, 0)) == (42, 42, 42)


def test_base642image_converts_non_rgb_to_rgb():
    """Test that base642image normalises mode so processors get consistent input."""
    from PIL import Image

    grayscale = Image.new("L", (4, 4), 200)

    decoded = base642image(image2base64(grayscale))

    assert decoded[0].mode == "RGB"


def test_base642image_rejects_invalid_base64():
    with pytest.raises(ValueError, match="Invalid base64 image payload"):
        base642image("not base64!")


# ---------------------------------------------------------------------------
# Prompt expansion (Phase 1)
# ---------------------------------------------------------------------------


def test_process_vision_prompt_expands_placeholders(tokenizer, processor):
    """Test that the prompt is vision-expanded rather than left as one token."""
    image = make_image(7)
    messages = [user_message_with_image(image, "what is this?")]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )

    result = _process_vision_prompt(processor, text, image_data)

    assert result.input_ids.count(IMAGE_PAD_ID) == PATCHES_PER_IMAGE
    assert len(result.mm_token_type_ids) == len(result.input_ids)
    assert sum(result.mm_token_type_ids) == PATCHES_PER_IMAGE
    # One dict for the whole sequence, matching the non-agent vision workflows.
    assert len(result.multi_modal_input) == 1
    assert result.multi_modal_input[0]["pixel_values"].shape == (
        PATCHES_PER_IMAGE,
        PATCH_DIM,
    )
    assert result.multi_modal_input[0]["image_grid_thw"].shape == (1, 3)


def test_process_vision_prompt_accumulates_multiple_images(tokenizer, processor):
    """Test that every image in the context lands in one multi_modal_input dict."""
    messages = [
        user_message_with_image(make_image(11), "first"),
        {"role": "assistant", "content": "ok"},
        user_message_with_image(make_image(22), "second"),
    ]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )

    result = _process_vision_prompt(processor, text, image_data)

    assert len(image_data) == 2
    assert result.input_ids.count(IMAGE_PAD_ID) == 2 * PATCHES_PER_IMAGE
    assert len(result.multi_modal_input) == 1
    assert result.multi_modal_input[0]["pixel_values"].shape[0] == (
        2 * PATCHES_PER_IMAGE
    )
    assert result.multi_modal_input[0]["image_grid_thw"].shape == (2, 3)


def test_process_vision_prompt_falls_back_to_token_type_ids(tokenizer):
    class TokenTypeProcessor(StubProcessor):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result["token_type_ids"] = result.pop("mm_token_type_ids")
            return result

    messages = [user_message_with_image(make_image(7), "what is this?")]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )

    result = _process_vision_prompt(TokenTypeProcessor(), text, image_data)

    assert len(result.mm_token_type_ids) == len(result.input_ids)
    assert sum(result.mm_token_type_ids) == PATCHES_PER_IMAGE


def test_process_vision_prompt_rejects_missing_pixels(tokenizer):
    class NoPixelProcessor(StubProcessor):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result.pop("pixel_values")
            return result

    messages = [user_message_with_image(make_image(7), "what is this?")]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )

    with pytest.raises(ValueError, match="did not return pixel_values"):
        _process_vision_prompt(NoPixelProcessor(), text, image_data)


def test_require_processor_rejects_images_without_processor():
    """Test that images without a processor fail loudly instead of corrupting data."""
    with pytest.raises(ValueError, match="no processor is configured"):
        _require_processor(None)


def test_process_vision_prompt_rejects_remote_urls(processor):
    """Test that remote image URLs are refused rather than fetched by the proxy."""
    with pytest.raises(ValueError, match="Remote image URLs are not supported"):
        _process_vision_prompt(
            processor, "prompt <|image|>", ["https://example.com/cat.png"]
        )


# ---------------------------------------------------------------------------
# Tensor-dict export
# ---------------------------------------------------------------------------


def _make_interaction(
    input_ids: list[int],
    mm_token_type_ids: list[int] | None,
    multi_modal_input: list[dict] | None,
    output_len: int = 3,
    reward: float = 1.0,
) -> InteractionWithTokenLogpReward:
    response = ModelResponse(
        input_tokens=list(input_ids),
        output_tokens=[5] * output_len,
        output_logprobs=[-0.5] * output_len,
        output_versions=[0] * output_len,
    )
    interaction = InteractionWithTokenLogpReward(
        model_response=response,
        reward=reward,
        mm_token_type_ids=mm_token_type_ids,
        multi_modal_input=multi_modal_input,
    )
    interaction.interaction_id = f"chatcmpl-{id(interaction)}"
    return interaction


def test_to_tensor_dict_emits_vision_payload(tokenizer, processor):
    """Test that the exported tensor dict carries the keys the engines require."""
    image = make_image(3)
    messages = [user_message_with_image(image, "hi")]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )
    vision = _process_vision_prompt(processor, text, image_data)

    interaction = _make_interaction(
        vision.input_ids, vision.mm_token_type_ids, vision.multi_modal_input
    )
    tensor_dict = interaction.to_tensor_dict()

    assert "multi_modal_input" in tensor_dict
    assert "mm_token_type_ids" in tensor_dict
    # mm ids are prompt-scoped on the interaction and zero-extended over output.
    assert tensor_dict["mm_token_type_ids"].shape == tensor_dict["input_ids"].shape
    assert tensor_dict["mm_token_type_ids"][0, len(vision.input_ids) :].sum() == 0
    assert (
        tensor_dict["mm_token_type_ids"][0, : len(vision.input_ids)].sum()
        == PATCHES_PER_IMAGE
    )


def test_to_tensor_dict_omits_vision_keys_for_text_only():
    """Test that text-only interactions keep their original key set."""
    interaction = _make_interaction([1, 2, 3], None, None)

    tensor_dict = interaction.to_tensor_dict()

    assert "multi_modal_input" not in tensor_dict
    assert "mm_token_type_ids" not in tensor_dict


def test_to_tensor_dict_realigns_mismatched_mm_ids(caplog):
    """Test that a prompt/mm-id length mismatch warns instead of breaking shapes."""
    interaction = _make_interaction([1, 2, 3, 4], [1, 1], [{"pixel_values": None}])

    tensor_dict = interaction.to_tensor_dict()

    assert tensor_dict["mm_token_type_ids"].shape == tensor_dict["input_ids"].shape


def test_mixed_vision_and_text_turns_batch_together():
    """Test that an episode mixing text-only and image turns can be concatenated."""
    text_only = _make_interaction([1, 2, 3], None, None)
    with_image = _make_interaction(
        [1, 2, 3, IMAGE_PAD_ID],
        [0, 0, 0, 1],
        [{"pixel_values": torch.ones(PATCHES_PER_IMAGE, PATCH_DIM)}],
    )

    trajectory = interactions_to_trajectory(
        {"a": text_only, "b": with_image},
    )

    assert trajectory["input_ids"].shape[0] == 2
    assert len(trajectory["multi_modal_input"]) == 2
    # The text-only row gets an empty placeholder the engines skip over.
    assert trajectory["multi_modal_input"][0] == {}
    assert "pixel_values" in trajectory["multi_modal_input"][1]
    assert trajectory["mm_token_type_ids"].shape == trajectory["input_ids"].shape


# ---------------------------------------------------------------------------
# Serialization + dedup (Phase 2)
# ---------------------------------------------------------------------------


def test_serialization_round_trips_vision_payload():
    """Test that multi_modal_input survives the proxy's HTTP serialization."""
    pixel_values = torch.arange(PATCHES_PER_IMAGE * PATCH_DIM, dtype=torch.float32)
    pixel_values = pixel_values.reshape(PATCHES_PER_IMAGE, PATCH_DIM)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    interaction = _make_interaction(
        [1, 2, IMAGE_PAD_ID],
        [0, 0, 1],
        [{"pixel_values": pixel_values, "image_grid_thw": grid}],
    )

    restored = deserialize_interactions(serialize_interactions({"a": interaction}))

    tensor_dict = restored["a"].to_tensor_dict()
    torch.testing.assert_close(
        tensor_dict["multi_modal_input"][0]["pixel_values"], pixel_values
    )
    torch.testing.assert_close(
        tensor_dict["multi_modal_input"][0]["image_grid_thw"], grid
    )
    torch.testing.assert_close(
        tensor_dict["mm_token_type_ids"],
        torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.long),
    )


def test_serialization_dedups_repeated_images():
    """Test that a shared image is transmitted once, not once per turn."""
    pixel_values = torch.full((PATCHES_PER_IMAGE, PATCH_DIM), 4.0)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    interactions = {
        f"turn-{i}": _make_interaction(
            [1, 2, IMAGE_PAD_ID],
            [0, 0, 1],
            # Distinct tensor objects with identical content, exactly as a
            # multi-turn agent re-processing the same image would produce.
            [
                {
                    "pixel_values": pixel_values.clone(),
                    "image_grid_thw": grid.clone(),
                }
            ],
        )
        for i in range(4)
    }

    payload = serialize_interactions(interactions)

    assert payload["__format__"] == "mm-blobs-v1"
    # Two distinct tensors (pixel_values, image_grid_thw) shared by all 4 turns.
    assert len(payload["blobs"]) == 2

    restored = deserialize_interactions(payload)
    assert len(restored) == 4
    for interaction in restored.values():
        torch.testing.assert_close(
            interaction.to_tensor_dict()["multi_modal_input"][0]["pixel_values"],
            pixel_values,
        )


def test_serialization_keeps_distinct_images_separate():
    """Test that different images are not collapsed by content addressing."""
    interactions = {
        f"turn-{i}": _make_interaction(
            [1, IMAGE_PAD_ID],
            [0, 1],
            [{"pixel_values": torch.full((PATCHES_PER_IMAGE, PATCH_DIM), float(i))}],
        )
        for i in range(3)
    }

    payload = serialize_interactions(interactions)

    assert len(payload["blobs"]) == 3

    restored = deserialize_interactions(payload)
    for i, key in enumerate(sorted(restored)):
        values = restored[key].to_tensor_dict()["multi_modal_input"][0]["pixel_values"]
        assert values.unique().tolist() == [float(i)]


def test_serialization_round_trips_text_only_interactions():
    """Test that the new envelope does not disturb text-only export."""
    interaction = _make_interaction([1, 2, 3], None, None, reward=0.25)

    restored = deserialize_interactions(serialize_interactions({"a": interaction}))

    tensor_dict = restored["a"].to_tensor_dict()
    assert "multi_modal_input" not in tensor_dict
    assert restored["a"].reward == 0.25


# ---------------------------------------------------------------------------
# concat chat template (Phase 3)
# ---------------------------------------------------------------------------


def _turn_one(tokenizer, processor, image):
    messages = [user_message_with_image(image, "look at this")]
    vision = concat_vision_prompt_with_parent(messages, None, tokenizer, processor)
    output_tokens = [7, 8, EOS_ID]
    parent = InteractionWithTokenLogpReward(
        messages=messages,
        output_message_list=[{"role": "assistant", "content": "a square"}],
        model_response=ModelResponse(
            input_tokens=list(vision.input_ids),
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            stop_reason="stop",
            tokenizer=tokenizer,
        ),
        chat_template_type="concat",
        collapsed_input_ids=list(vision.collapsed_input_ids),
        mm_token_type_ids=vision.mm_token_type_ids,
        multi_modal_input=vision.multi_modal_input,
    )
    return parent, vision


def test_concat_text_path_still_splices_on_parent_tokens(tokenizer):
    """Test that extracting the splice helper left the text-only path intact.

    ``tests/experimental/openai/test_concat_prompt.py`` covers this against a
    real tokenizer, but needs a model download; this keeps the refactor honest
    on CPU.
    """
    from areal.experimental.openai.client import concat_prompt_token_ids_with_parent

    messages = [{"role": "user", "content": "first"}]
    parent_prompt = apply_chat_template(
        tokenizer, messages, add_generation_prompt=True, tokenize=True
    )
    output_tokens = [7, 8, EOS_ID]
    parent = InteractionWithTokenLogpReward(
        messages=messages,
        output_message_list=[{"role": "assistant", "content": "reply"}],
        model_response=ModelResponse(
            input_tokens=parent_prompt,
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            stop_reason="stop",
            tokenizer=tokenizer,
        ),
        chat_template_type="concat",
    )
    expected_prefix = parent_prompt + [7, 8] + [EOS_ID]

    prompt = concat_prompt_token_ids_with_parent(
        [{"role": "user", "content": "second"}], parent, tokenizer
    )

    assert prompt[: len(expected_prefix)] == expected_prefix
    assert len(prompt) > len(expected_prefix)


def test_concat_vision_prompt_preserves_parent_tokens(tokenizer, processor):
    """Test that the child's prompt keeps the parent's real tokens verbatim."""
    parent, parent_vision = _turn_one(tokenizer, processor, make_image(9))
    parent_tokens = (
        parent.model_response.input_tokens
        + parent.model_response.output_tokens_without_stop
        + [EOS_ID]
    )

    child = concat_vision_prompt_with_parent(
        [{"role": "user", "content": "and now?"}], parent, tokenizer, processor
    )

    assert child.input_ids[: len(parent_tokens)] == parent_tokens
    assert len(child.input_ids) > len(parent_vision.input_ids)


def test_concat_vision_prompt_aligns_mm_token_type_ids(tokenizer, processor):
    """Test that mm ids stay index-aligned with the spliced prompt."""
    parent, _ = _turn_one(tokenizer, processor, make_image(9))

    child = concat_vision_prompt_with_parent(
        [{"role": "user", "content": "and now?"}], parent, tokenizer, processor
    )

    assert len(child.mm_token_type_ids) == len(child.input_ids)
    flagged = {i for i, v in enumerate(child.mm_token_type_ids) if v == 1}
    pads = {i for i, t in enumerate(child.input_ids) if t == IMAGE_PAD_ID}
    assert flagged == pads


def test_concat_vision_prompt_carries_all_context_images(tokenizer, processor):
    """Test that a later turn's pixels cover images introduced in earlier turns."""
    parent, _ = _turn_one(tokenizer, processor, make_image(9))

    child = concat_vision_prompt_with_parent(
        [user_message_with_image(make_image(21), "compare with this")],
        parent,
        tokenizer,
        processor,
    )

    # Both the parent's image and the new one must be present, in order.
    assert child.multi_modal_input[0]["image_grid_thw"].shape == (2, 3)
    assert child.multi_modal_input[0]["pixel_values"].shape[0] == 2 * PATCHES_PER_IMAGE
    markers = child.multi_modal_input[0]["pixel_values"][:, 0].unique().tolist()
    assert sorted(markers) == [9.0, 21.0]


class _ThinkStrippingTokenizer(StubTokenizer):
    """Renders like a thinking model: drops <think> blocks from assistant turns.

    Qwen3.5/3.6 templates do this. It must not disturb concat, which splices the
    parent's *real* tokens and locates the seam by EOS count rather than by
    diffing successive renders.
    """

    def apply_chat_template(self, messages, **kwargs):
        stripped = []
        for message in messages:
            content = message.get("content")
            if (
                message.get("role") == "assistant"
                and isinstance(content, str)
                and "<think>" in content
            ):
                message = {
                    **message,
                    "content": content.split("</think>", 1)[-1].lstrip(),
                }
            stripped.append(message)
        return super().apply_chat_template(stripped, **kwargs)


def _thinking_parent(tokenizer, processor, image):
    """Turn 0 whose assistant output contains a <think> block."""
    messages = [user_message_with_image(image, "solve it")]
    vision = concat_vision_prompt_with_parent(messages, None, tokenizer, processor)
    answer = "<think>reasoning</think>the answer is 12"
    output_tokens = tokenizer.encode(answer) + [EOS_ID]
    return InteractionWithTokenLogpReward(
        messages=messages,
        output_message_list=[{"role": "assistant", "content": answer}],
        model_response=ModelResponse(
            input_tokens=list(vision.input_ids),
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            stop_reason="stop",
            tokenizer=tokenizer,
        ),
        chat_template_type="concat",
        collapsed_input_ids=list(vision.collapsed_input_ids),
        mm_token_type_ids=vision.mm_token_type_ids,
        multi_modal_input=vision.multi_modal_input,
    )


@pytest.mark.parametrize("strip_think", [False, True])
def test_concat_survives_a_think_stripping_template(processor, strip_think):
    """Test that concat is unaffected by templates that drop prior <think>.

    Stripping changes assistant *content* but not the number of EOS boundaries,
    so the seam still lands correctly and the parent's real tokens (including
    the think block) carry through.
    """
    tokenizer = _ThinkStrippingTokenizer() if strip_think else StubTokenizer()
    parent = _thinking_parent(tokenizer, processor, make_image(51))
    parent_tokens = (
        parent.model_response.input_tokens
        + parent.model_response.output_tokens_without_stop
        + [EOS_ID]
    )

    child = concat_vision_prompt_with_parent(
        [{"role": "user", "content": "wrong, retry"}], parent, tokenizer, processor
    )

    assert child.input_ids[: len(parent_tokens)] == parent_tokens
    assert len(child.input_ids) > len(parent_tokens)
    # mm ids stay index-aligned across the seam...
    assert len(child.mm_token_type_ids) == len(child.input_ids)
    flagged = {i for i, v in enumerate(child.mm_token_type_ids) if v == 1}
    pads = {i for i, t in enumerate(child.input_ids) if t == IMAGE_PAD_ID}
    assert flagged == pads
    # ...and the image-pad count still matches image_grid_thw.
    n_images = child.multi_modal_input[0]["image_grid_thw"].shape[0]
    assert len(pads) == n_images * PATCHES_PER_IMAGE


def test_concat_vision_prompt_without_parent_matches_direct_processing(
    tokenizer, processor
):
    """Test that the first turn of a concat episode equals plain hf processing."""
    image = make_image(5)
    messages = [user_message_with_image(image, "hello")]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    direct = _process_vision_prompt(
        processor,
        apply_chat_template(
            tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
        ),
        image_data,
    )

    concat = concat_vision_prompt_with_parent(messages, None, tokenizer, processor)

    assert concat.input_ids == direct.input_ids
    assert concat.mm_token_type_ids == direct.mm_token_type_ids


# ---------------------------------------------------------------------------
# End-to-end wiring through ArealOpenAI
# ---------------------------------------------------------------------------


class _EchoEngine:
    """Fake engine recording the request and echoing its prompt back."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.last_request = None

    async def agenerate(self, req):
        self.last_request = req
        output_tokens = [11, 12, EOS_ID]
        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=output_tokens,
            output_logprobs=[-0.1] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            stop_reason="stop",
            tokenizer=self.tokenizer,
        )


def _make_client(tokenizer, processor):
    from areal.experimental.openai import ArealOpenAI

    engine = _EchoEngine(tokenizer)
    client = ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        processor=processor,
        api_key="test",
    )
    return client, engine


@pytest.mark.asyncio
async def test_chat_completion_exports_vision_tensors(tokenizer, processor):
    """Test that a VLM chat completion lands in the cache with its vision payload."""
    client, engine = _make_client(tokenizer, processor)
    image = make_image(31)

    completion = await client.chat.completions.create(
        messages=[user_message_with_image(image, "describe")],
        model="default",
        max_completion_tokens=8,
    )

    # The engine must receive the expanded prompt plus the raw image payload,
    # and the collapsed companion the vLLM transport puts on the wire. Messages
    # no longer cross the inference boundary at all.
    assert engine.last_request.input_ids.count(IMAGE_PAD_ID) == PATCHES_PER_IMAGE
    assert len(engine.last_request.image_data) == 1
    assert engine.last_request.collapsed_input_ids.count(IMAGE_PAD_ID) == 1

    interaction = client.get_interaction(completion.id)
    assert interaction.multi_modal_input is not None
    tensor_dict = interaction.to_tensor_dict()
    assert tensor_dict["mm_token_type_ids"].shape == tensor_dict["input_ids"].shape
    assert tensor_dict["multi_modal_input"][0]["pixel_values"].shape[0] == (
        PATCHES_PER_IMAGE
    )


@pytest.mark.asyncio
async def test_chat_completion_without_images_is_unchanged(tokenizer, processor):
    """Test that text-only requests keep the tokenizer-only path."""
    client, _ = _make_client(tokenizer, processor)

    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "hello"}],
        model="default",
        max_completion_tokens=8,
    )

    interaction = client.get_interaction(completion.id)
    assert interaction.multi_modal_input is None
    assert "multi_modal_input" not in interaction.to_tensor_dict()


@pytest.mark.asyncio
async def test_chat_completion_rejects_images_without_processor(tokenizer):
    """Test that a VLM request against a text-only client fails loudly."""
    client, _ = _make_client(tokenizer, None)

    with pytest.raises(ValueError, match="no processor is configured"):
        await client.chat.completions.create(
            messages=[user_message_with_image(make_image(1), "describe")],
            model="default",
            max_completion_tokens=8,
        )


def test_data_uri_helper_produces_extractable_payload():
    """Test that the data-URI form used by agents survives image extraction."""
    image = make_image(64)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri(image)}}
            ],
        }
    ]

    image_data, tok_messages, vllm_messages = _extract_images_from_messages(messages)

    assert len(image_data) == 1
    assert base642image(image_data)[0].getpixel((0, 0)) == (64, 64, 64)
    assert tok_messages[0]["content"][0] == {"type": "image"}
    assert vllm_messages[0]["content"][0]["type"] == "image_url"


# ---------------------------------------------------------------------------
# Generated vision-pad tokens
# ---------------------------------------------------------------------------


def test_vision_pad_token_ids_discovers_both_placeholders(tokenizer):
    """Test that both reserved vision placeholders are resolved from a VLM tokenizer."""
    assert vision_pad_token_ids(tokenizer) == {IMAGE_PAD_ID, VIDEO_PAD_ID}


def test_vision_pad_token_ids_empty_without_the_conversion_api():
    """Test that a tokenizer lacking the conversion API yields no banned ids."""

    class _Bare:
        pass

    assert vision_pad_token_ids(_Bare()) == frozenset()


def test_vision_pad_tokens_prefers_what_the_processor_declares(tokenizer):
    """Test that the model's own placeholder names win over the fallback list.

    vLLM builds its expansion target from ``hf_processor.image_token``, so a ban
    derived from the same attribute aims at the token vLLM will actually expand.
    """

    class _Processor:
        image_token = "<|image_pad|>"
        video_token = "<|video_pad|>"

    assert vision_pad_tokens(tokenizer, processor=_Processor()) == (
        "<|image_pad|>",
        "<|video_pad|>",
    )


def test_vision_pad_tokens_falls_back_when_the_processor_declares_none(tokenizer):
    """Test that a processor without placeholder attributes uses the defaults."""

    class _Processor:
        pass

    assert vision_pad_tokens(tokenizer, processor=_Processor()) == VISION_PAD_TOKENS
    assert vision_pad_tokens(tokenizer) == VISION_PAD_TOKENS


def test_vision_pad_tokens_drops_names_the_tokenizer_lacks():
    """Test that an unresolvable placeholder is dropped rather than banned blind.

    ``bad_words`` takes strings and vLLM re-encodes them, so banning a name this
    vocabulary lacks would silently prohibit an ordinary sub-word sequence.
    """

    class _ForeignTokenizer:
        def convert_tokens_to_ids(self, token):
            return None

        def convert_ids_to_tokens(self, token_id):
            return None

    assert vision_pad_tokens(_ForeignTokenizer()) == ()


def test_a_bare_tokenizer_is_not_mistaken_for_a_processor():
    """Test that a text-only model is not treated as multimodal.

    ``AutoProcessor.from_pretrained`` returns the plain tokenizer for a
    text-only model instead of raising. Treating that as a processor would run
    the vision canary against something that cannot process an image.
    """

    class _Tokenizer:
        def convert_tokens_to_ids(self, token):
            return None

    assert not _is_multimodal_processor(_Tokenizer())
    assert not _is_multimodal_processor(None)


def test_a_real_processor_is_recognised():
    """Test that a processor delegating to a tokenizer is accepted."""

    class _Processor:
        tokenizer = object()
        image_processor = object()

    assert _is_multimodal_processor(_Processor())


def test_collapsed_marker_differs_from_the_expanded_pad(tokenizer):
    """Test that a Gemma3-style processor is counted by its collapsed marker.

    Gemma3 rewrites ``boi_token`` into a run built from ``image_token``, so
    counting pads in a collapsed prompt would find none.
    """

    class _Gemma3Like:
        boi_token = "<|video_pad|>"  # stands in for boi; distinct from the pad
        image_token = "<|image_pad|>"

    proc = _Gemma3Like()
    assert media_marker_token_ids(tokenizer, processor=proc) == {VIDEO_PAD_ID}
    assert vision_pad_token_ids(tokenizer, processor=proc) == {IMAGE_PAD_ID}


def test_collapsed_marker_falls_back_to_the_pad_when_undeclared(tokenizer):
    """Test that Qwen-style processors keep one token for both roles."""

    class _QwenLike:
        image_token = "<|image_pad|>"

    assert media_marker_token_ids(tokenizer, processor=_QwenLike()) == {IMAGE_PAD_ID}


def test_vision_pad_tokens_unchecked_without_a_tokenizer():
    """Test that a missing tokenizer still bans the declared placeholders.

    Nothing can be verified without a tokenizer, and banning nothing would leave
    such a request less protected than it was before the check existed.
    """
    assert vision_pad_tokens(None) == VISION_PAD_TOKENS


def test_drop_vision_pad_tokens_filters_all_parallel_sequences(tokenizer):
    """Test that tokens, logprobs and versions stay index aligned after filtering."""
    response = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[11, IMAGE_PAD_ID, 12, VIDEO_PAD_ID, EOS_ID],
        output_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
        output_versions=[0, 1, 2, 3, 4],
        stop_reason="stop",
        tokenizer=tokenizer,
    )

    drop_vision_pad_tokens(response, vision_pad_token_ids(tokenizer))

    assert response.output_tokens == [11, 12, EOS_ID]
    assert response.output_logprobs == [-0.1, -0.3, -0.5]
    assert response.output_versions == [0, 2, 4]


def test_drop_vision_pad_tokens_leaves_a_clean_response_untouched(tokenizer):
    """Test that a response without reserved placeholders is not rewritten."""
    response = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[11, 12, EOS_ID],
        output_logprobs=[-0.1, -0.2, -0.3],
        output_versions=[0, 0, 0],
        stop_reason="stop",
        tokenizer=tokenizer,
    )
    before = list(response.output_tokens)

    drop_vision_pad_tokens(response, vision_pad_token_ids(tokenizer))

    assert response.output_tokens == before
    assert response.output_logprobs == [-0.1, -0.2, -0.3]


@pytest.mark.asyncio
async def test_generated_pad_token_never_reaches_the_exported_trajectory(
    tokenizer, processor
):
    """Test that a sampled image pad is stripped before the interaction is cached.

    A generated pad would add a visual position no image tensor describes,
    desyncing the placeholder count from image_grid_thw on the next turn.
    """
    client, engine = _make_client(tokenizer, processor)

    async def _agenerate(req):
        engine.last_request = req
        output_tokens = [11, IMAGE_PAD_ID, 12, EOS_ID]
        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=output_tokens,
            output_logprobs=[-0.1] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            stop_reason="stop",
            tokenizer=tokenizer,
        )

    engine.agenerate = _agenerate

    completion = await client.chat.completions.create(
        messages=[user_message_with_image(make_image(37), "describe")],
        model="default",
        max_completion_tokens=8,
    )

    interaction = client.get_interaction(completion.id)
    output_tokens = interaction.model_response.output_tokens
    assert IMAGE_PAD_ID not in output_tokens
    assert output_tokens == [11, 12, EOS_ID]
    # The parallel sequences must have been trimmed with the tokens.
    assert len(interaction.model_response.output_logprobs) == len(output_tokens)
    assert len(interaction.model_response.output_versions) == len(output_tokens)

    # And the exported tensor dict must stay internally consistent.
    tensor_dict = interaction.to_tensor_dict()
    n_pads = int((tensor_dict["input_ids"][0] == IMAGE_PAD_ID).sum())
    assert n_pads == PATCHES_PER_IMAGE


# ---------------------------------------------------------------------------
# Collapsed prompt (exact-token transport)
# ---------------------------------------------------------------------------


def test_collapsed_prompt_keeps_one_placeholder_per_image(tokenizer, processor):
    """Test that the collapsed prompt is the unexpanded form vLLM expects."""
    messages = [user_message_with_image(make_image(9), "what is this?")]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )

    result = _process_vision_prompt(processor, text, image_data)

    assert result.collapsed_input_ids.count(IMAGE_PAD_ID) == 1
    assert result.input_ids.count(IMAGE_PAD_ID) == PATCHES_PER_IMAGE
    # Same prompt either way once the media span is excluded.
    pad_ids = vision_pad_token_ids(processor.tokenizer)
    assert [t for t in result.collapsed_input_ids if t not in pad_ids] == [
        t for t, mm in zip(result.input_ids, result.mm_token_type_ids) if not mm
    ]


def test_collapsed_prompt_scales_with_image_count(tokenizer, processor):
    """Test that each image contributes exactly one collapsed placeholder."""
    messages = [
        user_message_with_image(make_image(11), "first"),
        {"role": "assistant", "content": "ok"},
        user_message_with_image(make_image(22), "second"),
    ]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )

    result = _process_vision_prompt(processor, text, image_data)

    assert result.collapsed_input_ids.count(IMAGE_PAD_ID) == 2
    assert result.input_ids.count(IMAGE_PAD_ID) == 2 * PATCHES_PER_IMAGE


def test_concat_splices_the_collapsed_form_in_parallel(tokenizer, processor):
    """Test that the child's collapsed prompt extends the parent's collapsed one.

    The two splices locate the same conversational seam by EOS count, at
    different token indices, so neither index may be shared.
    """
    image = make_image(31)
    turn0 = [user_message_with_image(image, "find x")]
    vision = concat_vision_prompt_with_parent(turn0, None, tokenizer, processor)

    output_tokens = [11, 12, EOS_ID]
    parent = InteractionWithTokenLogpReward(
        messages=turn0,
        output_message_list=[{"role": "assistant", "content": "x = 12"}],
        model_response=ModelResponse(
            input_tokens=list(vision.input_ids),
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            stop_reason="stop",
            tokenizer=tokenizer,
        ),
        chat_template_type="concat",
        collapsed_input_ids=list(vision.collapsed_input_ids),
        mm_token_type_ids=vision.mm_token_type_ids,
        multi_modal_input=vision.multi_modal_input,
    )

    child = concat_vision_prompt_with_parent(
        [{"role": "user", "content": "wrong, retry"}], parent, tokenizer, processor
    )

    # Each form extends its own parent representation verbatim.
    expanded_prefix = vision.input_ids + output_tokens[:-1] + [EOS_ID]
    collapsed_prefix = vision.collapsed_input_ids + output_tokens[:-1] + [EOS_ID]
    assert child.input_ids[: len(expanded_prefix)] == expanded_prefix
    assert child.collapsed_input_ids[: len(collapsed_prefix)] == collapsed_prefix
    # The image is still described once in the collapsed form, and in full in the
    # expanded one.
    assert child.collapsed_input_ids.count(IMAGE_PAD_ID) == 1
    assert child.input_ids.count(IMAGE_PAD_ID) == PATCHES_PER_IMAGE


def test_concat_without_a_parent_collapsed_snapshot_fails(tokenizer, processor):
    """Test that a parent missing its snapshot raises instead of guessing.

    Falling back to the expanded prompt would make the server expand the media
    placeholders a second time.
    """
    image = make_image(41)
    turn0 = [user_message_with_image(image, "find x")]
    vision = concat_vision_prompt_with_parent(turn0, None, tokenizer, processor)
    output_tokens = [11, EOS_ID]
    parent = InteractionWithTokenLogpReward(
        messages=turn0,
        output_message_list=[{"role": "assistant", "content": "x"}],
        model_response=ModelResponse(
            input_tokens=list(vision.input_ids),
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            stop_reason="stop",
            tokenizer=tokenizer,
        ),
        chat_template_type="concat",
        collapsed_input_ids=None,
        mm_token_type_ids=vision.mm_token_type_ids,
        multi_modal_input=vision.multi_modal_input,
    )

    with pytest.raises(ValueError, match="collapsed_input_ids"):
        concat_vision_prompt_with_parent(
            [{"role": "user", "content": "retry"}], parent, tokenizer, processor
        )


def test_collapsed_snapshot_survives_request_prompt_growth(tokenizer, processor):
    """Test that resuming generation does not mutate the interaction's snapshot.

    ModelRequest.extend_prompt appends in place after every response, so a
    shared list would make the next turn splice the parent's output twice.
    """
    from areal.api.cli_args import GenerationHyperparameters
    from areal.api.io_struct import ModelRequest

    messages = [user_message_with_image(make_image(5), "hi")]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )
    vision = _process_vision_prompt(processor, text, image_data)

    interaction = InteractionWithTokenLogpReward(
        collapsed_input_ids=list(vision.collapsed_input_ids)
    )
    request = ModelRequest(
        input_ids=list(vision.input_ids),
        collapsed_input_ids=list(vision.collapsed_input_ids),
        gconfig=GenerationHyperparameters(max_new_tokens=4),
    )
    snapshot = list(interaction.collapsed_input_ids)

    request.extend_prompt([42, EOS_ID])

    assert interaction.collapsed_input_ids == snapshot
    assert request.collapsed_input_ids[-2:] == [42, EOS_ID]


def test_collapsed_snapshot_never_reaches_training_data(tokenizer, processor):
    """Test that the collapsed prompt stays out of exports.

    It is inference-transport state. Training must only ever see the expanded
    prompt, and deserialized interactions are training inputs rather than
    future concat parents, so the proxy must not ship it either.
    """
    messages = [user_message_with_image(make_image(13), "hi")]
    image_data, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )
    vision = _process_vision_prompt(processor, text, image_data)

    interaction = _make_interaction(
        vision.input_ids, vision.mm_token_type_ids, vision.multi_modal_input
    )
    interaction.collapsed_input_ids = list(vision.collapsed_input_ids)

    tensor_dict = interaction.to_tensor_dict()
    # Matched by substring rather than by exact key: this guards the concept,
    # so renaming the field cannot quietly turn the check into a no-op.
    assert not [key for key in tensor_dict if "collapsed" in key]
    # The exported prompt is the expanded one.
    assert tensor_dict["input_ids"].shape[1] == len(vision.input_ids) + 3

    payload = serialize_interactions({"a": interaction})
    assert "collapsed" not in json.dumps(payload, default=str)


def test_collapsed_prompt_helper_returns_a_fresh_list(processor, tokenizer):
    """Test that repeated calls do not share one mutable list.

    The engine appends generated tokens to this list in place while resuming an
    interrupted generation, so a cached list would leak one rollout's output
    into the next prompt that renders identically. Callers that assign the
    result straight into a ModelRequest depend on this.
    """
    from areal.utils.hf_utils import collapsed_prompt_token_ids

    messages = [user_message_with_image(make_image(17), "same prompt")]
    _, tok_messages, _ = _extract_images_from_messages(messages)
    text = apply_chat_template(
        tokenizer, tok_messages, add_generation_prompt=True, tokenize=False
    )

    first = collapsed_prompt_token_ids(processor, text)
    second = collapsed_prompt_token_ids(processor, text)

    assert first == second
    assert first is not second
    first.extend([1234, 5678])
    assert collapsed_prompt_token_ids(processor, text) == second


def test_collapsed_check_counts_markers_not_expanded_pads(tokenizer):
    """Test that a Gemma3-shaped prompt is validated against its collapsed marker.

    Its collapsed prompt holds boi_token and its expanded prompt holds a run of
    image_token, so counting pads on the collapsed side would find none and
    reject a perfectly valid request before it ever reached vLLM.
    """
    BOI = VIDEO_PAD_ID  # stands in for boi_token: a real id, distinct from the pad

    class _Gemma3Like:
        boi_token = "<|video_pad|>"
        image_token = "<|image_pad|>"
        tokenizer = None

    proc = _Gemma3Like()
    proc.tokenizer = tokenizer

    collapsed = [10, BOI, 11]
    expanded = [10, IMAGE_PAD_ID, IMAGE_PAD_ID, 11]
    mm_mask = [0, 1, 1, 0]

    # One marker, one image: accepted.
    _check_collapsed_matches_expanded(collapsed, expanded, mm_mask, proc, 1)

    # Two images against one marker: rejected, with the counts named.
    with pytest.raises(ValueError, match="1 media placeholders for 2 media"):
        _check_collapsed_matches_expanded(collapsed, expanded, mm_mask, proc, 2)


def test_loader_rejects_the_tokenizer_autoprocessor_returns_for_text_models(
    monkeypatch,
):
    """Test that load_hf_processor returns None for a text-only model.

    ``AutoProcessor.from_pretrained`` does not raise there; it returns the
    model's tokenizer. Returning that would make a text-only deployment look
    multimodal and run the vision canary against it.
    """
    import transformers

    from areal.utils import hf_utils

    class _Tokenizer:
        def convert_tokens_to_ids(self, token):
            return None

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _Tokenizer()),
    )
    hf_utils.load_hf_processor.cache_clear()
    try:
        assert hf_utils.load_hf_processor("some/text-only-model") is None
    finally:
        hf_utils.load_hf_processor.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "with_image, chat_template_type, retained",
    [
        (True, "concat", True),  # only a concat child can splice a parent
        (True, "hf", False),  # hf re-renders each turn; no parent is read
        (False, "concat", False),  # text-only has nothing to collapse
        (False, "hf", False),
    ],
)
async def test_collapsed_prompts_are_retained_only_where_concat_uses_them(
    tokenizer, processor, with_image, chat_template_type, retained
):
    """Test the retention rule through create(), not through the helper.

    Asserting on the predicate alone would still pass if a request path stopped
    consulting it, or set the field unconditionally. This drives the real path
    and inspects what the interaction ends up holding.
    """
    from areal.experimental.openai import ArealOpenAI

    engine = _EchoEngine(tokenizer)
    client = ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        processor=processor if with_image else None,
        chat_template_type=chat_template_type,
        api_key="test",
    )
    messages = (
        [user_message_with_image(make_image(41), "describe")]
        if with_image
        else [{"role": "user", "content": "hello"}]
    )

    completion = await client.chat.completions.create(
        messages=messages, model="default", max_completion_tokens=8
    )

    interaction = client.get_interaction(completion.id)
    assert (interaction.collapsed_input_ids is not None) is retained


def test_the_two_request_paths_agree_on_the_retention_rule():
    """Test that Completions and Responses carry the same rule.

    They are separate classes with separate copies of it, so they can drift.
    """
    from areal.experimental.openai.client import (
        AsyncCompletionsWithReward,
        AsyncResponsesWithReward,
    )

    def _retains(cls, processor, chat_template_type):
        obj = cls.__new__(cls)
        obj.processor = processor
        obj.chat_template_type = chat_template_type
        return obj._retains_collapsed_prompts()

    for cls in (AsyncCompletionsWithReward, AsyncResponsesWithReward):
        assert _retains(cls, object(), "concat"), cls.__name__
        assert not _retains(cls, object(), "hf"), cls.__name__
        assert not _retains(cls, None, "concat"), cls.__name__
        assert not _retains(cls, None, "hf"), cls.__name__
