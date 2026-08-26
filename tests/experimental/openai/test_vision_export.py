# SPDX-License-Identifier: Apache-2.0

"""CPU tests for multimodal support on the agent/proxy path.

Covers the three defects the vision export closes: the prompt must be
vision-expanded, the exported tensor dict must carry ``multi_modal_input`` and
``mm_token_type_ids``, and the payload must survive the proxy's HTTP
serialization without shipping one copy of ``pixel_values`` per turn.
"""

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
from areal.utils.hf_utils import apply_chat_template
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

    # The engine must receive the expanded prompt plus the raw image payload.
    assert engine.last_request.input_ids.count(IMAGE_PAD_ID) == PATCHES_PER_IMAGE
    assert len(engine.last_request.image_data) == 1
    assert engine.last_request.vision_msg_vllm is not None

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
