# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from areal.api import ModelResponse
from areal.experimental.openai.client import (
    ArealOpenAI,
    _extract_images_from_messages,
    _prepare_prompt,
    _process_multimodal_prompt,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=True,
        **_kwargs,
    ):
        tokens = [10, 2, 20] if len(messages) == 1 else [10, 2, 20, 30, 2, 40]
        return {"input_ids": tokens} if tokenize else f"message_count={len(messages)}"

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class _FakeEngine:
    def __init__(self):
        self.requests = []

    async def agenerate(self, request):
        self.requests.append(request)
        return ModelResponse(
            input_tokens=list(request.input_ids),
            output_tokens=[77],
            output_logprobs=[-0.1],
            output_versions=[0],
            stop_reason="length",
            tokenizer=request.tokenizer,
            processor=request.processor,
        )


class _FakeVLLMEngine(_FakeEngine):
    config = SimpleNamespace(backend="vllm:d1")


class _FakeProcessor:
    image_processor = SimpleNamespace(image_processor_type="fake")

    def __call__(self, *, text, images, padding, return_tensors):
        assert padding is False
        assert return_tensors == "pt"
        assert len(text) == 1
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        if text[0] == "message_count=1":
            input_ids = [10, 2, 20]
            token_types = [1, 1, 1]
        else:
            input_ids = [10, 2, 20, 30, 2, 40]
            token_types = [1, 1, 1, 0, 0, 2]
        return {
            "input_ids": torch.tensor([input_ids]),
            "mm_token_type_ids": torch.tensor([token_types]),
            "pixel_values": torch.ones(1, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }


def _png_base64() -> str:
    with BytesIO() as buffer:
        Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _response(input_tokens: list[int], output_tokens: list[int]) -> ModelResponse:
    return ModelResponse(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_logprobs=[-0.1] * len(output_tokens),
        output_versions=[0] * len(output_tokens),
        tokenizer=_FakeTokenizer(),
    )


def test_extract_images_preserves_remote_url_for_backend_forwarding():
    """Tokenizer-only clients should keep backend-managed image URLs unchanged."""
    url = "https://example.com/image.png"
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": url}}],
        }
    ]

    image_data, tokenizer_messages, _ = _extract_images_from_messages(messages)

    assert image_data == [url]
    assert tokenizer_messages[0]["content"] == [{"type": "image"}]


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "https://internal.example/image.png",
        "file:///etc/passwd",
    ],
)
def test_local_multimodal_processing_rejects_non_inline_urls(url):
    """Local VLM processing must not fetch external resources."""
    with pytest.raises(ValueError, match="requires valid base64-encoded image data"):
        _process_multimodal_prompt(
            _FakeProcessor(),
            _FakeTokenizer(),
            [{"role": "user", "content": [{"type": "image"}]}],
            [url],
            None,
            {},
        )


def test_local_multimodal_processing_rejects_existing_local_image_path(tmp_path):
    """Local VLM processing must not open filesystem paths from requests."""
    image_path = tmp_path / "private-image.png"
    Image.new("RGB", (2, 2), color="red").save(image_path)

    with pytest.raises(ValueError, match="requires valid base64-encoded image data"):
        _process_multimodal_prompt(
            _FakeProcessor(),
            _FakeTokenizer(),
            [{"role": "user", "content": [{"type": "image"}]}],
            [str(image_path)],
            None,
            {},
        )


@pytest.mark.asyncio
async def test_prepare_prompt_strict_mode_rejects_image_without_processor():
    """An image request must not silently export a text-only trajectory."""
    message = {"role": "user", "content": [{"type": "image"}]}

    with pytest.raises(ValueError, match="require a multimodal processor"):
        await _prepare_prompt(
            tokenizer=_FakeTokenizer(),
            processor=None,
            tokenizer_messages=[message],
            concat_messages=[message],
            image_data=[_png_base64()],
            parent=None,
            chat_template_type="hf",
            tools=None,
            extra_body={},
            require_multimodal_processor=True,
        )


@pytest.mark.asyncio
async def test_prepare_prompt_default_mode_preserves_image_forwarding_without_processor():
    """Tokenizer-only clients should retain the legacy backend image path."""
    message = {"role": "user", "content": [{"type": "image"}]}

    prepared = await _prepare_prompt(
        tokenizer=_FakeTokenizer(),
        processor=None,
        tokenizer_messages=[message],
        concat_messages=[message],
        image_data=[_png_base64()],
        parent=None,
        chat_template_type="hf",
        tools=None,
        extra_body={},
    )

    assert prepared.input_ids == [10, 2, 20]
    assert prepared.mm_token_type_ids is None
    assert prepared.multi_modal_input is None


@pytest.mark.asyncio
async def test_prepare_prompt_with_image_returns_processor_tokens_and_tensors():
    """Image prompts should use processor-expanded IDs and vision tensors."""
    message = {"role": "user", "content": [{"type": "image"}]}

    prepared = await _prepare_prompt(
        tokenizer=_FakeTokenizer(),
        processor=_FakeProcessor(),
        tokenizer_messages=[message],
        concat_messages=[message],
        image_data=[_png_base64()],
        parent=None,
        chat_template_type="hf",
        tools=None,
        extra_body={},
    )

    assert prepared.input_ids == [10, 2, 20]
    assert prepared.mm_token_type_ids == [1, 1, 1]
    assert prepared.multi_modal_input is not None
    torch.testing.assert_close(
        prepared.multi_modal_input["pixel_values"],
        torch.ones(1, 3, 2, 2),
        rtol=0,
        atol=0,
    )


@pytest.mark.asyncio
async def test_prepare_concat_prompt_preserves_parent_and_uses_final_vision_data():
    """Concat should preserve sampled parent IDs without duplicating vision data."""
    root_message = {"role": "user", "content": [{"type": "image"}]}
    parent = InteractionWithTokenLogpReward(
        model_response=_response([10, 2, 20], [30, 2]),
        messages=[root_message],
        output_message_list=[{"role": "assistant", "content": "answer"}],
        chat_template_type="concat",
        prompt_token_ids=[10, 2, 20],
        mm_token_type_ids=[1, 1, 1],
        multi_modal_input={"pixel_values": torch.ones(1, 3, 2, 2)},
    )
    full_messages = [
        root_message,
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "continue"},
    ]

    prepared = await _prepare_prompt(
        tokenizer=_FakeTokenizer(),
        processor=_FakeProcessor(),
        tokenizer_messages=full_messages,
        concat_messages=[{"role": "user", "content": "continue"}],
        image_data=[_png_base64()],
        parent=parent,
        chat_template_type="concat",
        tools=None,
        extra_body={},
    )

    assert prepared.input_ids == [10, 2, 20, 30, 2, 40]
    assert prepared.mm_token_type_ids == [1, 1, 1, 0, 0, 2]
    assert prepared.multi_modal_input is not None
    assert prepared.multi_modal_input["pixel_values"].shape == (1, 3, 2, 2)


def test_multimodal_interaction_exports_training_fields():
    """Tensor export should append output token types and retain one image item."""
    interaction = InteractionWithTokenLogpReward(
        model_response=_response([10, 2, 20], [30, 2]),
        prompt_token_ids=[10, 2, 20],
        mm_token_type_ids=[1, 1, 1],
        multi_modal_input={
            "pixel_values": torch.ones(1, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        },
    )

    result = interaction.to_tensor_dict()

    torch.testing.assert_close(
        result["mm_token_type_ids"],
        torch.tensor([[1, 1, 1, 0, 0]]),
        rtol=0,
        atol=0,
    )
    assert len(result["multi_modal_input"]) == 1
    assert result["multi_modal_input"][0] is interaction.multi_modal_input


def test_multimodal_interaction_rejects_processor_rollout_token_mismatch():
    """A trajectory with different processor and rollout IDs must be rejected."""
    interaction = InteractionWithTokenLogpReward(
        model_response=_response([10, 2, 99], [30, 2]),
        prompt_token_ids=[10, 2, 20],
        mm_token_type_ids=[1, 1, 1],
        multi_modal_input={"pixel_values": torch.ones(1, 3, 2, 2)},
    )

    with pytest.raises(ValueError, match="processor prompt tokens"):
        interaction.to_tensor_dict()


@pytest.mark.asyncio
async def test_areal_openai_shares_processor_with_both_generation_apis():
    """Chat Completions and Responses should use the same loaded processor."""
    processor = _FakeProcessor()
    client = ArealOpenAI(
        engine=SimpleNamespace(),
        tokenizer=_FakeTokenizer(),
        processor=processor,
        require_multimodal_processor=True,
        api_key="test-key",
        base_url="http://test.invalid/v1",
    )
    try:
        assert client.chat.completions.processor is processor
        assert client.responses.processor is processor
        assert client.chat.completions.require_multimodal_processor is True
        assert client.responses.require_multimodal_processor is True
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("api_type", ["chat", "responses"])
async def test_generation_api_exports_processor_multimodal_trajectory(api_type):
    """Both OpenAI generation APIs should produce trainable VLM fields."""
    processor = _FakeProcessor()
    engine = _FakeEngine()
    client = ArealOpenAI(
        engine=engine,
        tokenizer=_FakeTokenizer(),
        processor=processor,
        api_key="test-key",
        base_url="http://test.invalid/v1",
    )
    data_uri = f"data:image/png;base64,{_png_base64()}"
    try:
        if api_type == "chat":
            completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                            {"type": "text", "text": "describe"},
                        ],
                    }
                ],
                max_completion_tokens=1,
            )
            interaction_id = completion.id
        else:
            response = await client.responses.create(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": data_uri,
                            },
                            {"type": "input_text", "text": "describe"},
                        ],
                    }
                ],
                max_output_tokens=1,
            )
            interaction_id = response.id

        interaction = client.get_interaction(interaction_id)
        assert interaction is not None
        result = interaction.to_tensor_dict()

        assert engine.requests[0].processor is processor
        assert engine.requests[0].input_ids == [10, 2, 20]
        assert result["mm_token_type_ids"].tolist() == [[1, 1, 1, 0]]
        assert len(result["multi_modal_input"]) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("api_type", ["chat", "responses"])
async def test_generation_api_rejects_vllm_multimodal_trajectory(api_type):
    """Trainable multimodal agent trajectories should fail fast on vLLM."""
    engine = _FakeVLLMEngine()
    client = ArealOpenAI(
        engine=engine,
        tokenizer=_FakeTokenizer(),
        processor=_FakeProcessor(),
        require_multimodal_processor=True,
        api_key="test-key",
        base_url="http://test.invalid/v1",
    )
    data_uri = f"data:image/png;base64,{_png_base64()}"
    try:
        with pytest.raises(ValueError, match="supported only with.*SGLang"):
            if api_type == "chat":
                await client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_uri},
                                }
                            ],
                        }
                    ],
                    max_completion_tokens=1,
                )
            else:
                await client.responses.create(
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "detail": "auto",
                                    "image_url": data_uri,
                                }
                            ],
                        }
                    ],
                    max_output_tokens=1,
                )
        assert engine.requests == []
    finally:
        await client.close()
