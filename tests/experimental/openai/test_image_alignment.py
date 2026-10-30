# SPDX-License-Identifier: Apache-2.0

import base64
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from areal.api import ModelResponse
from areal.experimental.openai.client import (
    ArealOpenAI,
    _is_sglang_backend,
    _prepare_prompt,
)
from areal.infra.processor_cache import ProcessorCallCache
from areal.utils.sglang_image import IMAGE_SPANS_KEY, image_request_kwargs


class Qwen3VLProcessor:
    """Record the actual image supplied by the OpenAI input preparation path."""

    def __init__(self, model_path):
        self.tokenizer = SimpleNamespace(name_or_path=str(model_path))
        self.image_token_id = 3
        self.image_processor = SimpleNamespace(
            patch_size=16,
            merge_size=2,
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )
        self.calls = 0
        self.image_kwargs = []

    def __call__(self, *, images, **kwargs):
        self.calls += 1
        self.image_kwargs.append(kwargs.get("images_kwargs"))
        image = images[0]
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.from_numpy(np.asarray(image).copy()),
            "image_grid_thw": torch.tensor(
                [[1, image.height // 16, image.width // 16]]
            ),
        }


@pytest.fixture
def processor(tmp_path, monkeypatch):
    monkeypatch.delenv("AREAL_VLM_ALIGN_SGLANG", raising=False)
    monkeypatch.delenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", raising=False)
    monkeypatch.delenv("SGLANG_RESIZE_RESAMPLE", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    return Qwen3VLProcessor(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_template_type", ["hf", "concat"])
async def test_cached_prompt_keeps_wire_images_with_training_tensors(
    monkeypatch, chat_template_type, processor
):
    monkeypatch.delenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", raising=False)
    buffer = BytesIO()
    Image.new("RGB", (1600, 1200), "red").save(buffer, format="JPEG")
    original_images = [base64.b64encode(buffer.getvalue()).decode()]
    cache = ProcessorCallCache()
    kwargs = dict(
        processor=processor,
        tokenizer=SimpleNamespace(eos_token_id=2),
        tokenizer_messages=[],
        concat_messages=[],
        image_data=original_images,
        parent=None,
        chat_template_type=chat_template_type,
        tools=None,
        extra_body={},
        processor_cache=cache,
        align_sglang_images=True,
    )
    with patch(
        "areal.experimental.openai.client.apply_chat_template", return_value="prompt"
    ):
        first = await _prepare_prompt(**kwargs)
        second = await _prepare_prompt(**kwargs)
        unaligned = await _prepare_prompt(**{**kwargs, "align_sglang_images": False})
        monkeypatch.setenv("AREAL_VLM_ALIGN_SGLANG", "false")
        disabled = await _prepare_prompt(**kwargs)
        monkeypatch.setenv("AREAL_VLM_ALIGN_SGLANG", "true")
        reenabled = await _prepare_prompt(**kwargs)

    assert processor.calls == 2
    assert unaligned.image_data is None
    assert disabled.image_data is None
    assert reenabled.image_data == first.image_data
    assert (
        disabled.multi_modal_input["pixel_values"]
        is unaligned.multi_modal_input["pixel_values"]
    )
    assert (
        reenabled.multi_modal_input["pixel_values"]
        is first.multi_modal_input["pixel_values"]
    )
    assert unaligned.multi_modal_input["pixel_values"].shape == (1200, 1600, 3)
    assert first.image_data == second.image_data
    assert first.image_data is not second.image_data
    assert original_images == first.image_data
    assert processor.image_kwargs[0] == {
        "device": "cpu",
        "size": {"shortest_edge": 65536, "longest_edge": 1048576},
    }
    with Image.open(BytesIO(base64.b64decode(first.image_data[0]))) as wire:
        assert wire.size == (1600, 1200)
        expected = wire.convert("RGB").resize((1152, 864), resample=None)
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(expected).copy()),
            first.multi_modal_input["pixel_values"],
            rtol=0,
            atol=0,
        )
    cache.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("alignment", [None, "false", "true"])
async def test_create_forwards_images_for_backend(
    monkeypatch, api, backend, alignment, processor
):
    if alignment is None:
        monkeypatch.delenv("AREAL_VLM_ALIGN_SGLANG", raising=False)
    else:
        monkeypatch.setenv("AREAL_VLM_ALIGN_SGLANG", alignment)
    monkeypatch.delenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", raising=False)
    tokenizer = MagicMock(eos_token_id=2, pad_token_id=0)
    tokenizer.decode.return_value = "done"
    tokenizer.convert_tokens_to_ids.side_effect = {
        "<|image_pad|>": 3,
        "<|video_pad|>": 4,
        "<|vision_start|>": 5,
        "<|vision_end|>": 6,
    }.get
    processor.tokenizer.convert_tokens_to_ids = tokenizer.convert_tokens_to_ids
    requests = []

    class Engine:
        config = SimpleNamespace(backend=backend)

        async def agenerate(self, request):
            requests.append(request)
            return ModelResponse(
                input_tokens=request.input_ids,
                output_tokens=[3, 2],
                output_logprobs=[-0.1, -0.1],
                output_versions=[0, 0],
                tokenizer=request.tokenizer,
            )

    buffer = BytesIO()
    Image.new("RGB", (1600, 1200), "blue").save(buffer, format="JPEG")
    original = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    client = ArealOpenAI(
        engine=Engine(),
        tokenizer=tokenizer,
        processor=processor,
        api_key="test",
        reasoning_parser="",
    )
    try:
        with patch(
            "areal.experimental.openai.client.apply_chat_template",
            return_value="prompt",
        ):
            if api == "chat":
                await client.chat.completions.create(
                    model="test",
                    max_tokens=8,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": original}}
                            ],
                        }
                    ],
                )
            else:
                await client.responses.create(
                    model="test",
                    max_output_tokens=8,
                    input=[
                        {
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": original}],
                        }
                    ],
                )
        assert len(requests) == 1
        request = requests[0]
        # A sampled image_pad is text, even when resize alignment is disabled.
        # Retain real-image provenance so SGLang can distinguish the two.
        assert request.metadata[IMAGE_SPANS_KEY] == [[2, 2]]
        if backend == "sglang":
            wire = image_request_kwargs(request, accumulated_tokens=[3])
            assert wire["input_ids"] == request.input_ids + [3]
            assert wire["mm_sampling_kwargs"][IMAGE_SPANS_KEY] == [[2, 2]]
        if backend == "sglang" and alignment != "false":
            assert request.image_data == [original]
            expected_size = (1152, 864)
            assert processor.image_kwargs[0]["size"]["longest_edge"] == 1048576
        else:
            assert request.image_data == [original]
            expected_size = (1600, 1200)
            assert processor.image_kwargs[0] is None
        with Image.open(
            BytesIO(base64.b64decode(request.image_data[0].split(",", 1)[1]))
        ) as wire:
            assert wire.size == (1600, 1200)
            expected = wire.convert("RGB").resize(expected_size, resample=None)
            interaction = next(iter(client._cache.values()))
            torch.testing.assert_close(
                torch.from_numpy(np.asarray(expected).copy()),
                interaction.multi_modal_input["pixel_values"],
                rtol=0,
                atol=0,
            )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cached_prompt_respects_sglang_pixel_and_resampling_settings(
    monkeypatch, processor
):
    """A policy change cannot reuse a prompt from a different resize policy."""
    buffer = BytesIO()
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
    )
    image.save(buffer, format="PNG")
    cache = ProcessorCallCache()
    kwargs = dict(
        processor=processor,
        tokenizer=SimpleNamespace(eos_token_id=2),
        tokenizer_messages=[],
        concat_messages=[],
        image_data=[base64.b64encode(buffer.getvalue()).decode()],
        parent=None,
        chat_template_type="hf",
        tools=None,
        extra_body={},
        processor_cache=cache,
        align_sglang_images=True,
    )
    try:
        with patch(
            "areal.experimental.openai.client.apply_chat_template",
            return_value="prompt",
        ):
            first = await _prepare_prompt(**kwargs)
            monkeypatch.setenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", "false")
            full_budget = await _prepare_prompt(**kwargs)
            monkeypatch.delenv("SGLANG_VLM_QWEN_LIMIT_PIXEL")
            monkeypatch.setenv("SGLANG_RESIZE_RESAMPLE", "NEAREST")
            nearest = await _prepare_prompt(**kwargs)
            repeated = await _prepare_prompt(**kwargs)

        assert processor.calls == 3
        assert first.multi_modal_input["pixel_values"].shape == (864, 1152, 3)
        assert full_budget.multi_modal_input["pixel_values"].shape == (1216, 1600, 3)
        assert nearest.multi_modal_input["pixel_values"].shape == (864, 1152, 3)
        assert not torch.equal(
            first.multi_modal_input["pixel_values"],
            nearest.multi_modal_input["pixel_values"],
        )
        assert (
            repeated.multi_modal_input["pixel_values"]
            is nearest.multi_modal_input["pixel_values"]
        )
        assert first.image_data == full_budget.image_data == nearest.image_data
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_sglang_decode_keeps_raw_exif_orientation(processor):
    image = Image.new("RGB", (40, 80), "red")
    exif = image.getexif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    original = base64.b64encode(buffer.getvalue()).decode()
    with patch(
        "areal.experimental.openai.client.apply_chat_template", return_value="prompt"
    ):
        result = await _prepare_prompt(
            processor=processor,
            tokenizer=SimpleNamespace(eos_token_id=2),
            tokenizer_messages=[],
            concat_messages=[],
            image_data=[original],
            parent=None,
            chat_template_type="hf",
            tools=None,
            extra_body={},
            align_sglang_images=True,
        )
    assert result.image_data == [original]
    assert result.multi_modal_input["pixel_values"].shape == (384, 192, 3)


@pytest.mark.parametrize(
    ("engine_type", "config_backend", "backend_type", "expected"),
    [
        ("RolloutController", "sglang:d8", "NoneType", True),
        ("RolloutController", "vllm:d8", "NoneType", False),
        ("RemoteSGLangEngine", None, "NoneType", True),
        ("InfBridge", None, "SGLangBackend", True),
        ("InfBridge", None, "VLLMBackend", False),
        ("UnknownEngine", None, "NoneType", False),
    ],
)
def test_alignment_selects_sglang_backends(
    engine_type, config_backend, backend_type, expected
):
    engine = type(engine_type, (), {})()
    engine.config = SimpleNamespace(backend=config_backend)
    engine.backend = type(backend_type, (), {})()
    assert _is_sglang_backend(engine) is expected
