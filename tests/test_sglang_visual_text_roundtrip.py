# SPDX-License-Identifier: Apache-2.0
"""Real local Qwen/HF/SGLang preprocessing; no weights or server required."""

import asyncio
import base64
import os
import pickle
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from areal.api import ModelRequest, ModelResponse
from areal.experimental.openai.client import ArealOpenAI, _process_multimodal_prompt
from areal.utils.sglang_image import (
    IMAGE_SPANS_KEY,
    image_prompt_metadata,
    image_request_kwargs,
)


@pytest.fixture(scope="module")
def qwen_runtime():
    path = os.environ.get("AREAL_TEST_QWEN_MODEL_PATH")
    if not path:
        pytest.skip("Set AREAL_TEST_QWEN_MODEL_PATH to the local native Qwen model")
    pytest.importorskip("sglang.srt.multimodal.processors.qwen_vl")
    from sglang.srt import server_args
    from sglang.srt.multimodal.processors import base_processor
    from transformers import AutoConfig, AutoProcessor

    from areal.engine.sglang_processors.qwen_vl import AReaLQwenImageProcessor

    hf = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True)
    sg_hf = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    args = SimpleNamespace(
        model_path=path,
        device="cpu",
        mm_feature_transport="cpu",
        mm_process_config={},
        mm_processor_worker_num=1,
        mm_io_worker_num=1,
        mm_preprocess_cache_size_mb=100,
        tokenizer_worker_num=1,
        base_gpu_id=0,
        disable_fast_image_processor=False,
        keep_mm_feature_on_device=False,
        skip_tokenizer_init=False,
        rl_on_policy_target=None,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(server_args, "_global_server_args", args)
        patch.setattr(base_processor, "_is_cpu", True)
        processor = AReaLQwenImageProcessor(
            config, args, sg_hf, transport_mode="default", skip_mm_pool=True
        )
        try:
            yield hf, processor
        finally:
            for executor in (
                processor.io_executor,
                processor.cpu_executor,
                processor.load_mm_data_executor,
            ):
                executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    "literal",
    ["<|image_pad|>", "<|video_pad|>", "<|vision_start|><|image_pad|><|vision_end|>"],
)
@pytest.mark.parametrize("image_count", [1, 2])
def test_real_generated_visual_tokens_keep_full_ids_features_and_offsets(
    qwen_runtime, literal, image_count, caplog
):
    hf, sg = qwen_runtime
    images = []
    for index in range(image_count):
        buffer = BytesIO()
        Image.new("RGB", (511 + index * 2, 257), (20, 90, 150)).save(
            buffer, format="JPEG"
        )
        images.append(
            "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        )
    messages = [
        {"role": "assistant", "content": f"left{literal}right"},
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images]
            + [{"type": "text", "text": f"again{literal}end"}],
        },
    ]
    vocab = hf.tokenizer.get_vocab()
    prepared = _process_multimodal_prompt(
        hf, hf.tokenizer, messages, images, None, {}, align_sglang_images=True
    )
    metadata = image_prompt_metadata(
        {}, prepared.input_ids, prepared.mm_token_type_ids, images, hf
    )
    req = ModelRequest(
        input_ids=prepared.input_ids, image_data=images, processor=hf, metadata=metadata
    )
    wire = image_request_kwargs(req)
    assert wire["input_ids"] == prepared.input_ids
    result = asyncio.run(
        sg.process_mm_data_async(
            images,
            wire["input_ids"],
            SimpleNamespace(
                video_data=None,
                audio_data=None,
                rid="visual-text-regression",
                mm_sampling_kwargs=wire["mm_sampling_kwargs"],
            ),
        )
    )
    assert result.input_ids == prepared.input_ids
    assert len(result.mm_items) == image_count
    logical_bytes = sum(
        item.feature.numel() * item.feature.element_size() for item in result.mm_items
    )
    assert len(pickle.dumps(result)) < logical_bytes + 64 * 1024
    for item in result.mm_items:
        assert item.feature.untyped_storage().nbytes() == (
            item.feature.numel() * item.feature.element_size()
        )
    assert [item.offsets[0] for item in result.mm_items] == [
        tuple(span) for span in metadata[IMAGE_SPANS_KEY]
    ]
    assert result.mrope_positions.shape == (3, len(prepared.input_ids))
    torch.testing.assert_close(
        torch.cat([item.feature for item in result.mm_items]),
        prepared.multi_modal_input["pixel_values"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.cat(
            [item.model_specific_data["image_grid_thw"] for item in result.mm_items]
        ),
        prepared.multi_modal_input["image_grid_thw"],
        rtol=0,
        atol=0,
    )
    real_positions = {
        i for start, end in metadata[IMAGE_SPANS_KEY] for i in range(start, end + 1)
    }
    for i, token in enumerate(prepared.input_ids):
        if token == hf.image_token_id:
            assert prepared.mm_token_type_ids[i] == int(i in real_positions)
    assert hf.tokenizer.get_vocab() == vocab
    assert not any(
        "falling back to decode+retokenize" in r.message for r in caplog.records
    )


@pytest.mark.slow
def test_native_image_transport_without_explicit_spans_compacts_cpu_views(qwen_runtime):
    hf, sg = qwen_runtime
    buffer = BytesIO()
    Image.new("RGB", (511, 257), (20, 90, 150)).save(buffer, format="JPEG")
    images = [
        "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    ] * 2
    messages = [{"role": "user", "content": [{"type": "image"}] * 2}]
    prepared = _process_multimodal_prompt(
        hf, hf.tokenizer, messages, images, None, {}, align_sglang_images=True
    )
    wire = image_request_kwargs(
        ModelRequest(input_ids=prepared.input_ids, image_data=images, processor=hf)
    )
    assert "mm_sampling_kwargs" not in wire

    result = asyncio.run(
        sg.process_mm_data_async(
            images,
            wire["input_ids"],
            SimpleNamespace(video_data=None, audio_data=None, rid="native-storage"),
        )
    )

    assert result.input_ids == prepared.input_ids
    pixels = torch.cat([item.feature for item in result.mm_items])
    torch.testing.assert_close(
        pixels, prepared.multi_modal_input["pixel_values"], rtol=0, atol=0
    )
    logical_bytes = pixels.numel() * pixels.element_size()
    assert len(pickle.dumps(result)) < logical_bytes + 64 * 1024


@pytest.mark.slow
def test_external_processor_registration_uses_native_plugin_interface(qwen_runtime):
    from sglang.srt.managers.multimodal_processor import (
        PROCESSOR_MAPPING,
        import_processors,
    )

    from areal.engine.sglang_processors.qwen_vl import AReaLQwenImageProcessor

    import_processors("areal.engine.sglang_processors", overwrite=True)
    assert AReaLQwenImageProcessor.models
    assert all(
        PROCESSOR_MAPPING[model] is AReaLQwenImageProcessor
        for model in AReaLQwenImageProcessor.models
    )


@pytest.mark.slow
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("template", ["hf", "concat"])
def test_real_proxy_continues_after_sampling_image_pad(qwen_runtime, api, template):
    asyncio.run(_continue_after_sampling_image_pad(qwen_runtime, api, template))


async def _continue_after_sampling_image_pad(qwen_runtime, api, template):
    hf, _ = qwen_runtime
    requests = []
    sampled = (
        hf.tokenizer.encode("left", add_special_tokens=False)
        + [hf.image_token_id]
        + hf.tokenizer.encode("right", add_special_tokens=False)
        + [hf.tokenizer.eos_token_id]
    )

    class Engine:
        config = SimpleNamespace(backend="sglang")

        async def agenerate(self, request):
            requests.append(request)
            return ModelResponse(
                input_tokens=list(request.input_ids),
                output_tokens=sampled.copy(),
                output_logprobs=[-0.25] * len(sampled),
                output_versions=[7] * len(sampled),
                stop_reason="stop",
                tokenizer=hf.tokenizer,
                processor=hf,
            )

    buffer = BytesIO()
    Image.new("RGB", (320, 256), "red").save(buffer, format="JPEG")
    image = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    client = ArealOpenAI(
        engine=Engine(),
        tokenizer=hf.tokenizer,
        processor=hf,
        chat_template_type=template,
        reasoning_parser="",
        api_key="test",
        base_url="http://test.invalid/v1",
    )
    try:
        if api == "chat":
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image}}],
                }
            ]
            first = await client.chat.completions.create(
                messages=messages, max_completion_tokens=32
            )
            first_id = first.id
            messages += [
                {"role": "assistant", "content": first.choices[0].message.content},
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image}}],
                },
            ]
            second = await client.chat.completions.create(
                messages=messages, max_completion_tokens=32
            )
        else:
            message = {
                "role": "user",
                "content": [{"type": "input_image", "image_url": image}],
            }
            first = await client.responses.create(input=[message], max_output_tokens=32)
            first_id = first.id
            second = await client.responses.create(
                input=[
                    message,
                    *[item.model_dump(exclude_none=True) for item in first.output],
                    message,
                ],
                max_output_tokens=32,
            )
        spans = requests[1].metadata[IMAGE_SPANS_KEY]
        assert len(spans) == 2
        assert "mm_sampling_kwargs" in image_request_kwargs(requests[1])
        first_interaction = client.get_interaction(first_id)
        second_interaction = client.get_interaction(second.id)
        assert first_interaction.model_response.output_tokens == sampled
        assert first_interaction.model_response.output_logprobs == [-0.25] * len(
            sampled
        )
        if template == "concat":
            prefix = requests[0].input_ids + sampled
            assert requests[1].input_ids[: len(prefix)] == prefix
            start = len(requests[0].input_ids)
            assert second_interaction.mm_token_type_ids[
                start : start + len(sampled)
            ] == [0] * len(sampled)
        tensors = second_interaction.to_tensor_dict()
        assert tensors["mm_token_type_ids"].shape[1] == len(
            requests[1].input_ids
        ) + len(sampled)
    finally:
        await client.close()
