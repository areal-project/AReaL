"""Unit tests for v2 Data Proxy multimodal processor wiring."""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from PIL import Image

from areal.v2.inference_service.data_proxy.app import (
    _create_areal_client,
    _create_inf_bridge,
)
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.data_proxy.tokenizer_proxy import TokenizerProxy


class TestTokenizerProxyMultimodal:
    def test_loads_image_processor_and_tokenizer(self):
        tokenizer = MagicMock()
        processor = MagicMock(image_processor=MagicMock())

        with patch(
            "areal.utils.hf_utils.load_hf_processor_and_tokenizer",
            return_value=(processor, tokenizer),
        ) as load_processor:
            proxy = TokenizerProxy("mock-vlm")

        load_processor.assert_called_once_with("mock-vlm")
        assert proxy._tok is tokenizer
        assert proxy.processor is processor

    def test_ignores_processor_without_image_processor(self):
        tokenizer = MagicMock()
        processor = object()

        with patch(
            "areal.utils.hf_utils.load_hf_processor_and_tokenizer",
            return_value=(processor, tokenizer),
        ):
            proxy = TokenizerProxy("mock-text-model")

        assert proxy._tok is tokenizer
        assert proxy.processor is None


class TestDataProxyMultimodalClient:
    def test_client_without_processor_injects_strict_mode(self):
        bridge = MagicMock()
        tokenizer = MagicMock()
        tok = MagicMock(_tok=tokenizer, processor=None)
        config = DataProxyConfig()

        with patch(
            "areal.v2.inference_service.data_proxy.app.ArealOpenAI"
        ) as areal_openai:
            _create_areal_client(bridge, tok, config)

        assert areal_openai.call_args.kwargs["processor"] is None
        assert areal_openai.call_args.kwargs["require_multimodal_processor"] is True

    def test_multimodal_client_injects_processor_and_requires_it(self):
        bridge = MagicMock()
        tokenizer = MagicMock()
        processor = MagicMock()
        tok = MagicMock(_tok=tokenizer, processor=processor)
        config = DataProxyConfig()

        with patch(
            "areal.v2.inference_service.data_proxy.app.ArealOpenAI"
        ) as areal_openai:
            client = _create_areal_client(bridge, tok, config)

        assert client is areal_openai.return_value
        areal_openai.assert_called_once_with(
            engine=bridge,
            tokenizer=tokenizer,
            processor=processor,
            tool_call_parser=config.tool_call_parser,
            reasoning_parser=config.reasoning_parser,
            engine_max_tokens=config.engine_max_tokens,
            chat_template_type=config.chat_template_type,
            require_multimodal_processor=True,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("api_type", ["chat", "responses"])
    @pytest.mark.parametrize(
        "backend_type,has_image,has_processor,error",
        [
            ("vllm", True, True, "supported only with.*SGLang"),
            ("sglang", True, False, "require a multimodal processor"),
            ("sglang", True, True, None),
            ("vllm", False, False, None),
            ("sglang", False, False, None),
            ("vllm", False, True, None),
            ("sglang", False, True, None),
        ],
    )
    async def test_generation_with_inf_bridge_enforces_multimodal_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_type: str,
        backend_type: str,
        has_image: bool,
        has_processor: bool,
        error: str | None,
    ) -> None:
        """Exercise real v2 client/bridge wiring, mocking only HF objects and HTTP."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://test.invalid/v1")
        tokenizer = MagicMock(eos_token_id=2, pad_token_id=0)
        tokenizer.apply_chat_template.side_effect = (
            lambda *args, tokenize=True, **kwargs: (
                {"input_ids": [10, 2, 20]} if tokenize else "describe"
            )
        )
        tokenizer.decode.return_value = "answer"
        processor = MagicMock(
            image_processor=MagicMock(),
            return_value={
                "input_ids": torch.tensor(
                    [[10, 2, 20]], dtype=torch.long, device="cpu"
                ),
                "pixel_values": torch.ones(
                    1, 3, 2, 2, dtype=torch.float32, device="cpu"
                ),
                "image_grid_thw": torch.tensor(
                    [[1, 1, 1]], dtype=torch.long, device="cpu"
                ),
            },
        )
        tok = MagicMock(_tok=tokenizer, processor=processor if has_processor else None)
        config = DataProxyConfig(backend_type=backend_type)
        bridge = _create_inf_bridge("http://test.invalid", PauseState(), config)
        response = (
            {
                "meta_info": {
                    "finish_reason": {"type": "length"},
                    "output_token_logprobs": [(-0.1, 77)],
                }
            }
            if backend_type == "sglang"
            else {
                "choices": [
                    {
                        "finish_reason": "length",
                        "logprobs": {
                            "tokens": ["token:77"],
                            "token_logprobs": [-0.1],
                        },
                    }
                ]
            }
        )
        with patch.object(
            bridge, "_send_request", new_callable=AsyncMock, return_value=response
        ) as send_request:
            client = _create_areal_client(bridge, tok, config)
            try:
                if has_image:
                    with BytesIO() as buffer:
                        Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
                        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                    image_url = f"data:image/png;base64,{encoded}"
                    content = (
                        [{"type": "image_url", "image_url": {"url": image_url}}]
                        if api_type == "chat"
                        else [{"type": "input_image", "image_url": image_url}]
                    )
                else:
                    content = "describe"

                messages = [{"role": "user", "content": content}]
                request = (
                    client.chat.completions.create(
                        messages=messages, max_completion_tokens=1
                    )
                    if api_type == "chat"
                    else client.responses.create(input=messages, max_output_tokens=1)
                )
                if error is not None:
                    with pytest.raises(ValueError, match=error):
                        await request
                    send_request.assert_not_awaited()
                    processor.assert_not_called()
                else:
                    completion = await request
                    send_request.assert_awaited_once()
                    interaction = client.get_interaction(completion.id)
                    assert interaction is not None
                    trajectory = interaction.to_tensor_dict()
                    assert trajectory["input_ids"].tolist() == [[10, 2, 20, 77]]
                    if has_image:
                        processor.assert_called_once()
                        torch.testing.assert_close(
                            trajectory["multi_modal_input"][0]["pixel_values"],
                            processor.return_value["pixel_values"],
                            rtol=0,
                            atol=0,
                        )
                    else:
                        processor.assert_not_called()
            finally:
                await client.close()
                await bridge.aclose()
