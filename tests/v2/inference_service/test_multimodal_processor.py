"""Unit tests for v2 Data Proxy multimodal processor wiring."""

from unittest.mock import MagicMock, patch

from areal.v2.inference_service.data_proxy.app import _create_areal_client
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
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
    def test_client_requires_processor_for_image_requests(self):
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
