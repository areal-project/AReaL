"""Tests for VLM image input through the OpenAI-compatible API.

Uses real base64-encoded PNG images (tiny 1x1 and 2x2 pixels) generated
at test time -- no mocks for image data.
"""

from __future__ import annotations

import base64
import io
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from PIL import Image

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.experimental.inference_service.data_proxy.pause import PauseState
from areal.experimental.inference_service.inf_bridge import InfBridge
from areal.experimental.inference_service.sglang.bridge import SGLangBridgeBackend
from areal.experimental.inference_service.vllm.bridge import VLLMBridgeBackend
from areal.experimental.openai.client import (
    _build_messages_list,
    _extract_images_from_messages,
)

# ---------------------------------------------------------------------------
# Real image fixtures
# ---------------------------------------------------------------------------


def _make_png_base64(width: int = 1, height: int = 1, color="red") -> str:
    """Create a real PNG image and return its raw base64 string (no URI prefix)."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_data_uri(b64: str, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{b64}"


def _materialize(obj):
    """Recursively convert Pydantic ValidatorIterators (and other iterables) to plain lists/dicts."""
    if isinstance(obj, dict):
        return {k: _materialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_materialize(item) for item in obj]
    if isinstance(obj, str):
        return obj
    try:
        return [_materialize(item) for item in obj]
    except TypeError:
        return obj


@pytest.fixture
def red_pixel_b64():
    return _make_png_base64(1, 1, "red")


@pytest.fixture
def blue_pixel_b64():
    return _make_png_base64(1, 1, "blue")


# =========================================================================
# _extract_images_from_messages — pure function, real image data
# =========================================================================


class TestExtractImagesFromMessages:
    def test_single_base64_image(self, red_pixel_b64):
        data_uri = _make_data_uri(red_pixel_b64)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        image_data, tok_msgs, vllm_msgs = _extract_images_from_messages(messages)

        assert len(image_data) == 1
        assert image_data[0] == red_pixel_b64

        tok_content = tok_msgs[0]["content"]
        assert tok_content[0] == {"type": "text", "text": "Describe this image"}
        assert tok_content[1] == {"type": "image"}

        vllm_content = vllm_msgs[0]["content"]
        assert vllm_content[1]["type"] == "image_url"
        assert vllm_content[1]["image_url"]["url"] == "placeholder"

    def test_multiple_images_in_single_message(self, red_pixel_b64, blue_pixel_b64):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _make_data_uri(red_pixel_b64)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _make_data_uri(blue_pixel_b64)},
                    },
                ],
            }
        ]

        image_data, tok_msgs, vllm_msgs = _extract_images_from_messages(messages)

        assert len(image_data) == 2
        assert image_data[0] == red_pixel_b64
        assert image_data[1] == blue_pixel_b64

        tok_content = tok_msgs[0]["content"]
        assert len(tok_content) == 3
        assert tok_content[1] == {"type": "image"}
        assert tok_content[2] == {"type": "image"}

    def test_images_across_multiple_messages(self, red_pixel_b64, blue_pixel_b64):
        messages = [
            {"role": "system", "content": "You are a VLM assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _make_data_uri(red_pixel_b64)},
                    },
                ],
            },
            {"role": "assistant", "content": "I see a red pixel."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Second image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _make_data_uri(blue_pixel_b64)},
                    },
                ],
            },
        ]

        image_data, tok_msgs, vllm_msgs = _extract_images_from_messages(messages)

        assert len(image_data) == 2
        assert image_data[0] == red_pixel_b64
        assert image_data[1] == blue_pixel_b64

        assert tok_msgs[0] == {"role": "system", "content": "You are a VLM assistant."}
        assert tok_msgs[2] == {"role": "assistant", "content": "I see a red pixel."}

    def test_http_url_preserved_as_is(self):
        url = "https://example.com/photo.jpg"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]

        image_data, tok_msgs, _ = _extract_images_from_messages(messages)

        assert len(image_data) == 1
        assert image_data[0] == url
        assert tok_msgs[0]["content"][1] == {"type": "image"}

    def test_no_images_returns_empty(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        image_data, tok_msgs, vllm_msgs = _extract_images_from_messages(messages)

        assert image_data == []
        assert tok_msgs[0] == messages[0]
        assert tok_msgs[1] == messages[1]

    def test_string_content_passes_through(self):
        messages = [{"role": "user", "content": "Hello"}]

        image_data, tok_msgs, vllm_msgs = _extract_images_from_messages(messages)

        assert image_data == []
        assert tok_msgs[0]["content"] == "Hello"
        assert vllm_msgs[0]["content"] == "Hello"

    def test_original_messages_not_mutated(self, red_pixel_b64):
        data_uri = _make_data_uri(red_pixel_b64)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        original = deepcopy(messages)

        _extract_images_from_messages(messages)

        assert messages == original

    def test_jpeg_data_uri(self):
        jpeg_b64 = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg").decode()
        data_uri = _make_data_uri(jpeg_b64, mime="image/jpeg")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        image_data, _, _ = _extract_images_from_messages(messages)

        assert len(image_data) == 1
        assert image_data[0] == jpeg_b64

    def test_detail_parameter_ignored_for_extraction(self, red_pixel_b64):
        data_uri = _make_data_uri(red_pixel_b64)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "high"},
                    },
                ],
            }
        ]

        image_data, _, _ = _extract_images_from_messages(messages)

        assert len(image_data) == 1
        assert image_data[0] == red_pixel_b64

    def test_real_png_roundtrip(self):
        """Create a real 4x4 image, encode it, extract it, and decode back."""
        original = Image.new("RGB", (4, 4), color=(42, 128, 200))
        buf = io.BytesIO()
        original.save(buf, format="PNG")
        raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = _make_data_uri(raw_b64)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color?"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        image_data, _, _ = _extract_images_from_messages(messages)

        decoded_bytes = base64.b64decode(image_data[0])
        decoded_img = Image.open(io.BytesIO(decoded_bytes))
        assert decoded_img.size == (4, 4)
        assert decoded_img.getpixel((0, 0)) == (42, 128, 200)


# =========================================================================
# SGLangBridgeBackend — image_data forwarding
# =========================================================================


class TestSGLangBridgeBackendImageForwarding:
    def test_image_data_included_in_payload(self, red_pixel_b64):
        backend = SGLangBridgeBackend()
        gconfig = GenerationHyperparameters(
            n_samples=1,
            max_new_tokens=10,
            max_tokens=32768,
        )
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            image_data=[red_pixel_b64],
        )

        http_req = backend.build_generation_request(req, with_lora=False, version=0)

        assert http_req.endpoint == "/generate"
        assert http_req.payload["image_data"] == [red_pixel_b64]
        assert http_req.payload["input_ids"] == [1, 2, 3]

    def test_image_data_none_included_in_payload(self):
        backend = SGLangBridgeBackend()
        gconfig = GenerationHyperparameters(
            n_samples=1,
            max_new_tokens=10,
            max_tokens=32768,
        )
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            image_data=None,
        )

        http_req = backend.build_generation_request(req, with_lora=False, version=0)

        assert http_req.payload["image_data"] is None

    def test_multiple_images_in_payload(self, red_pixel_b64, blue_pixel_b64):
        backend = SGLangBridgeBackend()
        gconfig = GenerationHyperparameters(
            n_samples=1,
            max_new_tokens=10,
            max_tokens=32768,
        )
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            image_data=[red_pixel_b64, blue_pixel_b64],
        )

        http_req = backend.build_generation_request(req, with_lora=False, version=0)

        assert http_req.payload["image_data"] == [red_pixel_b64, blue_pixel_b64]


# =========================================================================
# VLLMBridgeBackend — vision message construction with real images
# =========================================================================


class TestVLLMBridgeBackendVisionMessages:
    def test_vision_messages_get_real_data_uris(self, red_pixel_b64):
        backend = VLLMBridgeBackend()
        gconfig = GenerationHyperparameters(
            n_samples=1,
            max_new_tokens=10,
            max_tokens=32768,
        )
        vision_msgs = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "placeholder"},
                        },
                    ],
                }
            ]
        ]
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            image_data=[red_pixel_b64],
            vision_msg_vllm=vision_msgs,
        )

        http_req = backend.build_generation_request(req, with_lora=False, version=0)

        assert http_req.endpoint == "/v1/chat/completions"
        msg_content = http_req.payload["messages"][0]["content"]
        image_part = msg_content[1]
        assert image_part["type"] == "image_url"
        expected_uri = f"data:image/png;base64,{red_pixel_b64}"
        assert image_part["image_url"]["url"] == expected_uri

    def test_multiple_vision_images_injected(self, red_pixel_b64, blue_pixel_b64):
        backend = VLLMBridgeBackend()
        gconfig = GenerationHyperparameters(
            n_samples=1,
            max_new_tokens=10,
            max_tokens=32768,
        )
        vision_msgs = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare"},
                        {"type": "image_url", "image_url": {"url": "placeholder"}},
                        {"type": "image_url", "image_url": {"url": "placeholder"}},
                    ],
                }
            ]
        ]
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            image_data=[red_pixel_b64, blue_pixel_b64],
            vision_msg_vllm=vision_msgs,
        )

        http_req = backend.build_generation_request(req, with_lora=False, version=0)

        msg_content = http_req.payload["messages"][0]["content"]
        assert (
            msg_content[1]["image_url"]["url"]
            == f"data:image/png;base64,{red_pixel_b64}"
        )
        assert (
            msg_content[2]["image_url"]["url"]
            == f"data:image/png;base64,{blue_pixel_b64}"
        )

    def test_mismatched_image_count_raises(self, red_pixel_b64):
        backend = VLLMBridgeBackend()
        gconfig = GenerationHyperparameters(
            n_samples=1,
            max_new_tokens=10,
            max_tokens=32768,
        )
        vision_msgs = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "placeholder"}},
                        {"type": "image_url", "image_url": {"url": "placeholder"}},
                    ],
                }
            ]
        ]
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            image_data=[red_pixel_b64],
            vision_msg_vllm=vision_msgs,
        )

        with pytest.raises(ValueError, match="Not enough images"):
            backend.build_generation_request(req, with_lora=False, version=0)

    def test_real_image_survives_data_uri_roundtrip(self):
        """Encode a real 2x2 PNG, build vLLM request, decode from the payload."""
        original = Image.new("RGB", (2, 2), color=(255, 0, 0))
        buf = io.BytesIO()
        original.save(buf, format="PNG")
        raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        backend = VLLMBridgeBackend()
        gconfig = GenerationHyperparameters(
            n_samples=1, max_new_tokens=10, max_tokens=32768
        )
        vision_msgs = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "placeholder"}},
                    ],
                }
            ]
        ]
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            image_data=[raw_b64],
            vision_msg_vllm=vision_msgs,
        )

        http_req = backend.build_generation_request(req, with_lora=False, version=0)

        full_url = http_req.payload["messages"][0]["content"][0]["image_url"]["url"]
        assert full_url.startswith("data:image/png;base64,")
        b64_payload = full_url.split(",", 1)[1]
        decoded = base64.b64decode(b64_payload)
        recovered = Image.open(io.BytesIO(decoded))
        assert recovered.size == (2, 2)


# =========================================================================
# InfBridge — image_data preserved through abort/resubmit loop
# =========================================================================


def _make_sglang_response(
    token_logprobs: list[tuple[float, int]],
    finish_reason_type: str = "stop",
) -> dict[str, Any]:
    return {
        "meta_info": {
            "finish_reason": {"type": finish_reason_type},
            "output_token_logprobs": token_logprobs,
        },
    }


class TestInfBridgeImagePreservation:
    @pytest.mark.asyncio
    async def test_image_data_present_in_initial_request(self, red_pixel_b64):
        """image_data is forwarded in the first HTTP payload."""
        captured_payloads: list[dict] = []

        async def mock_send(http_req, **kwargs):
            captured_payloads.append(dict(http_req.payload))
            return _make_sglang_response([(-0.5, 100)], "stop")

        pause_state = PauseState()
        bridge = InfBridge(
            backend=SGLangBridgeBackend(),
            backend_addr="http://mock",
            pause_state=pause_state,
            resubmit_wait=0.01,
        )
        bridge._send_request = mock_send

        gconfig = GenerationHyperparameters(
            n_samples=1, max_new_tokens=10, max_tokens=32768
        )
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            image_data=[red_pixel_b64],
        )

        resp = await bridge.agenerate(req)

        assert len(captured_payloads) == 1
        assert captured_payloads[0]["image_data"] == [red_pixel_b64]
        assert resp.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_image_data_persists_through_resubmit(self, red_pixel_b64):
        """image_data stays in payload across abort/resubmit iterations."""
        captured_payloads: list[dict] = []
        call_count = 0

        async def mock_send(http_req, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(dict(http_req.payload))
            if call_count == 1:
                return _make_sglang_response([(-0.5, 100)], "abort")
            return _make_sglang_response([(-0.3, 200)], "stop")

        pause_state = PauseState()
        bridge = InfBridge(
            backend=SGLangBridgeBackend(),
            backend_addr="http://mock",
            pause_state=pause_state,
            resubmit_wait=0.01,
        )
        bridge._send_request = mock_send

        gconfig = GenerationHyperparameters(
            n_samples=1, max_new_tokens=20, max_tokens=32768
        )
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            image_data=[red_pixel_b64],
        )

        resp = await bridge.agenerate(req)

        assert call_count == 2
        assert captured_payloads[0]["image_data"] == [red_pixel_b64]
        assert captured_payloads[1]["image_data"] == [red_pixel_b64]
        assert resp.output_tokens == [100, 200]


# =========================================================================
# Data proxy /chat/completions — image messages passed to ArealOpenAI
# =========================================================================


ADMIN_KEY = "areal-admin-key"


def _make_mock_areal_client_for_images():
    """Mock ArealOpenAI client that captures kwargs and returns a ChatCompletion."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    mock_client = MagicMock()
    captured_calls: list[dict] = []

    completion = ChatCompletion(
        id="chatcmpl-vision-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content="I see a red pixel.", role="assistant"
                ),
            )
        ],
        created=1234567890,
        model="vlm-model",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=10, total_tokens=15),
    )

    async def _mock_create(*, areal_cache=None, **kwargs):
        import torch

        raw_messages = kwargs.get("messages", [])
        messages = (
            list(raw_messages) if not isinstance(raw_messages, list) else raw_messages
        )
        kwargs["messages"] = messages
        captured_calls.append(kwargs)

        interaction = InteractionWithTokenLogpReward(
            messages=messages if isinstance(messages, list) else list(messages),
            completion=completion,
            output_message_list=[
                {"role": "assistant", "content": "I see a red pixel."}
            ],
        )
        interaction._cache = {
            "input_ids": torch.tensor([[100, 200, 300, 400, 500, 2]]),
            "loss_mask": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "logprobs": torch.tensor([[0.0, 0.0, 0.0, -0.5, -0.3, -0.1]]),
            "versions": torch.tensor([[-1, -1, -1, 0, 0, 0]]),
            "attention_mask": torch.ones(6, dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([0.0]),
        }
        if areal_cache is not None:
            areal_cache[completion.id] = interaction
        return completion

    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)
    mock_client._captured_calls = captured_calls
    return mock_client


@pytest_asyncio.fixture
async def image_test_client():
    """Data proxy test client wired with an image-capturing mock."""
    from areal.experimental.inference_service.data_proxy.app import create_app
    from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
    from areal.experimental.inference_service.data_proxy.session import SessionStore

    config = DataProxyConfig(
        host="127.0.0.1",
        port=18082,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
    )
    mock_client = _make_mock_areal_client_for_images()

    mock_tok = MagicMock()
    mock_tok._tok = MagicMock()
    mock_tok._tok.eos_token_id = 2
    mock_tok._tok.pad_token_id = 0

    app = create_app(config)

    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=5,
        resubmit_wait=0.01,
    )
    app.state.tokenizer = mock_tok
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = mock_client
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = SessionStore()
    app.state.version = 0

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, mock_client


class TestDataProxyImagePassthrough:
    @pytest.mark.asyncio
    async def test_image_url_messages_passed_through(self, image_test_client):
        client, mock_areal = image_test_client
        b64 = _make_png_base64(1, 1, "green")
        data_uri = _make_data_uri(b64)

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "vlm-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "I see a red pixel."

        assert len(mock_areal._captured_calls) == 1
        call_kwargs = mock_areal._captured_calls[0]
        messages = _materialize(call_kwargs["messages"])
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][1]["type"] == "image_url"
        assert messages[0]["content"][1]["image_url"]["url"] == data_uri

    @pytest.mark.asyncio
    async def test_session_lifecycle_with_image_messages(self, image_test_client):
        """Full lifecycle: start → image chat → reward → end → export."""
        client, mock_areal = image_test_client
        b64 = _make_png_base64(2, 2, "yellow")
        data_uri = _make_data_uri(b64)

        # 1. Start session
        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "vision-lifecycle"},
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        )
        assert resp.status_code == 201
        session_id = resp.json()["sessions"][0]["session_id"]
        api_key = resp.json()["sessions"][0]["session_api_key"]

        # 2. Chat with image
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "vlm-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What do you see?"},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["object"] == "chat.completion"

        # 3. Set reward (finishes the session since set_reward_finish_timeout=0)
        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["interaction_count"] == 1

        # 4. Export trajectories
        resp = await client.post(
            "/export_trajectories",
            json={
                "session_ids": [session_id],
                "discount": 1.0,
                "style": "individual",
            },
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "interactions" in data
        assert len(data["interactions"]) == 1

    @pytest.mark.asyncio
    async def test_mixed_text_and_image_messages(self, image_test_client):
        """Multiple messages with only some containing images."""
        client, mock_areal = image_test_client
        b64 = _make_png_base64(1, 1, "white")
        data_uri = _make_data_uri(b64)

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "vlm-model",
                "messages": [
                    {"role": "system", "content": "You are a vision assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Look at this:"},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    },
                ],
            },
        )

        assert resp.status_code == 200
        call_kwargs = mock_areal._captured_calls[-1]
        msgs = _materialize(call_kwargs["messages"])
        assert msgs[0] == {"role": "system", "content": "You are a vision assistant."}
        assert msgs[1]["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_plain_text_still_works(self, image_test_client):
        """Non-image requests still work after image support is added."""
        client, mock_areal = image_test_client

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "text-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert resp.status_code == 200
        call_kwargs = mock_areal._captured_calls[-1]
        msgs = _materialize(call_kwargs["messages"])
        assert msgs[0] == {"role": "user", "content": "Hello"}


# =========================================================================
# _build_messages_list — Responses API input → Chat Completions messages
# =========================================================================


class TestBuildMessagesList:
    def test_string_content(self):
        result = _build_messages_list({"role": "user", "content": "Hello"})
        assert result == [{"role": "user", "content": "Hello"}]

    def test_output_text_flattened(self):
        item = {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Answer A"}],
        }
        result = _build_messages_list(item)
        assert result == [{"role": "assistant", "content": "Answer A"}]

    def test_multiple_output_text_each_becomes_message(self):
        item = {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Part 1"},
                {"type": "output_text", "text": "Part 2"},
            ],
        }
        result = _build_messages_list(item)
        assert result == [
            {"role": "assistant", "content": "Part 1"},
            {"role": "assistant", "content": "Part 2"},
        ]

    def test_input_text_flattened(self):
        item = {
            "role": "user",
            "content": [{"type": "input_text", "text": "Describe this"}],
        }
        result = _build_messages_list(item)
        assert result == [{"role": "user", "content": "Describe this"}]

    def test_input_image_produces_multimodal_message(self, red_pixel_b64):
        data_uri = _make_data_uri(red_pixel_b64)
        item = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "image_url": data_uri},
            ],
        }
        result = _build_messages_list(item)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0] == {"type": "text", "text": "What is this?"}
        assert msg["content"][1] == {
            "type": "image_url",
            "image_url": {"url": data_uri},
        }

    def test_input_image_with_detail_forwarded(self, red_pixel_b64):
        data_uri = _make_data_uri(red_pixel_b64)
        item = {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": data_uri, "detail": "high"},
            ],
        }
        result = _build_messages_list(item)
        assert len(result) == 1
        img_part = result[0]["content"][0]
        assert img_part == {
            "type": "image_url",
            "image_url": {"url": data_uri, "detail": "high"},
        }

    def test_input_image_without_detail_omits_key(self, red_pixel_b64):
        data_uri = _make_data_uri(red_pixel_b64)
        item = {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": data_uri},
            ],
        }
        result = _build_messages_list(item)
        img_part = result[0]["content"][0]
        assert "detail" not in img_part["image_url"]

    def test_input_image_missing_image_url_raises(self):
        item = {
            "role": "user",
            "content": [{"type": "input_image"}],
        }
        with pytest.raises(ValueError, match="image_url"):
            _build_messages_list(item)

    def test_input_image_empty_image_url_raises(self):
        item = {
            "role": "user",
            "content": [{"type": "input_image", "image_url": ""}],
        }
        with pytest.raises(ValueError, match="image_url"):
            _build_messages_list(item)

    def test_mixed_text_and_image(self, red_pixel_b64):
        data_uri = _make_data_uri(red_pixel_b64)
        item = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Look at this image"},
                {"type": "input_image", "image_url": data_uri},
                {"type": "input_text", "text": "and describe it"},
            ],
        }
        result = _build_messages_list(item)
        assert len(result) == 1
        parts = result[0]["content"]
        assert len(parts) == 3
        assert parts[0] == {"type": "text", "text": "Look at this image"}
        assert parts[1]["type"] == "image_url"
        assert parts[2] == {"type": "text", "text": "and describe it"}

    def test_unsupported_content_type_raises(self):
        item = {
            "role": "user",
            "content": [{"type": "input_file", "file_id": "file-123"}],
        }
        with pytest.raises(ValueError, match="Unsupported content format: input_file"):
            _build_messages_list(item)

    def test_non_dict_content_part_raises(self):
        item = {
            "role": "user",
            "content": ["plain string"],
        }
        with pytest.raises(ValueError, match="Unsupported content format"):
            _build_messages_list(item)

    def test_function_call_output_converted(self):
        item = {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": '{"result": 42}',
        }
        result = _build_messages_list(item)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == '{"result": 42}'
        assert result[0]["tool_call_id"] == "call_abc"

    def test_function_call_output_without_call_id(self):
        item = {
            "type": "function_call_output",
            "output": "done",
        }
        result = _build_messages_list(item)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "done"
        assert "tool_call_id" not in result[0]


# =========================================================================
# _extract_images_from_messages — empty URL validation
# =========================================================================


class TestExtractImagesValidation:
    def test_empty_url_raises(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}},
                ],
            }
        ]
        with pytest.raises(ValueError, match="empty or missing URL"):
            _extract_images_from_messages(messages)

    def test_missing_url_key_raises(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {}},
                ],
            }
        ]
        with pytest.raises(ValueError, match="empty or missing URL"):
            _extract_images_from_messages(messages)

    def test_missing_image_url_obj_raises(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url"},
                ],
            }
        ]
        with pytest.raises(ValueError, match="empty or missing URL"):
            _extract_images_from_messages(messages)

    def test_non_dict_image_url_obj_raises(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "not-a-dict"},
                ],
            }
        ]
        with pytest.raises(ValueError, match="empty or missing URL"):
            _extract_images_from_messages(messages)
