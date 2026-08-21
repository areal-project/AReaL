# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from areal.workflow.openai import geometry3k_agent


def _png_bytes() -> bytes:
    with BytesIO() as buffer:
        Image.new("RGB", (2, 2), color="blue").save(buffer, format="PNG")
        return buffer.getvalue()


def test_geometry3k_reward_combines_accuracy_and_format_scores(monkeypatch):
    """The agent workflow must preserve Geometry3K's 90/10 reward weighting."""
    monkeypatch.setattr(geometry3k_agent, "acc_reward", lambda *_: 0.5)
    monkeypatch.setattr(geometry3k_agent, "format_reward", lambda *_: 1.0)

    reward = geometry3k_agent.geometry3k_reward_fn(
        prompt="prompt",
        completions="completion",
        prompt_ids=[1],
        completion_ids=[2],
        answer="answer",
    )

    assert reward == pytest.approx(0.55)


def test_build_agent_input_preserves_text_and_embeds_image():
    """Geometry3K image placeholders should become Responses API image items."""
    data = {
        "images": [_png_bytes()],
        "messages_chat": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}},
                    {"type": "text", "text": "Find the missing angle."},
                ],
            }
        ],
    }

    result = geometry3k_agent._build_agent_input(data)

    assert result[0]["role"] == "user"
    image_part, text_part = result[0]["content"]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    encoded = image_part["image_url"]["url"].split(",", maxsplit=1)[1]
    assert base64.b64decode(encoded) == data["images"][0]
    assert text_part == {"type": "text", "text": "Find the missing angle."}


@pytest.mark.parametrize(
    ("images", "content", "message"),
    [
        (
            [],
            [{"type": "image_url", "image_url": {"url": ""}}],
            "more image placeholders",
        ),
        (
            [_png_bytes(), _png_bytes()],
            [
                {"type": "image_url", "image_url": {"url": ""}},
                {"type": "text", "text": "question"},
            ],
            "more images",
        ),
    ],
)
def test_build_agent_input_rejects_image_placeholder_mismatch(images, content, message):
    """Every image must correspond to exactly one message placeholder."""
    data = {
        "images": images,
        "messages_chat": [{"role": "user", "content": content}],
    }

    with pytest.raises(ValueError, match=message):
        geometry3k_agent._build_agent_input(data)


@pytest.mark.asyncio
async def test_geometry3k_agent_run_uses_proxy_client_and_returns_reward(monkeypatch):
    """The standalone agent should use proxy credentials and return final reward."""
    captured = {}
    fake_http_client = object()

    async def fake_create(**kwargs):
        captured["request"] = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="final answer"))]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    def fake_openai_client(**kwargs):
        captured["client_kwargs"] = kwargs
        return fake_client

    monkeypatch.setattr(geometry3k_agent, "AsyncOpenAI", fake_openai_client)
    monkeypatch.setattr(geometry3k_agent, "acc_reward", lambda *_: 1.0)
    monkeypatch.setattr(geometry3k_agent, "format_reward", lambda *_: 1.0)

    agent = geometry3k_agent.Geometry3KAgent(
        temperature=0.8,
        max_tokens=128,
        max_completion_tokens=64,
    )
    reward = await agent.run(
        {
            "answer": "42",
            "images": [_png_bytes()],
            "messages_chat": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": ""}},
                        {"type": "text", "text": "Solve this geometry problem."},
                    ],
                }
            ],
        },
        base_url="http://proxy.example/v1",
        api_key="session-key",
        http_client=fake_http_client,
    )

    assert reward == pytest.approx(1.0)
    assert captured["client_kwargs"] == {
        "base_url": "http://proxy.example/v1",
        "api_key": "session-key",
        "http_client": fake_http_client,
        "max_retries": 0,
    }
    assert captured["request"]["model"] == "default"
    assert captured["request"]["temperature"] == pytest.approx(0.8)
    assert captured["request"]["max_completion_tokens"] == 64
    assert "max_tokens" not in captured["request"]
    assert captured["request"]["messages"][0]["content"][0]["type"] == "image_url"
