# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO

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
    assert image_part["type"] == "input_image"
    assert image_part["image_url"].startswith("data:image/png;base64,")
    encoded = image_part["image_url"].split(",", maxsplit=1)[1]
    assert base64.b64decode(encoded) == data["images"][0]
    assert text_part == {"type": "input_text", "text": "Find the missing angle."}


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
