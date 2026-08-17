# SPDX-License-Identifier: Apache-2.0

import base64
import os
import re
from io import BytesIO
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from PIL.Image import Image as ImageObject

from areal.api import AsyncRewardWrapper
from areal.api.io_struct import detect_image_mime


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def geometry3k_reward_fn(completions: str, answer: str, **_: Any) -> float:
    format_reward_val = format_reward(completions)
    acc_reward_val = acc_reward(completions, answer)
    format_score = 0.1
    score = (1.0 - format_score) * (acc_reward_val) + format_score * format_reward_val
    return score


def _image_to_data_uri(image: bytes | str | ImageObject) -> str:
    if isinstance(image, str):
        if image.startswith(("data:image/", "http://", "https://")):
            return image
        encoded = image
    elif isinstance(image, bytes):
        encoded = base64.b64encode(image).decode("utf-8")
    elif isinstance(image, ImageObject):
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise TypeError(f"Unsupported Geometry3K image type: {type(image).__name__}")
    return f"data:{detect_image_mime(encoded)};base64,{encoded}"


def _build_agent_input(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Inject Geometry3K images into Chat Completions messages."""
    images = iter(data["images"])
    result: list[dict[str, Any]] = []
    image_count = 0

    for message in data["messages_chat"]:
        content: list[dict[str, Any]] = []
        for part in message["content"]:
            if part.get("type") == "image_url":
                try:
                    image = next(images)
                except StopIteration as exc:
                    raise ValueError(
                        "Geometry3K messages contain more image placeholders "
                        "than images."
                    ) from exc
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_to_data_uri(image),
                            "detail": "auto",
                        },
                    }
                )
                image_count += 1
            elif part.get("type") == "text":
                content.append({"type": "text", "text": part["text"]})
            else:
                raise ValueError(
                    f"Unsupported Geometry3K message content: {part.get('type')}"
                )
        result.append({"role": message["role"], "content": content})

    try:
        next(images)
    except StopIteration:
        pass
    else:
        raise ValueError(
            "Geometry3K samples contain more images than image placeholders."
        )
    if image_count == 0:
        raise ValueError("Geometry3K agent input must contain at least one image.")
    return result


class Geometry3KAgent:
    """Run a single OpenAI Chat Completions request on Geometry3K images."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs.copy()
        max_new_tokens = self.kwargs.pop("max_new_tokens", None)
        self.kwargs.pop("max_turns", None)
        if max_new_tokens is not None and "max_tokens" not in self.kwargs:
            self.kwargs["max_tokens"] = max_new_tokens
        else:
            self.kwargs.pop("max_tokens", None)
        self._reward_fn = AsyncRewardWrapper(geometry3k_reward_fn)

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )
        completion: ChatCompletion = await client.chat.completions.create(
            messages=_build_agent_input(data),
            model="default",
            **self.kwargs,
        )
        output = completion.choices[0].message.content
        if output is None:
            raise ValueError("The Geometry3K completion did not contain text output.")
        return await self._reward_fn(
            completions=output,
            answer=data["answer"],
        )
