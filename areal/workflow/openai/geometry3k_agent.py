# SPDX-License-Identifier: Apache-2.0

import base64
import re
from io import BytesIO
from typing import Any

from agents import ModelSettings, OpenAIProvider, RunConfig, Runner
from mathruler.grader import extract_boxed_content, grade_answer
from PIL.Image import Image as ImageObject
from transformers import ProcessorMixin

from areal import workflow_context
from areal.api import AsyncRewardWrapper, InferenceEngine, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import detect_image_mime
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.workflow.openai_agent.math_agent import build_math_agent


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def geometry3k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
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
    """Convert Geometry3K chat messages and images to Responses API input."""
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
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": _image_to_data_uri(image),
                    }
                )
                image_count += 1
            elif part.get("type") == "text":
                content.append({"type": "input_text", "text": part["text"]})
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


class Geometry3KAgentWorkflow(RolloutWorkflow):
    """Run the OpenAI multi-agent math workflow on Geometry3K images."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: str,
        processor: str,
    ):
        loaded_processor, loaded_tokenizer = load_hf_processor_and_tokenizer(processor)
        if loaded_processor is None:
            raise ValueError(f"Failed to load a VLM processor from {processor}.")
        if tokenizer != processor:
            _, loaded_tokenizer = load_hf_processor_and_tokenizer(tokenizer)
        self.processor: ProcessorMixin = loaded_processor
        self.tokenizer = loaded_tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.async_reward_fn = AsyncRewardWrapper(geometry3k_reward_fn)

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            processor=self.processor,
            tool_call_parser="qwen25",
            chat_template_type="concat",
        )
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
            model_settings=ModelSettings(
                temperature=self.gconfig.temperature,
                max_tokens=self.gconfig.max_new_tokens,
            ),
        )
        try:
            await Runner.run(
                build_math_agent(),
                input=_build_agent_input(data),
                run_config=run_config,
            )

            interactions = client.export_interactions(style="individual")
            if not interactions:
                raise RuntimeError(
                    "The Geometry3K agent produced no model interactions."
                )
            last_interaction = next(reversed(interactions.values()))
            response = last_interaction.model_response
            if response is None:
                raise RuntimeError(
                    "The final Geometry3K interaction has no model response."
                )

            reward = await self.async_reward_fn(
                prompt=self.tokenizer.decode(response.input_tokens),
                completions=self.tokenizer.decode(response.output_tokens),
                prompt_ids=response.input_tokens,
                completion_ids=response.output_tokens,
                answer=data["answer"],
                **{
                    key: value
                    for key, value in data.items()
                    if key not in {"answer", "prompt", "prompt_ids", "completion_ids"}
                },
            )
            client.set_last_reward(reward)
            client.apply_reward_discount(turn_discount=0.9)
            stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)
            return client.export_interactions(style="concat")
        finally:
            await client.close()
