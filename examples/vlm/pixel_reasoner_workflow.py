"""Multi-turn Pixel visual-tool episodes for RL training."""

import asyncio
import itertools
import json
import math
import re
import subprocess
import tempfile
import unicodedata
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.client import _process_multimodal_prompt
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra.processor_cache import ProcessorCallCache
from areal.utils import stats_tracker
from areal.utils.image import image2base64
from areal.utils.qwen_vl_image import (
    prepare_sglang_qwen_images,
    sglang_qwen_image_kwargs,
)
from areal.utils.sglang_image import image_prompt_metadata

SYSTEM = r"""Inspect the images and answer the question. You may use
crop_image_normalized with bbox_2d=[x1,y1,x2,y2] in [0,1] and target_image
(a 1-based image index). Returned crops receive new image indices and may
also be cropped. Request one crop at a time using exactly:
<tool_call>{"name":"crop_image_normalized","arguments":{"bbox_2d":[0,0,1,1],"target_image":1}}</tool_call>
End your response after </tool_call> and wait for the actual image feedback.
When ready, put your final answer in \boxed{}."""


VIDEO_SYSTEM = (
    SYSTEM
    + r"""
You are given a video as an ordered frame sequence. Use select_frames to
inspect selected frames in detail:
<tool_call>{"name":"select_frames","arguments":{"target_frames":[1,2]}}</tool_call>
target_frames contains 1-based indices into the original video, at most 8 per call.
Before any tool images exist, crop_image_normalized.target_image refers to a
video frame. Afterwards the video occupies index 1, and tool images are numbered
from 2 in the order returned. Put the final option letter within \boxed{}.
"""
)


async def prepare_video_data(frames, destination):
    # Use the same container implementation as the server, preserving its geometry.
    from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
    from sglang.srt.utils import load_video

    transport = await asyncio.to_thread(encode_video_frames, frames, destination)

    def preprocess():
        # The async SGLang helper performs CPU decoding/resizing synchronously.
        decoded = load_video(str(destination), use_gpu=False)
        return asyncio.run(preprocess_video(decoded))

    pixels, metadata = await asyncio.to_thread(preprocess)
    actual = [int(i) for i in metadata["frames_indices"]]
    if actual != list(range(transport["transport_frames"])):
        raise ValueError(f"Server preprocessing changed frame order: {actual}")
    return {
        "path": str(destination),
        "pixels": pixels,
        "metadata": metadata,
        "transport": transport,
    }


def bounded_image(image: Image.Image, max_pixels: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    if scale < 1:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def crop_image(image: Image.Image, arguments: dict) -> Image.Image:
    """Normalized coordinates, with 0.1 padding as in the original collector."""
    box = arguments.get("bbox_2d")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in box
        )
    ):
        raise ValueError("bbox_2d must contain four finite coordinates in [0,1]")
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Require x1<x2 and y1<y2")
    crop = image.crop(
        (
            max(0, x1 - 0.1) * image.width,
            max(0, y1 - 0.1) * image.height,
            min(1, x2 + 0.1) * image.width,
            min(1, y2 + 0.1) * image.height,
        )
    )
    if min(crop.size) <= 28:
        raise ValueError("Crop dimensions must both exceed 28 pixels")
    return crop


def encode_video_frames(frames: list[Image.Image], destination: Path) -> dict:
    """Lossless transport of the supplied frame sequence, not original video timing.

    SGLang samples at 2 fps by default. Encode at the same rate and pad only
    the final frame for Qwen's temporal patch size of two, retaining every
    original frame. Tools continue to use the unpadded source indices.
    """
    if not frames or any(frame.size != frames[0].size for frame in frames):
        raise ValueError("Video frames must be nonempty and have identical dimensions")
    width, height = frames[0].size
    transported = frames + frames[-1:] if len(frames) % 2 else frames
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "2",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            "-y",
            str(destination),
        ],
        input=b"".join(frame.convert("RGB").tobytes() for frame in transported),
        capture_output=True,
        check=True,
        timeout=120,
    )
    return {
        "source_frames": len(frames),
        "transport_frames": len(transported),
        "transport_fps": 2,
        "timing": "synthetic; original timestamps unavailable",
        "padded_last_frame": len(transported) != len(frames),
    }


def select_frames(
    frames: list[Image.Image], target_frames: list[int]
) -> list[Image.Image]:
    """Official Pixel select_frames: 1-based indices, at most eight frames.

    Preserve order and repeats, as upstream does. Reject nonpositive indices
    explicitly instead of accidentally accepting Python's negative indexing.
    """
    if (
        not isinstance(target_frames, list)
        or not 1 <= len(target_frames) <= 8
        or any(type(i) is not int or not 1 <= i <= len(frames) for i in target_frames)
    ):
        raise ValueError(f"Select 1 to 8 frames with indices in [1,{len(frames)}]")
    return [frames[i - 1] for i in target_frames]


def video_crop_source(
    frames: list[Image.Image], tool_images: list[Image.Image], target_image: int
) -> Image.Image:
    """Match upstream RL indexing: video occupies item 1 after tool images exist."""
    if type(target_image) is not int:
        raise ValueError("target_image must be an integer")
    if not tool_images:
        if not 1 <= target_image <= len(frames):
            raise ValueError("target_image is outside the video frame range")
        return frames[target_image - 1]
    if not 2 <= target_image <= len(tool_images) + 1:
        raise ValueError("target_image must refer to a tool image numbered from 2")
    return tool_images[target_image - 2]


def split_reasoning(text: str) -> tuple[str, str]:
    """Handle an opening think token supplied either by the prompt or generation."""
    text = text.lstrip()
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.removeprefix("<think>"), content
    if text.startswith("<think>"):
        return text[len("<think>") :], ""
    return "", text


def get_action(content: str) -> dict | None:
    if "<tool_call>" not in content and "</tool_call>" not in content:
        return None
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.S)
    if len(matches) != 1 or not content.rstrip().endswith("</tool_call>"):
        raise ValueError("Expected one complete tool_call at the end")
    action = json.loads(matches[0])
    if (
        not isinstance(action, dict)
        or action.get("name") not in {"crop_image_normalized", "select_frames"}
        or not isinstance(action.get("arguments"), dict)
    ):
        raise ValueError("Expected crop_image_normalized with arguments")
    return action


def boxed_answer(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += len(r"\boxed{")
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index]
    return None


def verify(content: str, answers: list[str]) -> dict:
    """Normalized boxed-answer exact match for a single reference label."""
    prediction = boxed_answer(content)
    if prediction is None or len(answers) != 1:
        return {"status": "needs_review", "prediction": prediction}

    def normalize(value):
        return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()

    gold = boxed_answer(answers[0])
    matched = normalize(prediction) == normalize(
        gold if gold is not None else answers[0]
    )
    return {
        "status": "matched" if matched else "needs_review",
        "prediction": prediction,
    }


def pixel_reward_fn(completions: str, answer: list[str]) -> float:
    """Example outcome reward, without Pixel's curiosity/exploration bonuses."""
    return float(verify(completions, answer)["status"] == "matched")


class _CachedVideoProcessor:
    """Reuse one immutable video's features while rendering fresh turn tokens."""

    def __init__(self, processor, cache: ProcessorCallCache):
        self.processor = processor
        self.cache = cache

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        # This cache belongs to one prepared video, never to arbitrary inputs.
        key = self.cache.make_key(id(self.processor), kwargs)

        def process():
            return {
                name: value.detach().cpu() if torch.is_tensor(value) else value
                for name, value in self.processor(*args, **kwargs).items()
            }

        result = self.cache.get_or_compute(key, process)
        # HF pops video_metadata and may change its fps. Only tensors are shared.
        return {
            name: value if torch.is_tensor(value) else deepcopy(value)
            for name, value in result.items()
        }


class PixelReasonerWorkflow(RolloutWorkflow):
    """One independent training row per turn, one logical rollout per episode.

    Re-rendered history need not preserve the sampled token prefix. Earlier
    responses are masked as context in later rows; their actions are trained
    exactly once in the row that generated them.
    """

    def __init__(
        self,
        processor: Any,
        gconfig: GenerationHyperparameters,
        data_root: str,
        max_turns: int | None = None,
        max_output_tokens: int | None = None,
        original_max_pixels: int = 1568000,
        crop_max_pixels: int = 784000,
        video_temp_root: str | None = None,
    ):
        if max_turns is not None and max_turns < 1:
            raise ValueError("max_turns must be positive or null")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive or null")
        if min(original_max_pixels, crop_max_pixels) < 1:
            raise ValueError("Pixel limits must be positive")
        if isinstance(processor, str):
            processor = AutoProcessor.from_pretrained(processor)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.data_root = Path(data_root).resolve()
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.original_max_pixels = original_max_pixels
        self.crop_max_pixels = crop_max_pixels
        self.video_temp_root = video_temp_root
        self.reward_fn = AsyncRewardWrapper(pixel_reward_fn)
        self._video_groups: dict[ProcessorCallCache, dict[tuple, asyncio.Task]] = {}

    def prepare(self, messages: list[dict], images: list[Image.Image], video=None):
        """Prepare exact actor tokens/features and separate native video wire IDs."""
        template_kwargs = dict(enable_thinking=True, preserve_thinking=True)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        for message in messages:
            reasoning = message.get("reasoning_content", "").strip()
            if reasoning and reasoning not in prompt:
                raise ValueError("Chat template removed reasoning_content")
        image_data = image2base64(images)
        if video is None:
            prepared = _process_multimodal_prompt(
                self.processor,
                self.tokenizer,
                messages,
                image_data,
                None,
                template_kwargs,
                align_sglang_images=True,
                processor_cache=workflow_context.get().processor_cache,
            )
            return (
                prepared.input_ids,
                prepared.image_data or image_data,
                prepared.mm_token_type_ids,
                prepared.multi_modal_input,
                None,
            )

        # Native video expands timestamps and frame tokens on the server. Keep
        # its compact template separately; actor training needs the expanded IDs.
        # The image-only explicit-spans transport cannot disambiguate video text.
        visual_tokens = (
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        )
        for message in messages:
            text = message.get("reasoning_content", "")
            content = message.get("content", "")
            text += (
                content
                if isinstance(content, str)
                else "".join(part.get("text", "") for part in content)
            )
            if any(token in text for token in visual_tokens):
                raise ValueError(
                    "Literal visual special tokens in video text are unsupported"
                )
        aligned = prepare_sglang_qwen_images(
            self.processor,
            images,
            cache=workflow_context.get().processor_cache,
            image_keys=image_data,
        )
        processor = self.processor
        if getattr(processor, "video_processor", None) is not None:
            # A per-call shallow copy leaves the shared processor unchanged.
            processor = copy(processor)
            processor.video_processor = _CachedVideoProcessor(
                self.processor.video_processor,
                video.setdefault("processor_cache", ProcessorCallCache()),
            )
        processed = processor(
            text=[prompt],
            images=(aligned[0] if aligned else images) or None,
            videos=[video["pixels"]],
            video_metadata=[video["metadata"]],
            do_sample_frames=False,
            padding=False,
            return_tensors="pt",
            **(
                {"images_kwargs": sglang_qwen_image_kwargs(self.processor)}
                if aligned
                else {}
            ),
        )
        input_ids = processed["input_ids"][0].tolist()
        token_types = processed.get("mm_token_type_ids")
        if token_types is None:
            token_types = torch.tensor(
                [
                    [
                        1
                        if token == self.processor.image_token_id
                        else 2
                        if token == self.processor.video_token_id
                        else 0
                        for token in input_ids
                    ]
                ]
            )
        payload = {
            key: (
                processed[key]
                if key in {"pixel_values_videos", "video_grid_thw"}
                and processor is not self.processor
                else processed[key].detach().cpu()
            )
            for key in (
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
            )
            if key in processed
        }
        if not {"pixel_values_videos", "video_grid_thw"} <= payload.keys():
            raise ValueError("Video processor must return pixels and video_grid_thw")
        wire_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if wire_ids.count(self.processor.video_token_id) != 1 or wire_ids.count(
            self.processor.image_token_id
        ) != len(images):
            raise ValueError(
                "Native video template must have one placeholder per visual input"
            )
        return input_ids, image_data, token_types[0].tolist(), payload, wire_ids

    async def load_image(self, name: str) -> Image.Image:
        path = (self.data_root / name).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError("Image path outside data_root")

        def load():
            with Image.open(path) as image:
                return image.convert("RGB")

        return await asyncio.to_thread(load)

    async def arun_episode(
        self, engine, data: dict[str, Any]
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        if not data.get("is_video", False):
            return await self._run_episode(engine, data, None)
        group_cache = workflow_context.get().processor_cache
        if group_cache is None:
            video = await self._prepare_video(data)
            try:
                return await self._run_episode(engine, data, video)
            finally:
                await self._release_video(video)

        preparations = self._video_groups.setdefault(group_cache, {})
        key = tuple(data.get("image", []))
        if key not in preparations:
            preparations[key] = asyncio.create_task(self._prepare_video(data))
        # One cancelled candidate must not cancel its siblings' preparation.
        video = await asyncio.shield(preparations[key])
        return await self._run_episode(engine, data, video)

    async def _prepare_video(self, data: dict[str, Any]) -> dict:
        if not data.get("image"):
            raise ValueError("Task has no visual inputs")
        if self.video_temp_root is not None:
            await asyncio.to_thread(
                Path(self.video_temp_root).mkdir, parents=True, exist_ok=True
            )
        directory = await asyncio.to_thread(
            tempfile.TemporaryDirectory, dir=self.video_temp_root
        )
        try:
            frames = [await self.load_image(name) for name in data["image"]]
            video = await prepare_video_data(frames, Path(directory.name) / "video.mkv")
        except BaseException:
            await asyncio.to_thread(directory.cleanup)
            raise
        video.update(
            frames=frames, directory=directory, processor_cache=ProcessorCallCache()
        )
        return video

    async def _release_video(self, video: dict) -> None:
        video["processor_cache"].close()
        await asyncio.to_thread(video["directory"].cleanup)

    async def _afinalize_processor_cache_group(self, context) -> None:
        tasks = self._video_groups.pop(context.processor_cache, {})
        try:
            # Shielded preparation can outlive a cancelled sample. Drain it
            # before removing files still needed by sibling model requests.
            for result in await asyncio.gather(*tasks.values(), return_exceptions=True):
                if not isinstance(result, BaseException):
                    await self._release_video(result)
        finally:
            if context.processor_cache is not None:
                context.processor_cache.close()

    async def _run_episode(self, engine, data: dict[str, Any], video: dict | None):
        if not data.get("image"):
            raise ValueError("Task has no visual inputs")
        originals = (
            [] if video else [await self.load_image(name) for name in data["image"]]
        )
        video_frames = video["frames"] if video else []
        images = []
        content = [{"type": "text", "text": data["question"]}]
        if video_frames:
            content = [
                {"type": "video", "video": video["path"]},
                {
                    "type": "text",
                    "text": data["question"]
                    + f"\nThe original video contains {len(video_frames)} frames.",
                },
            ]
        else:
            for original in originals:
                images.append(
                    await asyncio.to_thread(
                        bounded_image, original, self.original_max_pixels
                    )
                )
                content.extend(
                    [
                        {"type": "text", "text": f"Image {len(images)}:"},
                        {"type": "image"},
                    ]
                )
        messages = [
            {"role": "system", "content": VIDEO_SYSTEM if video else SYSTEM},
            {"role": "user", "content": content},
        ]
        interactions = {}
        used = 0
        reward = 0.0
        truncated = True
        turns = itertools.count() if self.max_turns is None else range(self.max_turns)
        for turn in turns:
            (
                input_ids,
                image_data,
                token_types,
                mm_input,
                wire_ids,
            ) = await asyncio.to_thread(self.prepare, messages, images, video)
            budget = min(
                self.gconfig.max_new_tokens,
                self.gconfig.max_tokens - len(input_ids),
            )
            if self.max_output_tokens is not None:
                budget = min(budget, self.max_output_tokens - used)
            if budget <= 0:
                break
            metadata = (
                image_prompt_metadata(
                    {}, input_ids, token_types, image_data, self.processor
                )
                if video is None
                else {
                    "sglang_video_input_ids": wire_ids,
                    "sglang_video_prompt_length": len(input_ids),
                }
            )
            request = ModelRequest(
                input_ids=input_ids,
                image_data=image_data,
                metadata=metadata,
                video_data=[video["path"]] if video else None,
                processor=self.processor,
                tokenizer=self.tokenizer,
                gconfig=self.gconfig.new(max_new_tokens=budget, n_samples=1),
            )
            response = await engine.agenerate(request)
            if response.input_tokens != input_ids:
                raise ValueError(
                    "Rollout response does not preserve actor prompt tokens"
                )
            if not response.output_tokens or response.stop_reason == "abort":
                # Engine failures are not negative answer examples.
                return None
            if not (
                response.output_len
                == len(response.output_logprobs)
                == len(response.output_versions)
            ):
                raise ValueError(
                    "Sampled tokens, logprobs and weight versions must align"
                )
            interaction = InteractionWithTokenLogpReward(
                model_response=response,
                prompt_token_ids=input_ids,
                mm_token_type_ids=token_types,
                multi_modal_input=mm_input,
                metadata={"pixel": {"is_video": video is not None, "turn": turn}},
            )
            interactions[request.rid] = interaction
            used += response.output_len
            text = self.tokenizer.decode(
                response.output_tokens_without_stop, skip_special_tokens=False
            )
            reasoning, answer = split_reasoning(text)
            messages.append(
                {"role": "assistant", "reasoning_content": reasoning, "content": answer}
            )
            if response.stop_reason == "length":
                break
            try:
                action = get_action(answer)
                if action is None:
                    truncated = False
                    reward = await self.reward_fn(
                        completions=answer, answer=data["answer"]
                    )
                    break
                arguments = action["arguments"]
                if action["name"] == "select_frames":
                    if not video:
                        raise ValueError("select_frames requires video input")
                    selected = select_frames(
                        video_frames, arguments.get("target_frames")
                    )
                else:
                    index = arguments.get("target_image")
                    if not video and (
                        type(index) is not int or not 1 <= index <= len(originals)
                    ):
                        raise ValueError(
                            "target_image is outside available image indices"
                        )
                    original = (
                        video_crop_source(video_frames, originals, index)
                        if video
                        else originals[index - 1]
                    )
                    selected = [
                        await asyncio.to_thread(crop_image, original, arguments)
                    ]
                feedback = []
                for original in selected:
                    image = await asyncio.to_thread(
                        bounded_image, original, self.crop_max_pixels
                    )
                    originals.append(original)
                    images.append(image)
                    feedback.extend(
                        [
                            {
                                "type": "text",
                                "text": f"Tool result: Image {len(images) + int(video is not None)}.",
                            },
                            {"type": "image"},
                        ]
                    )
            except (ValueError, KeyError, TypeError) as error:
                feedback = [{"type": "text", "text": f"Execution error: {error}"}]
            # A real tool response keeps the last user query fixed in Qwen's
            # Thinking template, preserving prior reasoning without re-training it.
            messages.append({"role": "tool", "content": feedback})
        if not interactions:
            return None
        for turn, interaction in enumerate(interactions.values()):
            interaction.reward = interaction.rollout_reward = reward
            tensors = interaction.to_tensor_dict()
            tensors["is_truncated"] = torch.tensor([truncated], dtype=torch.bool)
            # Each row has its own context, while turn_ids retain episode order.
            tensors["turn_ids"][tensors["loss_mask"].bool()] = turn
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=reward,
            num_turns=len(interactions),
            truncated=int(truncated),
            output_tokens=used,
        )
        return interactions
