"""Pixel image/video rollouts with persistent thinking and official visual tools."""

import asyncio
import itertools
import json
import math
import re
import subprocess
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.image import image2base64

logger = logging.getLogger("PixelReasonerWorkflow")
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
    """Conservative final-label matching; mismatches require review, not rejection."""
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


class PixelReasonerWorkflow(RolloutWorkflow):
    """Collect traces only; return None rather than training tensors."""

    def __init__(
        self,
        processor,
        gconfig: GenerationHyperparameters,
        data_root: str,
        dump_dir: str,
        max_turns: int | None = None,
        max_output_tokens: int = 262144,
        original_max_pixels: int = 1568000,
        crop_max_pixels: int = 784000,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.gconfig = gconfig
        self.data_root = Path(data_root).resolve()
        self.dump_dir = Path(dump_dir)
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.original_max_pixels = original_max_pixels
        self.crop_max_pixels = crop_max_pixels
        if max_turns is not None and max_turns < 1:
            raise ValueError("max_turns must be positive or null")
        if min(max_output_tokens, original_max_pixels, crop_max_pixels) < 1:
            raise ValueError("Budgets must be positive")

    def prepare(self, messages: list[dict], images: list[Image.Image], video=None):
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            preserve_thinking=True,
        )
        # Fail visibly if a different template silently drops thinking history.
        for message in messages:
            reasoning = message.get("reasoning_content", "").strip()
            if reasoning and reasoning not in prompt:
                raise ValueError("Chat template removed reasoning_content")
        video_kwargs = (
            {}
            if video is None
            else {
                "videos": [video["pixels"]],
                "video_metadata": [video["metadata"]],
                "do_sample_frames": False,
            }
        )
        inputs = self.processor(
            text=[prompt],
            images=images or None if video else images,
            return_tensors="pt",
            **video_kwargs,
        )
        expanded_ids = inputs["input_ids"][0].tolist()
        # SGLang expands image placeholders itself. Sending the HF-expanded
        # tokens makes one image look like hundreds of separate image inputs.
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        image_id = self.tokenizer.convert_tokens_to_ids(image_token)
        if input_ids.count(image_id) != len(images):
            raise ValueError("Expected one image placeholder per supplied image")
        if video is not None:
            video_id = self.tokenizer.convert_tokens_to_ids(self.processor.video_token)
            if input_ids.count(video_id) != 1:
                raise ValueError(
                    "Expected one video placeholder for the supplied video"
                )
        return input_ids, image2base64(images), expanded_ids

    @staticmethod
    def save(directory: Path, episode: dict):
        temporary = directory / "episode.json.tmp"
        temporary.write_text(json.dumps(episode, ensure_ascii=False) + "\n")
        temporary.replace(directory / "episode.json")

    async def arun_episode(self, engine, data: dict[str, Any]) -> None:
        directory = self.dump_dir / uuid.uuid4().hex
        await asyncio.to_thread(directory.mkdir, parents=True)
        started = time.monotonic()
        episode = {"source": data, "status": "running", "turns": [], "tools": []}
        messages, originals, images = [], [], []
        used = 0
        video, video_frames = None, []
        try:
            if not data.get("image"):
                raise ValueError("Task has no visual inputs")
            content = [{"type": "text", "text": data["question"]}]
            for name in data["image"]:
                path = (self.data_root / name).resolve()
                if not path.is_relative_to(self.data_root):
                    raise ValueError("Image path outside data_root")

                def load(path=path):
                    with Image.open(path) as image:
                        return image.convert("RGB")

                original = await asyncio.to_thread(load)
                if data.get("is_video"):
                    video_frames.append(original)
                    continue
                originals.append(original)
                image = await asyncio.to_thread(
                    bounded_image, original, self.original_max_pixels
                )
                images.append(image)
                relative = f"image-{len(images)}.png"
                await asyncio.to_thread(image.save, directory / relative)
                content.extend(
                    [
                        {"type": "text", "text": f"Image {len(images)}:"},
                        {"type": "image", "image": relative},
                    ]
                )
            if video_frames:
                video = await prepare_video_data(video_frames, directory / "video.mkv")
                episode["video_transport"] = video["transport"]
                content = [
                    {"type": "video", "video": "video.mkv"},
                    {
                        "type": "text",
                        "text": data["question"]
                        + f"\nThe original video contains {len(video_frames)} frames.",
                    },
                ]
            messages = [
                {"role": "system", "content": VIDEO_SYSTEM if video else SYSTEM},
                {"role": "user", "content": content},
            ]
            turns = (
                itertools.count() if self.max_turns is None else range(self.max_turns)
            )
            for turn in turns:
                input_ids, image_data, expanded_ids = await asyncio.to_thread(
                    self.prepare, messages, images, video
                )
                budget = min(
                    self.gconfig.max_new_tokens,
                    self.max_output_tokens - used,
                    self.gconfig.max_tokens - len(expanded_ids),
                )
                if budget <= 0:
                    episode["status"] = "token_budget_exhausted"
                    break
                request = ModelRequest(
                    input_ids=input_ids,
                    image_data=image_data,
                    video_data=[video["path"]] if video else None,
                    processor=self.processor,
                    tokenizer=self.tokenizer,
                    gconfig=self.gconfig.new(max_new_tokens=budget, n_samples=1),
                )
                turn_record = {
                    "input_ids": input_ids,
                    "expanded_input_ids": expanded_ids,
                    "prompt_tokens": len(expanded_ids),
                    "prompt_tokens_source": "client_processor",
                    "images": [f"image-{i + 1}.png" for i in range(len(images))],
                    "max_new_tokens": budget,
                }
                if video:
                    turn_record["video"] = "video.mkv"
                episode["turns"].append(turn_record)
                episode["messages"] = messages
                await asyncio.to_thread(self.save, directory, episode)
                response = await engine.agenerate(request)
                raw = self.tokenizer.decode(
                    response.output_tokens, skip_special_tokens=False
                )
                text = self.tokenizer.decode(
                    response.output_tokens_without_stop, skip_special_tokens=False
                )
                reasoning, answer = split_reasoning(text)
                used += response.output_len
                turn_record.update(
                    {
                        "output_ids": response.output_tokens,
                        "raw_text": raw,
                        "reasoning_content": reasoning,
                        "content": answer,
                        "finish_reason": response.stop_reason,
                        "latency": response.latency,
                        "completion_tokens": response.output_len,
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "reasoning_content": reasoning,
                        "content": answer,
                    }
                )
                # Checkpoint every response before executing its tool.
                episode["messages"] = messages
                await asyncio.to_thread(self.save, directory, episode)
                if response.stop_reason != "stop":
                    episode["status"] = "incomplete_generation"
                    break
                try:
                    action = get_action(answer)
                    if action is None:
                        episode["status"] = (
                            "completed" if answer.strip() else "missing_final"
                        )
                        episode["final_content"] = answer
                        episode["verification"] = verify(answer, data.get("answer", []))
                        break
                    arguments = action["arguments"]
                    if action["name"] == "select_frames":
                        if not video:
                            raise ValueError("select_frames requires video input")
                        selected = select_frames(
                            video_frames, arguments.get("target_frames")
                        )
                        tool_started = time.monotonic()
                        first_index = len(images) + 2
                        feedback, files = [], []
                        for original in selected:
                            image = await asyncio.to_thread(
                                bounded_image, original, self.crop_max_pixels
                            )
                            relative = f"image-{len(images) + 1}.png"
                            await asyncio.to_thread(image.save, directory / relative)
                            originals.append(original)
                            images.append(image)
                            files.append(relative)
                            feedback.append({"type": "image", "image": relative})
                        feedback.insert(
                            0,
                            {
                                "type": "text",
                                "text": f"Here are the selected frames (Numbered {first_index} to {len(images) + 1}):",
                            },
                        )
                        episode["tools"].append(
                            {
                                "turn": turn,
                                "action": action,
                                "images": files,
                                "frame_indices": arguments["target_frames"],
                                "elapsed_seconds": time.monotonic() - tool_started,
                            }
                        )
                        messages.append({"role": "user", "content": feedback})
                        continue
                    index = arguments.get("target_image")
                    if not video and (
                        type(index) is not int or not 1 <= index <= len(originals)
                    ):
                        raise ValueError(
                            "target_image is outside available image indices"
                        )
                    tool_started = time.monotonic()
                    crop = await asyncio.to_thread(
                        crop_image,
                        video_crop_source(video_frames, originals, index)
                        if video
                        else originals[index - 1],
                        arguments,
                    )
                    crop = await asyncio.to_thread(
                        bounded_image, crop, self.crop_max_pixels
                    )
                    relative = f"image-{len(images) + 1}.png"
                    await asyncio.to_thread(crop.save, directory / relative)
                    originals.append(crop)
                    images.append(crop)
                    episode["tools"].append(
                        {
                            "turn": turn,
                            "action": action,
                            "image": relative,
                            "elapsed_seconds": time.monotonic() - tool_started,
                        }
                    )
                    feedback = [
                        {
                            "type": "text",
                            "text": f"Tool result: Image {len(images) + int(video is not None)}, cropped from Image {index}.",
                        },
                        {"type": "image", "image": relative},
                    ]
                except (ValueError, KeyError, TypeError) as error:
                    episode["tools"].append({"turn": turn, "error": str(error)})
                    feedback = [{"type": "text", "text": f"Execution error: {error}"}]
                messages.append({"role": "user", "content": feedback})
            else:
                episode["status"] = "turn_budget_exhausted"
        except Exception as error:
            episode.update(status="failed", error=f"{type(error).__name__}: {error}")
            logger.exception("Pixel rollout failed: %s", directory)
        finally:
            episode.update(
                messages=messages,
                completion_tokens_sum=used,
                prompt_tokens_sum=sum(t["prompt_tokens"] for t in episode["turns"]),
                peak_context_tokens=max(
                    (
                        t["prompt_tokens"] + len(t.get("output_ids", []))
                        for t in episode["turns"]
                    ),
                    default=0,
                ),
                successful_crops=sum("image" in tool for tool in episode["tools"]),
                elapsed_seconds=time.monotonic() - started,
            )
            await asyncio.to_thread(self.save, directory, episode)
            stats_tracker.get("pixel").scalar(
                completed=int(episode["status"] == "completed"),
                output_tokens=used,
                rounds=len(episode["turns"]),
            )
        return None
