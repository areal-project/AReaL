# SPDX-License-Identifier: Apache-2.0

import math
import os
from io import BytesIO
from typing import Any

import torch.distributed as dist
from datasets import load_dataset
from PIL.Image import Image as ImageObject

from areal.utils import logging

logger = logging.getLogger("CLEVR70KDataset")

DATASET_NUM_PROC = 16


def convert_image(
    image: dict[str, Any] | ImageObject | str,
    max_pixels: int | None,
) -> ImageObject:
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = (
            int(image.width * resize_factor),
            int(image.height * resize_factor),
        )
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    with BytesIO() as output:
        image.save(output, format="JPEG")
        return output.getvalue()


def get_clevr_count_70k_sft_dataset(
    path: str,
    split: str,
    processor,
    max_length: int | None = None,
):
    """
    "clevr_count_70k": {
        "image_key": "images",
        "question_key": "problem",
        "answer_key": "answer"
    },
    """

    def _do_preprocess(
        path: str,
        split: str,
        processor,
        max_length: int | None = None,
        num_proc: int | None = None,
    ):
        dataset = load_dataset(path=path, split=split)

        tokenizer = processor.tokenizer

        def process_example(example):
            # Add query_id column
            images = example["images"]
            image_processor_type = (
                processor.image_processor.image_processor_type.lower()
            )
            if "qwen" in image_processor_type:
                image_token = "<|vision_start|><|image_pad|><|vision_end|>"
            elif "gemma3" in image_processor_type:
                image_token = processor.boi_token
            else:
                image_token = (
                    processor.image_token if processor is not None else "<image>"
                )
            example["problem"] = (
                example["problem"]
                .replace("<image>", image_token)
                .replace("different", "")
            )
            processed_images = []
            for image in images:
                processed_images.append(convert_image(image, 336 * 336))
            example["images"] = processed_images
            example["seq"] = (
                example["problem"] + example["answer"] + tokenizer.eos_token
            )

            return example

        num_proc = max(1, min(os.cpu_count(), 16))
        dataset = dataset.map(
            lambda example: process_example(example),
            num_proc=num_proc,
        )

        def _process(example):
            text = example["seq"]
            processed_input = processor(
                text=[text],
                images=example["images"],
                padding=False,
                return_tensors="pt",
                return_length=True,
                return_attention_mask=False,
            )

            example["input_ids"] = processed_input["input_ids"].squeeze(0)
            example["mm_token_type_ids"] = processed_input.get("token_type_ids", None)
            multi_modal_input = {}
            multi_modal_input["pixel_values"] = processed_input["pixel_values"]
            if "image_grid_thw" in processed_input:
                multi_modal_input["image_grid_thw"] = processed_input["image_grid_thw"]
            example["multi_modal_input"] = [multi_modal_input]
            answer_token = tokenizer.encode(example["answer"])
            loss_mask = [0] * (len(example["input_ids"]) - len(answer_token)) + [
                1
            ] * len(answer_token)
            example["loss_mask"] = loss_mask
            return example

        dataset = dataset.map(
            lambda x: _process(x),
            remove_columns=["images", "seq", "problem", "answer"],
            num_proc=num_proc,
        )

        if max_length is not None:
            # Filter out sequences longer than max_length
            dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

        return dataset

    if dist.is_initialized():
        # Use multi-processing to accelerate data-processing
        # FIXME: processor process data extremely slowly in transformers > 4.53.1
        num_proc = max(1, min(os.cpu_count(), DATASET_NUM_PROC))
        logger.warning("Please set HF_HOME to your NFS directory")
        if int(os.getenv("RANK", "0")) == 0:
            # First process data in rank 0, and use HF cache to load pre-processed dataset in other ranks
            dataset = _do_preprocess(path, split, processor, max_length, num_proc)
        dist.barrier()
    else:
        # Do not use multi-processing (slow)
        num_proc = None

    # If use multiprocessing, it will load dataset in HF cache
    dataset = _do_preprocess(path, split, processor, max_length, num_proc)
    return dataset


def get_clevr_count_70k_rl_dataset(
    path: str,
    split: str,
    processor,
    max_length: int | None = None,
):
    def _do_preprocess(
        path: str,
        split: str,
        processor,
        max_length: int | None = None,
        num_proc: int | None = None,
    ):
        dataset = load_dataset(path=path, split=split)

        def process(sample):
            processed_images = [
                convert_image(image, 336 * 336) for image in sample["images"]
            ]
            image_processor_type = (
                processor.image_processor.image_processor_type.lower()
            )
            if "qwen" in image_processor_type:
                image_token = "<|vision_start|><|image_pad|><|vision_end|>"
            elif "gemma3" in image_processor_type:
                image_token = processor.boi_token
            else:
                image_token = (
                    processor.image_token if processor is not None else "<image>"
                )
            system_prompt = {
                "role": "system",
                "content": (
                    "Solve the following question: count the number of items in the image and provide the final answer in [ ] format, ensuring that only the number is inside the brackets without any additional text or explanations. "
                ),
            }

            messages = [
                {
                    "role": "user",
                    "content": sample["problem"]
                    .replace("<image>", image_token)
                    .replace("different", ""),
                }
            ]
            messages.insert(0, system_prompt)
            messages = processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            return {"messages": messages, "images": processed_images}

        dataset = dataset.map(process, num_proc=num_proc).remove_columns(["problem"])

        # Filter out sequences longer than max_length if max_length is provided
        if max_length is not None:

            def filter_length(sample):
                # Process the sample to get the total token count including image tokens
                processed_input = processor(
                    text=[sample["messages"]],
                    images=sample["images"],
                    padding=False,
                    return_tensors="pt",
                    return_length=True,
                    return_attention_mask=False,
                )
                total_tokens = len(processed_input["input_ids"].squeeze(0))
                return total_tokens <= max_length

            dataset = dataset.filter(filter_length)
        return dataset

    if dist.is_initialized():
        # Use multi-processing to accelerate data-processing
        # FIXME: processor process data extremely slowly in transformers > 4.53.1
        num_proc = max(1, min(os.cpu_count(), DATASET_NUM_PROC))
        logger.warning("Please set HF_HOME to your NFS directory")
        if int(os.getenv("RANK", "0")) == 0:
            # First process data in rank 0, and use HF cache to load pre-processed dataset in other ranks
            dataset = _do_preprocess(path, split, processor, max_length, num_proc)
        dist.barrier()
    else:
        # Do not use multi-processing (slow)
        num_proc = None

    # If use multiprocessing, it will load dataset in HF cache
    dataset = _do_preprocess(path, split, processor, max_length, num_proc)
    return dataset
