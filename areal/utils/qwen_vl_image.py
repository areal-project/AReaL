# SPDX-License-Identifier: Apache-2.0

"""Select native or legacy Qwen image preprocessing without changing wire images.

``AREAL_VLM_ALIGN_SGLANG`` defaults to true. Set it to false to use the
original HF image preprocessing in the OpenAI proxy. With alignment enabled,
``SGLANG_VLM_QWEN_LIMIT_PIXEL`` defaults to false, preserving native image bounds.
Set it to true to restore the legacy preliminary PIL resize and 16-fold
pixel-budget reduction. The HF processor still handles model sizing, and
original wire bytes are retained in both modes.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from PIL.Image import Image

    from areal.infra.processor_cache import ProcessorCallCache


def sglang_image_alignment_enabled() -> bool:
    """Read the AReaL image-alignment switch without changing server settings."""
    value = os.environ.get("AREAL_VLM_ALIGN_SGLANG", "true").strip().lower()
    if value not in ("true", "false", "1", "0", "yes", "no", "y", "n"):
        raise ValueError(
            "Invalid AREAL_VLM_ALIGN_SGLANG boolean value; use true or false"
        )
    return value in ("true", "1", "yes", "y")


@lru_cache(maxsize=8)
def _processor_model_type(model_name_or_path: str) -> str | None:
    # Read raw config: Qwen4-Exp need not be registered with AutoConfig here.
    from transformers import PretrainedConfig

    config, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    return config.get("model_type")


def _sglang_image_size(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Match QwenVLImageProcessor.smart_resize_for_image in SGLang 0.5.18."""
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError(
            "Qwen-VL images must have positive dimensions and aspect ratio <= 200"
        )
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h = math.floor(height / scale / factor) * factor
        w = math.floor(width / scale / factor) * factor
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    if min(h, w) <= 0:
        raise ValueError(
            "Qwen-VL pixel budget is too small for this image's aspect ratio"
        )
    return h, w


class _ImagePolicy(NamedTuple):
    factor: int
    min_pixels: int
    max_pixels: int
    compress_images: bool
    resample: Any


def _image_policy(processor: Any) -> _ImagePolicy | None:
    if not any(cls.__name__ == "Qwen3VLProcessor" for cls in type(processor).__mro__):
        return None
    model_path = getattr(getattr(processor, "tokenizer", None), "name_or_path", None)
    if not isinstance(model_path, (str, os.PathLike)) or not model_path:
        return None
    if _processor_model_type(os.fspath(model_path)) not in (
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
    ):
        return None

    image_processor = processor.image_processor
    factor = int(image_processor.patch_size) * int(image_processor.merge_size)
    min_pixels = int(image_processor.size["shortest_edge"])
    max_pixels = int(image_processor.size["longest_edge"])
    limit_pixels = (
        os.environ.get("SGLANG_VLM_QWEN_LIMIT_PIXEL", "false").strip().lower()
    )
    if limit_pixels not in ("true", "false", "1", "0", "yes", "no", "y", "n"):
        raise ValueError("Invalid SGLANG_VLM_QWEN_LIMIT_PIXEL boolean value")
    compress_images = limit_pixels in ("true", "1", "yes", "y")
    resample = None
    if compress_images:
        from PIL import Image as PILImage

        if max_pixels >= (32 * factor) ** 2:
            max_pixels //= 16
        resample = getattr(PILImage, os.environ.get("SGLANG_RESIZE_RESAMPLE", ""), None)
    if factor <= 0 or not 0 < min_pixels <= max_pixels:
        raise ValueError("Invalid Qwen-VL image factor or pixel bounds")

    return _ImagePolicy(factor, min_pixels, max_pixels, compress_images, resample)


def sglang_qwen_image_kwargs(processor: Any) -> dict[str, Any] | None:
    """Use proxy CPU HF processing with the selected image pixel bounds.

    The default matches native SGLang 0.5.20. Enabling the legacy switch
    restores the training-side reduction without mutating the shared processor
    or changing the inference server. Callers also use this independently of
    AREAL_VLM_ALIGN_SGLANG to recognize models requiring real-image metadata.
    """
    policy = _image_policy(processor)
    if policy is None:
        return None
    return {
        "device": "cpu",
        "size": {
            "shortest_edge": policy.min_pixels,
            "longest_edge": policy.max_pixels,
        },
    }


def prepare_sglang_qwen_images(
    processor: Any,
    images: list[Image],
    *,
    cache: ProcessorCallCache | None = None,
    image_keys: list[str] | None = None,
) -> tuple[list[Image], list[str]] | None:
    """Prepare native or legacy-resized images while retaining wire bytes.

    By default image dimensions are retained until HF preprocessing.
    SGLANG_VLM_QWEN_LIMIT_PIXEL=true restores the old training-side PIL resize
    and pixel-budget reduction; SGLANG_RESIZE_RESAMPLE selects its PIL filter.

    Return RGB training images and an unchanged copy of image_keys for
    inference transport. No image is encoded again. With no image_keys the
    second return value is empty, since the original wire bytes are unavailable.
    Qwen4-Exp and unknown processor/model families keep their own policy.
    """
    policy = _image_policy(processor)
    if policy is None:
        return None
    factor, min_pixels, max_pixels, compress_images, resample = policy
    if image_keys is not None and len(image_keys) != len(images):
        raise ValueError("Image cache keys must match the input images")
    prepared = []
    for index, image in enumerate(images):

        def prepare(image=image):
            prepared_image = image.convert("RGB")
            if compress_images:
                height, width = _sglang_image_size(
                    image.height, image.width, factor, min_pixels, max_pixels
                )
                prepared_image = prepared_image.resize(
                    (width, height), resample=resample
                )
            return prepared_image

        if cache is not None and image_keys is not None:
            # Reuse decoded historical images when text or appended images
            # change. The original JPEG/PNG transport remains unchanged.
            key = cache.make_key(
                "sglang_qwen_image_compression",
                id(processor),
                factor,
                min_pixels,
                max_pixels,
                compress_images,
                resample,
                image_keys[index],
            )
            prepared_image = cache.get_or_compute(key, prepare)
            # Processors may mutate PIL inputs. Never expose the cached image.
            prepared_image = prepared_image.copy()
        else:
            prepared_image = prepare()
        prepared.append(prepared_image)
    return prepared, list(image_keys) if image_keys is not None else []
