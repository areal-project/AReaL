# SPDX-License-Identifier: Apache-2.0

"""Follow SGLang's CPU/PIL Qwen image pipeline without changing wire images."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from PIL.Image import Image

    from areal.infra.processor_cache import ProcessorCallCache


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

    from PIL import Image as PILImage

    image_processor = processor.image_processor
    factor = int(image_processor.patch_size) * int(image_processor.merge_size)
    min_pixels = int(image_processor.size["shortest_edge"])
    max_pixels = int(image_processor.size["longest_edge"])
    limit_pixels = os.environ.get("SGLANG_VLM_QWEN_LIMIT_PIXEL", "true").lower()
    if limit_pixels not in ("true", "false", "1", "0", "yes", "no", "y", "n"):
        raise ValueError("Invalid SGLANG_VLM_QWEN_LIMIT_PIXEL boolean value")
    if limit_pixels in ("true", "1", "yes", "y") and max_pixels >= (32 * factor) ** 2:
        max_pixels //= 16
    if factor <= 0 or not 0 < min_pixels <= max_pixels:
        raise ValueError("Invalid Qwen-VL image factor or pixel bounds")

    # Match the resampling filter used by SGLang's preliminary PIL resize.
    resample = getattr(PILImage, os.environ.get("SGLANG_RESIZE_RESAMPLE", ""), None)
    return _ImagePolicy(factor, min_pixels, max_pixels, resample)


def sglang_qwen_image_kwargs(processor: Any) -> dict[str, Any] | None:
    """Use proxy CPU HF processing with SGLang's effective size, without mutation.

    SGLang reduces its own processor's maximum once at startup. The proxy keeps
    its shared processor unchanged and passes the effective bounds per call,
    so repeated requests cannot apply the pixel reduction more than once.
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
    """Reproduce native SGLang PIL resize, retaining original request bytes.

    The SGLang model startup patch selects CPU/PIL decoding while retaining
    native pixel bounds, preliminary resize, and HF device dispatch. The proxy
    uses CPU HF processing after the same preliminary PIL resize.
    Both processes must use the same SGLANG_VLM_QWEN_LIMIT_PIXEL and
    SGLANG_RESIZE_RESAMPLE settings, without server-side image-size overrides.

    Return resized training images and an unchanged copy of image_keys for
    inference transport. No image is encoded again. With no image_keys the
    second return value is empty, since the original wire bytes are unavailable.
    Qwen4-Exp and unknown processor/model families keep their own policy.
    """
    policy = _image_policy(processor)
    if policy is None:
        return None
    factor, min_pixels, max_pixels, resample = policy
    if image_keys is not None and len(image_keys) != len(images):
        raise ValueError("Image cache keys must match the input images")
    prepared = []
    for index, image in enumerate(images):

        def prepare(image=image):
            height, width = _sglang_image_size(
                image.height, image.width, factor, min_pixels, max_pixels
            )
            resized = image.convert("RGB").resize((width, height), resample=resample)
            return resized

        if cache is not None and image_keys is not None:
            # Reuse resized historical images when text or appended images
            # change. The original JPEG/PNG transport remains unchanged.
            key = cache.make_key(
                "sglang_qwen_image",
                id(processor),
                factor,
                min_pixels,
                max_pixels,
                resample,
                image_keys[index],
            )
            resized = cache.get_or_compute(key, prepare)
            # Processors may mutate PIL inputs. Never expose the cached image.
            resized = resized.copy()
        else:
            resized = prepare()
        prepared.append(resized)
    return prepared, list(image_keys) if image_keys is not None else []
