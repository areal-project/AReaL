# SPDX-License-Identifier: Apache-2.0

import base64
import binascii
from contextlib import suppress
from io import BytesIO

from PIL import Image, ImageOps
from PIL.Image import Image as ImageObject


def image2base64(images: list[ImageObject] | ImageObject) -> list[str]:
    if isinstance(images, ImageObject):
        images = [images]

    byte_images = []
    for image in images:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            buffer.seek(0)
            byte_image = base64.b64encode(buffer.read()).decode("utf-8")
            byte_images.append(byte_image)

    return byte_images


def base642image(images: list[str] | str) -> list[ImageObject]:
    """Decode base64 image payloads (without data-URI prefix) into PIL images.

    Match vLLM's default RGB decoder for EXIF orientation and transparency.
    """
    if isinstance(images, str):
        images = [images]

    decoded = []
    for byte_image in images:
        try:
            raw = base64.b64decode(byte_image, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 image payload.") from exc
        with BytesIO(raw) as buffer:
            image = Image.open(buffer)
            image.load()
            # vLLM ignores unreadable EXIF. In-place avoids copying upright images.
            with suppress(Exception):
                ImageOps.exif_transpose(image, in_place=True)
        if image.mode != "RGB":
            if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
                if image.mode != "RGBA":
                    image = image.convert("RGBA")
                background = Image.new("RGB", image.size, "white")
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
        decoded.append(image)

    return decoded


def pad_images_batch_to_max_size(images):
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)

    padded_images = []

    for image in images:
        width, height = image.size

        padding_left = (max_width - width) // 2
        padding_top = (max_height - height) // 2

        padded_image = Image.new("RGB", (max_width, max_height), (0, 0, 0))
        padded_image.paste(image, (padding_left, padding_top))

        padded_images.append(padded_image)

    return padded_images
