# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

from PIL import Image
from PIL.Image import Image as ImageObject


def image2base64(images: list[ImageObject] | ImageObject) -> list[str]:
    """Encode Pillow images as Base64 strings, one per image.

    Each image is serialized to PNG in memory and Base64-encoded, preserving
    its mode, palette, and dimensions after decoding. A single image is
    accepted for convenience and wrapped into a one-element list; an empty
    list yields an empty result.

    Args:
        images: A single Pillow image or a list of images.

    Returns:
        A list of Base64-encoded PNG strings in the same order as the input.
    """
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


def pad_images_batch_to_max_size(images):
    """Center-pad a batch of images to the batch's maximum width and height.

    Every image is pasted onto a new black RGB canvas sized to the largest
    image in the batch. When a dimension difference is odd, the extra pixel
    is left over as padding on the right/bottom (integer division places the
    image at ``(max - size) // 2``). Input images are not modified and not
    consumed: only their pixels and sizes are read.

    Args:
        images: A non-empty iterable of Pillow images.

    Returns:
        A list of new RGB images of identical ``(max_width, max_height)``
        size, in the same order as the input.

    Raises:
        ValueError: If ``images`` is empty, because the maximum size is then
            undefined.
    """
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
