# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

from PIL import Image
from PIL.Image import Image as ImageObject


def image2base64(images: list[ImageObject] | ImageObject) -> list[str]:
    """Encode one image, or a list of images, as Base64-encoded PNG strings.

    Every image is re-encoded as PNG regardless of the format it was loaded
    from. The returned list preserves input order, a single image yields a
    single-element list, and an empty list yields an empty list. The input
    images are not modified.
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
    """Center every image on a black canvas as large as the biggest one.

    The canvas is ``(max_width, max_height)`` across the batch, filled with
    black, and each image is pasted with floor-divided offsets. An odd size
    difference therefore keeps the extra pixel on the right and bottom rather
    than splitting it evenly. Output order matches input order.

    The results are new ``RGB`` images, so grayscale and RGBA inputs are
    converted to RGB and the inputs themselves are left untouched. An empty
    batch raises ``ValueError``, because the canvas size is derived from
    ``max(...)`` over the sequence.
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
