# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for image conversion and padding helpers."""

import base64
from io import BytesIO

from PIL import Image

from areal.utils.image import image2base64, pad_images_batch_to_max_size


def _solid(size: tuple[int, int], color: tuple[int, int, int] = (255, 0, 0)):
    return Image.new("RGB", size, color)


def _decode(payload: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(payload)))


def test_image2base64_wraps_single_image_in_png_payload():
    source = _solid((4, 3), (255, 0, 0))

    encoded = image2base64(source)

    assert isinstance(encoded, list)
    assert len(encoded) == 1
    decoded = _decode(encoded[0])
    assert decoded.format == "PNG"
    assert decoded.size == (4, 3)
    assert decoded.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_image2base64_preserves_order_for_multiple_images():
    first = _solid((2, 2), (255, 0, 0))
    second = _solid((3, 3), (0, 0, 255))

    encoded = image2base64([first, second])

    assert len(encoded) == 2
    assert _decode(encoded[0]).size == (2, 2)
    assert _decode(encoded[1]).size == (3, 3)


def test_pad_images_batch_to_max_size_uniforms_width_and_height():
    images = [_solid((2, 2)), _solid((4, 4)), _solid((3, 5))]

    padded = pad_images_batch_to_max_size(images)

    assert len(padded) == 3
    assert [image.size for image in padded] == [(4, 5)] * 3


def test_pad_images_batch_to_max_size_centers_padding_on_black():
    small = _solid((2, 2), (255, 0, 0))
    large = _solid((4, 4), (0, 255, 0))

    small_padded, large_padded = pad_images_batch_to_max_size([small, large])

    assert small_padded.size == (4, 4)
    # (4 - 2) // 2 == 1, so the 2x2 image starts at (1, 1).
    assert small_padded.getpixel((0, 0)) == (0, 0, 0)
    assert small_padded.getpixel((1, 1)) == (255, 0, 0)
    assert small_padded.getpixel((2, 2)) == (255, 0, 0)
    assert small_padded.getpixel((3, 3)) == (0, 0, 0)
    assert large_padded.getpixel((0, 0)) == (0, 255, 0)


def test_pad_images_batch_to_max_size_rounds_odd_gap_towards_top_left():
    small = _solid((2, 2), (255, 0, 0))
    wide = _solid((3, 2), (0, 255, 0))

    padded = pad_images_batch_to_max_size([small, wide])

    assert padded[0].size == (3, 2)
    # (3 - 2) // 2 == 0, so the odd column of padding lands on the right.
    assert padded[0].getpixel((0, 0)) == (255, 0, 0)
    assert padded[0].getpixel((2, 0)) == (0, 0, 0)
