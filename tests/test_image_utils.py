# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the image conversion and padding helpers.

These cover only observable behaviour of :mod:`areal.utils.image` on synthetic
in-memory images: no disk IO, no network, no GPU.
"""

import base64
from io import BytesIO

import pytest
from PIL import Image

from areal.utils.image import image2base64, pad_images_batch_to_max_size


def _decode(payload: str) -> Image.Image:
    """Decode a Base64 PNG payload back into a fully loaded image.

    ``Image.copy()`` drops the ``.format`` attribute, so load in place instead:
    ``load()`` pulls all pixel data into memory and closes the buffer, while
    ``.format`` stays populated for assertions.
    """
    image = Image.open(BytesIO(base64.b64decode(payload)))
    image.load()
    return image


def _pattern_rgb(width: int, height: int) -> Image.Image:
    """Deterministic RGB image so pixel comparisons are meaningful."""
    image = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            image.putpixel(
                (x, y), ((x * 37) % 256, (y * 53) % 256, ((x + y) * 29) % 256)
            )
    return image


def _pattern_rgba(width: int, height: int) -> Image.Image:
    image = Image.new("RGBA", (width, height))
    for x in range(width):
        for y in range(height):
            image.putpixel(
                (x, y), ((x * 11) % 256, (y * 17) % 256, 90, (x * 60 + y * 5) % 256)
            )
    return image


def _pattern_gray(width: int, height: int) -> Image.Image:
    image = Image.new("L", (width, height))
    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), (x * 40 + y * 7) % 256)
    return image


def _bounding_box_of_non_black(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Inclusive bounding box of pixels that are not pure black, or None."""
    width, height = image.size
    xs, ys = [], []
    for y in range(height):
        for x in range(width):
            if image.getpixel((x, y)) != (0, 0, 0):
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


# --------------------------------------------------------------------------- #
# image2base64
# --------------------------------------------------------------------------- #


def test_image2base64_single_image_returns_single_decodable_png():
    # Arrange
    source = _pattern_rgb(3, 2)

    # Act
    payloads = image2base64(source)

    # Assert
    assert len(payloads) == 1
    decoded = _decode(payloads[0])
    assert decoded.format == "PNG"
    assert decoded.size == (3, 2)


def test_image2base64_multiple_images_preserve_order():
    # Arrange: distinct widths so order is observable after decoding
    first = _pattern_rgb(2, 2)
    second = _pattern_rgb(4, 3)
    third = _pattern_rgb(1, 5)

    # Act
    payloads = image2base64([first, second, third])

    # Assert
    assert [p.size for p in (_decode(x) for x in payloads)] == [(2, 2), (4, 3), (1, 5)]
    assert [_decode(x).getpixel((0, 0)) for x in payloads] == [
        first.getpixel((0, 0)),
        second.getpixel((0, 0)),
        third.getpixel((0, 0)),
    ]


def test_image2base64_empty_list_returns_empty_list():
    assert image2base64([]) == []


def test_image2base64_preserves_rgb_mode_and_pixels():
    # Arrange
    source = _pattern_rgb(5, 4)

    # Act
    decoded = _decode(image2base64(source)[0])

    # Assert
    assert decoded.mode == "RGB"
    assert decoded.size == source.size
    assert list(decoded.getdata()) == list(source.getdata())


def test_image2base64_preserves_rgba_mode_and_pixels():
    # Arrange
    source = _pattern_rgba(4, 4)

    # Act
    decoded = _decode(image2base64(source)[0])

    # Assert
    assert decoded.mode == "RGBA"
    assert decoded.size == source.size
    assert list(decoded.getdata()) == list(source.getdata())


def test_image2base64_preserves_grayscale_mode_and_pixels():
    # Arrange
    source = _pattern_gray(3, 4)

    # Act
    decoded = _decode(image2base64(source)[0])

    # Assert
    assert decoded.mode == "L"
    assert decoded.size == source.size
    assert list(decoded.getdata()) == list(source.getdata())


def test_image2base64_leaves_input_image_unchanged():
    # Arrange
    source = _pattern_rgb(4, 3)
    snapshot_size = source.size
    snapshot_pixels = list(source.getdata())

    # Act
    image2base64([source, source])

    # Assert
    assert source.size == snapshot_size
    assert list(source.getdata()) == snapshot_pixels


# --------------------------------------------------------------------------- #
# pad_images_batch_to_max_size
# --------------------------------------------------------------------------- #


def test_pad_images_singleton_returns_same_geometry():
    # Arrange
    source = _pattern_rgb(3, 2)

    # Act
    (padded,) = pad_images_batch_to_max_size([source])

    # Assert
    assert padded.size == (3, 2)
    assert padded.mode == "RGB"
    assert list(padded.getdata()) == list(source.getdata())


def test_pad_images_equal_sizes_are_not_padded():
    # Arrange
    first = _pattern_rgb(3, 3)
    second = _pattern_rgb(3, 3)

    # Act
    padded = pad_images_batch_to_max_size([first, second])

    # Assert
    assert [p.size for p in padded] == [(3, 3), (3, 3)]
    assert [list(p.getdata()) for p in padded] == [
        list(first.getdata()),
        list(second.getdata()),
    ]


def test_pad_images_mixed_sizes_center_the_smaller_image():
    # Arrange: 2x2 inside a 5x5 batch. (5 - 2) // 2 == 1 on both axes.
    small = _pattern_rgb(2, 2)
    large = _pattern_rgb(5, 5)

    # Act
    padded_small, padded_large = pad_images_batch_to_max_size([small, large])

    # Assert: exactly the documented region is non-black
    assert padded_small.size == (5, 5)
    assert _bounding_box_of_non_black(padded_small) == (1, 1, 2, 2)
    assert all(
        padded_small.getpixel((x, y)) == (0, 0, 0)
        for x in range(5)
        for y in range(5)
        if not (1 <= x <= 2 and 1 <= y <= 2)
    )


def test_pad_images_odd_difference_floors_the_offset():
    # Arrange: 3x3 into 4x4 -> (4 - 3) // 2 == 0, so the remainder lands on the
    # right/bottom rather than being distributed evenly.
    small = _pattern_rgb(3, 3)
    large = _pattern_rgb(4, 4)

    # Act
    padded_small, _ = pad_images_batch_to_max_size([small, large])

    # Assert
    assert _bounding_box_of_non_black(padded_small) == (0, 0, 2, 2)


def test_pad_images_copy_source_pixels_into_the_padded_region():
    # Arrange
    small = _pattern_rgb(2, 2)
    large = _pattern_rgb(5, 5)

    # Act
    padded_small, _ = pad_images_batch_to_max_size([small, large])

    # Assert
    for sx in range(2):
        for sy in range(2):
            assert padded_small.getpixel((sx + 1, sy + 1)) == small.getpixel((sx, sy))


def test_pad_images_preserve_batch_order():
    # Arrange: three different sizes, each with a unique solid colour
    red = Image.new("RGB", (1, 1), (255, 0, 0))
    green = Image.new("RGB", (2, 2), (0, 255, 0))
    blue = Image.new("RGB", (3, 3), (0, 0, 255))

    # Act
    padded = pad_images_batch_to_max_size([red, green, blue])

    # Assert: every output is 3x3 and still carries its own colour
    assert [p.size for p in padded] == [(3, 3)] * 3
    assert [_bounding_box_of_non_black(p) is not None for p in padded] == [
        True,
        True,
        True,
    ]
    assert padded[0].getpixel((1, 1)) == (255, 0, 0)
    assert padded[1].getpixel((0, 0)) == (0, 255, 0)
    assert padded[2].getpixel((0, 0)) == (0, 0, 255)


def test_pad_images_output_is_rgb_even_for_grayscale_and_rgba_inputs():
    # Arrange
    gray = _pattern_gray(2, 2)
    rgba = _pattern_rgba(3, 3)

    # Act
    padded = pad_images_batch_to_max_size([gray, rgba])

    # Assert
    assert [p.mode for p in padded] == ["RGB", "RGB"]
    assert [p.size for p in padded] == [(3, 3), (3, 3)]


def test_pad_images_empty_input_raises_value_error():
    # The helper sizes itself from ``max(...)`` over the batch, which raises for
    # an empty sequence. Assert the type only; the message is not a contract.
    with pytest.raises(ValueError):
        pad_images_batch_to_max_size([])


def test_pad_images_leaves_input_images_unchanged():
    # Arrange
    small = _pattern_rgb(2, 2)
    large = _pattern_rgba(5, 5)
    before = [
        (image.size, image.mode, list(image.getdata())) for image in (small, large)
    ]

    # Act
    pad_images_batch_to_max_size([small, large])

    # Assert
    after = [
        (image.size, image.mode, list(image.getdata())) for image in (small, large)
    ]
    assert after == before
