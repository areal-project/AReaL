# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for the image helpers in ``areal.utils.image``.

The helpers are exercised with small Pillow images created in memory. Base64
round trips are verified by decoding the output and comparing the
reconstructed image (format, size, mode, pixels) instead of comparing encoded
byte strings, so the tests describe observable behavior only.
"""

import base64
from io import BytesIO

import pytest
from PIL import Image

from areal.utils.image import image2base64, pad_images_batch_to_max_size


def make_image(mode: str, size: tuple[int, int], fill) -> Image.Image:
    """Create a tiny solid-color RGB/RGBA/L image for tests."""
    return Image.new(mode, size, fill)


def decode_base64_png(encoded: str) -> Image.Image:
    """Decode one Base64 string produced by ``image2base64`` back to an image."""
    return Image.open(BytesIO(base64.b64decode(encoded)))


# ---------------------------------------------------------------------------
# image2base64
# ---------------------------------------------------------------------------


def test_image2base64_empty_list_returns_empty_list():
    """Encoding no images yields an empty list, not an error."""
    assert image2base64([]) == []


def test_image2base64_single_image_wrapped_into_list():
    """A lone image argument is treated as a one-element batch."""
    image = make_image("RGB", (2, 2), (255, 0, 0))

    result = image2base64(image)

    assert isinstance(result, list)
    assert len(result) == 1


@pytest.mark.parametrize(
    ("mode", "fill"),
    [
        ("RGB", (10, 20, 30)),
        ("RGBA", (10, 20, 30, 40)),
        ("L", 128),
    ],
    ids=["rgb", "rgba", "grayscale"],
)
def test_image2base64_round_trip_preserves_png_format_size_mode_and_pixels(mode, fill):
    """Decoding the output reconstructs a PNG with identical size/mode/pixels."""
    image = make_image(mode, (3, 2), fill)

    [encoded] = image2base64(image)
    decoded = decode_base64_png(encoded)

    assert decoded.format == "PNG"
    assert decoded.size == image.size
    assert decoded.mode == mode
    assert list(decoded.getdata()) == list(image.getdata())


def test_image2base64_multiple_images_preserves_order():
    """Each output string encodes the image at the same input position."""
    images = [
        make_image("RGB", (1, 2), (255, 0, 0)),
        make_image("RGB", (3, 4), (0, 255, 0)),
        make_image("RGB", (5, 1), (0, 0, 255)),
    ]

    result = image2base64(images)

    assert len(result) == len(images)
    for encoded, expected in zip(result, images):
        decoded = decode_base64_png(encoded)
        assert decoded.size == expected.size
        assert list(decoded.getdata()) == list(expected.getdata())


def test_image2base64_does_not_modify_inputs():
    """Encoding leaves the input images' dimensions and pixels untouched."""
    images = [make_image("RGBA", (2, 3), (1, 2, 3, 4))]
    sizes_before = [image.size for image in images]
    pixels_before = [list(image.getdata()) for image in images]

    image2base64(images)

    assert [image.size for image in images] == sizes_before
    assert [list(image.getdata()) for image in images] == pixels_before


# ---------------------------------------------------------------------------
# pad_images_batch_to_max_size
# ---------------------------------------------------------------------------


def test_pad_images_empty_batch_raises_value_error():
    """Padding an empty batch has no maximum size and raises the existing error."""
    with pytest.raises(ValueError):
        pad_images_batch_to_max_size([])


def test_pad_images_singleton_batch_returns_rgb_copy_at_max_size():
    """A one-image batch defines the max size; output is RGB with same pixels."""
    image = make_image("L", (3, 2), 128)

    [padded] = pad_images_batch_to_max_size([image])

    assert padded.size == image.size
    assert padded.mode == "RGB"
    gray = (128, 128, 128)
    assert list(padded.getdata()) == [gray] * (3 * 2)


def test_pad_images_equal_sizes_is_pixel_identity():
    """Images already at the batch max size are copied without black borders."""
    images = [
        make_image("RGB", (4, 4), (200, 10, 10)),
        make_image("RGB", (4, 4), (10, 10, 200)),
    ]

    padded = pad_images_batch_to_max_size(images)

    assert [p.size for p in padded] == [(4, 4), (4, 4)]
    for result, expected in zip(padded, images):
        assert list(result.getdata()) == list(expected.getdata())


def test_pad_images_mixed_sizes_even_difference_centers_image():
    """An even difference pads symmetrically: a 2x2 image lands at x=y=1 in 4x4."""
    small = make_image("RGB", (2, 2), (255, 0, 0))
    large = make_image("RGB", (4, 4), (0, 255, 0))

    padded_small, padded_large = pad_images_batch_to_max_size([small, large])

    assert padded_small.size == (4, 4)
    assert padded_large.size == (4, 4)
    red = (255, 0, 0)
    black = (0, 0, 0)
    green = (0, 255, 0)
    for x in range(4):
        for y in range(4):
            expected = red if x in (1, 2) and y in (1, 2) else black
            assert padded_small.getpixel((x, y)) == expected
    # The largest image defines the canvas, so it is placed at (0, 0) fully.
    assert all(
        padded_large.getpixel((x, y)) == green for x in range(4) for y in range(4)
    )


def test_pad_images_odd_difference_leaves_extra_pixels_on_right_and_bottom():
    """For a 2x2 image in a 5x5 batch, placement is x=1..2, y=1..2; rest black."""
    small = make_image("RGB", (2, 2), (255, 0, 0))
    large = make_image("RGB", (5, 5), (0, 255, 0))

    padded_small, padded_large = pad_images_batch_to_max_size([small, large])

    assert padded_small.size == (5, 5)
    assert padded_small.mode == "RGB"
    red = (255, 0, 0)
    black = (0, 0, 0)
    for x in range(5):
        for y in range(5):
            expected = red if x in (1, 2) and y in (1, 2) else black
            assert padded_small.getpixel((x, y)) == expected
    # Right/bottom keep the odd remainder (2 px) versus left/top (1 px).
    assert padded_small.getpixel((3, 1)) == black
    assert padded_small.getpixel((1, 3)) == black
    # Ordering and full coverage for the largest image.
    assert all(
        padded_large.getpixel((x, y)) == (0, 255, 0) for x in range(5) for y in range(5)
    )


def test_pad_images_output_order_matches_input_order():
    """Outputs align positionally with inputs for a mixed-size batch."""
    images = [
        make_image("RGB", (1, 1), (255, 0, 0)),
        make_image("RGB", (3, 3), (0, 255, 0)),
        make_image("RGB", (2, 2), (0, 0, 255)),
    ]

    padded = pad_images_batch_to_max_size(images)

    assert [p.size for p in padded] == [(3, 3)] * 3
    # 1x1 red image: (3-1)//2 == 1, so its single pixel lands at (1, 1).
    assert padded[0].getpixel((1, 1)) == (255, 0, 0)
    assert padded[0].getpixel((1, 0)) == (0, 0, 0)
    assert padded[1].getpixel((0, 0)) == (0, 255, 0)
    # 2x2 blue image: (3-2)//2 == 0, so it sits at the top-left corner.
    assert padded[2].getpixel((0, 0)) == (0, 0, 255)
    assert padded[2].getpixel((1, 1)) == (0, 0, 255)
    assert padded[2].getpixel((2, 0)) == (0, 0, 0)


def test_pad_images_does_not_modify_inputs():
    """Padding reads but never mutates the input images' sizes or pixels."""
    images = [
        make_image("RGB", (2, 2), (7, 7, 7)),
        make_image("RGB", (3, 5), (9, 9, 9)),
    ]
    sizes_before = [image.size for image in images]
    pixels_before = [list(image.getdata()) for image in images]

    pad_images_batch_to_max_size(images)

    assert [image.size for image in images] == sizes_before
    assert [list(image.getdata()) for image in images] == pixels_before
