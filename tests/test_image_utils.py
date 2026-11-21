# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

import pytest
from PIL import Image

from areal.utils.image import image2base64, pad_images_batch_to_max_size


@pytest.mark.parametrize("count", [0, 1, 3])
def test_png_round_trip_preserves_modes_pixels_and_order(count):
    """Decode PNGs from both accepted input forms and compare to pre-call snapshots."""
    images = [
        Image.new("RGBA", (2, 3), (19, 23, 73, 41)),
        Image.new("RGB", (3, 2), (41, 53, 67)),
        Image.new("L", (1, 2), 89),
    ][:count]
    before = [(im.mode, im.size, im.tobytes()) for im in images]
    encoded = image2base64(images[0] if count == 1 else images)
    assert isinstance(encoded, list) and len(encoded) == count
    for value, expected in zip(encoded, before, strict=True):
        with Image.open(BytesIO(base64.b64decode(value, validate=True))) as decoded:
            assert decoded.format == "PNG"
            assert (decoded.mode, decoded.size, decoded.tobytes()) == expected
    assert [(im.mode, im.size, im.tobytes()) for im in images] == before


@pytest.mark.parametrize(
    ("sizes", "canvas", "offsets"),
    [
        ([(3, 2)], (3, 2), [(0, 0)]),
        ([(2, 3), (2, 3)], (2, 3), [(0, 0), (0, 0)]),
        ([(2, 4), (4, 2)], (4, 4), [(1, 0), (0, 1)]),
        ([(2, 2), (5, 5)], (5, 5), [(1, 1), (0, 0)]),
    ],
)
def test_padding_geometry_order_and_source_pixels(sizes, canvas, offsets):
    """Compare every canvas pixel to independently specified placement and color."""
    images = [Image.new("L", size, 50 + i) for i, size in enumerate(sizes)]
    for im in images:
        im.putpixel((0, 0), 127)
    before = [(im.size, im.tobytes()) for im in images]
    outputs = pad_images_batch_to_max_size(images)
    for result, (size, pixels), (left, top) in zip(
        outputs, before, offsets, strict=True
    ):
        assert result.mode == "RGB" and result.size == canvas
        width, height = size
        for y in range(canvas[1]):
            for x in range(canvas[0]):
                value = 0
                if left <= x < left + width and top <= y < top + height:
                    value = pixels[(y - top) * width + x - left]
                assert result.getpixel((x, y)) == (value, value, value)
    assert [(im.size, im.tobytes()) for im in images] == before


def test_empty_padding_raises_value_error():
    """An empty batch has no maximum canvas dimensions."""
    with pytest.raises(ValueError):
        pad_images_batch_to_max_size([])
