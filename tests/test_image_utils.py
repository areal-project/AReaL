import base64
from io import BytesIO

import pytest
from PIL import Image

from areal.utils.image import image2base64, pad_images_batch_to_max_size


def _rgb_image() -> Image.Image:
    image = Image.new("RGB", (2, 2))
    image.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ]
    )
    return image


def _rgba_image() -> Image.Image:
    image = Image.new("RGBA", (2, 2))
    image.putdata(
        [
            (255, 0, 0, 255),
            (0, 255, 0, 128),
            (0, 0, 255, 0),
            (255, 255, 0, 32),
        ]
    )
    return image


def _l_image() -> Image.Image:
    image = Image.new("L", (2, 2))
    image.putdata([0, 128, 255, 64])
    return image


def _snapshot(image: Image.Image):
    return image.mode, image.size, list(image.getdata())


def _decode_png(encoded: str) -> Image.Image:
    image = Image.open(BytesIO(base64.b64decode(encoded, validate=True)))
    image.load()
    return image


def _solid_rgb(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def test_image2base64_single_rgb_image_round_trips_png():
    image = _rgb_image()
    before = _snapshot(image)

    encoded = image2base64(image)

    assert isinstance(encoded, list)
    assert len(encoded) == 1
    assert isinstance(encoded[0], str)
    decoded = _decode_png(encoded[0])
    assert decoded.format == "PNG"
    assert decoded.mode == "RGB"
    assert decoded.size == (2, 2)
    assert list(decoded.getdata()) == list(image.getdata())
    assert _snapshot(image) == before


def test_image2base64_list_round_trips_rgb_rgba_l_and_preserves_order():
    images = [_rgba_image(), _l_image(), _rgb_image()]
    before = [_snapshot(image) for image in images]

    encoded = image2base64(images)

    assert len(encoded) == 3
    assert all(isinstance(value, str) for value in encoded)
    for index, original in enumerate(images):
        decoded = _decode_png(encoded[index])
        assert decoded.format == "PNG"
        assert decoded.mode == original.mode
        assert decoded.size == original.size
        assert list(decoded.getdata()) == list(original.getdata())
        assert _snapshot(original) == before[index]


def test_image2base64_empty_list_returns_empty_list():
    assert image2base64([]) == []


def test_pad_single_l_image_returns_rgb_canvas_with_actual_pixels():
    image = _l_image()
    before = _snapshot(image)

    padded = pad_images_batch_to_max_size([image])

    assert len(padded) == 1
    assert padded[0].mode == "RGB"
    assert padded[0].size == (2, 2)
    assert list(padded[0].getdata()) == [
        (0, 0, 0),
        (128, 128, 128),
        (255, 255, 255),
        (64, 64, 64),
    ]
    assert _snapshot(image) == before


def test_pad_same_size_images_preserves_order_and_pixels():
    red = _solid_rgb((2, 2), (255, 0, 0))
    blue = _solid_rgb((2, 2), (0, 0, 255))
    before = [_snapshot(red), _snapshot(blue)]

    padded = pad_images_batch_to_max_size([red, blue])

    assert len(padded) == 2
    assert set(padded[0].getdata()) == {(255, 0, 0)}
    assert set(padded[1].getdata()) == {(0, 0, 255)}
    assert [_snapshot(red), _snapshot(blue)] == before


def test_pad_mixed_sizes_uses_black_canvas_and_centered_offsets():
    red = _solid_rgb((1, 2), (255, 0, 0))
    blue = _solid_rgb((3, 1), (0, 0, 255))
    green = _solid_rgb((2, 2), (0, 255, 0))
    before = [_snapshot(red), _snapshot(blue), _snapshot(green)]

    padded = pad_images_batch_to_max_size([red, blue, green])

    assert [image.size for image in padded] == [(3, 2), (3, 2), (3, 2)]
    assert all(image.mode == "RGB" for image in padded)

    red_canvas = padded[0]
    assert red_canvas.getpixel((1, 0)) == (255, 0, 0)
    assert red_canvas.getpixel((1, 1)) == (255, 0, 0)
    assert {
        red_canvas.getpixel((x, y))
        for x in range(3)
        for y in range(2)
        if (x, y) not in ((1, 0), (1, 1))
    } == {(0, 0, 0)}

    blue_canvas = padded[1]
    assert [blue_canvas.getpixel((x, 0)) for x in range(3)] == [(0, 0, 255)] * 3
    assert [blue_canvas.getpixel((x, 1)) for x in range(3)] == [(0, 0, 0)] * 3

    green_canvas = padded[2]
    assert {green_canvas.getpixel((x, y)) for x in range(2) for y in range(2)} == {
        (0, 255, 0)
    }
    assert {
        green_canvas.getpixel((x, y)) for x in range(3) for y in range(2) if x == 2
    } == {(0, 0, 0)}

    assert [_snapshot(red), _snapshot(blue), _snapshot(green)] == before


def test_pad_2x2_and_5x5_uses_floor_division_offsets():
    small = _solid_rgb((2, 2), (255, 255, 255))
    large = _solid_rgb((5, 5), (255, 0, 0))
    before = [_snapshot(small), _snapshot(large)]

    padded = pad_images_batch_to_max_size([small, large])

    assert [image.size for image in padded] == [(5, 5), (5, 5)]
    small_canvas = padded[0]
    for x in range(5):
        for y in range(5):
            expected = (255, 255, 255) if 1 <= x <= 2 and 1 <= y <= 2 else (0, 0, 0)
            assert small_canvas.getpixel((x, y)) == expected

    assert set(padded[1].getdata()) == {(255, 0, 0)}
    assert [_snapshot(small), _snapshot(large)] == before


def test_pad_empty_list_preserves_value_error():
    with pytest.raises(ValueError):
        pad_images_batch_to_max_size([])
