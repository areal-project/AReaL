# SPDX-License-Identifier: Apache-2.0

import base64
import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from areal.infra.processor_cache import ProcessorCallCache
from areal.utils.qwen_vl_image import (
    _sglang_image_size,
    prepare_sglang_qwen_images,
    sglang_qwen_image_kwargs,
)


class Qwen3VLProcessor:
    def __init__(self, model_path):
        self.tokenizer = SimpleNamespace(name_or_path=str(model_path))
        self.image_processor = SimpleNamespace(
            patch_size=16,
            merge_size=2,
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )


@pytest.fixture
def aligned_processor(monkeypatch, tmp_path):
    monkeypatch.delenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", raising=False)
    monkeypatch.delenv("SGLANG_RESIZE_RESAMPLE", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    return Qwen3VLProcessor(tmp_path)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((256, 256), (256, 256)),
        ((700, 900), (704, 896)),
        ((1200, 1600), (864, 1152)),
        ((56, 84), (224, 320)),
    ],
)
def test_image_size_matches_sglang_pixel_budget(shape, expected):
    assert _sglang_image_size(*shape, 32, 65536, 1048576) == expected


@pytest.mark.parametrize("shape", [(256, 256), (900, 700), (1600, 1200), (84, 56)])
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_resize_preserves_original_wire_bytes_and_matches_pil_policy(
    aligned_processor, shape, mode
):
    rng = np.random.default_rng(1)
    image = Image.fromarray(
        rng.integers(0, 256, (shape[1], shape[0], 3), dtype=np.uint8)
    ).convert(mode)
    stream = BytesIO()
    image.save(stream, format="JPEG" if mode == "RGB" else "PNG")
    wire = base64.b64encode(stream.getvalue()).decode()
    image = Image.open(BytesIO(stream.getvalue())).convert("RGB")
    original = np.asarray(image).copy()
    prepared, encoded = prepare_sglang_qwen_images(
        aligned_processor, [image], image_keys=[wire]
    )

    assert encoded == [wire]
    height, width = _sglang_image_size(image.height, image.width, 32, 65536, 1048576)
    expected = image.resize((width, height), resample=None)
    np.testing.assert_array_equal(np.asarray(prepared[0]), np.asarray(expected))
    assert width * height <= 1048576
    again, _ = prepare_sglang_qwen_images(aligned_processor, prepared)
    np.testing.assert_array_equal(np.asarray(again[0]), np.asarray(prepared[0]))
    np.testing.assert_array_equal(np.asarray(image), original)
    assert aligned_processor.image_processor.size["longest_edge"] == 16777216


def test_alignment_honors_disabled_sglang_pixel_limit(aligned_processor, monkeypatch):
    monkeypatch.setenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", "false")
    prepared, _ = prepare_sglang_qwen_images(
        aligned_processor, [Image.new("RGB", (1600, 1200))]
    )
    assert prepared[0].width * prepared[0].height > 1048576


def test_alignment_preserves_unsupported_processors():
    assert prepare_sglang_qwen_images(object(), [Image.new("RGB", (256, 256))]) is None


@pytest.mark.parametrize("model_type", ["qwen4_exp", "llama", None])
def test_shared_processor_does_not_apply_qwen35_pixel_cap_to_other_models(
    tmp_path, model_type
):
    # Both real models use Qwen3VLProcessor and the same original pixel bounds.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    processor = Qwen3VLProcessor(tmp_path)
    image = Image.new("RGB", (1600, 1200))

    assert prepare_sglang_qwen_images(processor, [image]) is None
    assert image.size == (1600, 1200)
    assert processor.image_processor.size["longest_edge"] == 16777216


def test_processor_without_model_identity_is_left_unchanged(tmp_path):
    processor = Qwen3VLProcessor(tmp_path)
    del processor.tokenizer.name_or_path
    assert prepare_sglang_qwen_images(processor, [Image.new("RGB", (256, 256))]) is None


def test_alignment_keeps_multiple_images_in_order(aligned_processor):
    prepared, encoded = prepare_sglang_qwen_images(
        aligned_processor,
        [Image.new("RGB", (900, 700), "red"), Image.new("L", (1600, 1200), 127)],
    )
    assert len(prepared) == 2
    assert encoded == []
    assert prepared[0].getpixel((0, 0)) == (255, 0, 0)
    assert prepared[1].getpixel((0, 0)) == (127, 127, 127)


def test_image_preparation_reuses_history_across_changed_prompts(
    aligned_processor, monkeypatch
):
    cache = ProcessorCallCache()
    image = Image.new("RGB", (900, 700), "red")
    expected, expected_wire = prepare_sglang_qwen_images(aligned_processor, [image])
    calls = []
    original_resize = Image.Image.resize

    def resize(self, *args, **kwargs):
        calls.append(self.size)
        return original_resize(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize)
    monkeypatch.setattr(Image.Image, "save", lambda *a, **k: pytest.fail("re-encoding"))
    first, first_wire = prepare_sglang_qwen_images(
        aligned_processor, [image], cache=cache, image_keys=["red-image"]
    )
    first[0].putpixel((0, 0), (0, 0, 0))
    second, second_wire = prepare_sglang_qwen_images(
        aligned_processor,
        [image, Image.new("RGB", (256, 256), "blue")],
        cache=cache,
        image_keys=["red-image", "blue-image"],
    )
    assert len(calls) == 2  # Each unique image resized once, despite new history.
    assert expected_wire == []
    assert first_wire == second_wire[:1] == ["red-image"]
    np.testing.assert_array_equal(np.asarray(second[0]), np.asarray(expected[0]))
    cache.close()
    prepare_sglang_qwen_images(
        aligned_processor, [image], cache=cache, image_keys=["red-image"]
    )
    assert len(calls) == 3


def test_hf_kwargs_match_effective_budget_without_repeated_reduction(aligned_processor):
    first = sglang_qwen_image_kwargs(aligned_processor)
    second = sglang_qwen_image_kwargs(aligned_processor)
    assert (
        first
        == second
        == {
            "device": "cpu",
            "size": {"shortest_edge": 65536, "longest_edge": 1048576},
        }
    )
    assert aligned_processor.image_processor.size["longest_edge"] == 16777216


def test_image_preparation_cache_respects_pixel_policy_changes(
    aligned_processor, monkeypatch
):
    cache = ProcessorCallCache()
    image = Image.new("RGB", (1600, 1200))
    first, _ = prepare_sglang_qwen_images(
        aligned_processor, [image], cache=cache, image_keys=["same-image"]
    )
    monkeypatch.setenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", "false")
    second, _ = prepare_sglang_qwen_images(
        aligned_processor, [image], cache=cache, image_keys=["same-image"]
    )
    assert first[0].width * first[0].height <= 1048576
    assert second[0].width * second[0].height > 1048576


@pytest.mark.parametrize("shape", [(1, 201), (0, 32)])
def test_alignment_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError, match="aspect ratio"):
        _sglang_image_size(*shape, 32, 65536, 1048576)
