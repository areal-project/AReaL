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
    sglang_image_alignment_enabled,
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
    monkeypatch.delenv("AREAL_VLM_ALIGN_SGLANG", raising=False)
    monkeypatch.delenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", raising=False)
    monkeypatch.delenv("SGLANG_RESIZE_RESAMPLE", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    return Qwen3VLProcessor(tmp_path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("true", True),
        (" TRUE ", True),
        ("1", True),
        ("yes", True),
        ("y", True),
        ("false", False),
        (" FALSE ", False),
        ("0", False),
        ("no", False),
        ("n", False),
    ],
)
def test_alignment_switch_accepts_boolean_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("AREAL_VLM_ALIGN_SGLANG", raising=False)
    else:
        monkeypatch.setenv("AREAL_VLM_ALIGN_SGLANG", value)
    assert sglang_image_alignment_enabled() is expected


@pytest.mark.parametrize("value", ["", "treu", "2"])
def test_alignment_switch_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("AREAL_VLM_ALIGN_SGLANG", value)
    with pytest.raises(ValueError, match="AREAL_VLM_ALIGN_SGLANG"):
        sglang_image_alignment_enabled()


def test_disabled_alignment_keeps_sglang_metadata_policy(
    aligned_processor, monkeypatch
):
    monkeypatch.setenv("AREAL_VLM_ALIGN_SGLANG", "false")
    assert not sglang_image_alignment_enabled()
    assert sglang_qwen_image_kwargs(aligned_processor) is not None
    assert aligned_processor.image_processor.size["longest_edge"] == 16777216


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
def test_preparation_preserves_original_dimensions_pixels_and_wire_bytes(
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
    assert prepared[0].size == image.size
    np.testing.assert_array_equal(np.asarray(prepared[0]), original)
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


def test_enabled_compression_restores_legacy_pixels_and_preserves_wire(
    aligned_processor, monkeypatch
):
    monkeypatch.setenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", " TRUE ")
    monkeypatch.setenv("SGLANG_RESIZE_RESAMPLE", "BILINEAR")
    rng = np.random.default_rng(2)
    image = Image.fromarray(rng.integers(0, 256, (1200, 1600, 3), dtype=np.uint8))
    original = np.asarray(image).copy()

    prepared, wire = prepare_sglang_qwen_images(
        aligned_processor, [image], image_keys=["original-wire-bytes"]
    )

    assert prepared[0].size == (1152, 864)
    expected = image.resize((1152, 864), resample=Image.Resampling.BILINEAR)
    np.testing.assert_array_equal(np.asarray(prepared[0]), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(image), original)
    assert wire == ["original-wire-bytes"]


@pytest.mark.parametrize("value", ["", "treu", "2"])
def test_image_compression_rejects_invalid_switch_values(
    aligned_processor, monkeypatch, value
):
    monkeypatch.setenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", value)
    with pytest.raises(ValueError, match="SGLANG_VLM_QWEN_LIMIT_PIXEL"):
        sglang_qwen_image_kwargs(aligned_processor)


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
    monkeypatch.setattr(Image.Image, "resize", lambda *a, **k: pytest.fail("resize"))
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
    assert expected_wire == []
    assert first_wire == second_wire[:1] == ["red-image"]
    np.testing.assert_array_equal(np.asarray(second[0]), np.asarray(expected[0]))
    cache.close()
    prepare_sglang_qwen_images(
        aligned_processor, [image], cache=cache, image_keys=["red-image"]
    )


@pytest.mark.parametrize(
    ("value", "max_pixels"), [("false", 16777216), ("true", 1048576)]
)
def test_hf_kwargs_match_effective_budget_without_repeated_reduction(
    aligned_processor, monkeypatch, value, max_pixels
):
    monkeypatch.setenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", value)
    first = sglang_qwen_image_kwargs(aligned_processor)
    second = sglang_qwen_image_kwargs(aligned_processor)
    assert (
        first
        == second
        == {
            "device": "cpu",
            "size": {"shortest_edge": 65536, "longest_edge": max_pixels},
        }
    )
    assert aligned_processor.image_processor.size["longest_edge"] == 16777216


@pytest.mark.parametrize("max_pixels", [1048575, 16777216])
def test_image_preparation_switch_does_not_reuse_other_mode_cache(
    aligned_processor, monkeypatch, max_pixels
):
    aligned_processor.image_processor.size["longest_edge"] = max_pixels
    cache = ProcessorCallCache()
    image = Image.new("RGB", (900, 700))
    monkeypatch.setenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", "true")
    first, _ = prepare_sglang_qwen_images(
        aligned_processor, [image], cache=cache, image_keys=["same-image"]
    )
    monkeypatch.setenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", "false")
    second, _ = prepare_sglang_qwen_images(
        aligned_processor, [image], cache=cache, image_keys=["same-image"]
    )
    assert first[0].size == (896, 704)
    assert second[0].size == image.size
    assert sglang_qwen_image_kwargs(aligned_processor)["size"] == (
        aligned_processor.image_processor.size
    )
    monkeypatch.setenv("SGLANG_VLM_QWEN_LIMIT_PIXEL", "true")
    third, _ = prepare_sglang_qwen_images(
        aligned_processor, [image], cache=cache, image_keys=["same-image"]
    )
    assert third[0].size == first[0].size
    cache.close()


@pytest.mark.parametrize("shape", [(1, 201), (0, 32)])
def test_alignment_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError, match="aspect ratio"):
        _sglang_image_size(*shape, 32, 65536, 1048576)
