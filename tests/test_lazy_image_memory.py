from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace

import pytest
from PIL import Image

from areal.infra.processor_cache import ProcessorCallCache
from areal.utils import qwen_vl_image as module


@pytest.fixture
def processor(monkeypatch):
    monkeypatch.setattr(module, "_processor_model_type", lambda _: "qwen3_vl")
    cls = type("Qwen3VLProcessor", (), {})
    obj = cls()
    obj.tokenizer = SimpleNamespace(name_or_path="local-test")
    obj.image_processor = SimpleNamespace(
        patch_size=14, merge_size=2, size={"shortest_edge": 784, "longest_edge": 3136}
    )
    return obj


def test_history_lazy_decode_preserves_pixels_and_isolation(processor):
    cache = ProcessorCallCache()
    calls = []

    def factory(index):
        def decode():
            calls.append(index)
            return Image.new("RGB", (123, 187), (index, 44, 91))

        return decode

    first, wire = module.prepare_sglang_qwen_images(
        processor, [factory(1), factory(2)], cache=cache, image_keys=["a", "b"]
    )
    first[0].putpixel((0, 0), (255, 255, 255))
    second, wire = module.prepare_sglang_qwen_images(
        processor,
        [factory(1), factory(2), factory(3)],
        cache=cache,
        image_keys=["a", "b", "c"],
    )
    expected, _ = module.prepare_sglang_qwen_images(
        processor, [Image.new("RGB", (123, 187), (i, 44, 91)) for i in (1, 2, 3)]
    )
    assert calls == [1, 2, 3]
    assert wire == ["a", "b", "c"]
    assert [im.tobytes() for im in second] == [im.tobytes() for im in expected]
    processor.image_processor.size["longest_edge"] = 6272
    module.prepare_sglang_qwen_images(
        processor, [factory(1)], cache=cache, image_keys=["a"]
    )
    assert calls == [1, 2, 3, 1]


def test_concurrent_history_singleflight(processor):
    cache, lock, calls = ProcessorCallCache(), Lock(), []

    def decode():
        with lock:
            calls.append(1)
        return Image.new("RGB", (128, 128))

    def prepare(_):
        return module.prepare_sglang_qwen_images(
            processor, [decode], cache=cache, image_keys=["same"]
        )[0][0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(prepare, range(32)))
    assert len(calls) == 1
    assert len({id(im) for im in results}) == 32


def test_decode_error_and_unknown_processor(processor):
    def bad():
        raise ValueError("invalid image")

    assert module.prepare_sglang_qwen_images(object(), [bad]) is None
    with pytest.raises(ValueError, match="invalid image"):
        module.prepare_sglang_qwen_images(
            processor, [bad], cache=ProcessorCallCache(), image_keys=["bad"]
        )
