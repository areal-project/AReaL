# SPDX-License-Identifier: Apache-2.0
"""Dense Qwen keeps model pixel bounds and lets HF do the only resize."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from areal.engine import sglang_model_patches as startup
from areal.engine import sglang_qwen35_vl as patch


@pytest.fixture
def source(monkeypatch):
    text = (
        "class Processor:\n"
        "    def configure(self, envs):\n"
        "        if True:\n"
        + patch.CAP_OLD
        + "                self.max_pixels //= 16\n"
        + "    def should_resize(self):\n"
        + "        return (\n"
        + patch.RESIZE_OLD
        + "        )\n"
    ).encode()
    result = text.replace(patch.CAP_OLD.encode(), patch.CAP_NEW.encode()).replace(
        patch.RESIZE_OLD.encode(), patch.RESIZE_NEW.encode()
    )
    monkeypatch.setattr(patch, "EXPECTED_SHA256", hashlib.sha256(text).hexdigest())
    monkeypatch.setattr(patch, "PATCHED_SHA256", hashlib.sha256(result).hexdigest())
    return text


@pytest.mark.parametrize("limit", [True, False])
@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_vl", "qwen3_5_moe"])
def test_pixel_budget_and_resize_are_model_specific(source, limit, model_type):
    namespace = {}
    result = patch.patched_source(source)
    exec(compile(result, "<processor>", "exec"), namespace)
    processor = namespace["Processor"]()
    processor.model_type = model_type
    processor.max_pixels = 16777216
    processor.configure(
        SimpleNamespace(SGLANG_VLM_QWEN_LIMIT_PIXEL=SimpleNamespace(get=lambda: limit))
    )
    expected = 16777216 if model_type == "qwen3_5" or not limit else 1048576
    assert processor.max_pixels == expected
    assert processor.should_resize() == (model_type != "qwen3_5")
    assert patch.patched_source(result) == result


def test_unknown_source_is_rejected():
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        patch.patched_source(b"# unsupported source")


def test_dense_startup_only_patches_images_and_registers_token_processor(
    source, base_source, tmp_path, monkeypatch
):
    root = tmp_path / "sglang"
    target = root / patch.TARGET_RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    base_target = root / patch.BASE_RELATIVE
    base_target.write_bytes(base_source)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setattr(startup, "installed_sglang_root", lambda: root)
    monkeypatch.delenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", raising=False)
    startup.apply_model_patches(["--model-path", str(model)])
    assert target.read_bytes() == patch.patched_source(source)
    assert base_target.read_bytes() == patch.patched_base_source(base_source)
    mtime = target.stat().st_mtime_ns
    startup.apply_model_patches(["--model-path", str(model)])
    assert target.stat().st_mtime_ns == mtime
    import os

    assert (
        os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"]
        == "areal.engine.sglang_processors"
    )


@pytest.fixture
def base_source(monkeypatch):
    text = (
        "class Processor:\n"
        "    def device(self):\n"
        "        if True:\n"
        + patch.DEVICE_OLD
        + '                return "cpu"\n'
        + '            return "cuda:0"\n'
    ).encode()
    result = text.replace(patch.DEVICE_OLD.encode(), patch.DEVICE_NEW.encode())
    monkeypatch.setattr(patch, "BASE_EXPECTED_SHA256", hashlib.sha256(text).hexdigest())
    monkeypatch.setattr(
        patch, "BASE_PATCHED_SHA256", hashlib.sha256(result).hexdigest()
    )
    return text


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_vl", "qwen3_5_moe"])
def test_gpu_server_uses_cpu_hf_only_for_dense_qwen(base_source, model_type):
    namespace = {
        "_is_cpu": False,
        "get_global_server_args": lambda: SimpleNamespace(rl_on_policy_target=None),
    }
    result = patch.patched_base_source(base_source)
    exec(compile(result, "<processor>", "exec"), namespace)
    processor = namespace["Processor"]()
    processor.model_type = model_type
    assert processor.device() == ("cpu" if model_type == "qwen3_5" else "cuda:0")
    assert patch.patched_base_source(result) == result
