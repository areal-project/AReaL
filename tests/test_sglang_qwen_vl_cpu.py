# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import pytest

from areal.engine import sglang_model_patches as startup
from areal.engine import sglang_qwen_vl_cpu as patch


@pytest.fixture
def cpu_runtime(tmp_path, monkeypatch):
    root = tmp_path / "sglang"
    qwen = (
        "class Processor:\n"
        + patch.DECODE_OLD
        + "    def configure(self):\n"
        + "        self.max_pixels = 16777216 // 16\n"
        + "    def resize(self):\n"
        + "        return True\n"
    ).encode()
    base = (
        "class Base:\n"
        "    def process(self):\n"
        "        if True:\n"
        + patch.NATIVE_DEVICE
        + '                return "cpu"\n'
        + '            return "cuda:0"\n'
    ).encode()
    expected_qwen = qwen.replace(patch.DECODE_OLD.encode(), patch.DECODE_NEW.encode())
    expected_base = base.replace(
        patch.NATIVE_DEVICE.encode(), patch.LEGACY_CPU_DEVICE.encode()
    )
    for name, data in [
        ("EXPECTED_SHA256", qwen),
        ("PATCHED_SHA256", expected_qwen),
        ("BASE_EXPECTED_SHA256", base),
        ("LEGACY_CPU_SHA256", expected_base),
    ]:
        monkeypatch.setattr(patch, name, hashlib.sha256(data).hexdigest())
    for relative, source in [
        (patch.TARGET_RELATIVE, qwen),
        (patch.BASE_RELATIVE, base),
    ]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    monkeypatch.setattr(startup, "installed_sglang_root", lambda: root)
    return root, qwen, base


@pytest.mark.parametrize("model_type", patch.MODEL_TYPES)
def test_cpu_profile_keeps_native_resize_and_budget(cpu_runtime, tmp_path, model_type):
    root, qwen, base = cpu_runtime
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))

    startup.apply_model_patches(["--model", str(model)])
    qwen_path = root / patch.TARGET_RELATIVE
    base_path = root / patch.BASE_RELATIVE
    assert qwen_path.read_bytes() == patch.patched_source(qwen)
    assert base_path.read_bytes() == base
    assert base_path.read_bytes() == patch.patched_base_source(base)
    namespace = {}
    exec(compile(qwen_path.read_bytes(), "qwen_vl.py", "exec"), namespace)
    processor = namespace["Processor"]()
    processor.configure()
    assert processor.max_pixels == 1048576
    assert processor.resize()
    assert processor.gpu_image_decode is False

    namespace = {
        "_is_cpu": False,
        "get_global_server_args": lambda: SimpleNamespace(rl_on_policy_target=None),
    }
    exec(compile(base_path.read_bytes(), "base_processor.py", "exec"), namespace)
    processor = namespace["Base"]()
    processor.model_type = model_type
    assert processor.process() == "cuda:0"
    processor.model_type = "llama"
    assert processor.process() == "cuda:0"

    mtimes = (qwen_path.stat().st_mtime_ns, base_path.stat().st_mtime_ns)
    startup.apply_model_patches(["--model-path", str(model)])
    assert (qwen_path.stat().st_mtime_ns, base_path.stat().st_mtime_ns) == mtimes


def test_unknown_base_source_does_not_partially_install(cpu_runtime, tmp_path):
    root, qwen, _ = cpu_runtime
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    target = root / patch.BASE_RELATIVE
    target.write_bytes(b"# unknown source\n")
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        startup.apply_model_patches(["--model-path", str(model)])
    assert (root / patch.TARGET_RELATIVE).read_bytes() == qwen
    assert target.read_bytes() == b"# unknown source\n"


@pytest.mark.parametrize("model_type", patch.MODEL_TYPES)
def test_existing_cpu_hf_override_is_restored_to_native_dispatch(
    cpu_runtime, tmp_path, model_type
):
    root, qwen, base = cpu_runtime
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    base_path = root / patch.BASE_RELATIVE
    legacy = base.replace(
        patch.NATIVE_DEVICE.encode(), patch.LEGACY_CPU_DEVICE.encode()
    )
    base_path.write_bytes(legacy)

    startup.apply_model_patches(["--model", str(model)])

    assert base_path.read_bytes() == base
    assert (root / patch.TARGET_RELATIVE).read_bytes() == patch.patched_source(qwen)
    namespace = {
        "_is_cpu": False,
        "get_global_server_args": lambda: SimpleNamespace(rl_on_policy_target=None),
    }
    exec(compile(base_path.read_bytes(), "base_processor.py", "exec"), namespace)
    processor = namespace["Base"]()
    processor.model_type = model_type
    assert processor.process() == "cuda:0"
    namespace["_is_cpu"] = True
    assert processor.process() == "cpu"
    namespace["_is_cpu"] = False
    namespace["get_global_server_args"] = lambda: SimpleNamespace(
        rl_on_policy_target="target"
    )
    assert processor.process() == "cpu"
