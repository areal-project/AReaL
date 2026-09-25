# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from examples.swe.qwen38_flash_next import patch_sglang_qwen4_vl as patch


def test_unknown_processor_version_is_not_modified(tmp_path, monkeypatch):
    source = tmp_path / "qwen_vl.py"
    source.write_bytes(b"# an unsupported SGLang version\n")
    before = source.read_bytes()
    monkeypatch.setattr("sys.argv", ["patch", "--target", str(source)])
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        patch.main()
    assert source.read_bytes() == before


def test_processor_patch_is_idempotent_and_keeps_other_model_policies(monkeypatch):
    # Isolate the source transformer from imports of the optional GPU runtime.
    source = (
        "class Processor:\n    def configure(self):\n"
        + patch.OLD
        + "            self.factor = 32\n"
        + '        if self.model_type == "qwen3_vl":\n'
        + "            self.max_pixels //= 16\n"
    ).encode()
    expected = source.replace(patch.OLD.encode(), patch.NEW.encode())
    monkeypatch.setattr(patch, "EXPECTED_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(patch, "PATCHED_SHA256", hashlib.sha256(expected).hexdigest())
    patched = patch.patched_source(source)
    assert patch.patched_source(patched) == patched
    namespace = {}
    exec(compile(patched, "qwen_vl.py", "exec"), namespace)
    for model, factor, pixels in [
        ("qwen4_exp", 32, 16777216),
        ("qwen3_vl", 32, 1048576),
        ("qwen2_vl", 28, 16777216),
    ]:
        processor = namespace["Processor"]()
        processor.model_type = model
        processor.factor = 28
        processor.max_pixels = 16777216
        processor.configure()
        assert (processor.factor, processor.max_pixels) == (factor, pixels)


def test_recipe_enables_processor_patch_and_group_filter():
    from pathlib import Path

    import yaml

    config = yaml.safe_load(
        Path(patch.__file__).with_name("swe_mm_rl.yaml").read_text()
    )
    assert config["should_accept_fn"] == "examples.swe.filter_function.filter_function"
    commands = config["rollout"]["scheduling_spec"][0]["additional_bash_cmds"]
    assert "python3 -m examples.swe.qwen38_flash_next.patch_sglang_qwen4_vl" in commands
