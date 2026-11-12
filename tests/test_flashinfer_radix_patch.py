# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

# This preparation tool intentionally works before the training dependencies load.
spec = importlib.util.spec_from_file_location(
    "flashinfer_radix_patch",
    Path(__file__).parents[1] / "areal/engine/flashinfer_radix_patch.py",
)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    source = tmp_path / "installed"
    source.mkdir()
    (source / "header.cuh").write_bytes(b"old barrier\n")
    (source / "untouched.py").write_bytes(b"payload\n")
    specification = {
        "upstream": "test",
        "files": [
            {
                "path": "header.cuh",
                "before": hashlib.sha256(b"old barrier\n").hexdigest(),
                "after": hashlib.sha256(b"last exit\n").hexdigest(),
                "replacements": [("old barrier", "last exit")],
            }
        ],
    }
    monkeypatch.setattr(patch, "_specification", lambda: specification)
    return source, tmp_path / "output"


def test_prepare_runtime_preserves_installation_and_records_exact_output(runtime):
    source, output = runtime
    manifest = patch.prepare_runtime(source, output)
    assert (source / "header.cuh").read_bytes() == b"old barrier\n"
    assert (output / "flashinfer/header.cuh").read_bytes() == b"last exit\n"
    assert (output / "flashinfer/untouched.py").read_bytes() == b"payload\n"
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_prepare_runtime_unknown_source_creates_no_output(runtime):
    source, output = runtime
    (source / "header.cuh").write_bytes(b"different version\n")
    with pytest.raises(ValueError, match="Unsupported"):
        patch.prepare_runtime(source, output)
    assert not output.exists()


def test_prepare_runtime_does_not_follow_symlink_when_patching(runtime, tmp_path):
    source, output = runtime
    original = tmp_path / "original.cuh"
    (source / "header.cuh").rename(original)
    (source / "header.cuh").symlink_to(original)
    patch.prepare_runtime(source, output)
    assert original.read_bytes() == b"old barrier\n"
    assert not (output / "flashinfer/header.cuh").is_symlink()


def test_prepare_runtime_existing_output_is_never_replaced(runtime):
    source, output = runtime
    output.mkdir()
    (output / "marker").touch()
    with pytest.raises(FileExistsError):
        patch.prepare_runtime(source, output)
    assert (output / "marker").exists()


def test_prepare_runtime_copy_failure_leaves_no_partial_output(runtime, monkeypatch):
    source, output = runtime

    def fail_copy(*args, **kwargs):
        raise OSError("copy failure")

    monkeypatch.setattr(patch.shutil, "copytree", fail_copy)
    with pytest.raises(OSError, match="copy failure"):
        patch.prepare_runtime(source, output)
    assert not output.exists()
    assert not list(output.parent.glob(".output-*"))


def test_prepare_runtime_rejects_output_inside_installation(runtime):
    source, _ = runtime
    with pytest.raises(ValueError, match="outside"):
        patch.prepare_runtime(source, source / "nested")


def test_production_spec_uses_distinct_module_names_and_no_diagnostic_delay():
    specification = json.loads(
        Path(patch.__file__).with_name("flashinfer_radix_reset.json").read_text()
    )
    assert {item["path"] for item in specification["files"]} == {
        "data/include/flashinfer/topk.cuh",
        "jit/sampling.py",
        "jit/topk.py",
    }
    text = json.dumps(specification)
    assert "__nanosleep" not in text
    assert "RadixGroupResetStateLastCTA" in text
    for item in specification["files"]:
        if item["path"].startswith("jit/"):
            assert "_areal_radix_reset_9e032f75" in item["replacements"][0][1]
