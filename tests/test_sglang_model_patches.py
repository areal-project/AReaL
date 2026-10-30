# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from areal.engine import (
    _sglang_source_patch as installer,
)
from areal.engine import (
    sglang_mm_executor as executor,
)
from areal.engine import (
    sglang_model_patches as startup,
)
from areal.engine import (
    sglang_qsa_compress_gather as qsa,
)
from areal.engine import (
    sglang_qwen4_vl as qwen,
)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """Small compilable call sites, with the real transforms and source guards."""
    root = tmp_path / "sglang"
    sources = {
        executor: ("class Runner:\n    def run(self):\n" + executor.OLD).encode(),
        qsa: ("def gather():\n    if True:\n" + qsa.OLD).encode(),
        qwen: (
            "class Processor:\n    def configure(self):\n"
            + qwen.OLD
            + "            pass\n"
            + "    def should_resize(self, images):\n        return (\n"
            + qwen.RESIZE_OLD
            + "            and bool(images)\n        )\n"
            + qwen.DEVICE_OLD
            + "        self, *args, **kwargs\n    ):\n        pass\n"
        ).encode(),
    }
    for patch, source in sources.items():
        target = root / patch.TARGET_RELATIVE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
        target.chmod(0o640)
        patched = source.replace(patch.OLD.encode(), patch.NEW.encode())
        if patch is qwen:
            patched = patched.replace(
                qwen.RESIZE_OLD.encode(), qwen.RESIZE_NEW.encode()
            )
            patched = patched.replace(
                qwen.DEVICE_OLD.encode(), qwen.DEVICE_NEW.encode()
            )
        monkeypatch.setattr(
            patch, "EXPECTED_SHA256", hashlib.sha256(source).hexdigest()
        )
        monkeypatch.setattr(
            patch, "PATCHED_SHA256", hashlib.sha256(patched).hexdigest()
        )
    monkeypatch.setattr(startup, "installed_sglang_root", lambda: root)
    model = tmp_path / "arbitrary-model-name"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    return root, model, sources


def test_flash_next_startup_patches_all_sources_and_repeated_start_is_write_free(
    runtime, monkeypatch
):
    root, model, sources = runtime
    startup.apply_model_patches(["--model-path", str(model), "--tp-size", "8"])
    mtimes = {}
    for patch, source in sources.items():
        target = root / patch.TARGET_RELATIVE
        assert target.read_bytes() == patch.patched_source(source)
        assert target.stat().st_mode & 0o777 == 0o640
        mtimes[target] = target.stat().st_mtime_ns
    lock = root / ".areal-sglang-patches.lock"
    lock.unlink()
    monkeypatch.setattr(
        installer, "_atomic_write", lambda *args: pytest.fail("rewrite")
    )

    startup.apply_model_patches([f"--model-path={model}"])

    assert not lock.exists()
    assert all(target.stat().st_mtime_ns == mtime for target, mtime in mtimes.items())


@pytest.mark.parametrize("model_type", ["qwen2_vl", "llama", None])
def test_other_model_types_skip_sglang_even_with_flash_next_directory_name(
    tmp_path, monkeypatch, model_type
):
    model = tmp_path / "Qwen3.8-Flash-Next"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    monkeypatch.setattr(
        startup, "installed_sglang_root", lambda: pytest.fail("SGLang was accessed")
    )

    startup.apply_model_patches(["--model", str(model)])


def test_hub_model_detection_reads_raw_config_with_selected_revision(
    runtime, monkeypatch
):
    from transformers import PretrainedConfig

    root, _, sources = runtime
    calls = []

    def get_config(path, **kwargs):
        calls.append((path, kwargs))
        return {"model_type": "qwen4_exp"}, {}

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", get_config)
    startup.apply_model_patches(["--model-path", "org/model", "--revision", "revision"])

    assert calls == [("org/model", {"revision": "revision"})]
    for patch, source in sources.items():
        assert (root / patch.TARGET_RELATIVE).read_bytes() == patch.patched_source(
            source
        )


def test_unknown_source_rejects_entire_profile_without_partial_modification(runtime):
    root, model, sources = runtime
    target = root / qwen.TARGET_RELATIVE
    target.write_bytes(b"# unsupported processor\n")
    before = {
        root / patch.TARGET_RELATIVE: (root / patch.TARGET_RELATIVE).read_bytes()
        for patch in sources
    }

    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        startup.apply_model_patches(["--model-path", str(model)])

    assert all(target.read_bytes() == source for target, source in before.items())
    assert not (root / ".areal-sglang-patches.lock").exists()


@pytest.mark.parametrize("patch", [qsa, qwen, executor])
def test_patch_cli_applies_fix_and_repeats_safely(runtime, monkeypatch, patch):
    """Recipe commands use the guarded, idempotent engine patch entry points."""
    root, _, sources = runtime
    target = root / patch.TARGET_RELATIVE
    monkeypatch.setattr(sys, "argv", [patch.__name__, "--target", str(target)])

    patch.main()
    assert target.read_bytes() == patch.patched_source(sources[patch])
    mtime = target.stat().st_mtime_ns
    patch.main()

    assert target.stat().st_mtime_ns == mtime
    assert target.read_bytes() == patch.patched_source(sources[patch])


def test_concurrent_starts_install_each_source_once(runtime, monkeypatch):
    root, model, sources = runtime
    writes = []
    original_write = installer._atomic_write

    def write(target, source):
        writes.append(target)
        original_write(target, source)

    monkeypatch.setattr(installer, "_atomic_write", write)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda _: startup.apply_model_patches(["--model-path", str(model)]),
                range(8),
            )
        )

    assert sorted(writes) == sorted(root / patch.TARGET_RELATIVE for patch in sources)
    for patch, source in sources.items():
        assert (root / patch.TARGET_RELATIVE).read_bytes() == patch.patched_source(
            source
        )


@pytest.mark.parametrize("entrypoint", startup._ENTRYPOINTS)
def test_bootstrap_patches_before_exec_and_preserves_server_arguments(
    runtime, monkeypatch, entrypoint
):
    root, model, sources = runtime
    command = [
        "inference-python",
        "-m",
        entrypoint,
        "--model-path",
        str(model),
        "--tp-size",
        "8",
        "--headless",
    ]
    wrapped = startup.wrap_server_command(command)
    monkeypatch.setattr(sys, "argv", [wrapped[2], *wrapped[3:]])
    executions = []

    def execute(interpreter, argv):
        for patch, source in sources.items():
            assert (root / patch.TARGET_RELATIVE).read_bytes() == patch.patched_source(
                source
            )
        executions.append((interpreter, argv))

    monkeypatch.setattr(startup.os, "execv", execute)
    startup.main()

    assert executions == [(sys.executable, [sys.executable, *command[1:]])]
    assert startup.wrap_server_command(wrapped) == wrapped


@pytest.mark.parametrize(
    "server_args", [["--help"], ["--model-path", "org/model", "--help"]]
)
def test_help_delegates_without_model_config_access(monkeypatch, server_args):
    entrypoint = "sglang.launch_server"
    monkeypatch.setattr(
        sys, "argv", ["bootstrap", "--entrypoint", entrypoint, "--", *server_args]
    )
    executions = []
    monkeypatch.setattr(startup.os, "execv", lambda *args: executions.append(args))
    startup.main()

    assert executions == [
        (sys.executable, [sys.executable, "-m", entrypoint, *server_args])
    ]


@pytest.mark.parametrize(
    "model_path",
    ["model.gguf", "s3://bucket/model", "https://host/model", "dummy", "none"],
)
def test_non_hf_formats_delegate_without_raw_config_lookup(
    tmp_path, monkeypatch, model_path
):
    from transformers import PretrainedConfig

    if model_path.endswith(".gguf"):
        target = tmp_path / model_path
        target.write_bytes(b"GGUF\x00\x00")
        model_path = str(target)
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *args, **kwargs: pytest.fail("binary/connector config lookup"),
    )
    monkeypatch.setattr(
        startup, "installed_sglang_root", lambda: pytest.fail("SGLang was accessed")
    )

    startup.apply_model_patches(["--model-path", model_path])


@pytest.mark.parametrize(
    "error", [OSError("non-HF config"), ValueError("non-HF config")]
)
def test_unresolved_model_config_delegates_to_original_server(monkeypatch, error):
    from transformers import PretrainedConfig

    def get_config(*args, **kwargs):
        raise error

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", get_config)
    monkeypatch.setattr(
        startup, "installed_sglang_root", lambda: pytest.fail("SGLang was accessed")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bootstrap",
            "--entrypoint",
            "sglang.launch_server",
            "--",
            "--model-path",
            "org/model",
        ],
    )
    executions = []
    monkeypatch.setattr(startup.os, "execv", lambda *args: executions.append(args))

    startup.main()

    assert executions == [
        (
            sys.executable,
            [sys.executable, "-m", "sglang.launch_server", "--model-path", "org/model"],
        )
    ]


def test_launcher_wraps_selected_entrypoint_and_keeps_child_environment(monkeypatch):
    from areal.infra.launcher import sglang_server

    command = [
        "inference-python",
        "-m",
        "areal.engine.awex.plugins.sglang_plugin",
        "--model-path",
        "model",
    ]
    executions = []
    monkeypatch.setattr(
        sglang_server.subprocess,
        "Popen",
        lambda cmd, **kwargs: executions.append((cmd, kwargs)),
    )
    sglang_server.launch_server_cmd(
        command, custom_env={"AWEX_META_SERVER_ADDR": "meta"}
    )

    assert executions[0][0] == startup.wrap_server_command(command)
    assert executions[0][1]["env"]["AWEX_META_SERVER_ADDR"] == "meta"


def test_recipe_uses_automatic_model_dispatch_and_preserves_group_filter():
    import yaml

    config = yaml.safe_load(
        (
            Path(__file__).parents[1] / "examples/swe/qwen38_flash_next/swe_mm_rl.yaml"
        ).read_text()
    )
    commands = config["rollout"]["scheduling_spec"][0]["additional_bash_cmds"]
    assert not any(
        "patch_sglang" in command or "sglang_mm_executor" in command
        for command in commands
    )
    assert config["should_accept_fn"] == "examples.swe.filter_function.filter_function"
