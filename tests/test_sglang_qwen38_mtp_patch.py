# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from areal.engine import _sglang_source_patch as installer
from areal.engine import sglang_model_patches as startup
from areal.engine import sglang_qwen38_mtp as patch

SHAPE = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "vocab_size": 248320,
}


def model_config():
    return {"model_type": "qwen3_5", "text_config": dict(SHAPE)}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Small compilable call sites; the real source guard and installer run.
    # Production file fingerprints are separately checked against SGLang sources.
    sources = [
        "def positions():\n    if True:\n        if True:\n" + patch.MROPE_OLD,
        "class BaseFinishReason:\n    pass\n\n" + patch.ABORT_INIT_OLD,
        "class Req:\n    def update_finish_state(self, new_accepted_len=1):\n"
        + patch.FINISH_OLD
        + "        if self.natural_finish:\n"
        + "            self.finished_reason = 'stop'\n"
        + "            return\n\n"
        + patch.FINISH_END_OLD
        + "        pass\n",
        "def abort_request():\n    if True:\n        if True:\n            if True:\n                if True:\n"
        + patch.SCHEDULER_OLD,
    ]
    root = tmp_path / "sglang"
    definitions = []
    before = {}
    after = {}
    for (relative, _, _, replacements), text in zip(
        patch.PATCHES, sources, strict=True
    ):
        source = text.encode()
        patched = source
        for old, new in replacements:
            patched = patched.replace(old.encode(), new.encode())
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
        before[target] = source
        after[target] = patched
        definitions.append(
            (
                relative,
                hashlib.sha256(source).hexdigest(),
                hashlib.sha256(patched).hexdigest(),
                replacements,
            )
        )
    monkeypatch.setattr(patch, "PATCHES", tuple(definitions))
    monkeypatch.setattr(startup, "installed_sglang_root", lambda: root)
    monkeypatch.delenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", raising=False)
    model = tmp_path / "checkpoint-epoch-7-step-32"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(model_config()))
    return root, model, before, after


@pytest.mark.parametrize("mtp_on", [True, False])
def test_matching_checkpoint_automatically_patches_with_or_without_mtp(runtime, mtp_on):
    _, model, _, after = runtime
    args = ["--model-path", str(model)]
    if mtp_on:
        args += ["--speculative-algorithm", "EAGLE"]
    startup.apply_model_patches(args)
    assert all(path.read_bytes() == content for path, content in after.items())


def test_repeated_start_needs_no_writes_or_lock(runtime, monkeypatch):
    root, model, _, after = runtime
    startup.apply_model_patches(["--model", str(model)])
    (root / ".areal-sglang-patches.lock").unlink()
    mtimes = {path: path.stat().st_mtime_ns for path in after}
    monkeypatch.setattr(
        installer, "_atomic_write", lambda *args: pytest.fail("rewritten")
    )
    startup.apply_model_patches([f"--model-path={model}"])
    assert not (root / ".areal-sglang-patches.lock").exists()
    assert all(path.stat().st_mtime_ns == value for path, value in mtimes.items())


def test_unknown_runtime_rejects_entire_patch_set_before_writing(runtime):
    _, model, before, _ = runtime
    last = list(before)[-1]
    last.write_bytes(before[last] + b"\n# unverified fork\n")
    before[last] = last.read_bytes()
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        startup.apply_model_patches(["--model", str(model)])
    assert all(path.read_bytes() == content for path, content in before.items())


def test_concurrent_matching_starts_install_one_consistent_patch_set(runtime):
    _, model, _, after = runtime
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda _: startup.apply_model_patches(["--model", str(model)]), range(4)
            )
        )
    assert all(path.read_bytes() == content for path, content in after.items())


@pytest.mark.parametrize("field", list(SHAPE))
def test_other_model_sizes_are_not_patched(runtime, monkeypatch, field):
    _, model, before, _ = runtime
    config = model_config()
    config["text_config"][field] += 1
    (model / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        startup,
        "installed_sglang_root",
        lambda: pytest.fail("other model accessed SGLang"),
    )
    startup.apply_model_patches(["--model-path", str(model)])
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model_type": "qwen3_5", "text_config": None},
        {"model_type": "qwen4_exp", "text_config": SHAPE},
        {"model_type": "llama", "text_config": SHAPE},
    ],
)
def test_incomplete_or_other_architectures_do_not_match(config):
    assert not patch.matches_model(config)


def test_external_abort_defers_to_natural_finish_but_internal_errors_win(runtime):
    _, _, _, after = runtime
    namespace = {}
    exec(
        next(value for path, value in after.items() if path.name == "req_types.py"),
        namespace,
    )
    exec(
        next(
            value for path, value in after.items() if path.name == "schedule_batch.py"
        ),
        namespace,
    )
    abort_type, request_type = namespace["FINISH_ABORT"], namespace["Req"]
    for external, natural, new_tokens, expected in [
        (True, True, 1, "stop"),
        (True, False, 1, "abort"),
        (False, True, 1, "abort"),
        (True, True, 0, "abort"),
    ]:
        req = request_type()
        abort = abort_type(check_completion=external)
        req.to_finish = abort
        req.finished_reason = None
        req.output_ids = [2]
        req.natural_finish = natural
        req.finished = lambda: req.finished_reason is not None
        req.update_finish_state(new_accepted_len=new_tokens)
        assert req.finished_reason == ("stop" if expected == "stop" else abort)


def test_partially_prepatched_runtime_installs_remaining_files(runtime):
    _, model, _, after = runtime
    first = next(iter(after))
    first.write_bytes(after[first])
    startup.apply_model_patches(["--model", str(model)])
    assert all(path.read_bytes() == content for path, content in after.items())


def test_read_only_unpatched_runtime_reports_actionable_error(runtime, monkeypatch):
    _, model, before, _ = runtime

    def deny_write(*args):
        raise PermissionError("read-only runtime")

    monkeypatch.setattr(installer, "_atomic_write", deny_write)
    with pytest.raises(
        RuntimeError, match="writable installation or prepatched runtime"
    ):
        startup.apply_model_patches(["--model", str(model)])
    assert all(path.read_bytes() == content for path, content in before.items())
