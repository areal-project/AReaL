# SPDX-License-Identifier: Apache-2.0
"""End-to-end single-NPU PointerHead decision SFT on a local data slice.

Set local assets before running the NPU job:

    DECISION_SUITE=/path/to/decision-suite
    DECISION_MODEL=/path/to/Qwen3.5-4B-Base

Or pass already-sliced JSONL:

    DECISION_TRAIN=/path/to/train.jsonl
    DECISION_VALID=/path/to/development.jsonl
    DECISION_MODEL=/path/to/Qwen3.5-4B-Base
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset.decision import load_labelled_path

_REPO = Path(__file__).resolve().parents[1]
_SUCCESS = re.compile(r"Train step 2/\d+ done\.")


def _env_path(*names: str) -> Path | None:
    for name in names:
        raw = os.environ.get(name)
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path
    return None


def _resolve_suite() -> Path | None:
    return _env_path("DECISION_SUITE", "AREAL_DECISION_SUITE", "SUITE_PATH")


def _resolve_model() -> Path | None:
    found = _env_path("DECISION_MODEL", "AREAL_DECISION_MODEL", "MODEL_PATH")
    if found is not None:
        return found
    for candidate in (
        Path("/storage/openpsi/models/Qwen__Qwen3.5-4B-Base"),
        Path("/data/models/Qwen3.5-4B-Base"),
    ):
        if candidate.is_dir():
            return candidate
    return None


def _resolve_jsonl() -> tuple[Path, Path] | None:
    train = _env_path("DECISION_TRAIN", "TRAIN_PATH")
    valid = _env_path("DECISION_VALID", "VALID_PATH")
    if train is None or valid is None:
        return None
    if not train.is_file() or not valid.is_file():
        return None
    return train, valid


_SUITE = _resolve_suite()
_MODEL = _resolve_model()
_JSONL = _resolve_jsonl()


def _prepare_decision_sample():
    path = _REPO / "examples/decision/prepare_v7_sample.py"
    spec = importlib.util.spec_from_file_location("prepare_v7_sample", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.prepare_decision_sample


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def test_prepare_decision_sample_writes_one_record_per_source(tmp_path):
    if _SUITE is None or not _SUITE.is_dir():
        pytest.skip(
            "set DECISION_SUITE to a local decision suite directory to run this test"
        )
    train_path, dev_path = _prepare_decision_sample()(
        suite=_SUITE,
        out_dir=tmp_path,
        n_per_source=1,
        dev_per_source=1,
        verify_suite=False,
    )
    train = load_labelled_path(train_path, split="train")
    dev = load_labelled_path(dev_path, split="development")
    assert train
    assert dev
    train_sources = {row["_meta"]["source"] for row in train if "_meta" in row}
    if train_sources:
        assert len(train) == len(train_sources)
    assert all("questions" in row and "state" in row for row in train)


def test_npu_poc_yaml_loads_as_single_npu_decision_sft():
    config, _ = load_expr_config(
        [
            "--config",
            str(_REPO / "examples/decision/qwen3_5_decision_sft_npu_poc.yaml"),
        ],
        SFTConfig,
    )
    assert config.actor.is_decision_model
    assert config.actor.use_lora
    assert config.actor.lora_rank == 16
    assert config.actor.decision_head_dim == 256
    assert config.actor.optimizer.lr == pytest.approx(5e-5)
    assert config.cluster.n_gpus_per_node == 1
    assert config.actor.backend.startswith("megatron")
    assert config.train_dataset.type == "decision"
    assert config.total_train_epochs == 2
    assert config.total_train_steps is None
    assert config.actor.path == "Qwen/Qwen3.5-4B-Base"
    assert config.train_dataset.path == "/data/decision/train.jsonl"


@pytest.mark.npu
@pytest.mark.slow
def test_qwen35_decision_sft_single_npu_poc(tmp_path):
    from areal.infra.platforms import current_platform

    if current_platform.device_type != "npu" or current_platform.device_count() < 1:
        pytest.skip("NPU is not available")
    if _MODEL is None or not _MODEL.is_dir():
        pytest.skip("set DECISION_MODEL to a local Qwen3.5 checkpoint to run this test")
    if (
        not (_MODEL / "model.safetensors.index.json").exists()
        and not (_MODEL / "config.json").exists()
    ):
        pytest.skip(f"Qwen3.5 checkpoint is not available: {_MODEL}")

    data_dir = tmp_path / "data"
    if _JSONL is not None:
        train_path, valid_path = _JSONL
    elif _SUITE is not None and _SUITE.is_dir():
        train_path, valid_path = _prepare_decision_sample()(
            suite=_SUITE,
            out_dir=data_dir,
            n_per_source=1,
            dev_per_source=1,
            verify_suite=False,
        )
    else:
        pytest.skip(
            "set DECISION_SUITE or DECISION_TRAIN + DECISION_VALID to run this test"
        )
    fileroot = tmp_path / "experiments"
    name_resolve = tmp_path / "name_resolve"
    log_path = tmp_path / "poc.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["ASCEND_RT_VISIBLE_DEVICES"] = env.get("ASCEND_RT_VISIBLE_DEVICES", "0")
    env["TORCH_COMPILE_DISABLE"] = "1"
    extra_noproxy = "127.0.0.1,localhost,::1"
    try:
        from areal.utils.network import gethostip

        extra_noproxy = f"{extra_noproxy},{gethostip()}"
    except Exception:
        pass
    for key in ("NO_PROXY", "no_proxy"):
        current = env.get(key, "")
        env[key] = f"{current},{extra_noproxy}" if current else extra_noproxy

    cmd = [
        sys.executable,
        str(_REPO / "examples/decision/train_decision.py"),
        "--config",
        str(_REPO / "examples/decision/qwen3_5_decision_sft_npu_poc.yaml"),
        "experiment_name=qwen35-decision-sft-e2e",
        "trial_name=poc",
        "scheduler.type=local",
        "cluster.n_nodes=1",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={fileroot}",
        f"cluster.name_resolve.nfs_record_root={name_resolve}",
        f"actor.path={_MODEL}",
        f"tokenizer_path={_MODEL}",
        "total_train_steps=2",
        "saver.freq_steps=2",
        "evaluator.freq_steps=2",
        "train_dataset.batch_size=2",
        "valid_dataset.batch_size=2",
        f"train_dataset.path={train_path}",
        f"valid_dataset.path={valid_path}",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=_REPO,
            env=env,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
    captured = _read_text(log_path)
    log_blobs = [captured]
    if fileroot.exists():
        for extra in fileroot.rglob("*.log"):
            log_blobs.append(_read_text(extra))
    output = "\n".join(log_blobs)
    if result.returncode != 0:
        pytest.fail(f"decision NPU PoC exited {result.returncode}\n{output[-8000:]}")
    assert _SUCCESS.search(output), output[-4000:]
    assert "decision_loss" in output
    checkpoints = list((fileroot / "checkpoints").rglob("head.pt"))
    assert checkpoints, f"expected head.pt under {fileroot}, got none"
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    head = payload.get("head", payload)
    assert "q.weight" in head


_MULTI_NPU_CASES = (
    ("dp2", "megatron:d2", 2),
    ("tp2", "megatron:d1t2", 2),
    ("cp2", "megatron:d1c2", 2),
)


@pytest.mark.multi_npu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("parallel", "backend", "n_devices"),
    _MULTI_NPU_CASES,
    ids=[case[0] for case in _MULTI_NPU_CASES],
)
def test_qwen35_decision_sft_multi_npu(parallel, backend, n_devices, tmp_path):
    """Two-step decision SFT for DP, TP+SP, and CP, same checks as 1 NPU."""
    from areal.infra.platforms import current_platform

    if (
        current_platform.device_type != "npu"
        or current_platform.device_count() < n_devices
    ):
        pytest.skip(f"need {n_devices} NPUs for {parallel}")
    if _MODEL is None or not _MODEL.is_dir():
        pytest.skip("set DECISION_MODEL to a local Qwen3.5 checkpoint to run this test")
    if not (_MODEL / "config.json").exists():
        pytest.skip(f"Qwen3.5 checkpoint is not available: {_MODEL}")

    fileroot = tmp_path / "experiments"
    log_path = tmp_path / "poc.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["MODEL_PATH"] = str(_MODEL)
    env["PARALLEL"] = parallel
    env["DEVICES"] = ",".join(str(i) for i in range(n_devices))
    env["FILEROOT"] = str(fileroot)
    env["NAME_RESOLVE"] = str(tmp_path / "name_resolve")
    env["SLICE_DIR"] = str(tmp_path / "data")
    env["EPOCHS"] = "1"
    env["STEPS"] = "2"
    env["BATCH"] = "2"
    env["N_PER_SOURCE"] = "1"
    env["DEV_PER_SOURCE"] = "1"
    env["EXPERIMENT_NAME"] = f"qwen35-decision-npu-{parallel}"
    env["TRIAL_NAME"] = parallel
    if _JSONL is not None:
        train_path, valid_path = _JSONL
        env["TRAIN_PATH"] = str(train_path)
        env["VALID_PATH"] = str(valid_path)
    elif _SUITE is not None and _SUITE.is_dir():
        env["SUITE_PATH"] = str(_SUITE)
    else:
        pytest.skip(
            "set DECISION_SUITE or DECISION_TRAIN + DECISION_VALID to run this test"
        )

    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            ["bash", str(_REPO / "examples/decision/run_npu_parallel.sh")],
            cwd=_REPO,
            env=env,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
    output = _read_text(log_path)
    if fileroot.exists():
        for extra in fileroot.rglob("*.log"):
            output += "\n" + _read_text(extra)
    if result.returncode != 0:
        pytest.fail(f"{parallel} exited {result.returncode}\n{output[-8000:]}")
    assert f"backend={backend}" in output
    if parallel == "tp2":
        assert "Sequence parallel is enabled" in output
    assert _SUCCESS.search(output), output[-4000:]
    assert "decision_loss" in output
    checkpoints = list((fileroot / "checkpoints").rglob("head.pt"))
    assert checkpoints, f"expected head.pt under {fileroot}, got none"
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    head = payload.get("head", payload)
    assert "q.weight" in head
