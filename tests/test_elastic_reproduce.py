# SPDX-License-Identifier: Apache-2.0

"""Public commands must preserve identical workloads and reject changed inputs."""

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from examples.elastic.reproduce import build_command, check_inputs, run_command


def test_reproduction_policies_share_workload():
    lock = json.loads(
        (
            Path(__file__).parents[1] / "examples/elastic/reproduction/inputs.lock.json"
        ).read_text()
    )
    args = argparse.Namespace(
        model=Path("model with spaces"),
        dataset=Path("data.jsonl"),
        swe_runtime=Path("runtime.json"),
        output=Path("new run"),
        policy="resident",
        prepare_only=True,
    )
    resident = build_command(args, lock)
    args.policy = "agent"
    agent = build_command(args, lock)
    differences = [(a, b) for a, b in zip(resident, agent, strict=True) if a != b]
    assert differences == [("resident", "agent")]
    assert resident[resident.index("--steps") + 1] == "8"
    assert resident[resident.index("--concurrency") + 1] == "2"
    assert resident[resident.index("--model") + 1] == "model with spaces"
    assert resident[-1] == "--prepare-only"


@pytest.mark.parametrize(
    "changed", ["tasks.jsonl", "tasks.images.json", "config.json", "model.safetensors"]
)
def test_reproduction_rejects_changed_content(tmp_path, changed):
    model = tmp_path / "model"
    model.mkdir()
    files = {
        tmp_path / "tasks.jsonl": b"task order",
        tmp_path / "tasks.images.json": b"pinned images",
        model / "config.json": b"model metadata",
        model / "model.safetensors": b"model weights",
    }
    hashes = {}
    for path, content in files.items():
        path.write_bytes(content)
        hashes[path.name] = hashlib.sha256(content).hexdigest()
    lock = {
        "dataset": {
            "tasks_sha256": hashes["tasks.jsonl"],
            "images_sha256": hashes["tasks.images.json"],
        },
        "model": {
            "sha256": {
                name: hashes[name] for name in ("config.json", "model.safetensors")
            }
        },
    }
    dataset = tmp_path / "tasks.jsonl"
    check_inputs(model, dataset, lock, full_model=True)
    next(path for path in files if path.name == changed).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input differs"):
        check_inputs(model, dataset, lock, full_model=True)


def test_reproduction_requires_all_model_shards(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_bytes(b"")
    dataset.with_suffix(".images.json").write_bytes(b"")
    empty_hash = hashlib.sha256(b"").hexdigest()
    lock = {
        "dataset": {"tasks_sha256": empty_hash, "images_sha256": empty_hash},
        "model": {"sha256": {"missing.safetensors": empty_hash}},
    }
    with pytest.raises(ValueError, match="Incomplete model"):
        check_inputs(tmp_path, dataset, lock, full_model=False)


def test_reproduction_stop_waits_for_child_cleanup(tmp_path):
    ready, cleaned = tmp_path / "ready", tmp_path / "cleaned"
    script = """
import signal, sys, time
from pathlib import Path
def stop(signum, frame):
    time.sleep(0.1)
    Path(sys.argv[2]).write_text('cleanup finished')
    raise SystemExit(7)
signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).touch()
time.sleep(5)
"""

    def send_stop():
        deadline = time.monotonic() + 4
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if ready.exists():
            os.kill(os.getpid(), signal.SIGTERM)

    original_handler = signal.getsignal(signal.SIGTERM)
    sender = threading.Thread(target=send_stop)
    sender.start()
    try:
        result = run_command([sys.executable, "-c", script, str(ready), str(cleaned)])
    finally:
        sender.join()
    assert result == 7
    assert cleaned.read_text() == "cleanup finished"
    assert signal.getsignal(signal.SIGTERM) == original_handler
