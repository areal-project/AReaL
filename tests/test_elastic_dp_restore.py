# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areal.api.alloc_mode import ModelAllocation
from areal.utils.atomic import atomic_json, recovery_digest
from areal.v2.training_service import checkpoint
from areal.v2.training_service.dp_restore import (
    DPStateRestorePlan,
    mapped_runtime_state,
)


def test_dp_preflight_maps_distinct_reproducible_streams_without_mutation(
    tmp_path, monkeypatch
):
    """All source sidecars and live RNG/shuffler state survive target preflight."""
    from tests.test_elastic_runtime_state import make_engine

    from areal.utils import seeding

    monkeypatch.setattr(seeding, "_SEED", 11)
    monkeypatch.setattr(seeding, "_BASE_SEED", 11)
    monkeypatch.setattr(seeding, "_SHUFFLER", seeding.Shuffler("fixture"))
    seeding._SHUFFLER.cnt = 7
    root = tmp_path / "step-00000001-fixture"
    (root / "dcp").mkdir(parents=True)
    (root / "dcp/.metadata").write_text("sidecar-only fixture")
    compatibility = {"actor": {"backend": "fsdp:d2"}}
    identity = recovery_digest(compatibility)
    atomic_json(
        root / "manifest.json",
        {
            "format_version": 1,
            "global_step": 1,
            "world_size": 2,
            "compatibility": compatibility,
        },
    )
    for rank in range(2):
        engine = make_engine("cpu")
        engine.rank, engine.world_size = rank, 2
        atomic_json(
            root / f"runtime-{rank}.json", checkpoint._capture(engine, "fsdp", identity)
        )
    source = {path: path.read_bytes() for path in root.glob("runtime-*.json")}
    target = make_engine("cpu")
    target.parallel_helper = SimpleNamespace(
        _ps=ModelAllocation.from_str("fsdp:d4").parallel
    )
    plan = asdict(DPStateRestorePlan(identity, root.name, 1, 2, 4, "a" * 64))
    before = recovery_digest(checkpoint._capture(target, "fsdp", identity))
    streams = []
    for rank in range(4):
        target.rank = rank
        kwargs = dict(path=root, identity_sha256=identity, version=1, plan=plan)
        state = mapped_runtime_state(target, **kwargs)
        assert recovery_digest(
            mapped_runtime_state(target, **kwargs)
        ) == recovery_digest(state)
        assert state["lr_scheduler"] == engine.lr_scheduler.state_dict()
        assert state["optimizer_group_lrs"] == [engine.optimizer.param_groups[0]["lr"]]
        assert state["policy_version"] == 1 and state["world_size"] == 4
        streams.append(recovery_digest(state["torch_rng"]))
    target.rank = 0
    assert recovery_digest(checkpoint._capture(target, "fsdp", identity)) == before
    assert len(set(streams)) == 4
    assert seeding._SHUFFLER.cnt == 7
    for path, content in source.items():
        assert path.read_bytes() == content


@pytest.mark.slow
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 4,
                reason="FSDP DP acceptance needs four visible CUDA devices",
            ),
        ),
    ],
)
def test_native_dcp_two_four_two_restores_parameters_and_adam(tmp_path, device):
    """Separate real process groups reshard DCP and continue optimizer updates."""
    project = Path(__file__).resolve().parents[1]
    worker = project / "tests/torchrun/run_elastic_dp_restore.py"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project), env.get("PYTHONPATH")))
    )
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    source_bytes = {}
    for stage, world_size in enumerate((2, 4, 2), start=1):
        # Avoid standalone rendezvous resolving the macOS IPv6 reverse-DNS
        # hostname; every rank in this single-node test uses explicit loopback.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            master_port = sock.getsockname()[1]
        with subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--node-rank=0",
                "--master-addr=127.0.0.1",
                f"--master-port={master_port}",
                f"--nproc-per-node={world_size}",
                str(worker),
                "--root",
                str(tmp_path),
                "--stage",
                str(stage),
                "--device",
                device,
            ],
            cwd=project,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                output, _ = process.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
                pytest.fail(f"DCP stage {stage} timed out:\n{output}")
            (tmp_path / f"stage-{stage}.log").write_text(output)
            assert process.returncode == 0, output
        status = json.loads((tmp_path / f"stage-{stage}.json").read_text())
        assert status["world_size"] == world_size and status["version"] == stage
        for path, content in source_bytes.items():
            assert path.read_bytes() == content
        source_bytes.update(
            {
                path: path.read_bytes()
                for path in (tmp_path / f"step-{stage:08d}-fixture").glob(
                    "runtime-*.json"
                )
            }
        )
