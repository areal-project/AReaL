# SPDX-License-Identifier: Apache-2.0

"""Numerical parity between staged and official Muon under real TP collectives."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WORKER = "tests/megatron/torchrun/run_gpu_staged_muon_tp.py"


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_muon_matches_official_tp2(tmp_path_factory) -> None:
    """Both matrix partition axes match official TensorParallelMuon at TP=2."""
    if current_platform.device_count() < 2:
        pytest.skip("staged Muon TP parity requires 2 GPUs")
    output = tmp_path_factory.mktemp("gpu_staged_muon_tp") / "result.txt"
    port = find_free_ports(1)[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "--master-addr=localhost",
            f"--master-port={port}",
            _WORKER,
            f"--output={output}",
        ],
        check=True,
        cwd=_PROJECT_ROOT,
        env=env,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    assert output.read_text().strip() == "Passed"
