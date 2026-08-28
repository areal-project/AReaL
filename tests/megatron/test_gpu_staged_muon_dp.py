# SPDX-License-Identifier: Apache-2.0

"""Layer-wise DP-owner parity for GPU-staged Muon."""

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
_WORKER = "tests/megatron/torchrun/run_gpu_staged_muon_dp.py"


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("world_size", "tp_size"),
    [(2, 1), (4, 2)],
    ids=["dp2-tp1", "dp2-tp2"],
)
def test_gpu_staged_muon_matches_layerwise_dp2_owners(
    tmp_path_factory, world_size: int, tp_size: int
) -> None:
    """Dense, expert, scalar, and empty-owner DP paths match native MCore."""
    if current_platform.device_count() < world_size:
        pytest.skip(f"staged Muon layer-wise parity requires {world_size} GPUs")
    output = (
        tmp_path_factory.mktemp(f"gpu_staged_muon_dp_{world_size}_{tp_size}")
        / "result.txt"
    )
    port = find_free_ports(1)[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["NCCL_DEBUG"] = "WARN"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--nnodes=1",
            "--master-addr=localhost",
            f"--master-port={port}",
            _WORKER,
            f"--output={output}",
            f"--tp-size={tp_size}",
        ],
        check=True,
        cwd=_PROJECT_ROOT,
        env=env,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    assert output.read_text().strip() == "Passed"
