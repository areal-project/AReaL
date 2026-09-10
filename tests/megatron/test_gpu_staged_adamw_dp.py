# SPDX-License-Identifier: Apache-2.0

"""MCore DP-shard parity for GPU-staged AdamW."""

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
_WORKER = "tests/megatron/torchrun/run_gpu_staged_adamw_dp.py"


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_adamw_matches_mcore_dp2_shards(tmp_path_factory) -> None:
    """A parameter crossing the DP boundary matches native full AdamW."""
    if current_platform.device_count() < 2:
        pytest.skip("staged AdamW DP parity requires 2 GPUs")
    output = tmp_path_factory.mktemp("gpu_staged_adamw_dp") / "result.txt"
    port = find_free_ports(1)[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["NCCL_DEBUG"] = "WARN"
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
