# SPDX-License-Identifier: Apache-2.0

import pathlib
import subprocess
import sys

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = [pytest.mark.npu, pytest.mark.multi_npu]

_TORCHRUN_SCRIPT = (
    pathlib.Path(__file__).parent / "torchrun" / "run_megatron_bridge_vision_cp.py"
).resolve()


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.ci
def test_vision_cp_allgather_backward_hccl() -> None:
    """Verify vision CP gradients and collective participation on two NPUs."""
    if current_platform.device_type != "npu":
        pytest.skip("HCCL test requires NPU")
    if current_platform.device_count() < 2:
        pytest.skip("Vision CP test requires at least two NPUs")

    port = find_free_ports(1)[0]
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        str(_TORCHRUN_SCRIPT),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stdout,
            text=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"Vision CP torchrun failed with exit code {exc.returncode}")
    except subprocess.TimeoutExpired:
        pytest.fail("Vision CP torchrun timed out after 300 seconds")
