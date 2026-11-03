# SPDX-License-Identifier: Apache-2.0

import pathlib
import subprocess
import sys

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = [pytest.mark.npu, pytest.mark.multi_npu]


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.parametrize("cp_size", [1, 2])
def test_native_mtp_cp_matches_reference(cp_size):
    if (
        current_platform.device_type != "npu"
        or current_platform.device_count() < cp_size
    ):
        pytest.skip(f"Native MTP test requires {cp_size} NPUs")
    script = pathlib.Path(__file__).parent / "torchrun" / "run_mtp_loss.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={cp_size}",
            "--nnodes=1",
            "--master-addr=127.0.0.1",
            f"--master-port={find_free_ports(1)[0]}",
            str(script),
        ],
        check=True,
        timeout=300,
    )
