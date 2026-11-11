# SPDX-License-Identifier: Apache-2.0
"""Real MindSpeed GDN CP checks using the AReaL-owned AscendC forward."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from areal.utils.network import find_free_ports

pytest.importorskip("torch_npu")
pytest.importorskip("fla_npu")

pytestmark = [pytest.mark.npu, pytest.mark.multi_npu, pytest.mark.slow, pytest.mark.ci]


@pytest.mark.parametrize("cp", [2, 4])
@pytest.mark.parametrize("chunk_size", [64, 128])
def test_native_gdn_cp_matches_cp1(tmp_path, cp, chunk_size):
    """Compare outputs, input gradients and summed parameter gradients."""
    if not torch.npu.is_available() or torch.npu.device_count() < cp:
        pytest.skip(f"Requires {cp} NPUs")
    from mindspeed.core.ssm import npu_chunk_gated_delta_rule as adapter

    if getattr(adapter, "_NATIVE_KKT", False):
        pytest.skip("MindSpeed already provides the generalized native integration")
    runner = Path(__file__).parent / "torchrun/run_mindspeed_gdn_native_cp.py"
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={cp}",
        "--nnodes=1",
        "--master-addr=127.0.0.1",
        f"--master-port={find_free_ports(1)[0]}",
        str(runner),
        "--output",
        str(output),
        f"--chunk-size={chunk_size}",
    ]
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            log, _ = process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                log, _ = process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                log, _ = process.communicate()
            pytest.fail(f"CP runner timed out:\n{log[-20000:]}")
    (tmp_path / "torchrun.log").write_text(log)
    assert process.returncode == 0, log[-20000:]
    for rank in range(cp):
        records = json.loads(output.with_name(f"result-rank{rank}.json").read_text())
        assert len(records) == 4
        assert all(
            record["cp"] == cp and record["chunk_size"] == chunk_size
            for record in records
        )
        assert all(record["packed"] for record in records)
        assert all(
            check["finite"] and check["elementwise_close"]
            for record in records
            for check in record["checks"].values()
        )
