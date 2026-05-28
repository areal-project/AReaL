import subprocess
import sys

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(
    model_type: str, alloc_mode: str, test_type: str, output: str, vpp_size: int = 1
):
    port = find_free_ports(1)[0]
    n_gpus = ModelAllocation.from_str(alloc_mode).parallel.world_size
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_megatron_engine_distributed.py",
                f"--model_type={model_type}",
                f"--backend={alloc_mode}",
                f"--output={output}",
                f"--test_type={test_type}",
                f"--vpp_size={vpp_size}",
            ],
            check=True,
            stdout=sys.stdout,
            stderr=sys.stdout,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")
    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3_tensor_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("tensor parallel requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_tensor_parallel.out"
    _run_test_with_torchrun(
        "qwen3", "megatron:d1p1t2", test_type="forward", output=str(output)
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3_grad_norm_mb_invariance(tmp_path_factory):
    """Regression guard: grad_norm must be invariant to num_microbatches.

    Guards the `loss_multiplier` fix in `MegatronEngine.train_batch` that
    compensates for Megatron Core's `loss /= num_microbatches` applied on the
    2-tuple `loss_func` return path in
    `megatron.core.pipeline_parallel.schedules._forward_step_helper`. Without
    the fix the reported grad_norm scales as `1 / num_microbatches`.
    """
    output = (
        tmp_path_factory.mktemp("test_output") / "qwen3_grad_norm_mb_invariance.out"
    )
    _run_test_with_torchrun(
        "qwen3",
        "megatron:d2p1t1",
        test_type="grad_norm_mb_invariance",
        output=str(output),
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3_pipeline_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("pipeline parallel requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_pipeline_parallel.out"
    _run_test_with_torchrun(
        "qwen3", "megatron:d1p2t1", test_type="forward", output=str(output)
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3_context_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("context parallel requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_context_parallel.out"
    _run_test_with_torchrun(
        "qwen3", "megatron:d1p1t1c2", test_type="forward", output=str(output)
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3_virtual_pipeline_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("virtual pipeline parallel requires 2 GPUs to run")
    output = (
        tmp_path_factory.mktemp("test_output") / "qwen3_virtual_pipeline_parallel.out"
    )
    _run_test_with_torchrun(
        "qwen3", "megatron:d1p2t1", test_type="forward", output=str(output), vpp_size=2
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3moe_expert_parallel(tmp_path_factory):
    if current_platform.device_count() < 4:
        pytest.skip("Qwen3 MoE expert parallel requires 4 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3moe_expert_parallel.out"
    _run_test_with_torchrun(
        "qwen3moe",
        "megatron:(attn:d1p1t2c2|ffn:d1p1t1e4)",
        test_type="forward",
        output=str(output),
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3_dcp_save_load(tmp_path_factory):
    if current_platform.device_count() < 8:
        pytest.skip("DCP save load requires 8 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_save_load.out"
    _run_test_with_torchrun(
        "qwen3",
        "megatron:d2p2t2",
        test_type="train_dcp_save_load",
        output=str(output),
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3moe_dcp_save_load(tmp_path_factory):
    if current_platform.device_count() < 8:
        pytest.skip("Qwen3 MoE DCP save load requires 8 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3moe_save_load.out"
    _run_test_with_torchrun(
        "qwen3moe",
        "megatron:(attn:d1p1t4c2|ffn:d1p1t2e4)",
        test_type="simple_dcp_save_load",
        output=str(output),
    )
