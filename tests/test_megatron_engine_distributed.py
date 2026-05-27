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


# ──────────────────────────────────────────────────────────────────────
# Qwen3.5 dense tests. Routed through bridge_type=megatron-bridge because
# its GDN hybrid attention is only handled by the megatron-bridge model
# definitions (mbridge would fall back to the qwen3 substring match and
# emit wrong shapes). The runner sets bridge_type automatically based on
# ``_MODEL_BRIDGE_OVERRIDES``.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_qwen3_5_single_gpu_forward(tmp_path_factory):
    """Smoke test on a single GPU: engine init + forward pass.

    Validates the megatron-bridge load path (including the AReaL-side
    ``with torch.device("cpu"):`` fix for GDN ChunkedMapping) and basic
    forward execution before exercising any parallelism.
    """
    if current_platform.device_count() < 1:
        pytest.skip("requires 1 GPU to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_5_single_gpu.out"
    _run_test_with_torchrun(
        "qwen3_5", "megatron:d1p1t1", test_type="forward", output=str(output)
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3_5_tensor_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("tensor parallel requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_5_tensor_parallel.out"
    _run_test_with_torchrun(
        "qwen3_5", "megatron:d1p1t2", test_type="forward", output=str(output)
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3_5_pipeline_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("pipeline parallel requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_5_pipeline_parallel.out"
    _run_test_with_torchrun(
        "qwen3_5", "megatron:d1p2t1", test_type="forward", output=str(output)
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skip(
    reason="megatron-bridge _broadcast_shared_embeddings does not support "
    "VPP + tied embeddings (TODO in model_bridge.py:1271). Not needed for "
    "initial Qwen3.5 support; VPP is an optional scheduling optimization."
)
def test_qwen3_5_virtual_pipeline_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("virtual pipeline parallel requires 2 GPUs to run")
    output = (
        tmp_path_factory.mktemp("test_output") / "qwen3_5_virtual_pipeline_parallel.out"
    )
    _run_test_with_torchrun(
        "qwen3_5",
        "megatron:d1p2t1",
        test_type="forward",
        output=str(output),
        vpp_size=2,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skip(
    reason="BSHD mode (use_padded_seq) lacks microbatch invariance: padding "
    "changes per MB boundary cause small grad_norm drift. verl sidesteps "
    "this by setting ppo_micro_batch_size_per_gpu=1 (1 seq/MB, no padding "
    "diff). See run_qwen3_5_35b_megatron.sh for the recommended config."
)
def test_qwen3_5_grad_norm_mb_invariance(tmp_path_factory):
    """Same regression guard as ``test_qwen3_grad_norm_mb_invariance`` but on
    Qwen3.5. Exercises full backward + optimizer step under DP=2 to verify the
    ``loss_multiplier`` fix still holds for GDN models.
    """
    if current_platform.device_count() < 2:
        pytest.skip("grad_norm_mb_invariance requires 2 GPUs to run")
    output = (
        tmp_path_factory.mktemp("test_output") / "qwen3_5_grad_norm_mb_invariance.out"
    )
    _run_test_with_torchrun(
        "qwen3_5",
        "megatron:d2p1t1",
        test_type="grad_norm_mb_invariance",
        output=str(output),
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_qwen3_5_hf_save_load(tmp_path_factory):
    """HF save/load round-trip under TP=2.

    Uses _save_model_to_hf / _load_model_from_hf (HF safetensors) instead of
    mcore DCP because mcore's dist_checkpointing does not support SSM/GDN
    ``flattened_range`` tensors yet. Validates train → save → zero → load →
    retrain produces identical weights.
    """
    if current_platform.device_count() < 2:
        pytest.skip("Qwen3.5 HF save load requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_5_hf_save_load.out"
    _run_test_with_torchrun(
        "qwen3_5",
        "megatron:d1p1t2",
        test_type="train_hf_save_load",
        output=str(output),
    )
