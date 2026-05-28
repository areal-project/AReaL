#!/usr/bin/env python3
"""Unified EP (Expert Parallel) test entry point.

This script consolidates all EP tests and supports different parallel configurations:
- EP+TP: ep=world_size, tp=world_size, cp=1 (2 GPU)
- EP Only: ep=world_size, tp=1, cp=1 (2 GPU)
- EP+CP: ep=2, tp=1, cp=2 (4 GPU)
- ETP: ep=world_size, tp=world_size, etp=world_size (2 GPU) - Expert Tensor Parallel

Run with:
    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_ep_tests.py \
        --test_type=ep_tp_forward --output=/tmp/result.out

Supported test types:
    - ep_tp_forward: EP+TP forward numerical correctness
    - ep_tp_weight_sync: EP+TP weight gather and roundtrip
    - ep_only_forward: EP only forward (tp=1)
    - ep_only_weight_sync: EP only weight sync (tp=1)
    - ep_cp_forward: EP+CP forward (4 GPU)
    - ep_cp_weight_sync: EP+CP weight sync (4 GPU)
    - etp_forward: ETP forward numerical correctness (ep>1, etp=tp)
    - etp_weight_sync: ETP weight sync (ep>1, etp=tp)
    - tp_only_forward: TP-only forward for MoE experts (ep=1, tp>1)
    - state_dict_update: State dict correctness after optimizer step
"""

import argparse

import torch
import torch.distributed as dist

from tests.experimental.archon.torchrun.dist_utils import (
    create_golden_model,
    create_moe_model_args,
    create_test_input,
    gather_full_state_dict,
    print_rank0,
    verify_outputs_match,
    write_result,
)

from areal.experimental.models.archon import ArchonParallelDims
from areal.experimental.models.archon.qwen3 import (
    Qwen3Model,
    Qwen3StateDictAdapter,
    parallelize_qwen3,
)
from areal.experimental.models.archon.ulysses import (
    ulysses_gather_output,
    ulysses_slice_inputs,
)

# =============================================================================
# Parallel Configuration Helpers
# =============================================================================


def get_parallel_dims_ep_tp(world_size: int) -> ArchonParallelDims:
    """EP+TP config: ep=ws, tp=ws."""
    return ArchonParallelDims(
        dp_shard=1,
        tp=world_size,
        cp=1,
        ep=world_size,
        world_size=world_size,
        device_type="cuda",
    )


def get_parallel_dims_ep_only(world_size: int) -> ArchonParallelDims:
    """EP only config: ep=ws, tp=1. dp_shard * tp * cp must equal world_size."""
    return ArchonParallelDims(
        dp_shard=world_size,
        tp=1,
        cp=1,
        ep=world_size,
        world_size=world_size,
        device_type="cuda",
    )


def get_parallel_dims_ep_cp(world_size: int) -> ArchonParallelDims:
    """EP+CP config: ep=4, cp=2, requires 4 GPUs."""
    assert world_size == 4, "EP+CP requires 4 GPUs"
    return ArchonParallelDims(
        dp_shard=2,
        tp=1,
        cp=2,
        ep=world_size,
        world_size=world_size,
        device_type="cuda",
    )


def get_parallel_dims_etp(world_size: int) -> ArchonParallelDims:
    """ETP config: tp=2, ep=2, etp=2, requires 4+ GPUs."""
    assert world_size >= 4 and world_size % 4 == 0, (
        "ETP requires at least 4 GPUs and world_size must be divisible by 4. "
        "Use tp_only_forward for 2 GPU TP-only MoE test."
    )
    dp_shard = world_size // 4
    return ArchonParallelDims(
        dp_shard=dp_shard * 2,
        tp=2,
        cp=1,
        ep=2,
        etp=2,
        world_size=world_size,
        device_type="cuda",
    )


def get_parallel_dims_tp_only(world_size: int) -> ArchonParallelDims:
    """TP-only MoE config: ep=1, tp=ws."""
    return ArchonParallelDims(
        dp_shard=1,
        tp=world_size,
        cp=1,
        ep=1,
        etp=1,
        world_size=world_size,
        device_type="cuda",
    )


def create_ep_model(
    model_args,
    parallel_dims: ArchonParallelDims,
    device: torch.device,
    seed: int = 42,
) -> Qwen3Model:
    """Create and parallelize model with given parallel dims."""
    torch.manual_seed(seed)
    model = Qwen3Model(model_args)
    model.init_weights()
    model.init_buffers(buffer_device=device)
    model = model.to(device)

    parallelize_qwen3(
        model=model,
        parallel_dims=parallel_dims,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        loss_parallel=False,
        cpu_offload=False,
        reshard_after_forward_policy="default",
        ac_config=None,
        enable_compile=False,
    )

    return model


# =============================================================================
# Forward Tests
# =============================================================================


def test_forward(
    parallel_dims: ArchonParallelDims,
    config_name: str,
    output: str | None = None,
    ep_model: Qwen3Model | None = None,
) -> tuple[bool, Qwen3Model]:
    """Test forward correctness: EP model vs non-EP golden model. Returns (success, ep_model)."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)

    golden_model = create_golden_model(model_args, device, seed=42)

    if ep_model is None:
        ep_model = create_ep_model(model_args, parallel_dims, device, seed=42)

    tokens, positions, cu_seqlens, seq_len_per_seq = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=123,
    )

    cp_group = parallel_dims.get_group("cp")
    cp_size = parallel_dims.cp
    cp_rank = parallel_dims.get_mesh("cp").get_local_rank() if cp_size > 1 else 0

    if cp_size > 1:
        inputs_dict = {"input_ids": tokens, "position_ids": positions}
        labels = torch.zeros_like(tokens)
        sliced_inputs, _ = ulysses_slice_inputs(inputs_dict, labels, cp_rank, cp_size)
        tokens_ep = sliced_inputs["input_ids"]
        positions_ep = sliced_inputs["position_ids"]
        cu_seqlens_ep = cu_seqlens
        seq_len_per_seq_ep = seq_len_per_seq
    else:
        tokens_ep = tokens
        positions_ep = positions
        cu_seqlens_ep = cu_seqlens
        seq_len_per_seq_ep = seq_len_per_seq

    with torch.no_grad():
        output_golden = golden_model(tokens, positions, cu_seqlens, seq_len_per_seq)
        output_ep = ep_model(tokens_ep, positions_ep, cu_seqlens_ep, seq_len_per_seq_ep)

    if cp_size > 1:
        output_ep = ulysses_gather_output(output_ep, cp_group, seq_dim=1)

    success, max_diff, mean_diff = verify_outputs_match(output_golden, output_ep)

    print_rank0(f"  {config_name} forward:")
    print_rank0(f"    Output shape: {output_ep.shape}")
    print_rank0(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

    if success:
        print_rank0(f"  {config_name}_forward: PASSED")
    else:
        print_rank0(f"  {config_name}_forward: FAILED (max_diff={max_diff})")

    dist.barrier()

    if rank == 0 and output:
        write_result(output, success)

    return success, ep_model


def test_output_consistency_across_ranks(
    parallel_dims: ArchonParallelDims,
    config_name: str,
    ep_model: Qwen3Model | None = None,
) -> bool:
    """Test that all ranks produce the same gathered output."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)

    if ep_model is None:
        ep_model = create_ep_model(model_args, parallel_dims, device, seed=42)
    model = ep_model

    tokens, positions, cu_seqlens, seq_len_per_seq = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=123,
    )

    cp_group = parallel_dims.get_group("cp")
    cp_size = parallel_dims.cp
    cp_rank = parallel_dims.get_mesh("cp").get_local_rank() if cp_size > 1 else 0

    if cp_size > 1:
        inputs_dict = {"input_ids": tokens, "position_ids": positions}
        labels = torch.zeros_like(tokens)
        sliced_inputs, _ = ulysses_slice_inputs(inputs_dict, labels, cp_rank, cp_size)
        tokens_model = sliced_inputs["input_ids"]
        positions_model = sliced_inputs["position_ids"]
        cu_seqlens_model = cu_seqlens
        seq_len_per_seq_model = seq_len_per_seq
    else:
        tokens_model = tokens
        positions_model = positions
        cu_seqlens_model = cu_seqlens
        seq_len_per_seq_model = seq_len_per_seq

    dist.barrier()

    with torch.no_grad():
        output = model(
            tokens_model, positions_model, cu_seqlens_model, seq_len_per_seq_model
        )

    if cp_size > 1:
        output = ulysses_gather_output(output, cp_group, seq_dim=1)

    output_list = [torch.zeros_like(output) for _ in range(world_size)]
    dist.all_gather(output_list, output)

    success = True
    for i in range(1, world_size):
        if not torch.allclose(output_list[0], output_list[i], rtol=1e-5, atol=1e-5):
            max_diff = (output_list[0] - output_list[i]).abs().max().item()
            print_rank0(
                f"  Output mismatch between rank 0 and rank {i}: max_diff={max_diff}"
            )
            success = False

    if success:
        print_rank0(f"  {config_name}_output_consistency: PASSED")
    else:
        print_rank0(f"  {config_name}_output_consistency: FAILED")

    dist.barrier()
    return success


# =============================================================================
# Weight Sync Tests
# =============================================================================


def test_weight_sync(
    parallel_dims: ArchonParallelDims,
    config_name: str,
    output: str | None = None,
) -> bool:
    """Test weight gather, roundtrip, and cross-rank consistency."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)

    torch.manual_seed(42)
    model = Qwen3Model(model_args)
    model.init_weights()
    model.init_buffers(buffer_device=device)
    model = model.to(device)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    parallelize_qwen3(
        model=model,
        parallel_dims=parallel_dims,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        loss_parallel=False,
        cpu_offload=False,
        reshard_after_forward_policy="default",
        ac_config=None,
        enable_compile=False,
    )

    gathered_state = gather_full_state_dict(model)

    success = True

    print_rank0(f"  {config_name} weight sync:")
    for name in original_state:
        if "experts" in name and name in gathered_state:
            original = original_state[name]
            gathered = gathered_state[name]

            if original.shape != gathered.shape:
                print_rank0(
                    f"    Shape mismatch for {name}: {original.shape} vs {gathered.shape}"
                )
                success = False
                continue

            if not torch.allclose(original, gathered, rtol=1e-5, atol=1e-5):
                max_diff = (original - gathered).abs().max().item()
                print_rank0(f"    Weight mismatch for {name}: max_diff={max_diff}")
                success = False

    expert_weight_names = [
        name for name in gathered_state.keys() if "experts.w1" in name
    ]

    for name in expert_weight_names[:2]:
        local_weight = gathered_state[name]
        weight_list = [torch.zeros_like(local_weight) for _ in range(world_size)]
        dist.all_gather(weight_list, local_weight)

        for i in range(1, world_size):
            if not torch.allclose(weight_list[0], weight_list[i], rtol=1e-5, atol=1e-5):
                max_diff = (weight_list[0] - weight_list[i]).abs().max().item()
                print_rank0(
                    f"    Cross-rank mismatch for {name} (rank 0 vs {i}): {max_diff}"
                )
                success = False

    torch.manual_seed(42)
    model_new = Qwen3Model(model_args)
    model_new.init_weights()
    model_new.init_buffers(buffer_device=device)
    model_new = model_new.to(device)

    loadable_state = {}
    model_new_state = model_new.state_dict()
    for name, tensor in gathered_state.items():
        if name in model_new_state and tensor.shape == model_new_state[name].shape:
            loadable_state[name] = tensor
    model_new.load_state_dict(loadable_state, strict=False)

    tokens, positions, cu_seqlens, seq_len_per_seq = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=456,
    )

    golden_model = create_golden_model(model_args, device, seed=42)

    with torch.no_grad():
        output_new = model_new(tokens, positions, cu_seqlens, seq_len_per_seq)
        output_ref = golden_model(tokens, positions, cu_seqlens, seq_len_per_seq)

    if not torch.allclose(output_new, output_ref, rtol=1e-5, atol=1e-5):
        max_diff = (output_new - output_ref).abs().max().item()
        print_rank0(f"    Roundtrip forward mismatch: max_diff={max_diff}")
        success = False

    if success:
        print_rank0(f"  {config_name}_weight_sync: PASSED")
    else:
        print_rank0(f"  {config_name}_weight_sync: FAILED")

    dist.barrier()

    if rank == 0 and output:
        write_result(output, success)

    return success


# =============================================================================
# State Dict Update Test
# =============================================================================


def test_state_dict_update(output: str | None = None) -> bool:
    """Test state dict correctness after optimizer step."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)

    parallel_dims = get_parallel_dims_ep_tp(world_size)
    model = create_ep_model(model_args, parallel_dims, device, seed=42)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokens, positions, cu_seqlens, seq_len_per_seq = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=123,
    )

    model_output = model(tokens, positions, cu_seqlens, seq_len_per_seq)
    loss = model_output.sum()
    loss.backward()
    optimizer.step()

    gathered_state = gather_full_state_dict(model)

    torch.manual_seed(99)
    model_new = Qwen3Model(model_args)
    model_new.init_weights()
    model_new.init_buffers(buffer_device=device)
    model_new = model_new.to(device)

    loadable_state = {}
    model_new_state = model_new.state_dict()
    for name, tensor in gathered_state.items():
        if name in model_new_state and tensor.shape == model_new_state[name].shape:
            loadable_state[name] = tensor
    model_new.load_state_dict(loadable_state, strict=False)

    tokens2, positions2, cu_seqlens2, seq_len_per_seq2 = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=789,
    )

    with torch.no_grad():
        output_ep = model(tokens2, positions2, cu_seqlens2, seq_len_per_seq2)
        output_new = model_new(tokens2, positions2, cu_seqlens2, seq_len_per_seq2)

    success, max_diff, mean_diff = verify_outputs_match(
        output_ep, output_new, rtol=1e-4, atol=1e-4
    )

    print_rank0("  state_dict_update:")
    print_rank0(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

    if success:
        print_rank0("  state_dict_update: PASSED")
    else:
        print_rank0(f"  state_dict_update: FAILED (max_diff={max_diff})")

    dist.barrier()

    if rank == 0 and output:
        write_result(output, success)

    return success


# =============================================================================
# Test Runners
# =============================================================================


def run_ep_tp_forward(output: str | None = None) -> bool:
    """Run EP+TP forward test."""
    print_rank0("\n=== EP+TP Forward Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_tp(world_size)
    success, ep_model = test_forward(parallel_dims, "ep_tp", output)
    if success:
        success = test_output_consistency_across_ranks(parallel_dims, "ep_tp", ep_model)
    return success


def run_ep_tp_weight_sync(output: str | None = None) -> bool:
    """Run EP+TP weight sync test."""
    print_rank0("\n=== EP+TP Weight Sync Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_tp(world_size)
    return test_weight_sync(parallel_dims, "ep_tp", output)


def run_ep_only_forward(output: str | None = None) -> bool:
    """Run EP only forward test."""
    print_rank0("\n=== EP Only Forward Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_only(world_size)
    success, ep_model = test_forward(parallel_dims, "ep_only", output)
    if success:
        success = test_output_consistency_across_ranks(
            parallel_dims, "ep_only", ep_model
        )
    return success


def run_ep_only_weight_sync(output: str | None = None) -> bool:
    """Run EP only weight sync test."""
    print_rank0("\n=== EP Only Weight Sync Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_only(world_size)
    return test_weight_sync(parallel_dims, "ep_only", output)


def run_ep_cp_forward(output: str | None = None) -> bool:
    """Run EP+CP forward test (requires 4 GPUs)."""
    print_rank0("\n=== EP+CP Forward Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_cp(world_size)
    success, ep_model = test_forward(parallel_dims, "ep_cp", output)
    if success:
        success = test_output_consistency_across_ranks(parallel_dims, "ep_cp", ep_model)
    return success


def run_ep_cp_weight_sync(output: str | None = None) -> bool:
    """Run EP+CP weight sync test (requires 4 GPUs)."""
    print_rank0("\n=== EP+CP Weight Sync Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_cp(world_size)
    return test_weight_sync(parallel_dims, "ep_cp", output)


def run_state_dict_update(output: str | None = None) -> bool:
    """Run state dict update test."""
    print_rank0("\n=== State Dict Update Test ===")
    return test_state_dict_update(output)


def run_etp_forward(output: str | None = None) -> bool:
    """Run ETP forward test (requires 4 GPUs).

    Tests ExpertTensorParallel with 2D weight sharding [Shard(0), Shard(1/2)].
    """
    print_rank0("\n=== ETP Forward Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_etp(world_size)
    success, ep_model = test_forward(parallel_dims, "etp", output)
    if success:
        success = test_output_consistency_across_ranks(parallel_dims, "etp", ep_model)
    return success


def run_etp_weight_sync(output: str | None = None) -> bool:
    """Run ETP weight sync test (requires 4 GPUs).

    Tests weight gather/roundtrip with ExpertTensorParallel.
    """
    print_rank0("\n=== ETP Weight Sync Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_etp(world_size)
    return test_weight_sync(parallel_dims, "etp", output)


def run_tp_only_forward(output: str | None = None) -> bool:
    """Run TP-only forward test for MoE experts (ep=1, tp>1).

    Tests TensorParallel class for MoE experts when EP is disabled.
    """
    print_rank0("\n=== TP-Only MoE Forward Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_tp_only(world_size)
    success, ep_model = test_forward(parallel_dims, "tp_only", output)
    if success:
        success = test_output_consistency_across_ranks(
            parallel_dims, "tp_only", ep_model
        )
    return success


# =============================================================================
# DTensor Checkpoint Roundtrip Tests
# =============================================================================


def test_dtensor_checkpoint_roundtrip(
    parallel_dims: ArchonParallelDims,
    config_name: str,
    output: str | None = None,
) -> bool:
    """Test DTensor checkpoint roundtrip: to_hf() -> from_hf() preserves weights."""
    from torch.distributed.tensor import DTensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)

    model = create_ep_model(model_args, parallel_dims, device, seed=42)

    original_dtensor_state = {}
    for name, param in model.named_parameters():
        original_dtensor_state[name] = param

    adapter = Qwen3StateDictAdapter(model_args)

    success = True
    print_rank0(f"  {config_name} DTensor checkpoint roundtrip:")

    hf_state_dict = adapter.to_hf(original_dtensor_state)

    moe_2d_count = 0
    for key, value in hf_state_dict.items():
        if ".mlp.experts." in key:
            moe_2d_count += 1
            if value.dim() != 2:
                print_rank0(f"    ERROR: {key} should be 2D, got {value.dim()}D")
                success = False

    print_rank0(f"    to_hf produced {moe_2d_count} 2D expert weights")

    reconstructed_state = adapter.from_hf(hf_state_dict)

    for name in original_dtensor_state:
        if "moe.experts.w" in name:
            original = original_dtensor_state[name]
            if name not in reconstructed_state:
                print_rank0(f"    ERROR: {name} missing in reconstructed state")
                success = False
                continue

            reconstructed = reconstructed_state[name]

            if isinstance(original, DTensor):
                original_local = original.to_local()
            else:
                original_local = original

            if isinstance(reconstructed, DTensor):
                reconstructed_local = reconstructed.to_local()
            else:
                reconstructed_local = reconstructed

            if original_local.shape != reconstructed_local.shape:
                print_rank0(
                    f"    ERROR: Shape mismatch for {name}: "
                    f"{original_local.shape} vs {reconstructed_local.shape}"
                )
                success = False
                continue

            if not torch.allclose(
                original_local, reconstructed_local, rtol=1e-5, atol=1e-5
            ):
                max_diff = (original_local - reconstructed_local).abs().max().item()
                print_rank0(
                    f"    ERROR: Value mismatch for {name}: max_diff={max_diff}"
                )
                success = False

    expert_keys = [k for k in reconstructed_state.keys() if "moe.experts.w1" in k]
    for key in expert_keys[:1]:
        value = reconstructed_state[key]
        if isinstance(value, DTensor):
            full_tensor = value.full_tensor()
        else:
            full_tensor = value

        tensor_list = [torch.zeros_like(full_tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, full_tensor)

        for i in range(1, world_size):
            if not torch.allclose(tensor_list[0], tensor_list[i], rtol=1e-5, atol=1e-5):
                max_diff = (tensor_list[0] - tensor_list[i]).abs().max().item()
                print_rank0(
                    f"    ERROR: Cross-rank mismatch for {key} "
                    f"(rank 0 vs {i}): max_diff={max_diff}"
                )
                success = False

    if success:
        print_rank0(f"  {config_name}_dtensor_checkpoint: PASSED")
    else:
        print_rank0(f"  {config_name}_dtensor_checkpoint: FAILED")

    dist.barrier()

    if rank == 0 and output:
        write_result(output, success)

    return success


def run_ep_tp_dtensor_checkpoint(output: str | None = None) -> bool:
    """Run EP+TP DTensor checkpoint roundtrip test."""
    print_rank0("\n=== EP+TP DTensor Checkpoint Roundtrip Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_tp(world_size)
    return test_dtensor_checkpoint_roundtrip(parallel_dims, "ep_tp", output)


def run_ep_only_dtensor_checkpoint(output: str | None = None) -> bool:
    """Run EP only DTensor checkpoint roundtrip test."""
    print_rank0("\n=== EP Only DTensor Checkpoint Roundtrip Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_only(world_size)
    return test_dtensor_checkpoint_roundtrip(parallel_dims, "ep_only", output)


def run_etp_dtensor_checkpoint(output: str | None = None) -> bool:
    """Run ETP DTensor checkpoint roundtrip test (requires 4 GPUs)."""
    print_rank0("\n=== ETP DTensor Checkpoint Roundtrip Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_etp(world_size)
    return test_dtensor_checkpoint_roundtrip(parallel_dims, "etp", output)


# =============================================================================
# Main
# =============================================================================

TEST_REGISTRY = {
    "ep_tp_forward": run_ep_tp_forward,
    "ep_tp_weight_sync": run_ep_tp_weight_sync,
    "ep_only_forward": run_ep_only_forward,
    "ep_only_weight_sync": run_ep_only_weight_sync,
    "ep_cp_forward": run_ep_cp_forward,
    "ep_cp_weight_sync": run_ep_cp_weight_sync,
    "etp_forward": run_etp_forward,
    "etp_weight_sync": run_etp_weight_sync,
    "tp_only_forward": run_tp_only_forward,
    "state_dict_update": run_state_dict_update,
    # DTensor checkpoint roundtrip tests
    "ep_tp_dtensor_checkpoint": run_ep_tp_dtensor_checkpoint,
    "ep_only_dtensor_checkpoint": run_ep_only_dtensor_checkpoint,
    "etp_dtensor_checkpoint": run_etp_dtensor_checkpoint,
}


def main():
    parser = argparse.ArgumentParser(description="EP Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result (Passed/Failed)",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    torch.cuda.set_device(rank)

    print_rank0("=" * 60)
    print_rank0(f"Running EP Test: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]
        success = test_fn(args.output)

        dist.barrier()

        if success:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"EP Test {args.test_type}: PASSED")
            print_rank0("=" * 60)
        else:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"EP Test {args.test_type}: FAILED")
            print_rank0("=" * 60)
            if rank == 0 and args.output:
                write_result(args.output, False)

    except Exception as e:
        print(f"Rank {rank} failed with: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False)
        raise

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
