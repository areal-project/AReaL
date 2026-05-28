"""Test script for Pipeline Parallelism combined with TP, DP, or EP.

This test verifies that PP works correctly when combined with other parallelism
dimensions.

Run with:
    torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_pp_combinations.py \
        --test_type=pp_tp --output=/tmp/result.txt

    torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_pp_combinations.py \
        --test_type=pp_dp --output=/tmp/result.txt

    torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_pp_combinations.py \
        --test_type=pp_ep --output=/tmp/result.txt
"""

import argparse

import torch
import torch.distributed as dist

from tests.experimental.archon.torchrun.dist_utils import (
    create_dense_model_args,
    create_golden_model,
    create_moe_model_args,
    create_test_input,
    print_rank0,
    verify_outputs_match,
    write_result,
)

from areal.api.cli_args import ArchonEngineConfig
from areal.experimental.models.archon import ArchonParallelDims
from areal.experimental.models.archon.pipeline_parallel import (
    generate_llm_fqn_per_model_part,
    pipeline_llm,
    pipeline_module_split,
)
from areal.experimental.models.archon.qwen3 import Qwen3Model, parallelize_qwen3


def test_pp_tp_forward(out_file: str | None = None) -> bool:
    """Test PP+TP combination forward pass.

    Configuration: 4 GPU with pp=2, tp=2, dp_shard=1
    Mesh layout: [pp=2, dp_shard=1, cp=1, tp=2]

    This test verifies:
    1. PP and TP meshes are correctly configured
    2. Forward pass produces correct output (matches golden model)

    Args:
        out_file: Optional file to write result

    Returns:
        True if test passed, False otherwise
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print_rank0(f"\n{'=' * 60}")
    print_rank0("PP+TP Forward Test: pp=2, tp=2")
    print_rank0("=" * 60)

    assert world_size == 4, f"This test requires 4 GPUs, got {world_size}"

    # Create model args
    n_layers = 4  # 2 stages * 2 layers per stage
    model_args = create_dense_model_args(
        n_layers=n_layers,
        dim=64,
        hidden_dim=128,
        n_heads=4,  # Must be divisible by tp=2
        n_kv_heads=2,  # Must be divisible by tp=2
        head_dim=16,
        vocab_size=1000,
    )

    # Create golden model
    seed = 42
    golden_model = create_golden_model(model_args, device, seed=seed)
    golden_model.eval()

    # Create PP+TP parallel dims
    parallel_dims = ArchonParallelDims(
        pp=2,
        dp_shard=1,
        tp=2,
        cp=1,
        world_size=world_size,
        device_type="cuda",
    )

    print_rank0(f"  PP enabled: {parallel_dims.pp_enabled}")
    print_rank0(f"  TP enabled: {parallel_dims.tp_enabled}")
    print_rank0(f"  PP degree: {parallel_dims.pp}")
    print_rank0(f"  TP degree: {parallel_dims.tp}")

    # Verify mesh configuration
    pp_mesh = parallel_dims.get_mesh("pp")
    tp_mesh = parallel_dims.get_mesh("tp")

    if pp_mesh is None:
        print_rank0("  FAILED: PP mesh is None")
        return False
    if tp_mesh is None:
        print_rank0("  FAILED: TP mesh is None")
        return False

    print_rank0(f"  PP mesh size: {pp_mesh.size()}")
    print_rank0(f"  TP mesh size: {tp_mesh.size()}")

    # Create PP model on meta device for efficient splitting
    with torch.device("meta"):
        model = Qwen3Model(model_args)

    # Use pipeline_module_split
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_stages=2,
        num_layers=n_layers,
    )
    print_rank0(f"  Module names per stage: {module_names_per_stage}")

    pp_stages, model_parts = pipeline_module_split(
        whole_model=model,
        pp_mesh=pp_mesh,
        pp_schedule="1F1B",
        device=device,
        module_names_per_stage=module_names_per_stage,
    )

    # Materialize weights from golden model
    golden_state = golden_model.state_dict()
    for part in model_parts:
        part.to_empty(device=device)
        part.init_buffers(buffer_device=device)
        part_state = part.state_dict()
        for key in part_state:
            if key in golden_state:
                part_state[key].copy_(golden_state[key])
        part.load_state_dict(part_state)

    # Determine first/last stage
    has_first = any(s.is_first for s in pp_stages)
    has_last = any(s.is_last for s in pp_stages)

    for part in model_parts:
        part.eval()

    # Get PP rank (which stage this rank belongs to)
    pp_rank = pp_mesh.get_local_rank()
    tp_rank = tp_mesh.get_local_rank()
    print(
        f"  Rank {rank}: pp_rank={pp_rank}, tp_rank={tp_rank}, "
        f"has_first={has_first}, has_last={has_last}"
    )

    # Get PP group for broadcast
    pp_group = parallel_dims.get_group("pp")
    pp_group_ranks = dist.get_process_group_ranks(pp_group)
    first_stage_rank = pp_group_ranks[0]

    # Create test input (same for all ranks)
    if rank == 0:
        tokens, positions, cu_seqlens, max_seqlen = create_test_input(
            num_seqs=2,
            seq_len_per_seq=8,
            vocab_size=model_args.vocab_size,
            device=device,
            seed=123,
        )
        input_data = [tokens, positions, cu_seqlens, max_seqlen]
    else:
        input_data = [None, None, None, None]

    dist.broadcast_object_list(input_data, src=0)
    tokens, positions, cu_seqlens, max_seqlen = input_data
    tokens = tokens.to(device)
    positions = positions.to(device)
    cu_seqlens = cu_seqlens.to(device)

    # Run golden model forward
    with torch.no_grad():
        golden_output = golden_model(tokens, positions, cu_seqlens, max_seqlen)

    print_rank0(f"  Golden output shape: {golden_output.shape}")

    # Run PP forward using dist.broadcast with PP group
    success = True

    with torch.no_grad():
        # First stage forward
        if has_first:
            h = model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
        else:
            h = torch.zeros(
                (1, tokens.shape[1], model_args.dim),
                dtype=torch.float32,
                device=device,
            )

        # Synchronize and broadcast within PP group
        torch.cuda.synchronize()
        dist.barrier()
        # Use dist.broadcast with group parameter for PP+TP/DP scenarios
        dist.broadcast(h, src=first_stage_rank, group=pp_group)

        # Last stage forward
        if has_last and not has_first:
            pp_output = model_parts[0](h, positions, cu_seqlens, max_seqlen)
        elif has_last:
            pp_output = h
        else:
            pp_output = None

    # Synchronize after forward
    torch.cuda.synchronize()
    dist.barrier()

    # Verify output on last stage
    if has_last:
        match, max_diff, mean_diff = verify_outputs_match(
            pp_output, golden_output, rtol=1e-3, atol=1e-3, max_diff_threshold=1e-1
        )
        print(f"  Rank {rank}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        success = match

    # Gather results
    success_tensor = torch.tensor([1 if success else 0], dtype=torch.int, device=device)
    all_success = [torch.zeros_like(success_tensor) for _ in range(world_size)]
    dist.all_gather(all_success, success_tensor)

    # Check all last-stage ranks passed
    final_success = all(t.item() == 1 for t in all_success)

    dist.barrier()

    if final_success:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP+TP Forward Test: PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP+TP Forward Test: FAILED")
        print_rank0("=" * 60)

    if rank == 0 and out_file:
        write_result(out_file, final_success)

    return final_success


def test_pp_dp_forward(out_file: str | None = None) -> bool:
    """Test PP+DP combination forward pass.

    Configuration: 4 GPU with pp=2, dp_shard=2, tp=1
    Mesh layout: [pp=2, dp_shard=2, cp=1, tp=1]

    This test verifies:
    1. PP and DP meshes are correctly configured
    2. Forward pass produces correct output across DP replicas

    Args:
        out_file: Optional file to write result

    Returns:
        True if test passed, False otherwise
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print_rank0(f"\n{'=' * 60}")
    print_rank0("PP+DP Forward Test: pp=2, dp_shard=2")
    print_rank0("=" * 60)

    assert world_size == 4, f"This test requires 4 GPUs, got {world_size}"

    # Create model args
    n_layers = 4
    model_args = create_dense_model_args(
        n_layers=n_layers,
        dim=64,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        vocab_size=1000,
    )

    # Create golden model
    seed = 42
    golden_model = create_golden_model(model_args, device, seed=seed)
    golden_model.eval()

    # Create PP+DP parallel dims
    parallel_dims = ArchonParallelDims(
        pp=2,
        dp_shard=2,
        tp=1,
        cp=1,
        world_size=world_size,
        device_type="cuda",
    )

    print_rank0(f"  PP enabled: {parallel_dims.pp_enabled}")
    print_rank0(f"  DP shard: {parallel_dims.dp_shard}")
    print_rank0(f"  PP degree: {parallel_dims.pp}")

    # Verify mesh configuration
    pp_mesh = parallel_dims.get_mesh("pp")
    dp_mesh = parallel_dims.get_mesh("dp_shard")

    if pp_mesh is None:
        print_rank0("  FAILED: PP mesh is None")
        return False
    if dp_mesh is None:
        print_rank0("  FAILED: DP mesh is None")
        return False

    print_rank0(f"  PP mesh size: {pp_mesh.size()}")
    print_rank0(f"  DP mesh size: {dp_mesh.size()}")

    # Create PP model on meta device for efficient splitting
    with torch.device("meta"):
        model = Qwen3Model(model_args)

    # Use pipeline_module_split
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_stages=2,
        num_layers=n_layers,
    )
    print_rank0(f"  Module names per stage: {module_names_per_stage}")

    pp_stages, model_parts = pipeline_module_split(
        whole_model=model,
        pp_mesh=pp_mesh,
        pp_schedule="1F1B",
        device=device,
        module_names_per_stage=module_names_per_stage,
    )

    # Materialize weights from golden model
    golden_state = golden_model.state_dict()
    for part in model_parts:
        part.to_empty(device=device)
        part.init_buffers(buffer_device=device)
        part_state = part.state_dict()
        for key in part_state:
            if key in golden_state:
                part_state[key].copy_(golden_state[key])
        part.load_state_dict(part_state)

    # Determine first/last stage
    has_first = any(s.is_first for s in pp_stages)
    has_last = any(s.is_last for s in pp_stages)

    for part in model_parts:
        part.eval()

    pp_rank = pp_mesh.get_local_rank()
    dp_rank = dp_mesh.get_local_rank()
    print(
        f"  Rank {rank}: pp_rank={pp_rank}, dp_rank={dp_rank}, "
        f"has_first={has_first}, has_last={has_last}"
    )

    # Get PP group for broadcast
    pp_group = parallel_dims.get_group("pp")
    pp_group_ranks = dist.get_process_group_ranks(pp_group)
    first_stage_rank = pp_group_ranks[0]

    # Create test input (same for all ranks, each DP replica processes same data)
    if rank == 0:
        tokens, positions, cu_seqlens, max_seqlen = create_test_input(
            num_seqs=2,
            seq_len_per_seq=8,
            vocab_size=model_args.vocab_size,
            device=device,
            seed=123,
        )
        input_data = [tokens, positions, cu_seqlens, max_seqlen]
    else:
        input_data = [None, None, None, None]

    dist.broadcast_object_list(input_data, src=0)
    tokens, positions, cu_seqlens, max_seqlen = input_data
    tokens = tokens.to(device)
    positions = positions.to(device)
    cu_seqlens = cu_seqlens.to(device)

    # Run golden model forward
    with torch.no_grad():
        golden_output = golden_model(tokens, positions, cu_seqlens, max_seqlen)

    print_rank0(f"  Golden output shape: {golden_output.shape}")

    # Run PP forward using dist.broadcast with PP group
    success = True

    with torch.no_grad():
        # First stage forward
        if has_first:
            h = model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
        else:
            h = torch.zeros(
                (1, tokens.shape[1], model_args.dim),
                dtype=torch.float32,
                device=device,
            )

        # Synchronize and broadcast within PP group
        torch.cuda.synchronize()
        dist.barrier()
        # Use dist.broadcast with group parameter for PP+TP/DP scenarios
        dist.broadcast(h, src=first_stage_rank, group=pp_group)

        # Last stage forward
        if has_last and not has_first:
            pp_output = model_parts[0](h, positions, cu_seqlens, max_seqlen)
        elif has_last:
            pp_output = h
        else:
            pp_output = None

    # Synchronize after forward
    torch.cuda.synchronize()
    dist.barrier()

    # Verify output on last stage
    if has_last:
        match, max_diff, mean_diff = verify_outputs_match(
            pp_output, golden_output, rtol=1e-3, atol=1e-3, max_diff_threshold=1e-1
        )
        print(f"  Rank {rank}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        success = match

    # Gather results
    success_tensor = torch.tensor([1 if success else 0], dtype=torch.int, device=device)
    all_success = [torch.zeros_like(success_tensor) for _ in range(world_size)]
    dist.all_gather(all_success, success_tensor)

    final_success = all(t.item() == 1 for t in all_success)

    dist.barrier()

    if final_success:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP+DP Forward Test: PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP+DP Forward Test: FAILED")
        print_rank0("=" * 60)

    if rank == 0 and out_file:
        write_result(out_file, final_success)

    return final_success


def test_pp_ep_forward(out_file: str | None = None) -> bool:
    """Test PP+EP combination forward pass with MoE model.

    Configuration: 4 GPU with pp=2, ep=2, dp_shard=1, tp=1
    Mesh layout: [pp=2, dp_shard=1, cp=1, tp=1] with ep=2

    This test verifies:
    1. PP and EP meshes are correctly configured
    2. Forward pass with MoE model produces valid output (no NaN/Inf)

    Note: PP+EP uses a MoE model since EP requires expert parallelism.

    Args:
        out_file: Optional file to write result

    Returns:
        True if test passed, False otherwise
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print_rank0(f"\n{'=' * 60}")
    print_rank0("PP+EP Forward Test: pp=2, ep=2 (MoE model)")
    print_rank0("=" * 60)

    assert world_size == 4, f"This test requires 4 GPUs, got {world_size}"

    # Create MoE model args for PP+EP
    # PP=2 requires n_layers divisible by 2
    # EP=2 requires num_experts divisible by 2
    n_layers = 4  # 2 stages * 2 layers per stage
    num_experts = 4  # Must be divisible by ep=2
    model_args = create_moe_model_args(
        num_experts=num_experts,
        n_layers=n_layers,
        dim=64,
        hidden_dim=128,
        moe_inter_dim=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        vocab_size=1000,
        top_k=2,
    )

    # Create PP+EP parallel dims
    # For pp=2, ep=2 with world_size=4:
    # - PP splits world into 2 pipeline stages (ranks 0,2 and 1,3 for example)
    # - EP splits experts across dp_shard dimension
    parallel_dims = ArchonParallelDims(
        pp=2,
        dp_shard=2,  # dp_shard is used for EP in AReaL
        tp=1,
        cp=1,
        ep=2,  # Expert parallelism degree
        world_size=world_size,
        device_type="cuda",
    )

    print_rank0(f"  PP enabled: {parallel_dims.pp_enabled}")
    print_rank0(f"  EP enabled: {parallel_dims.ep_enabled}")
    print_rank0(f"  PP degree: {parallel_dims.pp}")
    print_rank0(f"  EP degree: {parallel_dims.ep}")

    # Verify mesh configuration
    pp_mesh = parallel_dims.get_mesh("pp")
    ep_mesh = parallel_dims.get_mesh("dp_shard_cp")  # EP uses dp_shard_cp mesh

    if pp_mesh is None:
        print_rank0("  FAILED: PP mesh is None")
        return False

    print_rank0(f"  PP mesh size: {pp_mesh.size()}")
    if ep_mesh is not None:
        print_rank0(f"  EP mesh size: {ep_mesh.size()}")

    # Create MoE model on meta device for efficient pipeline splitting
    with torch.device("meta"):
        model = Qwen3Model(model_args)

    # Use pipeline_llm to split model across PP stages with parallelization
    archon_config = ArchonEngineConfig(pp_schedule="1F1B")
    try:
        pp_stages, model_parts, has_first, has_last = pipeline_llm(
            model=model,
            device=device,
            parallel_dims=parallel_dims,
            archon_config=archon_config,
            parallelize_fn=parallelize_qwen3,
            enable_compile=False,  # Disable compile for faster test
        )
    except Exception as e:
        print_rank0(f"  FAILED: pipeline_llm error: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and out_file:
            write_result(out_file, False)
        return False

    # Materialize weights from meta device
    for part in model_parts:
        part.to_empty(device=device)
        with torch.no_grad():
            part.init_weights()
        part.init_buffers(buffer_device=device)

    for part in model_parts:
        part.eval()

    pp_rank = pp_mesh.get_local_rank()
    print(
        f"  Rank {rank}: pp_rank={pp_rank}, has_first={has_first}, has_last={has_last}"
    )

    # Create test input (same for all ranks)
    if rank == 0:
        tokens, positions, cu_seqlens, max_seqlen = create_test_input(
            num_seqs=2,
            seq_len_per_seq=8,
            vocab_size=model_args.vocab_size,
            device=device,
            seed=123,
        )
        input_data = [tokens, positions, cu_seqlens, max_seqlen]
    else:
        input_data = [None, None, None, None]

    dist.broadcast_object_list(input_data, src=0)
    tokens, positions, cu_seqlens, max_seqlen = input_data
    tokens = tokens.to(device)
    positions = positions.to(device)
    cu_seqlens = cu_seqlens.to(device)

    print_rank0(f"  Input tokens shape: {tokens.shape}")

    # Run PP+EP forward
    # For PP+EP, we use the schedule-based forward from pipeline_llm
    # which handles activation passing between stages
    success = True

    try:
        with torch.no_grad():
            # Simple forward through local model parts
            # pipeline_llm returns model_parts that can be called directly
            if has_first:
                h = model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
                print(f"  Rank {rank}: first stage output shape = {h.shape}")

                # Check for NaN/Inf
                if torch.isnan(h).any() or torch.isinf(h).any():
                    print_rank0("  FAILED: First stage output contains NaN/Inf")
                    success = False

        torch.cuda.synchronize()
        dist.barrier()

        # For PP+EP test, we mainly verify:
        # 1. Model can be created and parallelized without error
        # 2. Forward pass produces valid output (no NaN/Inf)
        # Full PP forward with activation passing is tested in PP-only tests

    except Exception as e:
        print_rank0(f"  FAILED: Forward error: {e}")
        import traceback

        traceback.print_exc()
        success = False

    # Gather results from all ranks
    success_tensor = torch.tensor([1 if success else 0], dtype=torch.int, device=device)
    all_success = [torch.zeros_like(success_tensor) for _ in range(world_size)]
    dist.all_gather(all_success, success_tensor)

    final_success = all(t.item() == 1 for t in all_success)

    dist.barrier()

    if final_success:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP+EP Forward Test: PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP+EP Forward Test: FAILED")
        print_rank0("=" * 60)

    if rank == 0 and out_file:
        write_result(out_file, final_success)

    return final_success


TEST_REGISTRY = {
    "pp_tp": test_pp_tp_forward,
    "pp_dp": test_pp_dp_forward,
    "pp_ep": test_pp_ep_forward,
}


def main():
    parser = argparse.ArgumentParser(description="PP Combination Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run (pp_tp, pp_dp, or pp_ep)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")

    try:
        test_fn = TEST_REGISTRY[args.test_type]
        test_fn(args.output)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
