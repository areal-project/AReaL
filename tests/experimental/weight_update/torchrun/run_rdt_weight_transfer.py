#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.experimental.weight_update.torchrun.dist_utils import (  # noqa: E402
    print_rank0,
    write_result,
)

from areal.infra.platforms import current_platform  # noqa: E402

# Skip YR tests - only test NIXL (CUDA GPU)
# YR requires ray_ascend which may not be available in test environment
assert current_platform.device_type == "cuda", "RDT tests require CUDA GPU (NIXL)"


def run_rdt_weight_transfer_lifecycle(output=None):
    """Test: Full RDT weight transfer lifecycle with real WeightTransportActor + CUDA IPC.

    This test validates the complete RDT flow:
    1. TW creates WeightTransportActor with GPU binding
    2. TW creates tensors, uses CUDA IPC (share_memory_ + reduce_tensor)
    3. TW calls actor.store_ipc_handles.remote()
    4. IW receives TW actor handle, calls get_weights_tensor_nixl.remote()
    5. IW verifies transferred weights
    6. IW calls clear_ipc_handles.remote() to cleanup
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0(
        "=== RDT Weight Transfer Lifecycle Test (Real WeightTransportActor) ==="
    )

    infer_world_size = world_size // 2
    is_inference = rank < infer_world_size

    # All processes use cuda:0
    device = torch.device("cuda:0")

    print_rank0(
        f"  Inference ranks: 0..{infer_world_size - 1}, "
        f"Training ranks: {infer_world_size}..{world_size - 1}"
    )

    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from torch.multiprocessing.reductions import reduce_tensor

        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)

        from areal.experimental.weight_update.rdt import (
            deserialize_actor_handle_bytes,
            serialize_actor_handle_bytes,
        )
        from areal.experimental.weight_update.rdt.weight_transport_actor import (
            WeightTransportActor,
        )

        param_shapes = [(512, 256), (256,), (1024, 512), (2048, 1024)]
        pair_name = "lifecycle_pair"
        version = 1

        # Phase 1: TW creates WeightTransportActor and distributes handle
        tw_handles = {}  # IW will store these

        if not is_inference:
            # Training side: create WeightTransportActor
            tw_rank = rank - infer_world_size
            infer_rank = tw_rank  # Corresponding IW rank

            current_node_id = ray.get_runtime_context().get_node_id()
            current_visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

            print(
                f"[TW rank {rank}] CUDA_VISIBLE_DEVICES: '{current_visible_gpus}'",
                flush=True,
            )

            tw_actor = WeightTransportActor.options(
                name=f"tw-actor-{tw_rank}",
                num_gpus=0.0001,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=current_node_id, soft=False
                ),
                runtime_env={
                    "env_vars": {"CUDA_VISIBLE_DEVICES": current_visible_gpus}
                },
            ).remote()

            # Broadcast handle to all inference ranks
            encoded = serialize_actor_handle_bytes(tw_actor)

            for iw_rank in range(infer_world_size):
                length_tensor = torch.tensor([len(encoded)], dtype=torch.long)
                dist.send(length_tensor, dst=iw_rank)
                handle_tensor = torch.tensor(
                    [ord(c) for c in encoded], dtype=torch.long
                )
                dist.send(handle_tensor, dst=iw_rank)

            print_rank0(f"  TW rank {rank}: Distributed handle to all IW ranks")

        if is_inference:
            # Inference side: receive handles from all TW ranks
            for tw_idx in range(infer_world_size):
                tw_global_rank = tw_idx + infer_world_size

                length_tensor = torch.zeros(1, dtype=torch.long)
                dist.recv(length_tensor, src=tw_global_rank)
                handle_length = int(length_tensor.item())

                handle_tensor = torch.zeros(handle_length, dtype=torch.long)
                dist.recv(handle_tensor, src=tw_global_rank)
                encoded = "".join([chr(int(c.item())) for c in handle_tensor])

                tw_handle = deserialize_actor_handle_bytes(encoded)
                tw_handles[tw_idx] = tw_handle

            print_rank0(f"  IW rank {rank}: Received {len(tw_handles)} TW handles")

        dist.barrier()

        # Phase 2: TW creates tensors and stores IPC handles
        if not is_inference:
            tw_rank = rank - infer_world_size
            infer_rank = tw_rank

            # Create weights on cuda:0
            torch.manual_seed(100 + tw_rank)
            params = {
                f"model.layers.{i}.weight": torch.randn(shape, device=device)
                for i, shape in enumerate(param_shapes)
            }
            params["model.norm.weight"] = torch.randn(param_shapes[1][0], device=device)

            # Create IPC handles via share_memory_() + reduce_tensor()
            ipc_handles = {}
            for name, tensor in params.items():
                tensor.share_memory_()
                rebuild_fn, tensor_meta = reduce_tensor(tensor)
                ipc_handles[name] = {
                    "rebuild_fn": rebuild_fn,
                    "tensor_meta": tensor_meta,
                }

            # Store IPC handles in actor
            ray.get(
                tw_actor.store_ipc_handles.remote(
                    pair_name, infer_rank, version, ipc_handles
                )
            )

            print_rank0(
                f"  TW rank {rank}: Stored IPC handles for infer_rank {infer_rank}"
            )

        dist.barrier()

        # Phase 3: IW pulls weights from TW via Ray RPC (NIXL transport)
        if is_inference:
            tw_idx = rank
            infer_rank = tw_idx

            if tw_idx in tw_handles:
                tw_handle = tw_handles[tw_idx]

                # IW pulls weights via tensor_transport
                received_params = ray.get(
                    tw_handle.get_weights_tensor_nixl.remote(
                        pair_name, infer_rank, version
                    )
                )

                print_rank0(f"  IW rank {rank}: Pulled weights via Ray RPC")

                # Phase 4: Verify transferred weights
                torch.manual_seed(100 + tw_idx)
                expected_params = {
                    f"model.layers.{i}.weight": torch.randn(shape, device=device)
                    for i, shape in enumerate(param_shapes)
                }
                expected_params["model.norm.weight"] = torch.randn(
                    param_shapes[1][0], device=device
                )

                verify_success = True
                for name in expected_params:
                    expected = expected_params[name]
                    actual = received_params[name]
                    try:
                        torch.testing.assert_close(
                            actual, expected, rtol=1e-5, atol=1e-5
                        )
                    except AssertionError:
                        max_diff = (actual - expected).abs().max().item()
                        print_rank0(f"  MISMATCH {name}: max_diff={max_diff}")
                        verify_success = False

                print_rank0(
                    f"  IW rank {rank}: Weight verification "
                    f"{'PASSED' if verify_success else 'FAILED'}"
                )

                # Phase 5: Cleanup IPC handles
                ray.get(
                    tw_handle.clear_ipc_handles.remote(pair_name, infer_rank, version)
                )
                print_rank0(f"  IW rank {rank}: Cleaned up IPC handles")

        print_rank0("  RDT weight transfer lifecycle: PASSED")
        success = True

    except Exception as e:
        print_rank0(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        success = False

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)
    return success


TEST_REGISTRY = {
    "rdt_weight_transfer_lifecycle": run_rdt_weight_transfer_lifecycle,
}


def main():
    parser = argparse.ArgumentParser(description="RDT Weight Transfer Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Modify CUDA_VISIBLE_DEVICES BEFORE CUDA context initialization
    # Each process gets its own GPU (IW and TW on different GPUs)
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_list = all_gpus.split(",")

    # Each process has its own GPU (IW and TW on different GPUs)
    gpu_index = rank
    my_gpu = gpu_list[gpu_index] if gpu_index < len(gpu_list) else gpu_list[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = my_gpu

    dist.init_process_group(backend="gloo")
    torch.cuda.set_device(0)  # cuda:0 = my_gpu

    rank = dist.get_rank()

    print_rank0("=" * 60)
    print_rank0(f"Running: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]
        success = test_fn(args.output)

        dist.barrier()
        if success:
            print_rank0(f"\n{args.test_type}: PASSED")
        else:
            print_rank0(f"\n{args.test_type}: FAILED")
            if rank == 0 and args.output:
                write_result(args.output, False)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
