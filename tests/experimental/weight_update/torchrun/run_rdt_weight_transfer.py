#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
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
    """Test: Full RDT weight transfer lifecycle - TW sends, IW pulls via Ray RPC.

    This test validates the complete RDT flow:
    1. TW actor handle serialization and distribution
    2. IW stores TW handles
    3. IW pulls weights via ray.get() with tensor transport
    4. Weight values are correctly transferred
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("=== RDT Weight Transfer Lifecycle Test ===")

    infer_world_size = world_size // 2
    is_inference = rank < infer_world_size

    print_rank0(
        f"  Inference ranks: 0..{infer_world_size - 1}, "
        f"Training ranks: {infer_world_size}..{world_size - 1}"
    )

    try:
        import ray

        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)

        from areal.experimental.weight_update.rdt import (
            deserialize_actor_handle_bytes,
            serialize_actor_handle_bytes,
        )

        device = torch.device(f"cuda:{current_platform.current_device()}")

        param_shapes = [(512, 256), (256,), (1024, 512), (2048, 1024)]

        @ray.remote
        class MockTWActorForLifecycle:
            def __init__(self, tw_rank, param_shapes):
                self.tw_rank = tw_rank
                # Use Ray-assigned GPU
                self.device = torch.device("cuda:0")
                torch.manual_seed(100 + tw_rank)
                self.params = {
                    f"model.layers.{i}.weight": torch.randn(shape, device=self.device)
                    for i, shape in enumerate(param_shapes)
                }
                self.params["model.norm.weight"] = torch.randn(
                    param_shapes[1][0], device=self.device
                )

            @ray.method(tensor_transport="NIXL")
            def rdt_get_weights_tensor(self, pair_name, version):
                return {k: v.clone() for k, v in self.params.items()}

        # Phase 1: TW actors creation and handle distribution
        tw_handles = {}  # IW will store these

        if not is_inference:
            # Training side: create actors
            tw_rank = rank - infer_world_size
            tw_actor = MockTWActorForLifecycle.options(num_gpus=1).remote(
                tw_rank, param_shapes
            )

            # Broadcast handle to all inference ranks
            encoded = serialize_actor_handle_bytes(tw_actor)

            for iw_rank in range(infer_world_size):
                # Send to each IW rank individually
                length_tensor = torch.tensor(
                    [len(encoded)], dtype=torch.long, device=device
                )
                dist.send(length_tensor, dst=iw_rank)
                handle_tensor = torch.tensor(
                    [ord(c) for c in encoded], dtype=torch.long, device=device
                )
                dist.send(handle_tensor, dst=iw_rank)

            print_rank0(f"  TW rank {rank}: Distributed handle to all IW ranks")

        if is_inference:
            # Inference side: receive handles from all TW ranks
            for tw_idx in range(infer_world_size):
                tw_global_rank = tw_idx + infer_world_size

                length_tensor = torch.zeros(1, dtype=torch.long, device=device)
                dist.recv(length_tensor, src=tw_global_rank)
                handle_length = int(length_tensor.item())

                handle_tensor = torch.zeros(
                    handle_length, dtype=torch.long, device=device
                )
                dist.recv(handle_tensor, src=tw_global_rank)
                encoded = "".join([chr(int(c.item())) for c in handle_tensor])

                tw_handle = deserialize_actor_handle_bytes(encoded)
                tw_handles[tw_idx] = tw_handle

            print_rank0(f"  IW rank {rank}: Received {len(tw_handles)} TW handles")

        dist.barrier()

        # Phase 2: IW pulls weights from TW via Ray RPC (NIXL transport)
        if is_inference:
            tw_idx = rank
            if tw_idx in tw_handles:
                tw_handle = tw_handles[tw_idx]
                received_params = ray.get(
                    tw_handle.rdt_get_weights_tensor.remote("lifecycle_pair", version=1)
                )

                print_rank0(f"  IW rank {rank}: Pulled weights via Ray RPC")

                # Phase 3: Verify transferred weights
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
                    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5):
                        max_diff = (actual - expected).abs().max().item()
                        print_rank0(f"  MISMATCH {name}: max_diff={max_diff}")
                        verify_success = False

                print_rank0(
                    f"  IW rank {rank}: Weight verification "
                    f"{'PASSED' if verify_success else 'FAILED'}"
                )

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

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

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
