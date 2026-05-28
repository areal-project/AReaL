import argparse
import os
import sys
import time
import uuid

import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils.lock import DistributedLock


def setup_distributed_environment(backend: str):
    if dist.is_initialized():
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def _generate_namespace() -> str:
    namespace = None
    if dist.get_rank() == 0:
        namespace = f"dist_lock_test/{uuid.uuid4()}"
    obj_list = [namespace]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def distributed_lock_smoke_test(
    iters: int, timeout: float, hold_time: float, backoff: float
):
    namespace = _generate_namespace()
    lock = DistributedLock("unittest", namespace=namespace, backoff=backoff)

    for _ in range(iters):
        acquired = lock.acquire(timeout=timeout)
        if not acquired:
            raise RuntimeError(
                f"Rank {dist.get_rank()} failed to acquire lock within {timeout}s"
            )
        try:
            time.sleep(hold_time)
        finally:
            lock.release()
        dist.barrier()

    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--hold-time", type=float, default=0.01)
    parser.add_argument("--backoff", type=float, default=0.01)
    args = parser.parse_args()

    try:
        setup_distributed_environment(backend=args.backend)
    except Exception:
        raise RuntimeError("Failed to initialize distributed environment")

    rank = dist.get_rank()
    try:
        distributed_lock_smoke_test(
            args.iters,
            args.timeout,
            max(args.hold_time, 0.0),
            max(args.backoff, 1e-6),
        )
    finally:
        dist.destroy_process_group()

    if rank == 0:
        print("DistributedLock smoke test passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
