import os
import time

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform

HEADER_SIZE = 4
OWNER_RANK = 0
PULL_PARAM = 0
PUSH_GRAD = 1
DONE = 2


def init_distributed() -> None:
    if dist.is_initialized():
        return
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    current_platform.set_device(int(os.environ["LOCAL_RANK"]))


def request_plan(rank: int) -> list[tuple[int, float]]:
    # Different delays force workers to reach the owner in a dynamic order.
    plans = {
        1: [(0, 0.15), (1, 0.03)],
        2: [(1, 0.02), (0, 0.10)],
    }
    return plans[rank]


def build_owner_shards(device: torch.device) -> dict[int, torch.Tensor]:
    return {
        0: torch.tensor([1.0, 2.0, 3.0, 4.0], device=device),
        1: torch.tensor([10.0, 20.0, 30.0, 40.0], device=device),
    }


def run_worker(rank: int, device: torch.device) -> None:
    for req_id, (shard_id, delay_s) in enumerate(request_plan(rank)):
        time.sleep(delay_s)

        pull_header = torch.tensor(
            [PULL_PARAM, req_id, shard_id, 4], device=device, dtype=torch.int64
        )
        dist.isend(pull_header, dst=OWNER_RANK).wait()

        param = torch.empty(4, device=device, dtype=torch.float32)
        dist.irecv(param, src=OWNER_RANK).wait()

        # Fake local compute. Activations/grad activations stay local; only grad shard is pushed.
        grad = param * float(rank)
        grad_header = torch.tensor(
            [PUSH_GRAD, req_id, shard_id, grad.numel()],
            device=device,
            dtype=torch.int64,
        )
        dist.isend(grad_header, dst=OWNER_RANK).wait()
        dist.isend(grad, dst=OWNER_RANK).wait()

    done = torch.tensor([DONE, 0, 0, 0], device=device, dtype=torch.int64)
    dist.isend(done, dst=OWNER_RANK).wait()


def run_owner(world_size: int, device: torch.device) -> None:
    shards = build_owner_shards(device)
    grad_accum = {shard_id: torch.zeros_like(t) for shard_id, t in shards.items()}
    event_log: list[str] = []

    header_bufs = {
        src: torch.empty(HEADER_SIZE, device=device, dtype=torch.int64)
        for src in range(1, world_size)
    }
    header_works = {
        src: dist.irecv(header_bufs[src], src=src) for src in range(1, world_size)
    }
    done_workers: set[int] = set()

    while len(done_workers) < world_size - 1:
        progressed = False
        for src in range(1, world_size):
            if src in done_workers:
                continue
            work = header_works[src]
            if not work.is_completed():
                continue

            work.wait()
            header = header_bufs[src].clone()
            op, req_id, shard_id, numel = [int(x) for x in header.tolist()]
            progressed = True

            if op == PULL_PARAM:
                event_log.append(f"pull:worker={src},req={req_id},shard={shard_id}")
                payload = shards[shard_id]
                assert payload.numel() == numel
                dist.isend(payload, dst=src).wait()
                header_works[src] = dist.irecv(header_bufs[src], src=src)
            elif op == PUSH_GRAD:
                event_log.append(f"grad:worker={src},req={req_id},shard={shard_id}")
                grad = torch.empty(numel, device=device, dtype=torch.float32)
                dist.irecv(grad, src=src).wait()
                grad_accum[shard_id].add_(grad.view_as(grad_accum[shard_id]))
                header_works[src] = dist.irecv(header_bufs[src], src=src)
            elif op == DONE:
                done_workers.add(src)
            else:
                raise ValueError(f"Unexpected op={op} from worker {src}")

        if not progressed:
            time.sleep(0.001)

    expected = {
        0: shards[0] * (1.0 + 2.0),
        1: shards[1] * (1.0 + 2.0),
    }
    torch.testing.assert_close(grad_accum[0], expected[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(grad_accum[1], expected[1], atol=0.0, rtol=0.0)

    print("Observed dynamic event order:", flush=True)
    for event in event_log:
        print(f"  {event}", flush=True)
    print("Dynamic NCCL P2P mailbox test passed", flush=True)


def main() -> None:
    init_distributed()
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size >= 3, "This test requires at least 3 ranks"
        device = current_platform.current_device()

        if rank == OWNER_RANK:
            run_owner(world_size, device)
        else:
            run_worker(rank, device)

        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
