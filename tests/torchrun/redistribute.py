import argparse
import os
import pickle
import random

import torch
import torch.distributed as dist

from areal.infra.dist_rollout import DistRolloutCoordinator, redistribute_trajectories
from areal.infra.platforms import current_platform
from areal.utils.data import tensor_container_to


class _HybridTrainEngine:
    def __init__(self, rank, dp_group, model_group):
        self.rank = rank
        self.data_parallel_group = dp_group
        self.context_and_model_parallel_group = model_group

    def is_data_parallel_head(self):
        return self.rank % 2 == 0

    def current_data_parallel_head(self):
        return self.rank - self.rank % 2


class _HybridRolloutEngine:
    def __init__(self, rank):
        self.rank = rank

    def prepare_batch(self, *args, **kwargs):
        if self.rank == 0:
            raise ValueError("rank-local preparation failure")
        return []


def _test_hybrid_error(rank):
    head_dp_group = dist.new_group([0, 2])
    non_head_dp_group = dist.new_group([1, 3])
    first_model_group = dist.new_group([0, 1])
    second_model_group = dist.new_group([2, 3])
    train_engine = _HybridTrainEngine(
        rank,
        head_dp_group if rank % 2 == 0 else non_head_dp_group,
        first_model_group if rank < 2 else second_model_group,
    )
    coordinator = DistRolloutCoordinator(_HybridRolloutEngine(rank), train_engine)

    try:
        coordinator.prepare_batch(object(), object())
    except RuntimeError as exc:
        assert "rank-local preparation failure" in str(exc)
    else:
        raise AssertionError("Expected coordinated rank-local preparation failure")
    dist.barrier()

    trajectories = [] if train_engine.is_data_parallel_head() else None
    try:
        coordinator._broadcast_and_redistribute_trajectories(trajectories)
    except RuntimeError as exc:
        assert "Cannot redistribute 0 trainable trajectory groups" in str(exc)
    else:
        raise AssertionError("Expected coordinated rollout preparation failure")
    dist.barrier()


def main(args):
    dist.init_process_group(args.backend)
    rank = int(os.environ["LOCAL_RANK"])
    if args.hybrid_error:
        _test_hybrid_error(rank)
        return
    if args.backend == "nccl":
        current_platform.set_device(rank)
        device = f"{current_platform.device_type}:{rank}"
    else:
        device = "cpu"

    if args.ragged_empty:
        # One rank contributes zero trajectories; the gather must still work.
        bs = 3 * rank
    elif args.ragged:
        bs = rank + 1
    else:
        bs = 16
    prompt_lens = [random.randint(1, 10) for _ in range(bs)]
    ans_lens = [random.randint(1, 10) for _ in range(bs)]
    seqlens = [x + y for x, y in zip(prompt_lens, ans_lens)]

    data = []
    for prompt_len, ans_len, seqlen in zip(prompt_lens, ans_lens, seqlens):
        seq = torch.randint(0, 100, (seqlen,), dtype=torch.long, device=device)
        loss_mask = torch.tensor(
            [0] * prompt_len + [1] * ans_len, dtype=torch.bool, device=device
        )
        log_probs = torch.tensor(
            [0] * prompt_len + [-random.random() for _ in range(ans_len)],
            dtype=torch.float,
            device=device,
        )
        attention_mask = torch.ones(seqlen, dtype=torch.bool, device=device)
        d = dict(
            input_ids=seq.unsqueeze(0),
            loss_mask=loss_mask.unsqueeze(0),
            log_probs=log_probs.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )
        data.append(d)

    data = [tensor_container_to(x, device) for x in data]
    redistributed = redistribute_trajectories(data)

    redistributed.all_data = [
        tensor_container_to(x, "cpu") for x in redistributed.all_data
    ]
    redistributed.data = tensor_container_to(redistributed.data, "cpu")

    with open(
        os.path.join(args.dump_path, f"redistributed{dist.get_rank()}.pkl"), "wb"
    ) as f:
        pickle.dump(redistributed, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-path", type=str)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    parser.add_argument("--ragged", action="store_true")
    parser.add_argument("--ragged-empty", action="store_true")
    parser.add_argument("--hybrid-error", action="store_true")
    args = parser.parse_args()
    main(args)
