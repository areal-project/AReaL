import argparse
import os
import pickle
import random

import torch
import torch.distributed as dist

from areal.infra.dist_rollout import redistribute_trajectories
from areal.infra.platforms import current_platform
from areal.utils.data import tensor_container_to


def main(args):
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    current_platform.set_device(rank)
    device = f"{current_platform.device_type}:{rank}"

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
    args = parser.parse_args()
    main(args)
