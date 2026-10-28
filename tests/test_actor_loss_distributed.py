# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import LossTermConfig, PPOActorConfig
from areal.trainer.ppo.loss import ActorLoss
from areal.utils.data import make_transport_microbatch


def _check_named_prompt(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        n = 2 if rank == 0 else 1
        mask = torch.ones(n, 2, dtype=torch.bool)
        data = dict(
            input_ids=torch.ones(n, 2, dtype=torch.long),
            attention_mask=mask.clone(),
            loss_mask=mask,
            subset=mask.clone() if rank == 0 else torch.zeros_like(mask),
            empty=torch.zeros_like(mask),
        )
        cfg = PPOActorConfig(
            losses={
                name: LossTermConfig(
                    implementation="entropy", reduction="prompt_mean", mask_key=key
                )
                for name, key in [
                    ("all", "loss_mask"),
                    ("subset", "subset"),
                    ("empty", "empty"),
                ]
            }
        )
        prepared = ActorLoss(cfg).prepare(data, [n])
        batches = [
            {key: value[i : i + 1] for key, value in data.items()} for i in range(n)
        ]
        if n == 1:
            batches.append(make_transport_microbatch(data))
        steps = prepared.for_steps(batches, dp_group=dist.group.WORLD, device="cpu")
        totals = {name: 0.0 for name in ("all", "subset", "empty")}
        for i, (step, batch) in enumerate(zip(steps, batches)):
            entropy = torch.full((1, 2), float(1 + rank), requires_grad=True)
            if rank == 1 and i == 1:
                nums = {name: entropy.sum() * 0 for name in totals}
                masses = {name: torch.tensor(0.0) for name in totals}
            else:
                with patch("areal.trainer.ppo.loss.stats_tracker"):
                    nums = step(entropy * 0, entropy, batch)
                masses = step.loss_weight(batch)
            for name in totals:
                numerator = nums[name].detach().clone()
                mass = masses[name].float().clone()
                dist.all_reduce(numerator)
                dist.all_reduce(mass)
                totals[name] += (numerator / mass.clamp_min(1)).item() / 2
        assert totals == {"all": -1.5, "subset": -1.0, "empty": 0.0}
    finally:
        dist.destroy_process_group()


def test_named_prompt_asymmetric_support_and_transport_rank(tmp_path):
    mp.spawn(
        _check_named_prompt, args=(f"file://{tmp_path / 'rdzv'}",), nprocs=2, join=True
    )
