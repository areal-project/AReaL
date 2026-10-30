# SPDX-License-Identifier: Apache-2.0
"""One real FSDP SC optimizer update using a caller-provided local checkpoint."""

import argparse
import os

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
    RejectionSamplingConfig,
)
from areal.engine.fsdp_engine import FSDPEngine
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.utils import seeding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tis-cap", type=float, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    args = parser.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    tp = args.tp
    assert tp * args.cp == world
    backend = f"fsdp:d1t{tp}c{args.cp}"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    seeding.set_random_seed(42, "score-centering")
    config = PPOActorConfig(
        backend=backend,
        path=args.model,
        experiment_name="score-centering-test",
        trial_name="smoke",
        use_score_centering=True,
        rejection_sampling=RejectionSamplingConfig(action="clamp", upper=args.tis_cap)
        if args.tis_cap is not None
        else None,
        recompute_logprob=False,
        dtype="bfloat16",
        disable_dropout=True,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=128, n_mbs=4),
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0),
    )
    engine = FSDPEngine(config)
    engine.create_process_group(ModelAllocation.from_str(backend).parallel)
    engine.initialize(
        None, FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=2)
    )
    try:
        device = engine.device
        tokens = torch.tensor(
            [[31, 42, 53, 64, 75, 86, 97, 108], [41, 52, 63, 74, 85, 0, 0, 0]],
            device=device,
        )
        tokens = tokens.repeat(4, 1)
        attention = tokens != 0
        mask = attention.clone()
        mask[:, :2] = False
        mask[:, -1] = False
        mask[1::2, 4] = False
        vocab = engine.model_config.vocab_size
        heads = torch.tensor([0, vocab // 3, 2 * vocab // 3, vocab - 1], device=device)
        batch = dict(
            input_ids=tokens,
            attention_mask=attention,
            loss_mask=mask,
            logprobs=torch.full_like(tokens, -12.0, dtype=torch.float32),
            prox_logp=torch.full_like(tokens, -11.0, dtype=torch.float32),
            advantages=torch.ones_like(tokens, dtype=torch.float32),
            sampler_head_ids=heads.expand(8, 8, 4).clone(),
            sampler_head_logprobs=torch.full(
                (8, 8, 4), -20.0 if args.tis_cap is not None else -4.0, device=device
            ),
        )
        seen = []

        def loss_fn(logprobs, entropy, inputs, *, sc_correction=None, **kwargs):
            if sc_correction is not None:
                assert (
                    logprobs.shape == sc_correction.shape == inputs["loss_mask"].shape
                )
                assert torch.isfinite(sc_correction).all()
                seen.append(sc_correction.detach().abs().max().item())
            return grpo_loss_fn(
                logprobs,
                entropy,
                inputs,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
                use_score_centering=True,
                sc_correction=sc_correction,
                rejection_sampling=config.rejection_sampling,
                **kwargs,
            )

        engine.train()
        parameter = next(engine.model.parameters())
        before = parameter.to_local().detach().clone()
        stats = engine.train_batch(
            batch, loss_fn, lambda x: x["loss_mask"].count_nonzero()
        )
        torch.cuda.synchronize()
        assert seen and max(seen) > 0
        assert stats["update_successful"] == 1
        assert torch.isfinite(torch.tensor(stats["grad_norm"]))
        assert stats["grad_norm"] > 0
        delta = (parameter.to_local().detach().float() - before.float()).abs().max()
        dist.all_reduce(delta, op=dist.ReduceOp.MAX)
        assert delta > 0
        if dist.get_rank() == 0:
            from areal.utils import logging

            logging.getLogger("FSDPScoreCenteringTest").info(
                f"PASS {backend} tis_cap={args.tis_cap}, max_delta={delta.item()}, stats={stats}"
            )
    finally:
        engine.destroy()


if __name__ == "__main__":
    main()
