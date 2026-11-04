# SPDX-License-Identifier: Apache-2.0
"""One local-checkpoint Megatron SC update; no dataset/model download.

Run with torchrun --standalone --nproc_per_node=2 this_file --model /path/to/model.
This exercises packing, the engine loss hook, MCore backward and its optimizer.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
    RejectionSamplingConfig,
)
from areal.engine.core.model import lang_config
from areal.engine.megatron_engine import MegatronEngine
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.utils import seeding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tis-cap", type=float, default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--probe-dir", default=None)
    parser.add_argument("--no-sc", action="store_true")
    parser.add_argument("--dense-oracle", action="store_true")
    args = parser.parse_args()
    if args.dense_oracle:
        import areal.utils.score_centering as sc

        def dense_oracle(
            logits,
            labels,
            ids,
            q,
            chunk_size=256,
            *,
            tp_group=None,
            vocab_size=None,
            tis_cap=None,
        ):
            if tp_group is not None:
                shards = [
                    torch.empty_like(logits)
                    for _ in range(dist.get_world_size(tp_group))
                ]
                dist.all_gather(shards, logits.detach(), group=tp_group)
                shards[dist.get_rank(tp_group)] = logits
                logits = torch.cat(shards, dim=-1)
            if vocab_size is not None:
                logits = logits[:, :vocab_size]
            logp = logits.float().log_softmax(-1)
            heads = logp.gather(-1, ids)
            with torch.no_grad():
                p = heads.exp()
                rho = (1 - q.exp().sum(-1)).clamp_min(0) / (1 - p.sum(-1)).clamp_min(
                    1e-06
                )
                coefficient = q.exp() - rho[:, None] * p
                if tis_cap is not None:
                    coefficient = (
                        torch.minimum(p, tis_cap * q.exp())
                        - (tis_cap * rho).clamp_max(1)[:, None] * p
                    )
                entropy = -(logp.exp() * logp).sum(-1)
            return (
                logp.gather(-1, labels[:, None]).squeeze(-1),
                entropy,
                (coefficient * heads).sum(-1),
            )

        sc.score_centering_logprobs = dense_oracle
    world = int(os.environ["WORLD_SIZE"])
    tp = args.tp or world // (args.cp * args.pp)
    assert tp * args.cp * args.pp == world
    backend = f"megatron:d1t{tp}c{args.cp}p{args.pp}"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    seeding.set_random_seed(42, "score-centering")
    config = PPOActorConfig(
        backend=backend,
        path=args.model,
        experiment_name="score-centering-test",
        trial_name="smoke",
        use_score_centering=not args.no_sc,
        rejection_sampling=RejectionSamplingConfig(action="clamp", upper=args.tis_cap)
        if args.tis_cap is not None
        else None,
        recompute_logprob=False,
        dtype="bfloat16",
        disable_dropout=True,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=128, n_mbs=4),
        optimizer=OptimizerConfig(lr=0.0001, weight_decay=0),
        megatron=MegatronEngineConfig(bridge_type="mbridge"),
    )
    engine = MegatronEngine(config)
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
        vocab = lang_config(engine.hf_config).vocab_size
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
        # Ordinary logprob forwards must work without sampler metadata even
        # when the actor is configured for SC (including CP and PP).
        forward_inputs = dict(input_ids=tokens, attention_mask=attention)
        actual_lp = engine.forward_batch(forward_inputs)
        original_sc = config.use_score_centering
        try:
            config.use_score_centering = False
            expected_lp = engine.forward_batch(forward_inputs)
        finally:
            config.use_score_centering = original_sc
        torch.testing.assert_close(actual_lp, expected_lp, atol=1e-5, rtol=1e-5)
        seen = []

        def loss_fn(logprobs, entropy, inputs, *, sc_correction=None, **kwargs):
            if not args.no_sc:
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
                use_score_centering=not args.no_sc,
                sc_correction=sc_correction,
                rejection_sampling=config.rejection_sampling,
                **kwargs,
            )

        probe_parameters = {}
        probes = {}
        if args.probe_dir:
            models = engine.model
            for model_chunk in models:
                for module in model_chunk.modules():
                    if hasattr(module, "layer_number") and hasattr(
                        module, "self_attention"
                    ):
                        for name, p in module.named_parameters():
                            if (
                                "norm" in name
                                and p.ndim == 1
                                and (p.numel() == engine.hf_config.hidden_size)
                            ):
                                probe_parameters[
                                    f"layer{module.layer_number}.{name}"
                                ] = p
            original_step = engine.optimizer_step

            def capture_step():
                for name, p in sorted(probe_parameters.items()):
                    from megatron.core import parallel_state as mpu

                    g = torch.zeros_like(p.main_grad).flatten()
                    ownership = torch.zeros_like(g, dtype=torch.int32)
                    optimizers = getattr(
                        engine.optimizer, "chained_optimizers", [engine.optimizer]
                    )
                    for optimizer in optimizers:
                        if p in optimizer.model_param_gbuf_map:
                            bounds = optimizer._get_model_param_range_map(p)["param"]
                            g[bounds.start : bounds.end] = p.main_grad.flatten()[
                                bounds.start : bounds.end
                            ]
                            ownership[bounds.start : bounds.end] += 1
                    group = mpu.get_data_parallel_group(with_context_parallel=True)
                    dist.all_reduce(g, group=group)
                    dist.all_reduce(ownership, group=group)
                    assert (ownership == 1).all(), name
                    probes[name] = g.detach().float().cpu().clone()
                return original_step()

            engine.optimizer_step = capture_step
        engine.train()
        model = engine.model[0]
        parameter = next(model.parameters())

        def local_value(t):
            return t.to_local() if hasattr(t, "to_local") else t

        before = local_value(parameter.detach()).clone()
        stats = engine.train_batch(
            batch, loss_fn, lambda x: x["loss_mask"].count_nonzero()
        )
        torch.cuda.synchronize()
        assert stats["update_successful"] == 1
        assert torch.isfinite(torch.tensor(stats["grad_norm"]))
        assert stats["grad_norm"] > 0
        from megatron.core import parallel_state as mpu

        if not args.no_sc and mpu.is_pipeline_last_stage():
            assert seen and max(seen) > 0
        else:
            assert not seen
        delta = (local_value(parameter.detach()).float() - before.float()).abs().max()
        dist.all_reduce(delta, op=dist.ReduceOp.MAX)
        assert delta > 0
        if args.probe_dir:
            probe_dir = Path(args.probe_dir)
            probe_dir.mkdir(parents=True, exist_ok=True)
            keep = (
                mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_context_parallel_rank() == 0
            )
            if keep:
                assert probes
                torch.save(probes, probe_dir / f"rank{dist.get_rank()}.pt")
        if dist.get_rank() == 0:
            print(
                {
                    "result": "PASS",
                    "backend": "megatron",
                    "tp": tp,
                    "cp": args.cp,
                    "pp": args.pp,
                    "tis_cap": args.tis_cap,
                    "max_parameter_delta": delta.item(),
                    "stats": stats,
                },
                flush=True,
            )
    finally:
        engine.destroy()


if __name__ == "__main__":
    main()
