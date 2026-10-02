# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.engine import MegatronEngine


def digest(value):
    """Hash all state tensors, including FP32 masters, without retaining copies."""
    sha = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            sha.update(str((obj.dtype, tuple(obj.shape))).encode())
            sha.update(
                obj.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                sha.update(str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            for element in obj:
                visit(element)
        else:
            sha.update(repr(obj).encode())

    visit(value)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="megatron:d2p1t1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = TrainEngineConfig(
        experiment_name="muon-validation",
        trial_name="three-steps-resume",
        path=args.model,
        backend=args.backend,
        dtype="bfloat16",
        disable_dropout=True,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=128),
        optimizer=OptimizerConfig(
            type="muon", lr=1e-3, warmup_steps=0, lr_scheduler_type="linear"
        ),
    )
    config.megatron.ddp.use_distributed_optimizer = False
    config.megatron.use_checkpoint_opt_param_scheduler = True
    engine = MegatronEngine(config)
    engine.create_process_group(ModelAllocation.from_str(args.backend).parallel)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=40, train_batch_size=4)
    engine.initialize(None, ft_spec)
    engine.train()
    rank = dist.get_rank()
    parameters = dict(engine.model.named_parameters())
    names = {id(p): n for n, p in parameters.items()}
    groups = {}
    for opt in engine.optimizer.chained_optimizers:
        kind = type(opt.optimizer).__name__
        original = getattr(opt, "float16_groups", None)
        if original is None:
            original = [g["params"] for g in opt.param_groups]
        else:
            original = original + getattr(opt, "fp32_from_fp32_groups", [])
        groups[kind] = [names[id(p)] for group in original for p in group]
    muon_names = [n for kind, ns in groups.items() if "Muon" in kind for n in ns]
    adam_names = [n for kind, ns in groups.items() if "Adam" in kind for n in ns]
    assert muon_names and adam_names, groups
    assert all(parameters[name].ndim == 2 for name in muon_names)
    assert any("embedding" in n for n in adam_names), groups
    assert any("norm" in n for n in adam_names), groups
    assert any(n.endswith("bias") for n in adam_names), groups
    assert not any(
        "embedding" in n or "norm" in n or n.endswith("bias") for n in muon_names
    ), groups

    # DP ranks consume distinct data; TP peers consume identical data.
    generator = torch.Generator(device="cpu").manual_seed(
        23 + mpu.get_data_parallel_rank()
    )
    tokens = torch.randint(100, 10000, (4, 32), generator=generator, dtype=torch.long)
    batch = {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens, dtype=torch.bool),
    }
    metrics = []

    def model_digest():
        return digest(dict(engine.model.named_parameters()))

    def update():
        losses = []

        def loss_fn(logprobs, entropy, input_data, **kwargs):
            loss = -logprobs.mean()
            losses.append(float(loss.detach()))
            return loss

        result = engine.train_batch(batch, loss_fn, lambda x: x["cu_seqlens"][-1])
        assert result["update_successful"] == 1, result
        assert losses and all(math.isfinite(x) for x in losses), losses
        assert all(math.isfinite(result[k]) for k in ("grad_norm", "lr")), result
        engine.lr_scheduler_step()
        assert (
            torch.stack([torch.isfinite(p).all() for p in parameters.values()])
            .all()
            .item()
        )
        current = model_digest()
        replicas = [None] * mpu.get_data_parallel_world_size()
        dist.all_gather_object(replicas, current, group=mpu.get_data_parallel_group())
        assert len(set(replicas)) == 1, "DP parameters diverged after all-reduce"
        result["loss"] = sum(losses) / len(losses)
        result["scheduler_steps"] = engine.lr_scheduler.num_steps
        metrics.append(result)
        return current

    initial = model_digest()
    initial_muon = digest({name: parameters[name] for name in muon_names})
    for _ in range(3):
        step3_model = update()
    assert initial != step3_model, "weights did not change"
    step3_muon = digest({name: parameters[name] for name in muon_names})
    assert initial_muon != step3_muon, "Muon matrix weights did not change"
    momentum = [
        state["momentum_buffer"]
        for opt in engine.optimizer.chained_optimizers
        if "Muon" in type(opt.optimizer).__name__
        for state in opt.optimizer.state.values()
    ]
    assert momentum and all(torch.isfinite(t).all().item() for t in momentum)
    assert any(torch.count_nonzero(t).item() > 0 for t in momentum)
    step3_optimizer = digest(engine.optimizer.state_dict())
    step3_scheduler = engine.lr_scheduler.state_dict().copy()
    checkpoint = str(output / "checkpoint-step3")
    Path(checkpoint).mkdir(parents=True, exist_ok=True)
    engine.checkpointer.save_checkpoint(checkpoint)
    step4_model = update()
    step4_optimizer = digest(engine.optimizer.state_dict())
    step4_scheduler = engine.lr_scheduler.state_dict().copy()
    assert step4_model != step3_model
    # Exercise load-time allocation with a fresh optimizer, so stale momentum
    # buffers cannot hide missing state in the checkpoint template.
    engine._create_optimizer(ft_spec)
    engine._set_optimizer_grad_scale_func()
    assert engine.lr_scheduler.num_steps == 0
    assert digest(engine.optimizer.state_dict()) != step3_optimizer
    engine.checkpointer.load_checkpoint(checkpoint)
    assert model_digest() == step3_model, "model did not restore to step 3"
    assert digest(engine.optimizer.state_dict()) == step3_optimizer, (
        "optimizer state was not restored"
    )
    assert engine.lr_scheduler.state_dict() == step3_scheduler
    resumed_model = update()
    assert resumed_model == step4_model, (
        "resumed fourth update differs from uninterrupted update"
    )
    assert digest(engine.optimizer.state_dict()) == step4_optimizer
    assert engine.lr_scheduler.state_dict() == step4_scheduler
    report = {
        "rank": rank,
        "backend": args.backend,
        "model": args.model,
        "muon_path": "replicated native TensorParallelMuon with Adam fallback",
        "gradient_sync": "Megatron DDP all-reduce",
        "groups": groups,
        "parameter_shapes": {name: list(p.shape) for name, p in parameters.items()},
        "metrics": metrics,
        "state_restored": True,
        "fourth_update_identical": True,
        "initial_muon_sha256": initial_muon,
        "step3_muon_sha256": step3_muon,
        "muon_momentum_tensors": len(momentum),
        "step3_optimizer_sha256": step3_optimizer,
        "step4_optimizer_sha256": step4_optimizer,
        "step3_scheduler": step3_scheduler,
        "step4_scheduler": step4_scheduler,
    }
    with (output / f"rank-{rank}.json").open("w") as file:
        json.dump(report, file, indent=2)
    engine.destroy()


if __name__ == "__main__":
    main()
