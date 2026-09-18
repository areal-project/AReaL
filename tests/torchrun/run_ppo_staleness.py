# SPDX-License-Identifier: Apache-2.0

"""Real FSDP/NCCL optimizer checks for token-age filtering.

Run with torchrun on one or more nodes; --model must be a local HF checkpoint.
Uses synthetic trajectory tensors and real model/optimizer updates, not inference.
"""

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MicroBatchSpec,
    NormConfig,
    OptimizerConfig,
    PPOActorConfig,
)
from areal.engine import FSDPPPOActor
from areal.utils import logging, stats_tracker

logger = logging.getLogger("PPOActor")


def create_tiny_checkpoint(directory: Path) -> None:
    """Create an offline Qwen3 fixture that fits on small test GPUs."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    directory.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"t{index}": index for index in range(4, 1024)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    ).save_pretrained(directory)
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    Qwen3ForCausalLM(config).save_pretrained(directory)


def snapshot(engine: FSDPPPOActor) -> dict[str, torch.Tensor]:
    return {
        name: (param.to_local() if isinstance(param, DTensor) else param)
        .detach()
        .cpu()
        .clone()
        for name, param in engine.model.named_parameters()
    }


def optimizer_steps(engine: FSDPPPOActor) -> list[int]:
    return [
        int(state["step"].item())
        for state in engine.optimizer.state.values()
        if "step" in state
    ]


def trajectory(engine: FSDPPPOActor, version: int, profile: str) -> dict[str, Any]:
    rank = dist.get_rank(engine.data_parallel_group)
    generator = torch.Generator(device=engine.device).manual_seed(100 + rank)
    ids = torch.randint(100, 1000, (2, 16), generator=generator, device=engine.device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, :4] = False
    versions = torch.full_like(ids, version)
    if profile == "mixed":
        versions[0, 4:] = 0  # The short member finished entirely on v0.
        versions[1, 4:8] = 0
        versions[1, 8:12] = 1
    elif profile == "one_rank_empty" and rank == 0:
        versions[:, 4:] = 0
    elif profile == "all_empty":
        versions[:, 4:] = 0
    versions[:, :4] = -1
    data = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids, dtype=torch.bool),
        "loss_mask": mask,
        "versions": versions,
        "rewards": torch.tensor([1.0, 0.0], device=engine.device),
        "is_truncated": torch.zeros(2, dtype=torch.bool, device=engine.device),
    }
    prox = engine.compute_logp([data])[0]
    data["prox_logp"] = prox
    data["logprobs"] = torch.roll(prox, shifts=1, dims=-1)
    return engine.compute_advantages([data])[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank(dist.group.WORLD)
    world_size = dist.get_world_size(dist.group.WORLD)
    assert world_size >= 2
    config = PPOActorConfig(
        experiment_name="test",
        trial_name="staleness",
        path=args.model,
        backend=f"fsdp:d{world_size}p1t1",
        dtype="bfloat16",
        optimizer_dtype="float32",
        attn_impl="sdpa",
        gradient_checkpointing=False,
        ppo_n_minibatches=1,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=16),
        optimizer=OptimizerConfig(
            type="adam", lr=1e-4, weight_decay=0.1, lr_scheduler_type="constant"
        ),
        reward_norm=NormConfig(mean_level="group", std_level=None, group_size=2),
        adv_norm=None,
        kl_ctl=0.0,
        use_decoupled_loss=True,
    )
    engine = FSDPPPOActor(config)
    engine.create_process_group(
        parallel_strategy=ModelAllocation.from_str(config.backend).parallel
    )
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(total_train_epochs=1, dataset_size=32, train_batch_size=4),
    )
    results = []
    train_batch = engine.train_batch
    observed_counts = []

    def record_train_batch(data: dict[str, Any], **kwargs: Any) -> dict[str, float]:
        observed_counts.append(int(kwargs["loss_weight_fn"](data)))
        return train_batch(data, **kwargs)

    engine.train_batch = record_train_batch
    phases = [
        (0, "fresh", 24),
        (1, "fresh", 24),
        (2, "mixed", 24),
        (3, "mixed", 8),
        (3, "one_rank_empty", 0 if rank == 0 else 24),
        (3, "all_empty", None),
        (4, "fresh", 24),
    ]
    try:
        for version, profile, expected_count in phases:
            engine.set_version(version)
            data = trajectory(engine, version, profile)
            before = snapshot(engine)
            steps_before = optimizer_steps(engine)
            old_versions = data["versions"].clone()
            observed_counts.clear()
            optimizer_before = (
                copy.deepcopy(engine.optimizer.state_dict())
                if expected_count is None
                else None
            )
            scheduler_before = copy.deepcopy(engine.lr_scheduler.state_dict())

            engine.ppo_update([data], max_token_staleness=2)
            torch.cuda.synchronize()

            after = snapshot(engine)
            steps_after = optimizer_steps(engine)
            changed = any(
                not torch.equal(before[name], value) for name, value in after.items()
            )
            assert all(torch.isfinite(value).all() for value in after.values())
            torch.testing.assert_close(data["versions"], old_versions, rtol=0, atol=0)
            if expected_count is None:
                assert (
                    not observed_counts and not changed and steps_before == steps_after
                )
                assert engine.lr_scheduler.state_dict() == scheduler_before
                state_after = engine.optimizer.state_dict()
                for key, state in optimizer_before["state"].items():
                    for field, value in state.items():
                        other = state_after["state"][key][field]
                        if isinstance(value, torch.Tensor):
                            value = (
                                value.to_local()
                                if isinstance(value, DTensor)
                                else value
                            )
                            other = (
                                other.to_local()
                                if isinstance(other, DTensor)
                                else other
                            )
                            torch.testing.assert_close(value, other, rtol=0, atol=0)
                        else:
                            assert value == other
            else:
                assert observed_counts == [expected_count], (profile, observed_counts)
                assert changed, (profile, "parameters did not update")
                assert steps_after and all(
                    step == (steps_before[0] + 1 if steps_before else 1)
                    for step in steps_after
                )
            metrics = stats_tracker.export_all(
                reduce_group=engine.data_parallel_group, reset=True
            )
            result = {
                "rank": rank,
                "version": version,
                "profile": profile,
                "target_counts": list(observed_counts),
                "parameters_changed": changed,
                "optimizer_step": steps_after[0],
                "stale_metrics": {
                    key: value for key, value in metrics.items() if "stale" in key
                },
            }
            results.append(result)
            logger.info("STALENESS_TEST %s", json.dumps(result))
        Path(args.output).mkdir(parents=True, exist_ok=True)
        Path(args.output, f"rank-{rank}.json").write_text(json.dumps(results, indent=2))
    finally:
        engine.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
