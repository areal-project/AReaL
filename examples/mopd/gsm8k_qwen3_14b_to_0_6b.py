# SPDX-License-Identifier: Apache-2.0

"""Single-node Qwen3-14B -> Qwen3-0.6B MOPD example on GSM8K."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from areal import PPOTrainer
from areal.api import AsyncRewardWrapper
from areal.api.alloc_mode import ModelAllocation, ParallelStrategy
from areal.api.cli_args import (
    GRPOConfig,
    conf_as_dict,
    load_expr_config,
    to_structured_cfg,
)
from areal.dataset import get_custom_dataset
from areal.reward import gsm8k_reward_fn
from areal.trainer.mopd.compatibility import (
    model_fingerprint as core_model_fingerprint,
)
from areal.trainer.mopd.compatibility import (
    validate_mopd_model_compatibility,
)
from areal.utils.hf_utils import load_hf_tokenizer

MOPD_ROUTE = "gsm8k"
NO_THINK_SUFFIX = " /no_think"


def dynamic_filter(data: dict[str, Any]) -> bool:
    """Reject nearly all-correct rollout groups, matching the reference run."""
    return data["rewards"].mean() <= 0.95


class GSM8KRewardDistillationAgent:
    """Generate an on-policy response and report its GSM8K verifier reward."""

    def __init__(
        self,
        *,
        reward_timeout: float = 15.0,
        **generation_kwargs: Any,
    ):
        self.generation_kwargs = generation_kwargs
        self._reward = AsyncRewardWrapper(
            gsm8k_reward_fn,
            timeout_seconds=reward_timeout,
            max_workers=1,
            max_retries=1,
        )

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> dict[str, float]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL"),
            api_key=extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"),
            http_client=extra_kwargs.get("http_client"),
            max_retries=0,
        )
        response = await client.chat.completions.create(
            messages=data["messages"],
            model="default",
            **self.generation_kwargs,
        )
        completion = response.choices[0].message.content or ""
        reward = await self._reward(
            prompt=str(data["messages"]),
            completions=completion,
            prompt_ids=[],
            completion_ids=[],
            answer=data["answer"],
        )
        return {response.id: float(reward)}


# Keep existing launch commands and external imports working after the agent rename.
NoRewardDistillationAgent = GSM8KRewardDistillationAgent


def model_fingerprint(path: Path) -> dict[str, object]:
    """Compatibility wrapper for the core MOPD model fingerprint."""
    return core_model_fingerprint(path)


def add_mopd_route(sample: dict[str, Any]) -> dict[str, Any]:
    """Attach the single-teacher route and the reference no-think prompt suffix."""
    update: dict[str, Any] = {"task_type": MOPD_ROUTE}
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return update

    messages = [dict(message) for message in messages]
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str):
            if not content.rstrip().endswith("/no_think"):
                message["content"] = content.rstrip() + NO_THINK_SUFFIX
            break
    update["messages"] = messages
    return update


def load_routed_gsm8k_dataset(
    dataset_config: Any,
    *,
    split: str,
    tokenizer: Any,
):
    """Load either a local parquet mirror or a standard AReaL dataset snapshot."""
    path = Path(dataset_config.path)
    parquet_files = sorted((path / "main").glob(f"{split}-*.parquet"))
    if not parquet_files:
        return get_custom_dataset(
            split=split,
            dataset_config=dataset_config,
            tokenizer=tokenizer,
        ).map(add_mopd_route, desc="Attach Qwen3-14B MOPD route")

    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=[str(parquet_file) for parquet_file in parquet_files],
        split="train",
    )

    def process(sample: dict[str, Any]) -> dict[str, Any]:
        formatted = {
            "messages": [
                {
                    "role": "user",
                    "content": sample["question"]
                    + "\nPlease put your final answer within \\boxed{}.",
                }
            ],
        }
        return formatted | add_mopd_route(formatted)

    dataset = dataset.map(
        process,
        remove_columns=["question"],
        desc=f"Format routed GSM8K {split} split",
    )
    if dataset_config.max_length is not None:
        dataset = dataset.filter(
            lambda sample: len(tokenizer.encode(sample["messages"][0]["content"]))
            <= dataset_config.max_length,
            desc=f"Filter GSM8K {split} prompts by length",
        )
    return dataset


def validate_heterogeneous_models(
    actor_path: Path,
    teacher_paths: dict[str, Path],
) -> dict[str, dict[str, object]]:
    """Require core persistent-teacher and token compatibility invariants."""
    return validate_mopd_model_compatibility(actor_path, teacher_paths)


def dry_run(config_path: Path) -> None:
    """Validate the local topology and checkpoints without starting GPU workers."""
    raw = OmegaConf.load(config_path)
    config = OmegaConf.to_object(to_structured_cfg(raw, GRPOConfig))
    assert isinstance(config, GRPOConfig) and config.mopd is not None

    if config.scheduler.type != "local":
        raise ValueError("This example requires scheduler.type=local")
    if config.cluster.n_nodes != 1 or config.cluster.n_gpus_per_node != 8:
        raise ValueError("This example requires one node with eight GPUs")
    if MOPD_ROUTE not in config.mopd.routes:
        raise ValueError(f"Missing required MOPD route {MOPD_ROUTE!r}")

    actor = ModelAllocation.from_str(config.actor.backend, name="actor")
    teacher = ModelAllocation.from_str(
        config.mopd.teacher_engine.backend, name="mopd-teacher"
    )
    if not ParallelStrategy.parallelism_eq(actor.parallel, teacher.parallel):
        raise ValueError("actor and teacher parallel strategies differ")

    fingerprints = validate_heterogeneous_models(
        Path(config.actor.path),
        {
            teacher_id: Path(spec.path)
            for teacher_id, spec in config.mopd.teachers.items()
        },
    )
    dataset_paths = [config.train_dataset.path]
    if config.valid_dataset is not None:
        dataset_paths.append(config.valid_dataset.path)
    for dataset_path in dataset_paths:
        if not Path(dataset_path).is_dir():
            raise FileNotFoundError(f"Missing local dataset: {dataset_path}")

    report = {
        "scheduler": config.scheduler.type,
        "cluster": {
            "n_nodes": config.cluster.n_nodes,
            "n_gpus_per_node": config.cluster.n_gpus_per_node,
        },
        "parallel": {
            "world_size": actor.parallel.world_size,
            "tp": actor.parallel.tp_size,
            "pp": actor.parallel.pp_size,
            "dp": actor.parallel.dp_size,
        },
        "routes": config.mopd.routes,
        "fingerprints": fingerprints,
        "dataset": config.train_dataset.path,
        "resolved_config": conf_as_dict(config),
    }
    print(json.dumps(report, indent=2, default=str))


def train(argv: list[str]) -> None:
    """Run pure MOPD and report GSM8K rewards as rollout metrics."""
    config, _ = load_expr_config(argv, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = load_routed_gsm8k_dataset(
        config.train_dataset,
        split="train",
        tokenizer=tokenizer,
    )
    assert config.valid_dataset is not None
    valid_dataset = load_routed_gsm8k_dataset(
        config.valid_dataset,
        split="test",
        tokenizer=tokenizer,
    )

    workflow_kwargs = {
        "temperature": config.gconfig.temperature,
        "top_p": config.gconfig.top_p,
        "max_completion_tokens": config.gconfig.max_new_tokens,
    }
    eval_workflow_kwargs = workflow_kwargs | {"temperature": 0.6}

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=(
                "examples.mopd.gsm8k_qwen3_14b_to_0_6b.GSM8KRewardDistillationAgent"
            ),
            workflow_kwargs=workflow_kwargs,
            eval_workflow=(
                "examples.mopd.gsm8k_qwen3_14b_to_0_6b.GSM8KRewardDistillationAgent"
            ),
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn=("examples.mopd.gsm8k_qwen3_14b_to_0_6b.dynamic_filter"),
        )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    known, _ = parser.parse_known_args(argv)
    if known.dry_run:
        if known.config is None:
            raise ValueError("--config is required with --dry-run")
        dry_run(known.config)
        return
    train(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
