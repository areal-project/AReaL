# SPDX-License-Identifier: Apache-2.0

"""Single-node Qwen3-14B -> Qwen3-0.6B MOPD example on GSM8K."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from areal import PPOTrainer
from areal.api import AsyncRewardWrapper
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.reward import gsm8k_reward_fn
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


if __name__ == "__main__":
    train(sys.argv[1:])
