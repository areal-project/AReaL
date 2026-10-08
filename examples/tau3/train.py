"""Train a Qwen3.5 agent on text τ³-Bench tasks with the AReaL proxy."""

import sys
from dataclasses import asdict

from datasets import Dataset
from tau2.registry import registry

from examples.tau3.config import Tau3GRPOConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config


def get_tau3_dataset(domain: str, split: str, min_size: int = 0) -> Dataset:
    """Load task IDs; the simulator owns the task contents and tool environment."""
    task_ids = registry.get_task_splits_loader(domain)()[split]
    if min_size > len(task_ids):
        if not task_ids:
            raise ValueError(f"No tasks found for {domain}/{split}")
        task_ids = (task_ids * ((min_size + len(task_ids) - 1) // len(task_ids)))[
            :min_size
        ]
    return Dataset.from_list(
        [{"task_id": task_id, "split": split} for task_id in task_ids]
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, Tau3GRPOConfig)
    domain = config.econfig.domain
    split = config.train_dataset.path.rsplit("/", 1)[-1]
    train_dataset = get_tau3_dataset(domain, split, config.train_dataset.batch_size)
    workflow_kwargs = {
        "econfig": asdict(config.econfig),
        "gen_args": {
            "temperature": config.gconfig.temperature,
            "max_completion_tokens": config.gconfig.max_new_tokens,
        },
        "timeout": config.econfig.episode_timeout_seconds,
    }
    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=None) as trainer:
        trainer.train(
            workflow="examples.tau3.agent.Tau3AgentWorkflow",
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
