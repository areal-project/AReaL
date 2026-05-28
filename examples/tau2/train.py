"""Training script for Tau2 benchmark with AReaL proxy mode."""

import sys
import warnings
from typing import Any

from datasets import Dataset
from tau2.registry import registry

from examples.tau2.utils import Tau2PPOConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("Tau2Train")


def get_tau2_dataset(
    domain: str,
    type: str = "rl",
    split: str = "train",
) -> Dataset:
    """Create a HuggingFace Dataset from tau2 task IDs.

    Args:
        domain: The tau2 domain name, e.g., 'retail', 'airline', 'telecom'
        split: Dataset split (e.g., 'train', 'test', 'small')
        type: Dataset type (e.g., 'rl', 'sft'), only 'rl' is supported for now

    Returns:
        Dataset: HuggingFace Dataset containing task_id entries
    """
    assert type == "rl", "Only RL dataset is supported for now"

    splits_loader_fn = registry.get_task_splits_loader(domain)
    if splits_loader_fn is None:
        raise ValueError(f"No task splits loader found for domain {domain}")
    splits = splits_loader_fn()
    if split not in splits:
        raise ValueError(
            f"Split {split} not found for domain {domain}, "
            f"available splits: {list(splits.keys())}"
        )
    task_ids = splits[split]

    dataset_items = [{"task_id": task_id, "split": split} for task_id in task_ids]

    # Duplicate dataset if less than 128 items for efficient batching
    if len(dataset_items) < 128:
        original_items = dataset_items.copy()
        while len(dataset_items) < 128:
            dataset_items.extend(original_items)

    dataset = Dataset.from_list(dataset_items)
    logger.info(
        f"Created dataset with {len(dataset)} items for domain {domain}, split {split}"
    )
    return dataset


def group_filter(x: dict[str, Any]):
    return x["rewards"].mean() <= 0.95


def main(args):
    # Suppress pydantic UserWarning
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    config, _ = load_expr_config(args, Tau2PPOConfig)
    econfig = config.econfig
    domain = econfig.domain

    # Create dataset and dataloaders
    train_dataset = get_tau2_dataset(
        domain=domain,
        type=config.train_dataset.type,
        split=config.train_dataset.path.split("/")[-1],
    )
    valid_dataset = get_tau2_dataset(
        domain=domain,
        type=config.valid_dataset.type,
        split=config.valid_dataset.path.split("/")[-1],
    )

    # Convert econfig to dict for workflow kwargs
    from dataclasses import asdict

    econfig_dict = asdict(econfig)

    # Build workflow kwargs
    workflow_kwargs = dict(
        econfig=econfig_dict,
        gen_args=dict(
            temperature=config.gconfig.temperature,
            max_completion_tokens=config.gconfig.max_new_tokens,
        ),
        timeout=600.0,  # 10 minute timeout
    )

    # Eval workflow with lower temperature
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gen_args"] = dict(
        temperature=0.6,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.tau2.agent.Tau2AgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.tau2.agent.Tau2AgentWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn="examples.tau2.train.group_filter",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
