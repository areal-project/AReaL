"""Rollout-only script for Tau2 benchmark using RolloutControllerV2.

This example demonstrates how to run rollouts (data generation) without
training, using the gateway HTTP stack to route inference requests.

Usage:
    python3 examples/experimental/inference_service/tau2_rollout.py \
        --config examples/experimental/inference_service/tau2_rollout.yaml \
        econfig.user_llm_base_url=http://localhost:8000/v1/
"""

from __future__ import annotations

import sys
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from datasets import Dataset

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    BaseExperimentConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
    TrainDatasetConfig,
    load_expr_config,
)
from areal.experimental.inference_service.controller.controller import (
    RolloutControllerV2,
)
from areal.utils import logging

logger = logging.getLogger("Tau2GatewayRollout")


# ---------------------------------------------------------------------------
# Tau2 environment config (copied from examples/tau2/utils.py)
# ---------------------------------------------------------------------------


@dataclass
class Tau2EnvConfig:
    """Environment configuration for Tau2 benchmark."""

    domain: str = field(
        default="telecom",
        metadata={
            "help": "The tau2 domain name, e.g., 'retail', 'airline', 'telecom'."
        },
    )
    max_steps: int = field(
        default=100, metadata={"help": "Maximum number of steps per episode."}
    )
    add_thinking_tool: bool = field(
        default=False, metadata={"help": "Whether to add a thinking tool."}
    )
    solo_mode: bool = field(
        default=False, metadata={"help": "Whether to use solo mode."}
    )
    user_llm_base_url: str | None = field(
        default=None,
        metadata={"help": "The base URL of the user LLM."},
    )
    user_llm: str | None = field(
        default=None,
        metadata={"help": "The user LLM to use, default to the gpt-4.1 model."},
    )
    user_llm_args: dict | None = field(
        default=None, metadata={"help": "The arguments for the user LLM."}
    )
    turn_discount: float = field(
        default=1.0, metadata={"help": "Discount factor for turn-based learning."}
    )
    invalid_format_penalty: float = field(
        default=0.1, metadata={"help": "Penalty for invalid format in completions."}
    )


# ---------------------------------------------------------------------------
# Tau2 dataset helper (copied from examples/tau2/train.py)
# ---------------------------------------------------------------------------


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
    from tau2.registry import registry

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Tau2GatewayRolloutConfig(BaseExperimentConfig):
    """Configuration for Tau2 rollout-only with RolloutControllerV2."""

    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    model_path: str = ""
    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    train_dataset: TrainDatasetConfig = field(default_factory=TrainDatasetConfig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    config, _ = load_expr_config(argv, Tau2GatewayRolloutConfig)
    econfig = config.econfig
    rollout_cfg = deepcopy(config.rollout)
    rollout_cfg.model = config.model_path

    # --- Dataset ---
    train_dataset = get_tau2_dataset(
        domain=econfig.domain,
        type=config.train_dataset.type,
        split=config.train_dataset.path.split("/")[-1],
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=0,  # in-process; tau2 dataset is lightweight
    )

    # --- Scheduler ---
    from areal.infra.scheduler.local import LocalScheduler
    from areal.infra.scheduler.slurm import SlurmScheduler

    sched_type = config.scheduler.type
    if sched_type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif sched_type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise NotImplementedError(f"Unknown scheduler type: {sched_type}")

    # --- Controller ---
    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    if rollout_alloc.backend == "sglang":
        server_args = asdict(config.sglang)
    elif rollout_alloc.backend == "vllm":
        server_args = asdict(config.vllm)
    else:
        raise ValueError(f"Unsupported rollout backend: {rollout_alloc.backend}")

    ctrl = RolloutControllerV2(config=rollout_cfg, scheduler=scheduler)
    ctrl.initialize(
        role="rollout",
        server_args=server_args,
    )

    # --- Workflow kwargs (identical to examples/tau2/train.py) ---
    econfig_dict = asdict(econfig)
    workflow_kwargs: dict[str, Any] = dict(
        econfig=econfig_dict,
        gen_args=dict(
            temperature=config.gconfig.temperature,
            max_completion_tokens=config.gconfig.max_new_tokens,
        ),
        timeout=600.0,
    )

    # --- Rollout loop ---
    try:
        logger.info("Starting rollout loop")
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            # DataLoader yields column-oriented dicts; convert to list of row dicts
            keys = list(batch.keys())
            batch_size = len(batch[keys[0]])
            data = [{k: batch[k][i] for k in keys} for i in range(batch_size)]

            result = ctrl.rollout_batch(
                data=data,
                workflow="examples.tau2.agent.Tau2AgentWorkflow",
                workflow_kwargs=workflow_kwargs,
            )
            if result:
                # Localize RTensors and collect rewards across the batch
                import torch

                from areal.infra.rpc.rtensor import RTensor

                batch_rewards = []
                for traj in result:
                    local_traj = RTensor.localize(traj)
                    batch_rewards.append(local_traj["rewards"])
                all_rewards = torch.cat(batch_rewards, dim=0)
                logger.info(
                    "Batch %d: n_trajs=%d, rewards=%s, avg_reward=%.4f",
                    batch_idx,
                    len(result),
                    all_rewards,
                    all_rewards.mean().item(),
                )
            else:
                logger.warning("Batch %d: empty result (all rejected?)", batch_idx)
            batch_count += 1
        logger.info("Rollout complete (%d batches)", batch_count)
    finally:
        ctrl.destroy()
        scheduler.delete_workers(None)


if __name__ == "__main__":
    main(sys.argv[1:])
