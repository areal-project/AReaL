import json
import os
import sys

import torch.distributed as dist

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.workflow import RLVRWorkflow


class MinimalPPOTrainer(PPOTrainer):
    """Trainer that collects stats to JSON for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards_history = []

    def _export_and_commit_stats(self, epoch, epoch_step, global_step):
        # Collect stats before committing
        stats = self.actor.export_stats()
        self.rewards_history.append(stats["ppo_actor/task_reward/avg"])


def main() -> None:
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    with MinimalPPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
    ) as trainer:
        workflow = RLVRWorkflow
        workflow_kwargs = dict(
            reward_fn="areal.reward.gsm8k_reward_fn",
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            enable_thinking=False,
        )

        trainer.train(workflow, workflow_kwargs=workflow_kwargs)

        # Save rewards to JSON for test assertions
        if dist.get_rank() == 0:
            with open(os.path.join(config.cluster.fileroot, "rewards.json"), "w") as f:
                json.dump(trainer.rewards_history, f)


if __name__ == "__main__":
    main()
