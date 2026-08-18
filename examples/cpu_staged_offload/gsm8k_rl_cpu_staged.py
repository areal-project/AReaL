"""Train multi-turn GSM8K with a CPU-staged Qwen3-30B-A3B MoE actor."""

from __future__ import annotations

import sys

from examples.cpu_staged_offload.config import (
    CPUStagedGRPOConfig,
    install_cpu_staged_worker_environment,
)
from examples.cpu_staged_offload.engine import CPUStagedPPOTrainer

from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer

WORKFLOW_PATH = "examples.multi_turn_math.gsm8k_rl_mt.MultiturnRLVRWorkflow"
REWARD_PATH = "examples.multi_turn_math.gsm8k_rl_mt.gsm8k_reward_fn"


def main(args: list[str]) -> None:
    """Load GSM8K and launch the custom actor through the standard PPO flow."""
    config, _ = load_expr_config(args, CPUStagedGRPOConfig)
    install_cpu_staged_worker_environment(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = {
        "reward_fn": REWARD_PATH,
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "export_style": config.export_style,
        "max_turns": config.agent_run_args.get("max_turns", 2),
    }
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6, n_samples=1)

    with CPUStagedPPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=WORKFLOW_PATH,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=WORKFLOW_PATH,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
