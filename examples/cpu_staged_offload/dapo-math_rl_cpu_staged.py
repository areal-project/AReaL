"""Train DAPO-Math RL with a CPU-staged Qwen3-30B-A3B MoE actor."""

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

WORKFLOW_PATH = "areal.workflow.openai.math_agent.MathAgent"


def main(args: list[str]) -> None:
    """Load the configured math dataset and run the standard PPO flow."""
    config, _ = load_expr_config(args, CPUStagedGRPOConfig)
    install_cpu_staged_worker_environment(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split="test",
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )

    workflow_kwargs = {
        "temperature": config.gconfig.temperature,
        "top_p": config.gconfig.top_p,
        "max_completion_tokens": config.gconfig.max_new_tokens,
    }
    with CPUStagedPPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=WORKFLOW_PATH,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=None,
            eval_workflow_kwargs=None,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
