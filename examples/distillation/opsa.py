"""Train a language model with reward-free On-Policy Self-Adaptation."""

import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args: list[str]) -> None:
    """Launch OPSA training using prompts only; answer labels are not consumed."""
    config, _ = load_expr_config(args, GRPOConfig)
    if config.actor.opsa is None:
        raise ValueError("OPSA entrypoint requires actor.opsa to be configured")
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
        "reward_fn": "areal.reward.gsm8k.gsm8k_reward_fn",
        "gconfig": config.gconfig,
        "tokenizer": tokenizer,
        "enable_thinking": False,
    }

    eval_workflow_kwargs = workflow_kwargs.copy()

    eval_workflow_kwargs["gconfig"] = config.gconfig.new(
        temperature=0.7,
    )


    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:

        trainer.train(
            workflow=(
                "areal.workflow.self_adaptation."
                "SelfAdaptationWorkflow"
            ),
            workflow_kwargs=workflow_kwargs,

            eval_workflow=(
                "areal.workflow.self_adaptation."
                "SelfAdaptationWorkflow"
            ),
            eval_workflow_kwargs=eval_workflow_kwargs,
        )



if __name__ == "__main__":
    main(sys.argv[1:])
