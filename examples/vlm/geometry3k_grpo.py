import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.workflow.openai.geometry3k_agent import acc_reward as acc_reward
from areal.workflow.openai.geometry3k_agent import format_reward as format_reward
from areal.workflow.openai.geometry3k_agent import (
    geometry3k_reward_fn as _geometry3k_reward_fn,
)


def geometry3k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    """Keep the RLVR entrypoint compatible while sharing the agent's scoring."""
    return _geometry3k_reward_fn(
        prompt=prompt,
        completions=completions,
        answer=answer,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        **kwargs,
    )


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    workflow_kwargs = dict(
        reward_fn="examples.vlm.geometry3k_grpo.geometry3k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        processor=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
