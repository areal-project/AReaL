# SPDX-License-Identifier: Apache-2.0

"""Single-controller text/vision RL capture and inference-free replay driver."""

import sys

import areal
from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.logging import getLogger


def main(args: list[str]) -> None:
    # The standard loader saves the resolved, redacted effective configuration.
    config, _ = load_expr_config(args, GRPOConfig)
    getLogger("RLTrainer").info("Trajectory example uses AReaL at %s", areal.__file__)
    if config.trajectory_debug.mode == "replay":
        with PPOTrainer(config) as trainer:
            trainer.train()
        return

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split="test",
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
            processor=processor,
        )
    vision = "geometry3k" in config.train_dataset.path
    if config.rollout._version == "v2":
        # The v2 gateway accepts agents with run(), not RolloutWorkflow classes.
        workflow = (
            "areal.workflow.openai.geometry3k_agent.Geometry3KAgent"
            if vision
            else "areal.workflow.openai.math_agent.MathAgent"
        )
        kwargs = dict(
            temperature=config.gconfig.temperature,
            top_p=config.gconfig.top_p,
            max_tokens=config.gconfig.max_tokens,
            max_completion_tokens=config.gconfig.max_new_tokens,
        )
        eval_kwargs = {**kwargs, "temperature": 0.6}
    else:
        kwargs = dict(
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
            enable_thinking=False,
        )
        if vision:
            workflow = "areal.workflow.vision_rlvr.VisionRLVRWorkflow"
            kwargs.update(
                processor=config.tokenizer_path,
                reward_fn="examples.vlm.geometry3k_grpo.geometry3k_reward_fn",
            )
        else:
            workflow = "areal.workflow.rlvr.RLVRWorkflow"
            kwargs["reward_fn"] = "areal.reward.gsm8k.gsm8k_reward_fn"
        eval_kwargs = {**kwargs, "gconfig": config.gconfig.new(temperature=0.6)}

    with PPOTrainer(config, train_dataset, valid_dataset) as trainer:
        trainer.train(
            workflow=workflow,
            workflow_kwargs=kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
