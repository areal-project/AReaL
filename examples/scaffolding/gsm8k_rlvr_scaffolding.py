"""
RLVR (Reinforcement Learning with Verifiable Rewards) Example using Scaffolding Framework.

This example demonstrates how to use the scaffolding framework for RLVR training
on the GSM8K math dataset. The ScaffoldingWorkflow uses AReaL's engine for
generation and scaffolding controllers for reward computation.

Usage:
    python -m examples.scaffolding.gsm8k_rlvr_scaffolding \
        --config examples/scaffolding/gsm8k_rlvr_scaffolding.yaml \
        +scheduler.type=local experiment_name=areal trial_name=scaffolding
"""

import sys

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.dataset import get_custom_dataset
from areal.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer

from ._compat import (
    NativeGenerationController,
    ScaffoldingLlm,
)
from .controllers import (
    PipelineTrajectoryMaker,
    RLVRRewardController,
)
from .workflow import ScaffoldingWorkflow


class GSM8KScaffoldingWorkflow(ScaffoldingWorkflow):
    """ScaffoldingWorkflow customized for GSM8K RLVR training.

    Demonstrates overriding ``build_scaffolding_llm`` to control how the
    ScaffoldingLlm is constructed (e.g., swap controllers or workers).
    """

    def build_scaffolding_llm(self, engine: InferenceEngine) -> ScaffoldingLlm:
        sampling_params = {
            "max_tokens": self.gconfig.max_new_tokens,
            "temperature": self.gconfig.temperature or 1.0,
        }
        self.gen_controller = NativeGenerationController(
            sampling_params=sampling_params
        )
        self.reward_controller = RLVRRewardController(self.reward_fn)
        self.trajectory_maker = PipelineTrajectoryMaker(
            self.gen_controller, self.reward_controller
        )
        return ScaffoldingLlm(
            self.trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: self.worker},
        )


def main(args):
    """Main entry point for RLVR training with scaffolding."""
    config, _ = load_expr_config(args, GRPOConfig)
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

    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
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
            workflow="examples.scaffolding.gsm8k_rlvr_scaffolding.GSM8KScaffoldingWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.scaffolding.gsm8k_rlvr_scaffolding.GSM8KScaffoldingWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
