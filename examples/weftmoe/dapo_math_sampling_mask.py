# SPDX-License-Identifier: Apache-2.0

"""Compare sampling-mask transfer using the historical WeftMoE MathAgent."""

import sys

from math_verify import parse, verify

from examples.multi_turn_math.config import MultiTurnGRPOConfig

from areal import PPOTrainer
from areal.api import AsyncRewardWrapper
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.workflow.openai.math_agent import MathAgent


def reward_fn(completions: str, answer: str) -> float:
    """Preserve the historical baseline's gold-first verifier convention."""
    return float(verify(parse(answer), parse(completions)))


class BaselineMathAgent(MathAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reward_fn = AsyncRewardWrapper(reward_fn)


def main() -> None:
    config, _ = load_expr_config(sys.argv[1:], MultiTurnGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    with PPOTrainer(config, train_dataset=dataset, valid_dataset=None) as trainer:
        trainer.train(
            workflow="examples.weftmoe.dapo_math_sampling_mask.BaselineMathAgent",
            workflow_kwargs={
                "temperature": config.gconfig.temperature,
                "top_p": config.gconfig.top_p,
                "max_completion_tokens": config.gconfig.max_new_tokens,
                # The OpenAI SDK merges extra_body into the HTTP request body.
                # Pass this in BOTH arms; baseline must not silently omit top-k.
                "extra_body": {"top_k": config.gconfig.top_k},
            },
            eval_workflow=None,
            eval_workflow_kwargs=None,
        )


if __name__ == "__main__":
    main()
