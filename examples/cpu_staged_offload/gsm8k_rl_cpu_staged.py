"""Train multi-turn GSM8K with a CPU-staged Qwen3-30B-A3B MoE actor."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizerFast

from examples.cpu_staged_offload.config import (
    CPUStagedGRPOConfig,
    install_cpu_staged_worker_environment,
)
from examples.cpu_staged_offload.engine import CPUStagedPPOTrainer

from areal import workflow_context
from areal.api import AsyncRewardWrapper, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.openai import ArealOpenAI
from areal.reward import get_math_verify_worker
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer


def gsm8k_reward_fn(result: Any, answer: Any) -> float:
    """Verify a GSM8K answer, returning zero for unparsable responses."""
    try:
        worker = get_math_verify_worker()
        return float(worker.verify(str(result), str(answer)))
    except Exception:
        return 0.0


class MultiTurnMathAgent:
    """Retry an incorrect GSM8K answer for a bounded number of turns."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        reward_fn: Callable[[str, str], float | int],
        max_turns: int = 2,
    ) -> None:
        self.gconfig = gconfig
        self.max_turns = max_turns
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    async def run_agent(self, data: dict[str, Any], client: ArealOpenAI) -> float:
        """Run the original multi-turn correction loop and attach rewards."""
        messages = data["messages"].copy()
        reward = 0.0
        for _ in range(self.max_turns):
            response: ChatCompletion = await client.chat.completions.create(
                messages=messages,
                **self.gconfig.to_openai_args_dict(),
            )
            message = response.choices[0].message
            messages.append(message)
            reward = float(
                await self.async_reward_fn(
                    result=message.content, answer=data["answer"]
                )
            )
            client.set_reward(response.id, reward)
            if reward == 1:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your answer is either wrong or not parsable to the reward "
                        "function. You may misunderstand the original question. "
                        "Please carefully read the original question, check the "
                        "previous errors, and try to answer it again."
                    ),
                }
            )
        return reward


class MultiturnRLVRWorkflow(RolloutWorkflow):
    """Preserve the concat-style multi-turn GSM8K rollout semantics."""

    def __init__(
        self,
        reward_fn: Callable[[str, str], float | int] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        export_style: str = "concat",
        max_turns: int = 2,
    ) -> None:
        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        if isinstance(reward_fn, str):
            from areal.utils.dynamic_import import import_from_string

            reward_fn = import_from_string(reward_fn)
        self.tokenizer = tokenizer
        self.export_style = export_style
        if export_style not in {"individual", "concat"}:
            raise ValueError(f"invalid export style: {export_style}")
        self.chat_template_type = "concat" if export_style == "concat" else "hf"
        self.agent = MultiTurnMathAgent(
            gconfig=gconfig.new(n_samples=1),
            reward_fn=reward_fn,
            max_turns=max_turns,
        )

    async def arun_episode(
        self, engine: Any, data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Collect and export one discounted multi-turn trajectory."""
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            chat_template_type=self.chat_template_type,
        )
        reward = await self.agent.run_agent(data=data, client=client)
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)
        client.apply_reward_discount(turn_discount=0.9)
        return client.export_interactions(style=self.export_style)


def main(args: list[str]) -> None:
    """Load GSM8K and launch the custom actor with the standard PPO trainer flow."""
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

    max_turns = config.agent_run_args.get("max_turns", 2)
    workflow_kwargs = {
        "reward_fn": (
            "examples.cpu_staged_offload.gsm8k_rl_cpu_staged.gsm8k_reward_fn"
        ),
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "export_style": config.export_style,
        "max_turns": max_turns,
    }
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6, n_samples=1)
    workflow_path = (
        "examples.cpu_staged_offload.gsm8k_rl_cpu_staged.MultiturnRLVRWorkflow"
    )

    with CPUStagedPPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=workflow_path,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow_path,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
