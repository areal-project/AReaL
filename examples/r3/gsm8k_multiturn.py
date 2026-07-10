# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Callable

from openai.types.chat import ChatCompletion
from transformers import AutoConfig, PreTrainedTokenizerFast

from examples.multi_turn_math.config import MultiTurnGRPOConfig

from areal import PPOTrainer, workflow_context
from areal.api import AsyncRewardWrapper, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, load_expr_config
from areal.dataset import get_custom_dataset
from areal.engine.r3.config import resolve_r3_moe_config
from areal.experimental.openai import ArealOpenAI
from areal.experimental.openai.types import configure_r3_interactions
from areal.reward import get_math_verify_worker
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer


def _load_tokenizer_for_model(model_path: str) -> PreTrainedTokenizerFast:
    tokenizer = load_hf_tokenizer(model_path)
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    _align_tokenizer_token_id(tokenizer, "eos_token", hf_config.eos_token_id)
    _align_tokenizer_token_id(
        tokenizer,
        "pad_token",
        getattr(hf_config, "pad_token_id", None),
    )
    return tokenizer


def _align_tokenizer_token_id(
    tokenizer: PreTrainedTokenizerFast,
    token_attr: str,
    token_id: int | list[int] | None,
) -> None:
    if isinstance(token_id, list):
        token_id = token_id[0] if token_id else None
    if token_id is None or getattr(tokenizer, f"{token_attr}_id") == token_id:
        return
    token = tokenizer.convert_ids_to_tokens(token_id)
    if token is not None:
        setattr(tokenizer, token_attr, token)


def gsm8k_reward_fn(result: str, answer: str) -> float:
    try:
        worker = get_math_verify_worker()
        return float(worker.verify(str(result), str(answer)))
    except Exception:
        return 0.0


class MultiTurnMathAgent:
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        reward_fn: Callable[[str, str], float | int],
        max_turns: int = 2,
    ) -> None:
        self.gconfig = gconfig
        self.max_turns = max_turns
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    async def run_agent(self, data: dict, client: ArealOpenAI) -> float:
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
                    result=message.content,
                    answer=data["answer"],
                )
            )
            client.set_reward(response.id, reward)
            if reward == 1.0:
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


class R3MultiturnMathWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn: Callable[[str, str], float | int] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        export_style: str = "concat",
        max_turns: int = 2,
        r3_num_moe_layers: int | None = None,
        r3_topk: int | None = None,
    ) -> None:
        if isinstance(tokenizer, str):
            tokenizer = _load_tokenizer_for_model(tokenizer)
        if isinstance(reward_fn, str):
            from areal.utils.dynamic_import import import_from_string

            reward_fn = import_from_string(reward_fn)
        if export_style not in ["individual", "concat"]:
            raise ValueError(f"Invalid export style: {export_style}")

        self.tokenizer = tokenizer
        self.export_style = export_style
        self.chat_template_type = "concat" if export_style == "concat" else "hf"
        self.r3_num_moe_layers = r3_num_moe_layers
        self.r3_topk = r3_topk
        self.agent = MultiTurnMathAgent(
            gconfig=gconfig.new(n_samples=1),
            reward_fn=reward_fn,
            max_turns=max_turns,
        )

    async def arun_episode(self, engine, data: dict):
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            chat_template_type=self.chat_template_type,
        )

        reward = await self.agent.run_agent(data=data, client=client)
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        client.apply_reward_discount(turn_discount=0.9)
        completions_with_reward = client.export_interactions(style=self.export_style)
        configure_r3_interactions(
            completions_with_reward,
            num_moe_layers=self.r3_num_moe_layers,
            topk=self.r3_topk,
        )
        return completions_with_reward


def _r3_workflow_kwargs(config: MultiTurnGRPOConfig) -> dict[str, int]:
    if not config.rollout.return_routed_experts:
        return {}
    hf_config = AutoConfig.from_pretrained(
        config.actor.path,
        trust_remote_code=True,
    )
    r3_config = resolve_r3_moe_config(hf_config)
    return {
        "r3_num_moe_layers": r3_config.num_moe_layers,
        "r3_topk": r3_config.topk,
    }


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, MultiTurnGRPOConfig)
    tokenizer = _load_tokenizer_for_model(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = (
        get_custom_dataset(
            split="test",
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )
        if config.valid_dataset is not None
        else None
    )

    max_turns = config.agent_run_args.get("max_turns", 2)
    workflow_kwargs = dict(
        reward_fn="examples.r3.gsm8k_multiturn.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        export_style=config.export_style,
        max_turns=max_turns,
        **_r3_workflow_kwargs(config),
    )
    eval_workflow_kwargs = dict(
        workflow_kwargs,
        gconfig=config.eval_gconfig.new(temperature=0.6, n_samples=1),
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.r3.gsm8k_multiturn.R3MultiturnMathWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow=(
                "examples.r3.gsm8k_multiturn.R3MultiturnMathWorkflow"
                if valid_dataset is not None
                else None
            ),
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
