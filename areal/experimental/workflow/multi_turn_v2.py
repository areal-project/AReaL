# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any

from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import AsyncRewardWrapper, InferenceEngine, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging, stats_tracker

logger = logging.getLogger("MultiTurnV2Workflow")


class MultiTurnWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int,
        turn_discount: float,
    ):
        self.reward_fn = reward_fn
        # Enforce n_samples=1; grouping is handled by GroupedRolloutWorkflow
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer).new(
            n_samples=1
        )
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

        self.reflection_msg = [
            {
                "role": "user",
                "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                "Please carefully read the original question, check the previous errors, and try to answer it again.",
            }
        ]

    async def arun_episode(
        self, engine: InferenceEngine, data: dict
    ) -> tuple[dict, Any]:
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
        messages = deepcopy(data["messages"])
        # Run multi-turn rollout until correct
        t = reward = 0
        discount = 1
        while reward == 0 and t < self.max_turns:
            # Send generate request to get the response.
            _comp = await client.chat.completions.create(  # type: ignore[arg-type]
                messages=messages,
                frequency_penalty=self.gconfig.frequency_penalty,
                max_completion_tokens=self.gconfig.max_new_tokens,
                stop=self.gconfig.stop,
                store=True,
                temperature=self.gconfig.temperature,
                top_p=self.gconfig.top_p,
            )
            # _comp is an openai ChatCompletion object
            # but we also need to fetch the saved token IDs
            comp = client.get_interaction(_comp.id)
            reward = await self.async_reward_fn(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ),
                _comp.choices[0].message.content,
                comp.model_response.input_tokens,
                comp.model_response.output_tokens,
                **data,
            )
            # Increase counter
            t += 1
            # Amend a prompt if the previous answer is incorrect
            if reward == 0 and t < self.max_turns:
                messages += [
                    {
                        "role": "assistant",
                        "content": _comp.choices[0].message.content,
                    }
                ]
                messages += self.reflection_msg
                discount *= self.turn_discount

        reward = float(reward * discount)

        # Log reward.
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=reward, num_turns=t
        )

        client.set_reward(_comp.id, reward)
        return client.export_interactions()
