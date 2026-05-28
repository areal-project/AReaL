from dataclasses import dataclass, field

from camel.agents import ChatAgent
from transformers import PreTrainedTokenizerFast

from areal import PPOTrainer, workflow_context
from areal.api import AsyncRewardWrapper, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.camel.openai_model import AReaLOpenAICompatibleModel
from areal.experimental.openai import ArealOpenAI
from areal.reward import get_math_verify_worker
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer


@dataclass
class AgentRLConfig(GRPOConfig):
    max_tokens_per_trajectory: int = field(
        default=32768,
        metadata={
            "help": "Maximum number of tokens per trajectory. By default max_tokens_per_trajectory=32768."
        },
    )


def gsm8k_reward_fn(result, answer):
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(result), str(answer))
    except Exception:
        return 0.0


class CamelMathAgent:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_tokens_per_turn: int = 1024,
        max_total_tokens: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_total_tokens = max_total_tokens
        self.async_reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        model_config_dict = {"max_tokens": self.max_total_tokens}
        rollout_engine_request_timeout = client.engine.config.request_timeout

        messages = data["messages"].copy()
        agent = ChatAgent(
            model=AReaLOpenAICompatibleModel(
                openai_client=client,
                tokenizer=self.tokenizer,
                model_type="areal",
                model_config_dict=model_config_dict,
            ),
            step_timeout=rollout_engine_request_timeout,
        )
        response = await agent.astep(messages[-1]["content"])
        content = response.msg.content
        reward = await self.async_reward_fn(result=content, answer=data["answer"])
        client.set_last_reward(reward)
        return reward


class CamelRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        max_tokens: int = 32768,
    ):
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        # Search hyper-parameters
        self.agent = CamelMathAgent(
            tokenizer=self.tokenizer,
            max_tokens_per_turn=self.gconfig.max_new_tokens,
            max_total_tokens=max_tokens,
        )

    async def arun_episode(self, engine, data):
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)

        # Collect single trajectory
        reward = await self.agent.run_agent(
            data=data,
            client=client,
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        client.apply_reward_discount(turn_discount=0.9)
        interactions_with_reward = client.export_interactions(style="individual")
        return interactions_with_reward


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load dataset
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        max_tokens=config.max_tokens_per_trajectory,
    )

    # Create trainer (no valid_dataset for this example)
    with PPOTrainer(config, train_dataset, valid_dataset=None) as trainer:
        # Run training
        trainer.train(
            workflow="examples.camel.train.CamelRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow=None,
        )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
