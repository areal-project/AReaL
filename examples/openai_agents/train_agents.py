from agents import Agent as OpenAIAgent
from agents import ModelSettings, OpenAIProvider, RunConfig
from agents import Runner as OpenAIRunner
from transformers import PreTrainedTokenizerFast

from examples.openai_agents.config import AgentRLConfig

from areal import PPOTrainer, workflow_context
from areal.api import AsyncRewardWrapper, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer


class OpenAIAgentWrapper:
    def __init__(
        self,
        agent_builder_path: str,
        agent_builder_kwargs: dict,
        reward_fn_path: str,
        temperature: float = 1.0,
        max_tokens: int = 512,
    ):
        self.agent_builder = import_from_string(agent_builder_path)
        self.agent_builder_kwargs = agent_builder_kwargs
        self.async_reward_fn = AsyncRewardWrapper(import_from_string(reward_fn_path))
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def run_agent(self, data, client: ArealOpenAI):
        agent: OpenAIAgent = self.agent_builder(**self.agent_builder_kwargs)
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
            model_settings=ModelSettings(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
        )
        result = await OpenAIRunner.run(
            agent, input=data["messages"][-1]["content"], run_config=run_config
        )
        reward = await self.async_reward_fn(
            completions=result.final_output,
            answer=data["answer"],
            prompt=data.get("prompt"),
            prompt_ids=data.get("prompt_ids"),
            completion_ids=data.get("completion_ids"),
            **{
                k: v
                for k, v in data.items()
                if k
                not in ["messages", "answer", "prompt", "prompt_ids", "completion_ids"]
            },
        )
        client.set_last_reward(reward)
        return reward


class OpenAIAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        agent_builder_path: str,
        agent_builder_kwargs: dict,
        reward_fn_path: str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
    ):
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer

        # Search hyper-parameters
        self.agent = OpenAIAgentWrapper(
            agent_builder_kwargs=agent_builder_kwargs,
            temperature=gconfig.temperature,
            max_tokens=gconfig.max_tokens,
            agent_builder_path=agent_builder_path,
            reward_fn_path=reward_fn_path,
        )

    async def arun_episode(self, engine, data):
        client = ArealOpenAI(
            engine=engine, tokenizer=self.tokenizer, tool_call_parser="qwen25"
        )

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
        agent_builder_path=config.agent_builder_path,
        agent_builder_kwargs=config.agent_builder_kwargs,
        reward_fn_path=config.reward_fn_path,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.openai_agents.train_agents.OpenAIAgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.openai_agents.train_agents.OpenAIAgentWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
