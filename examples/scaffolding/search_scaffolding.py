"""
Search Agent Scaffolding Example.

This example demonstrates multi-turn search-based RL training using
the scaffolding framework.  A ``SearchAgentController`` drives a
tool-calling loop (search + visit) expressed as a scaffolding
``Controller``, while ``TraceTrajectoryMaker`` traces each LLM call
for PPO training.

The example uses real web search (via Serper API, requires SERPER_KEY_ID
env var) and basic HTTP fetching for page visits.  An LLM judge is used
for reward computation (the same inference engine is used for both agent
generation and judging).

Usage:
    python -m examples.scaffolding.search_scaffolding \\
        --config examples/scaffolding/search_scaffolding.yaml \\
        +scheduler.type=local experiment_name=areal trial_name=search_scaffolding
"""

import datetime
import sys
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.dataset import get_custom_dataset
from areal.trainer import PPOTrainer
from areal.utils import logging
from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer

from ._compat import (
    NativeGenerationController,
    ScaffoldingLlm,
)
from .controllers import (
    LLMJudgeController,
    TraceTrajectoryMaker,
)
from .search_agent_controller import SearchAgentController
from .workflow import ScaffoldingWorkflow

logger = logging.getLogger("SearchScaffoldingWorkflow")

# Reuse the system prompt from tongyi_deepresearch (search + visit only).
SYSTEM_PROMPT = (
    "You are a deep research assistant. Your core function is to conduct "
    "thorough, multi-source investigations into any topic. You must handle "
    "both broad, open-domain inquiries and queries within specialized academic "
    "fields. For every request, synthesize information from credible, diverse "
    "sources to deliver a comprehensive, accurate, and objective response. "
    "When you have gathered sufficient information and are ready to provide "
    "the definitive response, you must enclose the entire final answer within "
    "<answer></answer> tags.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "search", "description": '
    '"Perform Google web searches then returns a string of the top search '
    'results. Accepts multiple queries.", "parameters": {"type": "object", '
    '"properties": {"query": {"type": "array", "items": {"type": "string", '
    '"description": "The search query."}, "minItems": 1, "description": '
    '"The list of search queries."}}, "required": ["query"]}}}\n'
    '{"type": "function", "function": {"name": "visit", "description": '
    '"Visit webpage(s) and return the summary of the content.", "parameters": '
    '{"type": "object", "properties": {"url": {"type": "array", "items": '
    '{"type": "string"}, "description": "The URL(s) of the webpage(s) to '
    'visit. Can be a single URL or an array of URLs."}, "goal": {"type": '
    '"string", "description": "The specific information goal for visiting '
    'webpage(s)."}}, "required": ["url", "goal"]}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "Current date: "
)


class SearchScaffoldingWorkflow(ScaffoldingWorkflow):
    """ScaffoldingWorkflow for multi-turn search-agent RL training.

    The episode loop delegates to ``SearchAgentController`` (multi-turn
    tool calling) composed with ``TraceTrajectoryMaker`` (trajectory
    tracing) and ``LLMJudgeController`` (LLM-as-judge reward).

    Parameters
    ----------
    reward_fn : Callable | str
        Fallback reward function or importable path (used by parent class
        for non-LLM-judge scenarios; the LLM judge is the primary reward).
    gconfig : GenerationHyperparameters
        Generation hyperparameters.
    tokenizer : PreTrainedTokenizerFast | str
        Tokenizer or path.
    enable_thinking : bool
        Whether to enable thinking tokens.
    max_turns : int
        Maximum number of LLM calls per episode.
    max_total_tokens : int
        Soft token budget for the conversation.
    max_judge_tokens : int
        Maximum tokens for the LLM judge response.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        max_turns: int = 20,
        max_total_tokens: int = 32768,
        max_judge_tokens: int = 8192,
    ):
        super().__init__(
            reward_fn=reward_fn,
            gconfig=gconfig,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.max_judge_tokens = max_judge_tokens

    # ------------------------------------------------------------------
    # Scaffolding construction
    # ------------------------------------------------------------------

    def build_scaffolding_llm(self, engine: InferenceEngine) -> ScaffoldingLlm:
        """Build ``ScaffoldingLlm`` with ``SearchAgentController`` + ``TraceTrajectoryMaker``.

        Uses ``LLMJudgeController`` as the reward controller so that
        answer correctness is determined by the same LLM (via a judge
        prompt) rather than a deterministic string-matching function.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine (worker already initialised by parent).

        Returns
        -------
        ScaffoldingLlm
        """
        stop_strings = ["\n<tool_response>", "<tool_response>"]
        sampling_params: dict[str, Any] = {
            "max_tokens": self.gconfig.max_new_tokens,
            "temperature": self.gconfig.temperature or 1.0,
            "stop": stop_strings,
        }

        self.gen_controller = NativeGenerationController(
            sampling_params=sampling_params,
        )
        self.reward_controller = LLMJudgeController(
            max_judge_tokens=self.max_judge_tokens,
        )

        self.search_controller = SearchAgentController(
            generation_controller=self.gen_controller,
            tokenizer=self.tokenizer,
            max_turns=self.max_turns,
            max_total_tokens=self.max_total_tokens,
        )

        self.trajectory_maker = TraceTrajectoryMaker(
            rollout_controller=self.search_controller,
            reward_controller=self.reward_controller,
        )

        return ScaffoldingLlm(
            self.trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: self.worker},
        )

    # ------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run a single search-agent episode.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine.
        data : dict[str, Any]
            Must contain ``"question"`` and ``"answer"`` keys.

        Returns
        -------
        dict[str, torch.Tensor]
            Trajectory tensors for PPO training.
        """
        if self.worker is None:
            self._lazy_init_scaffolding(engine)

        # Build messages: system prompt + user question
        system_prompt = SYSTEM_PROMPT + datetime.date.today().strftime("%Y-%m-%d")
        question = data.get("question", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # Tokenize the original prompt
        input_ids = list(
            apply_chat_template(
                self.tokenizer,
                data["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )
        prompt_str = self.tokenizer.decode(input_ids)

        # Configure per-episode state on the search controller and reward controller.
        # ScaffoldingLlm.clone() deep-copies the controllers for each request,
        # so the per-episode data is isolated across concurrent episodes.
        self.search_controller.messages = messages
        self.search_controller.input_tokens = input_ids
        self.reward_controller.task_data = data

        # Run the full pipeline (generation + LLM judge reward)
        result = self.scaffolding_llm.generate_async(prompt_str)
        await result

        # Extract trace results
        scaffolding_output = result.outputs[0]
        trace_results = scaffolding_output.data

        # Get the final output text from the last traced interaction
        if trace_results:
            last_interaction = list(trace_results.values())[-1]
            output_str = ""
            if last_interaction.completion is not None:
                output_str = (
                    last_interaction.completion.choices[0].message.content or ""
                )
            # Reward is set by LLMJudgeController via TraceTrajectoryMaker
            reward = float(last_interaction.reward or 0.0)
        else:
            output_str = scaffolding_output.text or ""
            reward = 0.0

        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)

        # Build tensor dict for PPO training
        seq = input_ids + output_tokens
        logprobs = [0.0] * len(seq)
        loss_mask = [0] * len(input_ids) + [1] * len(output_tokens)
        versions = [-1] * len(seq)

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main(args):
    """Main entry point for search scaffolding training."""
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
        reward_fn="examples.scaffolding.search_reward.search_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
        max_turns=10,
        max_total_tokens=8192,
        max_judge_tokens=2048,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.scaffolding.search_scaffolding.SearchScaffoldingWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.scaffolding.search_scaffolding.SearchScaffoldingWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
