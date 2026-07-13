"""
RLVR Tasks for Scaffolding Framework.

This module provides task definitions for RLVR (Reinforcement Learning with
Verifiable Rewards) that integrate with the scaffolding framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._compat import (
    ChatTask,
    GenerationTask,
    ScaffoldingOutput,
    Task,
)

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward


@dataclass
class RLVRRewardTask(Task):
    """Task for computing RLVR (verifiable) rewards.

    This task contains the necessary information to verify whether a generated
    response is correct and compute the corresponding reward.

    Attributes
    ----------
    prompt_str : str
        The prompt string that was used for generation.
    completion_str : str
        The generated completion string to verify.
    input_tokens : list[int]
        The input token IDs.
    output_tokens : list[int]
        The output token IDs.
    output_logprobs : list[float]
        The log probabilities of output tokens.
    output_versions : list[int]
        The weight versions for output tokens.
    task_data : dict[str, Any]
        Additional task data containing ground truth (e.g., "answer" field).
    interaction : InteractionWithTokenLogpReward
        The interaction object to store the computed reward.
    reward : float
        The computed reward value (output field, set after processing).
    """

    # Input fields
    prompt_str: str = field(default="")
    completion_str: str = field(default="")
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    output_versions: list[int] = field(default_factory=list)
    task_data: dict[str, Any] = field(default_factory=dict)

    # The interaction object to update with the reward
    interaction: InteractionWithTokenLogpReward | None = None

    # Output field
    reward: float | None = None

    @staticmethod
    def create_from_generation_task(
        gen_task: GenerationTask,
        prompt_str: str,
        task_data: dict[str, Any],
        interaction: InteractionWithTokenLogpReward | None = None,
    ) -> RLVRRewardTask:
        """Create a reward task from a completed generation task.

        Parameters
        ----------
        gen_task : GenerationTask
            The completed generation task with output.
        prompt_str : str
            The original prompt string.
        task_data : dict[str, Any]
            Task data containing ground truth answer.
        interaction : InteractionWithTokenLogpReward, optional
            The interaction object to update with reward.

        Returns
        -------
        RLVRRewardTask
            The reward task ready for processing.
        """
        reward_task = RLVRRewardTask(
            prompt_str=prompt_str,
            completion_str=gen_task.output_str or "",
            input_tokens=list(gen_task.input_tokens or []),
            output_tokens=list(gen_task.output_tokens or []),
            output_logprobs=list(
                gen_task.customized_result_fields.get("output_logprobs", [])
            ),
            output_versions=list(
                gen_task.customized_result_fields.get("output_versions", [])
            ),
            task_data=task_data,
            interaction=interaction,
        )
        return reward_task


@dataclass
class TraceGenerationTask(Task):
    """Task for tracing multi-turn generation with ChatTracer.

    This task wraps a ChatTask (or GenerationTask) for tracing purposes.
    The trace results are stored after processing.

    Attributes
    ----------
    generation_task : ChatTask | GenerationTask
        The underlying task to be processed and traced.
    trace_results : dict[str, InteractionWithTokenLogpReward]
        The traced interaction results (output field, set after processing).
    """

    # The underlying generation/chat task
    generation_task: ChatTask | GenerationTask | None = None

    # Output field - trace results after processing
    trace_results: dict[str, InteractionWithTokenLogpReward] | None = None

    @staticmethod
    def create_from_prompt(prompt: str) -> TraceGenerationTask:
        """Create a TraceGenerationTask from a prompt string.

        Parameters
        ----------
        prompt : str
            The input prompt string.

        Returns
        -------
        TraceGenerationTask
            The task ready for processing.
        """
        # Create underlying ChatTask
        chat_task = ChatTask.create_from_prompt(prompt)
        return TraceGenerationTask(generation_task=chat_task)

    @staticmethod
    def create_from_chat_task(chat_task: ChatTask) -> TraceGenerationTask:
        """Create a TraceGenerationTask from an existing ChatTask.

        Parameters
        ----------
        chat_task : ChatTask
            The ChatTask to wrap.

        Returns
        -------
        TraceGenerationTask
            The task ready for processing.
        """
        return TraceGenerationTask(generation_task=chat_task)

    def create_scaffolding_output(self) -> ScaffoldingOutput:
        """Create a ScaffoldingOutput from the trace results.

        Returns
        -------
        ScaffoldingOutput
            The output containing traced results.
        """
        # Return the trace results as the output
        if self.generation_task is not None and hasattr(
            self.generation_task, "output_str"
        ):
            return ScaffoldingOutput(
                text=self.generation_task.output_str or "",
                token_ids=list(self.generation_task.output_tokens or []),
                data=self.trace_results,
            )
        return ScaffoldingOutput(text="", token_ids=[], data=self.trace_results)


@dataclass
class ChatRewardTask(Task):
    """Task for computing rewards on traced chat interactions.

    This task contains a traced InteractionWithTokenLogpReward and is used
    by the reward controller to compute and set rewards.

    Attributes
    ----------
    interaction : InteractionWithTokenLogpReward
        The traced interaction to compute reward for.
    interaction_id : str
        The ID of the interaction.
    reward : float
        The computed reward value (output field, set after processing).
    """

    # The traced interaction
    interaction: InteractionWithTokenLogpReward | None = None

    # Interaction ID for reference
    interaction_id: str = field(default="")

    # Output field
    reward: float | None = None

    @staticmethod
    def create_from_trace_result(
        interaction_id: str,
        interaction: InteractionWithTokenLogpReward,
    ) -> ChatRewardTask:
        """Create a ChatRewardTask from a trace result.

        Parameters
        ----------
        interaction_id : str
            The ID of the interaction.
        interaction : InteractionWithTokenLogpReward
            The traced interaction.

        Returns
        -------
        ChatRewardTask
            The reward task ready for processing.
        """
        return ChatRewardTask(
            interaction=interaction,
            interaction_id=interaction_id,
        )
