"""No feedback reflection strategy."""

import asyncio
from typing import Any, List, Optional

from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import ReflectionStrategy


class NoFeedbackReflection(ReflectionStrategy):
    """Reflection strategy that generates new solutions based on previous ones without evaluation feedback.

    This strategy asks the LLM to improve/refine the previous solution without providing
    any explicit evaluation or feedback about what's wrong with it.
    """

    def __init__(
        self, llm_service: LLMService, prompt_template: Optional[PromptTemplate],
        temperature: float = 1.0, reasoning_effort: Optional[str] = "auto"
    ):
        """Initialize no-feedback reflection.

        Args:
            llm_service: LLM service for generation
            prompt_template: PromptTemplate object with system and user prompts
                Should include placeholders for: {problem}, {solution}
            temperature: Sampling temperature for generation (default: 1.0)
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
        """
        super().__init__(llm_service, temperature=temperature, reasoning_effort=reasoning_effort)
        self.prompt_template = prompt_template

    async def reflect(
        self, problem: str, solutions: List[Solution], **kwargs: Any
    ) -> List[Solution]:
        """Generate new solutions based on previous solutions without evaluation feedback."""
        temperature = kwargs.get("temperature", self.temperature)

        tasks = [self._reflect_single(problem, solution, temperature) for solution in solutions]
        return await asyncio.gather(*tasks)

    async def _reflect_single(
        self, problem: str, solution: Solution, temperature: float
    ) -> Solution:
        """Generate a new solution based on a previous solution without feedback."""
        if self.prompt_template:
            # Use template with system and user prompts
            formatted = self.prompt_template.format_with_system(
                problem=problem, solution=solution.content
            )
            messages = [
                Message(role="system", content=formatted["system"]),
                Message(role="user", content=formatted["user"])
            ]
        else:
            # Fallback: use simple user prompt
            messages = [Message(role="user", content=f"Problem: {problem}\n\nPrevious Solution: {solution.content}\n\nGenerate an improved solution:")]

        # Build kwargs with reasoning effort
        gen_kwargs = self._build_generation_kwargs(temperature=temperature)
        response = await self.llm_service.generate(messages, **gen_kwargs)

        return Solution(
            content=response.content,
            metadata={
                "strategy": "no_feedback",
                "model": response.model,
                "original_solution": solution.content,
                "reasoning_content": response.reasoning_content,
                "usage": response.usage,  # Store usage including timing
            },
        )

    def get_strategy_name(self) -> str:
        """Get the name of the reflection strategy."""
        return "no_feedback"
