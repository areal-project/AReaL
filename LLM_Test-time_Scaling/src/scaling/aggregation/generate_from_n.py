"""Generate one solution from N candidates aggregation strategy."""

import random
from typing import Any, List, Optional

from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import AggregationStrategy


class GenerateFromNAggregation(AggregationStrategy):
    """Aggregation strategy that generates a new solution from N candidates."""

    def __init__(
        self,
        llm_service: LLMService,
        generation_prompt_template: Optional[PromptTemplate],
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = "auto",
    ):
        """Initialize generate from N aggregation.

        Args:
            llm_service: LLM service for generation
            generation_prompt_template: PromptTemplate for generation prompt
            temperature: Temperature for generation (default: 0.7)
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
        """
        super().__init__(llm_service, temperature=temperature, reasoning_effort=reasoning_effort)
        self.generation_prompt_template = generation_prompt_template

    async def aggregate(self, problem: str, solutions: List[Solution], **kwargs: Any) -> Solution:
        """Generate a new solution synthesizing from N candidates."""
        if len(solutions) == 1:
            return solutions[0]

        # Randomize solution order if requested (for prompt generation)
        randomize_order = kwargs.get("randomize_order", False)
        if randomize_order:
            solutions_for_prompt = solutions.copy()
            random.shuffle(solutions_for_prompt)
        else:
            solutions_for_prompt = solutions

        solutions_text = "\n\n".join(
            [f"Solution {i+1}:\n{sol.content}" for i, sol in enumerate(solutions_for_prompt)]
        )

        if self.generation_prompt_template:
            generation_formatted = self.generation_prompt_template.format_with_system(
                problem=problem, solutions=solutions_text
            )
            messages = [
                Message(role="system", content=generation_formatted["system"]),
                Message(role="user", content=generation_formatted["user"])
            ]
        else:
            # Fallback
            messages = [Message(role="user", content=f"Problem: {problem}\n\nCandidate Solutions:\n{solutions_text}\n\nGenerate the best solution:")]

        # Check input tokens before calling LLM
        exceeds_limit, input_tokens = self._check_input_tokens(messages)
        if exceeds_limit:
            print(f"  Warning: Input tokens ({input_tokens}) exceed limit (120000), skipping aggregation")
            return self._create_skip_solution(problem, solutions, input_tokens=input_tokens)

        # Build kwargs with reasoning effort
        gen_kwargs = self._build_generation_kwargs()
        response = await self.llm_service.generate(messages, **gen_kwargs)

        # Extract token usage information
        usage = response.usage or {}
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        
        # Store detailed information for each LLM call
        llm_call_details = [{
            "token_usage": token_usage,
            "reasoning_content": response.reasoning_content,
            "content": response.content,
        }]

        return Solution(
            content=response.content,
            metadata={
                "aggregation": "generate_from_n",
                "n_candidates": len(solutions),
                "model": response.model,
                "reasoning_content": response.reasoning_content,
                "token_usage": token_usage,
                "llm_call_details": llm_call_details,  # Store detailed info for each call
            },
        )

    def get_strategy_name(self) -> str:
        """Get the name of the aggregation strategy."""
        return "generate_from_n"
