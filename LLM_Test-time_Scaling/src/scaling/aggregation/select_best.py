"""Select best solution aggregation strategy."""

import random
import re
from typing import Any, List, Optional

from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import AggregationStrategy


class SelectBestAggregation(AggregationStrategy):
    """Aggregation strategy that selects the best solution from candidates."""

    def __init__(
        self, llm_service: LLMService, selection_prompt_template: Optional[PromptTemplate],
        temperature: float = 0.0, reasoning_effort: Optional[str] = "auto"
    ):
        """Initialize select best aggregation.

        Args:
            llm_service: LLM service for selection
            selection_prompt_template: PromptTemplate for selection prompt
            temperature: Temperature for selection (default: 0.0 for deterministic)
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
        """
        super().__init__(llm_service, temperature=temperature, reasoning_effort=reasoning_effort)
        self.selection_prompt_template = selection_prompt_template

    async def aggregate(self, problem: str, solutions: List[Solution], **kwargs: Any) -> Solution:
        """Select the best solution from candidates."""
        if len(solutions) == 1:
            return solutions[0]

        # Randomize solution order if requested (for prompt generation)
        randomize_order = kwargs.get("randomize_order", False)
        if randomize_order:
            # Create shuffled list for prompt, but keep mapping to original
            solution_indices = list(range(len(solutions)))
            random.shuffle(solution_indices)
            solutions_for_prompt = [solutions[i] for i in solution_indices]
            # Create reverse mapping: prompt_index -> original_index
            prompt_to_original = {i: solution_indices[i] for i in range(len(solution_indices))}
        else:
            solutions_for_prompt = solutions
            prompt_to_original = {i: i for i in range(len(solutions))}

        solutions_text = "\n\n".join(
            [f"Solution {i+1}:\n{sol.content}" for i, sol in enumerate(solutions_for_prompt)]
        )

        if self.selection_prompt_template:
            selection_formatted = self.selection_prompt_template.format_with_system(
                problem=problem, solutions=solutions_text
            )
            messages = [
                Message(role="system", content=selection_formatted["system"]),
                Message(role="user", content=selection_formatted["user"])
            ]
        else:
            # Fallback
            messages = [Message(role="user", content=f"Problem: {problem}\n\n{solutions_text}\n\nSelect the best solution:")]

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
        
        # Store detailed information for LLM call
        llm_call_details = [{
            "token_usage": token_usage,
            "reasoning_content": response.reasoning_content or "",
            "content": response.content,
        }]

        selected_prompt_idx = self._parse_selection(response.content, len(solutions))
        
        # Map back to original index if randomized
        selected_idx = prompt_to_original[selected_prompt_idx]
        selected = solutions[selected_idx]
        selected.metadata["aggregation"] = "select_best"
        selected.metadata["selection_reasoning"] = response.content
        selected.metadata["n_candidates"] = len(solutions)
        selected.metadata["token_usage"] = token_usage
        selected.metadata["llm_call_details"] = llm_call_details  # Store detailed info

        return selected

    def _parse_selection(self, response: str, n_solutions: int) -> int:
        """Parse the selected solution index from response."""
        match = re.search(r"[Ss]olution\s*(\d+)", response)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < n_solutions:
                return idx

        for i in range(n_solutions):
            if str(i + 1) in response[:100]:
                return i

        return 0

    def get_strategy_name(self) -> str:
        """Get the name of the aggregation strategy."""
        return "select_best"
