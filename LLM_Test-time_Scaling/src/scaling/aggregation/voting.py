"""Voting aggregation strategy."""

import re
from collections import Counter
from typing import Any, List, Optional

from ..base import Solution
from .base import AggregationStrategy


class VotingAggregation(AggregationStrategy):
    """Aggregation strategy using majority voting on final answers."""

    def __init__(self, answer_extractor: callable = None, llm_service: Any = None):
        """Initialize voting aggregation.

        Args:
            answer_extractor: Function to extract final answer from solution
                             Default: math-optimized extractor for IMO problems
            llm_service: Not used, but required by base class (can be None)
        """
        super().__init__(llm_service=llm_service or object(), temperature=0.0)
        self.answer_extractor = answer_extractor or self._math_answer_extractor

    async def aggregate(self, problem: str, solutions: List[Solution], **kwargs: Any) -> Solution:
        """Select solution by majority voting on final answers."""
        if len(solutions) == 1:
            return solutions[0]

        answers = [self.answer_extractor(sol.content) for sol in solutions]
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]

        for i, answer in enumerate(answers):
            if answer == most_common_answer:
                best_solution = solutions[i]
                best_solution.metadata["aggregation"] = "voting"
                best_solution.metadata["vote_count"] = count
                best_solution.metadata["total_votes"] = len(solutions)
                best_solution.metadata["answer_distribution"] = dict(answer_counts)
                return best_solution

        return solutions[0]

    def _math_answer_extractor(self, solution: str) -> str:
        """Math-optimized answer extraction method for IMO problems.
        
        Extracts answers using patterns common in mathematical solutions:
        - \\boxed{answer}
        - Final Answer: answer
        - Answer: answer
        - The answer is answer
        - Last line (common for math problems)
        """
        solution = solution.strip()

        # Try LaTeX boxed format (common in IMO solutions)
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try "Final Answer:" pattern
        if "Final Answer:" in solution:
            parts = solution.split("Final Answer:")
            answer = parts[-1].strip()
            # Extract just the answer part (before newline or period)
            answer = answer.split('\n')[0].split('.')[0].strip()
            if answer:
                return answer

        # Try "Answer:" pattern
        if "Answer:" in solution:
            parts = solution.split("Answer:")
            answer = parts[-1].strip()
            answer = answer.split('\n')[0].split('.')[0].strip()
            if answer:
                return answer

        # Try "the answer is" pattern
        answer_match = re.search(r"(?:the answer is|answer is)\s+([^\n\.]+)", solution, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()

        # Fallback: use last non-empty line
        lines = solution.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                return line

        return solution

    def _default_answer_extractor(self, solution: str) -> str:
        """Default answer extraction method (kept for backward compatibility)."""
        return self._math_answer_extractor(solution)

    def get_strategy_name(self) -> str:
        """Get the name of the aggregation strategy."""
        return "voting"
