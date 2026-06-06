"""Full pairwise comparison aggregation strategy that performs all n*(n-1) comparisons."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import AggregationStrategy
from .pairwise_comparison import PairwiseComparisonAggregation


class FullPairwiseComparisonAggregation(PairwiseComparisonAggregation):
    """Aggregation strategy using full pairwise comparisons (all n*(n-1) comparisons).
    
    This performs all possible pairwise comparisons including both orders (i,j) and (j,i),
    resulting in n*(n-1) total comparisons for n solutions.
    All comparison results are stored in metadata for later tournament simulation.
    """

    async def aggregate(self, problem: str, solutions: List[Solution], **kwargs: Any) -> Solution:
        """Select best solution using full pairwise comparisons (all n*(n-1) comparisons).
        
        Args:
            problem: Problem statement
            solutions: List of solutions to compare
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Best solution based on full pairwise comparisons, with all comparison results stored in metadata
        """
        if len(solutions) == 1:
            return solutions[0]

        n = len(solutions)
        wins = [0] * n

        comparison_tasks = []
        comparison_metadata = []  # Store all comparison details
        
        # Perform all n*(n-1) comparisons (including both orders)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # Skip self-comparison
                
                # Always compare (i, j) where i is solution1 and j is solution2
                comparison_tasks.append((i, j, self._compare_pair(problem, solutions[i], solutions[j])))
                comparison_metadata.append({
                    "solution1_idx": i,
                    "solution2_idx": j,
                })

        comparison_results = await asyncio.gather(*[task for _, _, task in comparison_tasks])

        # Extract token usage from comparison results
        total_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        
        comparison_values = []
        for result in comparison_results:
            if isinstance(result, tuple):
                # Result is (comparison_result, token_usage)
                comp_result, usage = result
                comparison_values.append(comp_result)
                if usage:
                    total_token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    total_token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    total_token_usage["total_tokens"] += usage.get("total_tokens", 0)
            else:
                # Backward compatibility if _compare_pair returns just int
                comparison_values.append(result)

        # Check if all comparisons were skipped (all results are 0 due to token limit)
        all_skipped = all(result == 0 for result in comparison_values) and len(comparison_tasks) > 0

        # Process comparison results and store all results
        # result_value = 1 means solution1 (i) wins, 2 means solution2 (j) wins, 0 means tie
        all_comparison_results = []
        for (i, j, _), result_value, metadata in zip(comparison_tasks, comparison_values, comparison_metadata):
            # Record the comparison result
            comparison_record = {
                "solution1_idx": i,
                "solution2_idx": j,
                "result": result_value,  # 1 = solution1 wins, 2 = solution2 wins, 0 = tie
            }
            all_comparison_results.append(comparison_record)
            
            # Update wins count
            if result_value == 1:
                wins[i] += 1  # solution1 (i) wins
            elif result_value == 2:
                wins[j] += 1  # solution2 (j) wins
            else:
                wins[i] += 0.5  # tie
                wins[j] += 0.5

        best_idx = max(range(n), key=lambda i: wins[i])
        best_solution = solutions[best_idx]
        best_solution.metadata["aggregation"] = "full_pairwise_comparison"
        best_solution.metadata["wins"] = wins[best_idx]
        best_solution.metadata["total_comparisons"] = n * (n - 1)
        best_solution.metadata["n_candidates"] = n
        best_solution.metadata["token_usage"] = total_token_usage
        best_solution.metadata["all_comparison_results"] = all_comparison_results  # Store all comparison results
        
        # Store solution correctness information for tournament simulation
        # This allows us to evaluate which solution was actually correct
        solution_correctness = []
        for i, sol in enumerate(solutions):
            # Get is_correct from solution metadata or score
            is_correct = sol.metadata.get("is_correct", None)
            if is_correct is None:
                # Try to get from original_metadata
                original_metadata = sol.metadata.get("original_metadata", {})
                is_correct = original_metadata.get("is_correct", None)
            solution_correctness.append({
                "solution_idx": i,
                "is_correct": bool(is_correct) if is_correct is not None else None,
                "score": sol.score if hasattr(sol, 'score') and sol.score is not None else None,
            })
        best_solution.metadata["solution_correctness"] = solution_correctness
        
        if all_skipped:
            best_solution.metadata["skipped"] = True
            best_solution.metadata["skip_reason"] = "input_tokens_exceeded"

        return best_solution
