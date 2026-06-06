"""Pairwise comparison aggregation strategy."""

import asyncio
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import AggregationStrategy


class PairwiseComparisonAggregation(AggregationStrategy):
    """Aggregation strategy using pairwise comparisons to rank solutions."""

    def __init__(
        self,
        llm_service: LLMService,
        comparison_prompt_template: Optional[PromptTemplate],
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = "auto",
    ):
        """Initialize pairwise comparison aggregation.

        Args:
            llm_service: LLM service for comparisons
            comparison_prompt_template: PromptTemplate for comparison prompt
            temperature: Temperature for comparisons (default: 0.0 for deterministic)
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
        """
        super().__init__(llm_service, temperature=temperature, reasoning_effort=reasoning_effort)
        self.comparison_prompt_template = comparison_prompt_template

    async def aggregate(self, problem: str, solutions: List[Solution], **kwargs: Any) -> Solution:
        """Select best solution using pairwise comparisons."""
        if len(solutions) == 1:
            return solutions[0]

        # Randomize comparison order if requested (for prompt generation)
        # Each solution should appear as solution1 in approximately half of its comparisons
        randomize_order = kwargs.get("randomize_order", False)
        
        n = len(solutions)
        wins = [0] * n

        # For large numbers of solutions (16 or 32), use partial comparison:
        # each solution compares with k=5 random opponents
        use_partial_comparison = (n == 32)
        k_opponents = 8 if use_partial_comparison else None
        
        comparison_tasks = []
        
        if use_partial_comparison:
            # Partial comparison: each solution compares with k random opponents as initiator
            # Use a fixed random seed based on problem hash for reproducibility
            import hashlib
            problem_hash = int(hashlib.md5(problem.encode()).hexdigest()[:8], 16)
            rng = random.Random(problem_hash)
            
            # Track which pairs have been compared to avoid duplicates
            compared_pairs = set()  # Set of (min(i,j), max(i,j)) tuples
            # Track how many times each solution has been initiator
            initiator_counts = [0] * n
            
            # For each solution, randomly select k opponents to compare with
            for i in range(n):
                # Get all possible opponents
                all_opponents = [j for j in range(n) if j != i]
                
                # Randomly select k opponents (or all if k >= n-1)
                if k_opponents >= len(all_opponents):
                    selected_opponents = all_opponents
                else:
                    selected_opponents = rng.sample(all_opponents, k_opponents)
                
                # Create comparison tasks for selected opponents
                for j in selected_opponents:
                    # Randomize which solution appears first (solution1) vs second (solution2)
                    if randomize_order:
                        first_is_i = rng.choice([True, False])
                    else:
                        # Default: always compare (i, j) where i < j
                        first_is_i = (i < j)
                    
                    # Record that i is the initiator of this comparison
                    initiator = i
                    initiator_counts[initiator] += 1
                    
                    if first_is_i:
                        # Compare solution i as solution1, solution j as solution2
                        comparison_tasks.append((i, j, True, initiator, self._compare_pair(problem, solutions[i], solutions[j])))
                    else:
                        # Compare solution j as solution1, solution i as solution2 (reversed order)
                        comparison_tasks.append((i, j, False, initiator, self._compare_pair(problem, solutions[j], solutions[i]))) 
        else:
            # Full pairwise comparison: compare all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    # Randomize which solution appears first (solution1) vs second (solution2)
                    if randomize_order:
                        # Randomly decide whether to compare (i, j) or (j, i)
                        # This ensures each solution appears as solution1 in ~50% of comparisons
                        first_is_i = random.choice([True, False])
                    else:
                        # Default: always compare (i, j) where i < j
                        first_is_i = True
                    
                    if first_is_i:
                        # Compare solution i as solution1, solution j as solution2
                        comparison_tasks.append((i, j, True, self._compare_pair(problem, solutions[i], solutions[j])))
                    else:
                        # Compare solution j as solution1, solution i as solution2 (reversed order)
                        comparison_tasks.append((i, j, False, self._compare_pair(problem, solutions[j], solutions[i])))

        # Extract tasks from comparison_tasks (format differs for partial vs full comparison)
        if use_partial_comparison:
            tasks = [task for _, _, _, _, task in comparison_tasks]
        else:
            tasks = [task for _, _, _, task in comparison_tasks]
        comparison_results = await asyncio.gather(*tasks)

        # Extract token usage from comparison results
        total_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        token_usage_list = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": []
        }
        
        comparison_values = []
        reasoning_contents = []  # Store reasoning for each comparison
        contents = []
        for result in comparison_results:
            if isinstance(result, tuple):
                if len(result) == 4:
                    # Result is (comparison_result, token_usage, reasoning_content)
                    comp_result, usage, reasoning, content = result
                    comparison_values.append(comp_result)
                    reasoning_contents.append(reasoning)
                    contents.append(content)
                elif len(result) == 2:
                    # Result is (comparison_result, token_usage) - backward compatibility
                    comp_result, usage = result
                    comparison_values.append(comp_result)
                    reasoning_contents.append("")
                    contents.append("")
                else:
                    # Fallback
                    comparison_values.append(result[0] if result else 0)
                    reasoning_contents.append("")
                    contents.append("")
                
                if usage:
                    total_token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    total_token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    total_token_usage["total_tokens"] += usage.get("total_tokens", 0)
                    token_usage_list["prompt_tokens"].append(usage.get("prompt_tokens", 0))
                    token_usage_list["completion_tokens"].append(usage.get("completion_tokens", 0))
                    token_usage_list["total_tokens"].append(usage.get("total_tokens", 0))
            else:
                # Backward compatibility if _compare_pair returns just int
                comparison_values.append(result)
                reasoning_contents.append("")
                contents.append("")

        # Check if all comparisons were skipped (all results are 0 due to token limit)
        all_spare = all(result == 0 for result in comparison_values) and len(comparison_tasks) > 0
        all_skipped = (all_spare and total_token_usage["completion_tokens"] == 0)
        
        # For partial comparison, track which comparisons count toward wins
        # Only count the first k_opponents comparisons where each solution is the initiator
        if use_partial_comparison:
            # Track how many comparisons each solution has counted as initiator
            initiator_comparison_counts = [0] * n
        
        # Process comparison results and store all results
        # result_value = 1 means solution1 (first in prompt) wins, 2 means solution2 (second in prompt) wins
        all_comparison_results = []
        
        if use_partial_comparison:
            # Partial comparison: only count wins for initiator, and only first k_opponents comparisons per initiator
            for (i, j, first_is_i, initiator, _), result_value, reasoning, content in zip(comparison_tasks, comparison_values, reasoning_contents, contents):
                # Determine which solution actually won (based on original indices, not prompt order)
                if first_is_i:
                    # Comparison was (i, j): solution i was solution1, solution j was solution2
                    if result_value == 1:
                        winner_idx = i
                    elif result_value == 2:
                        winner_idx = j
                    else:
                        winner_idx = None  # tie
                else:
                    # Comparison was (j, i): solution j was solution1, solution i was solution2
                    if result_value == 1:
                        winner_idx = j
                    elif result_value == 2:
                        winner_idx = i
                    else:
                        winner_idx = None  # tie
                
                # Only count wins for the initiator, and only if it hasn't exceeded k_opponents comparisons
                if initiator_comparison_counts[initiator] < k_opponents:
                    initiator_comparison_counts[initiator] += 1
                    
                    # Count wins for the initiator's comparison
                    if winner_idx == initiator:
                        wins[initiator] += 1
                    elif winner_idx is not None:
                        # The opponent won, initiator gets 0 (no win)
                        pass
                    else:
                        # Tie
                        wins[initiator] += 0.5
                
                # Record the comparison result with original indices and prompt order
                if first_is_i:
                    solution1_idx = i
                    solution2_idx = j
                else:
                    solution1_idx = j
                    solution2_idx = i
                
                comparison_record = {
                    "solution1_idx": solution1_idx,  # Index of solution1 (first in prompt)
                    "solution2_idx": solution2_idx,  # Index of solution2 (second in prompt)
                    "prompt_order": "i_j" if first_is_i else "j_i",  # Which order was used in prompt
                    "initiator": initiator,  # Index of solution that initiated this comparison
                    "result": result_value,  # 1 = solution1 (in prompt) wins, 2 = solution2 (in prompt) wins, 0 = tie
                    "winner_idx": winner_idx,  # Winner's original index (None if tie)
                    "reasoning": reasoning,  # LLM reasoning content for this comparison
                    "content": content
                }
                all_comparison_results.append(comparison_record)
        else:
            # Full pairwise comparison: count all wins normally
            for (i, j, first_is_i, _), result_value, reasoning, content in zip(comparison_tasks, comparison_values, reasoning_contents, contents):
                # Determine which solution actually won (based on original indices, not prompt order)
                if first_is_i:
                    # Comparison was (i, j): solution i was solution1, solution j was solution2
                    if result_value == 1:
                        wins[i] += 1  # solution1 (i) wins
                        winner_idx = i
                    elif result_value == 2:
                        wins[j] += 1  # solution2 (j) wins
                        winner_idx = j
                    else:
                        wins[i] += 0.5  # tie
                        wins[j] += 0.5
                        winner_idx = None  # tie
                else:
                    # Comparison was (j, i): solution j was solution1, solution i was solution2
                    if result_value == 1:
                        wins[j] += 1  # solution1 (j) wins
                        winner_idx = j
                    elif result_value == 2:
                        wins[i] += 1  # solution2 (i) wins
                        winner_idx = i
                    else:
                        wins[i] += 0.5  # tie
                        wins[j] += 0.5
                        winner_idx = None  # tie
                
                # Record the comparison result with original indices and prompt order
                comparison_record = {
                    "solution1_idx": i,  # Original index in solutions list
                    "solution2_idx": j,  # Original index in solutions list
                    "prompt_order": "i_j" if first_is_i else "j_i",  # Which order was used in prompt
                    "result": result_value,  # 1 = solution1 (in prompt) wins, 2 = solution2 (in prompt) wins, 0 = tie
                    "winner_idx": winner_idx,  # Winner's original index (None if tie)
                    "reasoning": reasoning,  # LLM reasoning content for this comparison
                    "content": content
                }
                all_comparison_results.append(comparison_record)

        best_idx = max(range(n), key=lambda i: wins[i])
        best_solution = solutions[best_idx]
        best_solution.metadata["aggregation"] = "pairwise_comparison"
        best_solution.metadata["selected_index"] = best_idx  # Index of selected solution in solutions list
        best_solution.metadata["wins"] = wins[best_idx]
        best_solution.metadata["all_wins"] = wins  # Store wins for all solutions for debugging
        best_solution.metadata["total_comparisons"] = len(comparison_tasks)  # Actual number of comparisons performed
        best_solution.metadata["n_candidates"] = n
        best_solution.metadata["use_partial_comparison"] = use_partial_comparison  # Whether partial comparison was used
        if use_partial_comparison:
            best_solution.metadata["k_opponents"] = k_opponents  # Number of opponents per solution
            best_solution.metadata["initiator_comparison_counts"] = initiator_comparison_counts  # How many comparisons each solution counted as initiator
        best_solution.metadata["token_usage"] = total_token_usage
        best_solution.metadata["token_usage_list"] = token_usage_list
        best_solution.metadata["all_comparison_results"] = all_comparison_results  # Store all comparison results for debugging
        
        if all_skipped:
            best_solution.metadata["skipped"] = True
            best_solution.metadata["skip_reason"] = "input_tokens_exceeded"

        return best_solution

    async def _compare_pair(self, problem: str, solution1: Solution, solution2: Solution) -> Tuple[int, Optional[Dict[str, int]], str, str]:
        """Compare two solutions and return which is better.

        Returns:
            Tuple of (comparison_result, token_usage_dict, reasoning_content, content)
            comparison_result: 1 if solution1 is better, 2 if solution2 is better, 0 if tie
            token_usage_dict: Token usage information
            reasoning_content: LLM reasoning content
            content: LLM response content
        """
        if self.comparison_prompt_template:
            comparison_formatted = self.comparison_prompt_template.format_with_system(
                problem=problem, solution1=solution1.content, solution2=solution2.content
            )
            messages = [
                Message(role="system", content=comparison_formatted["system"]),
                Message(role="user", content=comparison_formatted["user"])
            ]
        else:
            # Fallback
            messages = [Message(role="user", content=f"Problem: {problem}\n\nSolution 1:\n{solution1.content}\n\nSolution 2:\n{solution2.content}\n\nWhich is better?")]

        # Check input tokens before calling LLM
        exceeds_limit, input_tokens = self._check_input_tokens(messages)
        # if exceeds_limit:
        #     print(f"  Warning: Input tokens ({input_tokens}) exceed limit (120000), skipping comparison")
        #     # Return tie (0) when skipping
        #     return (0, None, "", "")

        # Build kwargs with reasoning effort
        gen_kwargs = self._build_generation_kwargs()
        response = await self.llm_service.generate(messages, **gen_kwargs)

        # Extract token usage
        usage = response.usage or {}
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        comparison_result = self._parse_comparison(response.content)
        # Store reasoning content for debugging
        reasoning_content = response.reasoning_content
        content = response.content
        return (comparison_result, token_usage, reasoning_content, content)

    def _parse_comparison(self, response: str) -> int:
        """Parse comparison result from response.
        
        Priority order:
        1. Look for explicit format: "Better Solution: Solution 1/2" or "**better solution:** solution 1/2"
        2. Look for "solution X is better" patterns
        3. Fallback to heuristics
        """
        response_lower = response.lower().strip()
        
        # Priority 1: Look for explicit format from config requirement
        # Pattern 1a: "**better solution:** **solution 1**" (both parts in bold)
        # Matches: "**better solution:** **solution 1**", "**better solution:** **1**", etc.
        markdown_bold_match = re.search(
            r"\*\*better\s+solution:\*\*\s*\*\*(?:solution\s+)?([12])\*\*",
            response_lower,
            re.IGNORECASE | re.MULTILINE
        )
        if markdown_bold_match:
            return int(markdown_bold_match.group(1))
        
        # Pattern 1b: "**better solution:** solution 1" (only first part in bold)
        # This handles: "**better solution:** solution 1" with flexible spacing
        # Matches: "**better solution:** solution 1", "**better solution:** solution1", etc.
        markdown_match = re.search(
            r"\*\*better\s+solution:\*\*\s*(?:solution\s+)?([12])\b",
            response_lower,
            re.IGNORECASE | re.MULTILINE
        )
        if markdown_match:
            return int(markdown_match.group(1))
        
        # Pattern 2: "Better Solution: Solution 1" or "Better Solution: [Solution 1 or Solution 2]"
        # Also handles without markdown formatting
        better_solution_match = re.search(
            r"better\s+solution\s*:\s*(?:\[?\s*)?(?:solution\s*)?([12])\b",
            response_lower,
            re.IGNORECASE | re.MULTILINE
        )
        if better_solution_match:
            return int(better_solution_match.group(1))
        
        # Priority 2: Look for "solution X is better" patterns
        if re.search(r"solution\s*1\s+is\s+better", response_lower):
            return 1
        if re.search(r"solution\s*2\s+is\s+better", response_lower):
            return 2
        
        # Priority 3: Look for conclusion patterns at the end
        # Check last 300 characters for conclusion
        conclusion = response_lower[-300:]
        if re.search(r"(?:conclusion|final|answer|choose|select).*solution\s*1", conclusion):
            return 1
        if re.search(r"(?:conclusion|final|answer|choose|select).*solution\s*2", conclusion):
            return 2
        
        # Priority 4: Fallback heuristics (only if no explicit format found)
        # Check if response mentions only one solution in the first 200 chars
        first_200 = response_lower[:200]
        has_solution_1 = "solution 1" in first_200 or "solution1" in first_200
        has_solution_2 = "solution 2" in first_200 or "solution2" in first_200
        
        if has_solution_1 and not has_solution_2:
            return 1
        if has_solution_2 and not has_solution_1:
            return 2
        print("no results: ", response_lower[:200])

        return 0

    def get_strategy_name(self) -> str:
        """Get the name of the aggregation strategy."""
        return "pairwise_comparison"
