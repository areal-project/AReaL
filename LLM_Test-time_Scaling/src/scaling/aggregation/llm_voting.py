"""LLM voting aggregation strategy.

This strategy uses LLM to extract answers from solutions and determine which answers
are mathematically equivalent (even if they have different forms), then selects the
answer that appears most frequently.
"""

import asyncio
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import AggregationStrategy


class LLMVotingAggregation(AggregationStrategy):
    """Aggregation strategy using LLM to extract and group equivalent answers, then vote."""

    def __init__(
        self,
        llm_service: LLMService,
        voting_prompt_template: Optional[PromptTemplate] = None,
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = "auto",
    ):
        """Initialize LLM voting aggregation.

        Args:
            llm_service: LLM service for answer extraction and equivalence checking
            voting_prompt_template: Optional PromptTemplate for voting prompt
            temperature: Temperature for LLM calls (default: 0.0 for deterministic)
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
        """
        super().__init__(llm_service, temperature=temperature, reasoning_effort=reasoning_effort)
        self.voting_prompt_template = voting_prompt_template

    async def aggregate(self, problem: str, solutions: List[Solution], **kwargs: Any) -> Solution:
        """Select solution by LLM-based voting on equivalent answers."""
        if len(solutions) == 1:
            return solutions[0]

        # Check input tokens before calling LLM
        messages = self._build_messages(problem, solutions)
        exceeds_limit, input_tokens = self._check_input_tokens(messages)
        if exceeds_limit:
            print(f"  Warning: Input tokens ({input_tokens}) exceed limit (128000), using fallback voting")
            return self._fallback_voting(solutions)

        # Call LLM to extract and group answers
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
        
        # Parse LLM response to get answer groups
        answer_groups = self._parse_answer_groups(response.content, len(solutions))
        
        if not answer_groups:
            # If parsing fails, fall back to simple voting
            print("  Warning: Failed to parse LLM response, using fallback voting")
            fallback_solution = self._fallback_voting(solutions)
            fallback_solution.metadata["token_usage"] = token_usage
            fallback_solution.metadata["llm_call_details"] = llm_call_details
            return fallback_solution

        # Count votes for each answer group
        answer_group_counts = Counter(answer_groups)
        most_common_group = answer_group_counts.most_common(1)[0][0]

        # Find the first solution with the most common answer
        for i, group_id in enumerate(answer_groups):
            if group_id == most_common_group:
                best_solution = solutions[i]
                best_solution.metadata["aggregation"] = "llm_voting"
                best_solution.metadata["vote_count"] = answer_group_counts[most_common_group]
                best_solution.metadata["total_votes"] = len(solutions)
                best_solution.metadata["answer_group_distribution"] = dict(answer_group_counts)
                best_solution.metadata["answer_groups"] = answer_groups
                best_solution.metadata["token_usage"] = token_usage
                best_solution.metadata["llm_call_details"] = llm_call_details  # Store detailed info
                return best_solution

        # Fallback: return first solution
        return solutions[0]

    def _build_messages(self, problem: str, solutions: List[Solution]) -> List[Message]:
        """Build messages for LLM voting."""
        if self.voting_prompt_template:
            # Format solutions as a list
            solutions_text = "\n\n".join(
                f"Solution {i+1}:\n{sol.content}" for i, sol in enumerate(solutions)
            )
            formatted = self.voting_prompt_template.format_with_system(
                problem=problem,
                solutions=solutions_text,
                num_solutions=len(solutions)
            )
            return [
                Message(role="system", content=formatted["system"]),
                Message(role="user", content=formatted["user"])
            ]
        else:
            # Default prompt
            solutions_text = "\n\n".join(
                f"Solution {i+1}:\n{sol.content}" for i, sol in enumerate(solutions)
            )
            user_content = f"""Problem: {problem}

{solutions_text}

Extract the final answer from each solution and determine which answers are mathematically equivalent (even if they have different forms, e.g., "1/2" and "0.5" are equivalent).

Provide your response in JSON format:
{{
  "answers": [
    {{"solution_id": 1, "answer": "extracted answer", "group_id": 1}},
    {{"solution_id": 2, "answer": "extracted answer", "group_id": 1}},
    {{"solution_id": 3, "answer": "extracted answer", "group_id": 2}},
    ...
  ]
}}

Use the same group_id for answers that are mathematically equivalent. The group_id should be a positive integer starting from 1."""
            
            return [Message(role="user", content=user_content)]

    def _parse_answer_groups(self, response: str, num_solutions: int) -> List[int]:
        """Parse answer groups from LLM response.
        
        Args:
            response: LLM response text
            num_solutions: Expected number of solutions
            
        Returns:
            List of group IDs, one for each solution (1-indexed)
        """
        # Try to extract JSON from response (more flexible pattern)
        # Look for JSON object containing "answers" array
        json_patterns = [
            r'\{[^{}]*"answers"\s*:\s*\[.*?\]\s*[^}]*\}',  # Standard JSON
            r'```json\s*(\{.*?\})\s*```',  # JSON in code block
            r'```\s*(\{.*?\})\s*```',  # JSON in code block without json tag
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
                try:
                    data = json.loads(json_str)
                    if "answers" in data and isinstance(data["answers"], list):
                        # Create mapping from solution_id to group_id
                        group_map = {}
                        for item in data["answers"]:
                            if isinstance(item, dict) and "solution_id" in item and "group_id" in item:
                                sol_id = int(item["solution_id"])
                                group_id = int(item["group_id"])
                                group_map[sol_id] = group_id
                        
                        # Return group IDs for each solution (1-indexed)
                        if len(group_map) == num_solutions:
                            return [group_map.get(i+1, i+1) for i in range(num_solutions)]
                        elif len(group_map) > 0:
                            # If we got some mappings, use them and assign remaining to new groups
                            result = []
                            next_group = max(group_map.values()) + 1 if group_map else 1
                            for i in range(num_solutions):
                                result.append(group_map.get(i+1, next_group))
                                if i+1 not in group_map:
                                    next_group += 1
                            return result
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    # Continue to next pattern
                    continue

        # Fallback: try to extract group IDs from a simpler format
        # Look for patterns like "Solution 1: answer (Group 1)" or "Group 1: Solution 1, Solution 2"
        group_patterns = [
            r"solution\s+(\d+).*?group\s+(\d+)",
            r"group\s+(\d+).*?solution\s+(\d+)",
        ]
        
        group_map = {}
        for pattern in group_patterns:
            matches = re.finditer(pattern, response.lower(), re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:
                    # Try both orders
                    try:
                        sol_id = int(match.group(1))
                        group_id = int(match.group(2))
                        if 1 <= sol_id <= num_solutions:
                            group_map[sol_id] = group_id
                    except ValueError:
                        try:
                            group_id = int(match.group(1))
                            sol_id = int(match.group(2))
                            if 1 <= sol_id <= num_solutions:
                                group_map[sol_id] = group_id
                        except ValueError:
                            continue

        if group_map and len(group_map) == num_solutions:
            return [group_map.get(i+1, i+1) for i in range(num_solutions)]

        # Last resort: assign each solution to its own group
        return list(range(1, num_solutions + 1))

    def _fallback_voting(self, solutions: List[Solution]) -> Solution:
        """Fallback to simple voting when LLM call fails or is skipped."""
        # Use simple answer extraction (similar to VotingAggregation)
        answers = [self._extract_answer(sol.content) for sol in solutions]
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]

        for i, answer in enumerate(answers):
            if answer == most_common_answer:
                best_solution = solutions[i]
                best_solution.metadata["aggregation"] = "llm_voting"
                best_solution.metadata["vote_count"] = count
                best_solution.metadata["total_votes"] = len(solutions)
                best_solution.metadata["answer_distribution"] = dict(answer_counts)
                best_solution.metadata["fallback"] = True
                return best_solution

        return solutions[0]

    def _extract_answer(self, solution: str) -> str:
        """Extract answer from solution (fallback method)."""
        solution = solution.strip()

        # Try LaTeX boxed format
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try "Final Answer:" pattern
        if "Final Answer:" in solution:
            parts = solution.split("Final Answer:")
            answer = parts[-1].strip()
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

    def get_strategy_name(self) -> str:
        """Get the name of the aggregation strategy."""
        return "llm_voting"
