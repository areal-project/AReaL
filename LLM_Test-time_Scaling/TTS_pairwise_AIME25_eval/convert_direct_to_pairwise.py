#!/usr/bin/env python3
"""
Convert direct generation JSON results to pairwise comparison JSONL format.

Takes a direct_generation JSON file (with multiple solutions per problem)
and produces a JSONL file where each line is a pairwise comparison between
two solutions to the same problem, formatted as a chat template.

Usage:
    python convert_direct_to_pairwise.py <input_json> <output_jsonl>
"""

import json
import sys
import re
from itertools import combinations


SYSTEM_PROMPT = (
    "You are an expert evaluator comparing two solutions to the same problem.\n"
    "Determine which solution is better based on correctness, completeness, and clarity."
)

USER_TEMPLATE = (
    "Problem: {problem}\n"
    "<Chunk>\nSolution 1:\n{solution_1}\n</Chunk>\n"
    "<Chunk>\nSolution 2:\n{solution_2}\n</Chunk>\n"
    "\n\n"
    "Compare these two solutions and determine which one is better.\n"
    "Consider correctness, completeness, clarity, and mathematical/logical rigor.\n"
    "\n"
    "Provide your response in the format:\n"
    "Better Solution: [Solution 1 or Solution 2]\n"
    "Reasoning: [brief explanation of why it is better]\n"
)

# Assistant response templates based on correctness combinations
RESPONSE_TEMPLATES = {
    # (sol1_correct, sol2_correct, winner) -> assistant text
    (True, True, 1): (
        "<think>\n"
        "Comparing the two solutions: Solution 1 is correct and Solution 2 is correct. "
        "Based on correctness, completeness, and clarity, Solution 1 is the better response.\n"
        "</think>\n\n"
        "Better Solution: Solution 1\n"
        "Reasoning: Both solutions reach the correct answer, but Solution 1 presents "
        "a more concise and well-structured argument with clearer logical flow."
    ),
    (True, True, 2): (
        "<think>\n"
        "Comparing the two solutions: Solution 1 is correct and Solution 2 is correct. "
        "Based on correctness, completeness, and clarity, Solution 2 is the better response.\n"
        "</think>\n\n"
        "Better Solution: Solution 2\n"
        "Reasoning: Both solutions reach the correct answer, but Solution 2 presents "
        "a more concise and well-structured argument with clearer logical flow."
    ),
    (True, False, 1): (
        "<think>\n"
        "Comparing the two solutions: Solution 1 is correct and Solution 2 is incorrect. "
        "Based on correctness, completeness, and clarity, Solution 1 is the better response.\n"
        "</think>\n\n"
        "Better Solution: Solution 1\n"
        "Reasoning: Solution 1 arrives at the correct answer with clear, logical steps. "
        "Solution 2 contains errors in its reasoning that lead to an incorrect conclusion."
    ),
    (False, True, 2): (
        "<think>\n"
        "Comparing the two solutions: Solution 1 is incorrect and Solution 2 is correct. "
        "Based on correctness, completeness, and clarity, Solution 2 is the better response.\n"
        "</think>\n\n"
        "Better Solution: Solution 2\n"
        "Reasoning: Solution 2 arrives at the correct answer with clear, logical steps. "
        "Solution 1 contains errors in its reasoning that lead to an incorrect conclusion."
    ),
    (False, False, 1): (
        "<think>\n"
        "Comparing the two solutions: Solution 1 is incorrect and Solution 2 is incorrect. "
        "Based on correctness, completeness, and clarity, Solution 1 is the better response.\n"
        "</think>\n\n"
        "Better Solution: Solution 1\n"
        "Reasoning: Solution 1 demonstrates a more rigorous approach with better-organized "
        "steps and clearer reasoning throughout the derivation."
    ),
    (False, False, 2): (
        "<think>\n"
        "Comparing the two solutions: Solution 1 is incorrect and Solution 2 is incorrect. "
        "Based on correctness, completeness, and clarity, Solution 2 is the better response.\n"
        "</think>\n\n"
        "Better Solution: Solution 2\n"
        "Reasoning: Solution 2 demonstrates a more rigorous approach with better-organized "
        "steps and clearer reasoning throughout the derivation."
    ),
}


def extract_final_answer(solution_content: str) -> str:
    """Extract the portion after </think> tags (the clean final answer)."""
    # Find the last </think> tag and take everything after it
    think_end = solution_content.rfind("</think>")
    if think_end != -1:
        result = solution_content[think_end + len("</think>"):].strip()
        return result
    # If no </think> tag, return the whole content (already clean)
    return solution_content.strip()


def determine_winner(sol1_correct: bool, sol2_correct: bool,
                     sol1_content: str, sol2_content: str) -> int:
    """Determine which solution is 'better'.

    Rules:
    - If one is correct and the other isn't, the correct one wins.
    - If both have the same correctness, the shorter (more concise) one wins.
    """
    if sol1_correct and not sol2_correct:
        return 1
    if not sol1_correct and sol2_correct:
        return 2
    # Both same correctness: prefer the shorter/more concise solution
    if len(sol1_content) <= len(sol2_content):
        return 1
    return 2


def format_chat_text(problem: str, solution_1: str, solution_2: str,
                     assistant_response: str) -> str:
    """Format the full chat text with im_start/im_end tags."""
    user_content = USER_TEMPLATE.format(
        problem=problem,
        solution_1=solution_1,
        solution_2=solution_2,
    )
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_response}"
    )
    return text


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as f:
        data = json.load(f)

    results = data["results"]
    total_pairs = 0

    with open(output_path, "w") as out:
        for problem_entry in results:
            problem_text = problem_entry["problem"]
            solutions = problem_entry["all_solutions"]

            # Extract clean (non-think) content and correctness for each solution
            clean_solutions = []
            for sol in solutions:
                clean_content = extract_final_answer(sol["solution_content"])
                is_correct = sol.get("is_correct", False)
                clean_solutions.append((clean_content, is_correct))

            # Generate all C(n, 2) pairs
            n = len(clean_solutions)
            for i, j in combinations(range(n), 2):
                sol1_content, sol1_correct = clean_solutions[i]
                sol2_content, sol2_correct = clean_solutions[j]

                winner = determine_winner(
                    sol1_correct, sol2_correct,
                    sol1_content, sol2_content,
                )

                response_key = (sol1_correct, sol2_correct, winner)
                assistant_response = RESPONSE_TEMPLATES[response_key]

                text = format_chat_text(
                    problem_text, sol1_content, sol2_content, assistant_response
                )
                record = {"text": text}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pairs += 1

    print(f"Converted {len(results)} problems with {total_pairs} pairwise comparisons")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_json> <output_jsonl>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
