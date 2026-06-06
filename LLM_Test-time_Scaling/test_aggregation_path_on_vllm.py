#!/usr/bin/env python3
"""Replicate the EXACT aggregation code path with training data.

This uses litellm (like the aggregation), the aggregation template,
prepare_request(), and the same model name format. If this works but
the real aggregation doesn't, the issue is in the input data content,
not the code path.

Differences from test_chat_completions_on_vllm.py:
  - Uses litellm.acompletion (not raw requests)
  - Uses the aggregation YAML template (not raw training text)
  - Goes through LiteLLMService.prepare_request() for token budgeting
  - Uses openai/ model name prefix (litellm routing)
  - Extracts problem + solutions from training data, re-formats via template
"""

import argparse
import asyncio
import json
import os
import re
import sys

# Add the project root to path so we can import the aggregation modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_service.litellm_service import LiteLLMService
from src.llm_service.base import Message
from src.prompts import PromptManager


def extract_parts_from_training_text(text: str) -> dict:
    """Extract problem, solution1, solution2 from training data text."""
    # Extract user content
    usr_start = text.find("<|im_start|>user\n")
    usr_end = text.find("<|im_end|>", usr_start)
    user_content = text[usr_start + len("<|im_start|>user\n"):usr_end]

    # Extract problem
    problem_match = re.search(r"Problem:\s*(.+?)\n<Chunk>", user_content, re.DOTALL)
    problem = problem_match.group(1).strip() if problem_match else ""

    # Extract solutions from chunks
    chunk_pattern = re.compile(r"<Chunk>\s*Solution \d+:\s*(.*?)\s*</Chunk>", re.DOTALL)
    chunks = chunk_pattern.findall(user_content)

    solution1 = chunks[0].strip() if len(chunks) > 0 else ""
    solution2 = chunks[1].strip() if len(chunks) > 1 else ""

    # Extract reference answer
    asst_start = text.find("<|im_start|>assistant\n")
    ref_answer = ""
    if asst_start != -1:
        ref_answer = text[asst_start + len("<|im_start|>assistant\n"):]
        # Remove trailing <|im_end|> if present
        if ref_answer.endswith("<|im_end|>"):
            ref_answer = ref_answer[:-len("<|im_end|>")]

    return {
        "problem": problem,
        "solution1": solution1,
        "solution2": solution2,
        "ref_answer": ref_answer.strip(),
    }


async def run_test(args):
    # Setup environment
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    # Load the aggregation template (same one the pipeline uses)
    prompt_manager = PromptManager()
    template = prompt_manager.get_template("aggregation_pairwise_new_encoding_comparison")
    print(f"Template: {template.name}")
    print(f"System:   {template.system_prompt[:80]}...")

    # Create LiteLLM service (same as aggregation pipeline)
    service = LiteLLMService(
        model_name=args.model_name,  # e.g., "openai/qwen3-4b-newenc"
        api_key="dummy",
        api_base=args.api_base,
    )
    print(f"Model:    {service.model_name}")
    print(f"API base: {service.api_bases}")
    print(f"Context:  {service.context_limit}")

    # Load training data
    lines = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if i >= args.n:
                break
            lines.append(json.loads(raw))

    for i, record in enumerate(lines):
        text = record["text"]
        parts = extract_parts_from_training_text(text)

        print(f"\n{'='*70}")
        print(f"Example {i+1}/{len(lines)}")
        print(f"Problem:   {parts['problem'][:100]}...")
        print(f"Sol1 len:  {len(parts['solution1'])} chars")
        print(f"Sol2 len:  {len(parts['solution2'])} chars")

        # Format using the SAME template the aggregation uses
        formatted = template.format_with_system(
            problem=parts["problem"],
            solution1=parts["solution1"],
            solution2=parts["solution2"],
        )

        messages = [
            Message(role="system", content=formatted["system"]),
            Message(role="user", content=formatted["user"]),
        ]

        # Show what prepare_request does (token budgeting + possible trimming)
        try:
            prepped_msgs, max_tokens = service.prepare_request(messages, None)
            print(f"prepare_request: {len(prepped_msgs)} msgs, max_tokens={max_tokens}")
            for m in prepped_msgs:
                print(f"  {m['role']}: {len(m['content'])} chars")
        except Exception as e:
            print(f"prepare_request error: {e}")

        # Call LLM through the exact same path as aggregation
        print(f"{'='*70}")
        try:
            response = await service.generate(messages, temperature=args.temperature)
            print(f"Finish reason: {response.finish_reason}")
            print(f"Usage: {response.usage}")
            if response.reasoning_content:
                print(f"Reasoning ({len(response.reasoning_content)} chars): "
                      f"{response.reasoning_content[:200]}...")
            print(f"Content:\n{response.content}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("Done.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="/storage/openpsi/users/zzy/new_aggre_dataset/"
                   "pairwise_io_imobench_20260417_222453_converted_filtered.jsonl")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model-name", default="openai/qwen3-4b-newenc",
                   help="Must include openai/ prefix for litellm routing")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()
    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
