#!/usr/bin/env python3
"""Send the first N lines of the training dataset to the vLLM server
using /v1/chat/completions (the same endpoint the aggregation uses).

Compare output with test_training_data_on_vllm.py (which uses /v1/completions)
to isolate the chat-template mismatch."""

import argparse
import json
import re
import requests


def parse_training_text(text: str) -> tuple[str, str]:
    """Parse the pre-formatted training text into (system, user) messages.

    Training text format:
      <|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n...
    """
    # Extract system content
    sys_start = text.find("<|im_start|>system\n")
    sys_end = text.find("<|im_end|>", sys_start)
    system_content = text[sys_start + len("<|im_start|>system\n"):sys_end]

    # Extract user content
    usr_start = text.find("<|im_start|>user\n")
    usr_end = text.find("<|im_end|>", usr_start)
    user_content = text[usr_start + len("<|im_start|>user\n"):usr_end]

    return system_content.strip(), user_content.strip()


def query_chat_completions(system: str, user: str, api_base: str,
                           model_name: str, max_tokens: int,
                           temperature: float) -> str:
    """Call /v1/chat/completions — same path as aggregation via litellm."""
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="/storage/openpsi/users/zzy/new_aggre_dataset/"
                   "pairwise_io_imobench_20260417_222453_converted_filtered.jsonl")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model-name", default="qwen3-4b-newenc",
                   help="Must match --served-model-name on the vLLM server")
    p.add_argument("--n", type=int, default=10,
                   help="Number of lines to send")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.6)
    args = p.parse_args()

    lines = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if i >= args.n:
                break
            lines.append(json.loads(raw))

    for i, record in enumerate(lines):
        text = record["text"]
        system, user = parse_training_text(text)

        # Show a short preview of the problem
        problem_match = re.search(r"Problem:\s*(.+?)(?:\n|<Chunk>)", user, re.DOTALL)
        problem_preview = (problem_match.group(1).strip()[:120] + "..."
                           if problem_match else "(could not extract)")

        print(f"\n{'='*70}")
        print(f"Example {i+1}/{len(lines)}")
        print(f"Problem: {problem_preview}")
        print(f"System:  {system[:80]}...")
        print(f"User:    {len(user)} chars")
        print(f"{'='*70}")

        try:
            response = query_chat_completions(
                system, user,
                args.api_base, args.model_name,
                args.max_tokens, args.temperature,
            )
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
