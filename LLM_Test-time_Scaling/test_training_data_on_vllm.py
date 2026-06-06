#!/usr/bin/env python3
"""Send the first N lines of the training dataset to the vLLM server
and print the responses. Uses /v1/completions with the raw pre-formatted
prompt so the tokenization matches training exactly."""

import argparse
import json
import re
import requests


def extract_prompt(text: str) -> str:
    """Extract everything up to and including '<|im_start|>assistant\n'
    from the pre-formatted training text, discarding the reference answer."""
    marker = "<|im_start|>assistant\n"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("Could not find assistant marker in text")
    return text[: idx + len(marker)]


def query_vllm(prompt: str, api_base: str, model_name: str,
               max_tokens: int, temperature: float) -> str:
    """Try /v1/completions first; fall back to /v1/chat/completions."""
    # --- attempt 1: raw completions endpoint (ideal for pre-formatted prompts)
    url = f"{api_base}/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, json=payload, timeout=600)
    if resp.status_code != 404:
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    # --- attempt 2: chat/completions (newer vLLM may only expose this)
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="/storage/openpsi/users/zzy/new_aggre_dataset/pairwise_io_imobench_20260417_222453_converted_filtered.jsonl")
    p.add_argument("--api-base", default="http://0.0.0.0:8000/v1")
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
        prompt = extract_prompt(text)

        # Show a short preview of the problem
        problem_match = re.search(r"Problem:\s*(.+?)(?:\n|<Chunk>)", prompt, re.DOTALL)
        problem_preview = (problem_match.group(1).strip()[:120] + "..."
                           if problem_match else "(could not extract)")

        print(f"\n{'='*70}")
        print(f"Example {i+1}/{len(lines)}")
        print(f"Problem: {problem_preview}")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"{'='*70}")

        try:
            response = query_vllm(
                prompt, args.api_base, args.model_name,
                args.max_tokens, args.temperature,
            )
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
