#!/usr/bin/env python3
"""Wrapper around test_training_data_on_vllm.py that:
1. Randomly selects 100 lines from the dataset
2. Sends them to vLLM
3. Parses 'better solution' from both the reference answer and model output
4. Saves all results (prompt, model output, reference, parsed verdicts) to a JSONL file
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
from datetime import datetime

import requests


def extract_prompt(text: str) -> str:
    """Extract everything up to and including '<|im_start|>assistant\\n'."""
    marker = "<|im_start|>assistant\n"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("Could not find assistant marker in text")
    return text[: idx + len(marker)]


def extract_reference(text: str) -> str:
    """Extract the reference answer after '<|im_start|>assistant\\n'."""
    marker = "<|im_start|>assistant\n"
    idx = text.find(marker)
    if idx == -1:
        return ""
    rest = text[idx + len(marker):]
    end_marker = "<|im_end|>"
    end_idx = rest.find(end_marker)
    if end_idx != -1:
        return rest[:end_idx]
    return rest


def parse_better_solution(text: str) -> int:
    """Parse 'better solution' verdict from text. Returns 1, 2, or 0 (unparseable)."""
    lower = text.lower().strip()

    # **better solution:** **solution 1**
    m = re.search(r"\*\*better\s+solution:\*\*\s*\*\*(?:solution\s+)?([12])\*\*", lower)
    if m:
        return int(m.group(1))

    # **better solution:** solution 1
    m = re.search(r"\*\*better\s+solution:\*\*\s*(?:solution\s+)?([12])\b", lower)
    if m:
        return int(m.group(1))

    # Better Solution: Solution 1
    m = re.search(r"better\s+solution\s*:\s*(?:\[?\s*)?(?:solution\s*)?([12])\b", lower)
    if m:
        return int(m.group(1))

    # solution X is better
    if re.search(r"solution\s*1\s+is\s+better", lower):
        return 1
    if re.search(r"solution\s*2\s+is\s+better", lower):
        return 2

    # Conclusion area
    conclusion = lower[-300:]
    if re.search(r"(?:conclusion|final|answer|choose|select).*solution\s*1", conclusion):
        return 1
    if re.search(r"(?:conclusion|final|answer|choose|select).*solution\s*2", conclusion):
        return 2

    return 0


def query_vllm(prompt: str, api_base: str, model_name: str,
               max_tokens: int, temperature: float) -> str:
    """Try /v1/completions first; fall back to /v1/chat/completions.

    api_base can be a single URL or one entry from a comma-separated list
    (the caller is responsible for picking which one to use)."""
    """Try /v1/completions first; fall back to /v1/chat/completions."""
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


def process_one_item(seq, orig_idx, record, api_bases, model_name,
                     max_tokens, temperature):
    """Process a single item: extract prompt, query vLLM, parse verdict.

    api_bases is a list of API base URLs; the item is routed to one of them
    round-robin by seq index.

    Returns (seq, result_dict).
    """
    api_base = api_bases[seq % len(api_bases)]
    text = record["text"]
    prompt = extract_prompt(text)
    reference = extract_reference(text)
    ref_verdict = parse_better_solution(reference)

    problem_match = re.search(r"Problem:\s*(.+?)(?:\n|<Chunk>)", prompt, re.DOTALL)
    preview = (problem_match.group(1).strip()[:120] + "..."
               if problem_match else "(could not extract)")

    model_output = ""
    model_verdict = -1  # -1 = error
    status = ""
    try:
        model_output = query_vllm(prompt, api_base, model_name,
                                  max_tokens, temperature)
        model_verdict = parse_better_solution(model_output)

        if model_verdict == ref_verdict and ref_verdict != 0:
            status = "MATCH"
        elif ref_verdict == 0:
            status = "REF_UNPARSEABLE"
        elif model_verdict == 0:
            status = "MODEL_UNPARSEABLE"
        else:
            status = "MISMATCH"
    except Exception as e:
        status = f"ERROR: {e}"

    result = {
        "line_index": orig_idx,
        "problem_preview": preview,
        "reference_answer": reference,
        "reference_verdict": ref_verdict,
        "model_output": model_output,
        "model_verdict": model_verdict,
        "status": status,
    }
    return seq, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",
                   default="/storage/openpsi/users/zzy/pairwise_from_direct_generation_AIME25.jsonl")
    p.add_argument("--api-base", default="http://0.0.0.0:8000/v1",
                   help="Comma-separated vLLM API base URLs; requests are round-robined across them")
    p.add_argument("--model-name", default="qwen3-4b-newenc")
    p.add_argument("--n", type=int, default=0,
                   help="Number of lines to process (0 = all lines, default: 0)")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path (default: auto-generated with timestamp)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Number of concurrent requests to send to the server (default: 1)")
    args = p.parse_args()

    random.seed(args.seed)

    # Parse API bases (comma-separated)
    api_bases = [base.strip() for base in args.api_base.split(",") if base.strip()]
    if not api_bases:
        print("Error: --api-base must specify at least one URL", file=sys.stderr)
        sys.exit(1)
    print(f"API bases ({len(api_bases)}): {api_bases}")

    # Load all lines and randomly sample
    print(f"Loading dataset: {args.dataset}")
    all_lines = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for raw in f:
            all_lines.append(json.loads(raw))
    print(f"Total lines: {len(all_lines)}")

    if args.n <= 0:
        sampled = [(idx, record) for idx, record in enumerate(all_lines)]
        print(f"Processing all {len(sampled)} lines")
    else:
        sample_size = min(args.n, len(all_lines))
        sampled_indices = sorted(random.sample(range(len(all_lines)), sample_size))
        sampled = [(idx, all_lines[idx]) for idx in sampled_indices]
        print(f"Randomly sampled {sample_size} lines (seed={args.seed})")

    # Output file
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(out_dir, f"eval_results_{timestamp}.jsonl")

    total = len(sampled)
    concurrency = max(1, args.concurrency)
    print(f"Concurrency: {concurrency}")

    # Process items concurrently
    results = [None] * total
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                process_one_item, seq, orig_idx, record,
                api_bases, args.model_name,
                args.max_tokens, args.temperature,
            ): seq
            for seq, (orig_idx, record) in enumerate(sampled)
        }
        for future in concurrent.futures.as_completed(futures):
            seq_idx, result = future.result()
            results[seq_idx] = result
            completed += 1
            print(f"[{completed}/{total}] Line {result['line_index']} | {result['status']}")

    # Write results in original order
    with open(args.output, "w", encoding="utf-8") as fout:
        for result in results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Compute stats
    match_count = 0
    mismatch_count = 0
    ref_parse_fail = 0
    model_parse_fail = 0
    error_count = 0

    for result in results:
        status = result["status"]
        if status == "MATCH":
            match_count += 1
        elif status == "MISMATCH":
            mismatch_count += 1
        elif status == "REF_UNPARSEABLE":
            ref_parse_fail += 1
        elif status == "MODEL_UNPARSEABLE":
            model_parse_fail += 1
        elif status.startswith("ERROR"):
            error_count += 1

    parseable = match_count + mismatch_count
    accuracy = (match_count / parseable * 100) if parseable > 0 else 0

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"  Total:             {total}")
    print(f"  Match:             {match_count}")
    print(f"  Mismatch:          {mismatch_count}")
    print(f"  Ref unparseable:   {ref_parse_fail}")
    print(f"  Model unparseable: {model_parse_fail}")
    print(f"  Errors:            {error_count}")
    print(f"  Accuracy (match/parseable): {accuracy:.1f}% ({match_count}/{parseable})")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
