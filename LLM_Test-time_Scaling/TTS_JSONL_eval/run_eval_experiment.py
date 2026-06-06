#!/usr/bin/env python3
"""JSONL pairwise evaluation experiment using test_training_data_eval.py logic.

For each line in the pairwise JSONL file:
  1. Extract the prompt (everything up to <|im_start|>assistant)
  2. Send to vLLM to get model output
  3. Extract the reference answer from the same line
  4. Parse 'better solution' verdict from both model output and reference
  5. Compare: match / mismatch / unparseable

Outputs a JSON file with aggregate_metrics.pass@1 = accuracy (match rate).

Usage:
    python3 TTS_JSONL_eval/run_eval_experiment.py \
        --jsonl-file pairwise_from_direct_generation_AIME25.jsonl \
        --ground-truth-file LLM_Test-time_Scaling/imobench.json \
        --model-name qwen3-4b-newenc \
        --api-base http://127.0.0.1:8000/v1 \
        --output-dir results/tts_jsonl_eval
"""

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


# ---------------------------------------------------------------------------
# Functions copied exactly from test_training_data_eval.py
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Async vLLM client (adapted from test_training_data_eval.py's query_vllm
# to use aiohttp for concurrent requests)
# ---------------------------------------------------------------------------

async def query_vllm(
    session: aiohttp.ClientSession,
    prompt: str,
    api_base: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    """Try /v1/completions first; fall back to /v1/chat/completions.

    This is the async version of test_training_data_eval.py's query_vllm().
    """
    url = f"{api_base}/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with session.post(url, json=payload, timeout=client_timeout) as resp:
        if resp.status != 404:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"vLLM returned {resp.status}: {body[:500]}")
            data = await resp.json()
            return data["choices"][0]["text"]

    # Fallback to chat completions
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with session.post(url, json=payload, timeout=client_timeout) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"vLLM chat returned {resp.status}: {body[:500]}")
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------

async def process_line(
    seq: int,
    orig_idx: int,
    record: dict,
    total: int,
    session: aiohttp.ClientSession,
    api_base: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    request_timeout: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single JSONL line — same logic as test_training_data_eval.py's main loop."""
    async with semaphore:
        text = record["text"]
        prompt = extract_prompt(text)
        reference = extract_reference(text)
        ref_verdict = parse_better_solution(reference)

        # Short preview (same regex as test_training_data_eval.py)
        problem_match = re.search(r"Problem:\s*(.+?)(?:\n|<Chunk>)", prompt, re.DOTALL)
        preview = (
            problem_match.group(1).strip()[:120] + "..."
            if problem_match
            else "(could not extract)"
        )

        model_output = ""
        model_verdict = -1  # -1 = error
        status = "ERROR"

        try:
            model_output = await query_vllm(
                session=session,
                prompt=prompt,
                api_base=api_base,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=request_timeout,
            )
            model_verdict = parse_better_solution(model_output)

            if model_verdict == ref_verdict and ref_verdict != 0:
                status = "MATCH"
            elif ref_verdict == 0:
                status = "REF_UNPARSEABLE"
            elif model_verdict == 0:
                status = "MODEL_UNPARSEABLE"
            else:
                status = "MISMATCH"

            print(
                f"  [{seq + 1}/{total}] Line {orig_idx} | "
                f"Ref: Solution {ref_verdict} | Model: Solution {model_verdict} | {status} "
                f"| {preview}",
                flush=True,
            )
        except Exception as e:
            status = f"ERROR: {e}"
            print(f"  [{seq + 1}/{total}] Line {orig_idx} | ERROR: {e}", flush=True)

        return {
            "line_index": orig_idx,
            "problem_preview": preview,
            "reference_answer": reference,
            "reference_verdict": ref_verdict,
            "model_output": model_output,
            "model_verdict": model_verdict,
            "status": status,
        }


async def run_experiment(args) -> Dict[str, Any]:
    """Run the full evaluation experiment."""
    start_time = datetime.now()

    # Load all lines from the JSONL file
    print(f"Loading dataset: {args.jsonl_file}")
    all_lines = []
    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                all_lines.append(json.loads(raw))
    total_lines = len(all_lines)
    print(f"  Total lines: {total_lines}")

    # Build the list of (orig_idx, record) for all lines
    sampled = [(idx, record) for idx, record in enumerate(all_lines)]

    total = len(sampled)
    print(f"  Processing all {total} lines")
    print(f"  Model: {args.model_name} @ {args.api_base}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print()

    # Process all lines concurrently
    semaphore = asyncio.Semaphore(args.max_concurrent)
    connector = aiohttp.TCPConnector(limit=args.max_concurrent * 2)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_line(
                seq=seq,
                orig_idx=orig_idx,
                record=record,
                total=total,
                session=session,
                api_base=args.api_base,
                model_name=args.model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                request_timeout=args.request_timeout,
                semaphore=semaphore,
            )
            for seq, (orig_idx, record) in enumerate(sampled)
        ]
        results = await asyncio.gather(*tasks)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Compute stats — same logic as test_training_data_eval.py
    match_count = 0
    mismatch_count = 0
    ref_parse_fail = 0
    model_parse_fail = 0
    error_count = 0

    for r in results:
        s = r["status"]
        if s == "MATCH":
            match_count += 1
        elif s == "MISMATCH":
            mismatch_count += 1
        elif s == "REF_UNPARSEABLE":
            ref_parse_fail += 1
        elif s == "MODEL_UNPARSEABLE":
            model_parse_fail += 1
        else:
            error_count += 1

    parseable = match_count + mismatch_count
    accuracy = (match_count / parseable) if parseable > 0 else 0.0

    # Build aggregate_metrics with pass@1 = accuracy
    # (so the shell pipeline's extract_pass1 function works unchanged)
    aggregate_metrics = {
        "pass@1": accuracy,
        "total_lines": total,
        "match": match_count,
        "mismatch": mismatch_count,
        "ref_unparseable": ref_parse_fail,
        "model_unparseable": model_parse_fail,
        "errors": error_count,
        "parseable": parseable,
    }

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"jsonl_eval_experiment_{timestamp}.json"

    output_data = {
        "source_file": str(args.jsonl_file),
        "ground_truth_file": str(args.ground_truth_file),
        "model_name": args.model_name,
        "temperature": args.temperature,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "aggregate_metrics": aggregate_metrics,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary — same format as test_training_data_eval.py
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"  Total:             {total}")
    print(f"  Match:             {match_count}")
    print(f"  Mismatch:          {mismatch_count}")
    print(f"  Ref unparseable:   {ref_parse_fail}")
    print(f"  Model unparseable: {model_parse_fail}")
    print(f"  Errors:            {error_count}")
    print(f"  Accuracy (match/parseable): {accuracy:.4f} ({match_count}/{parseable})")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Output: {output_file}")
    print(f"{'=' * 70}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Run JSONL pairwise evaluation experiment "
        "(test_training_data_eval.py style)"
    )
    parser.add_argument(
        "--jsonl-file", required=True, help="Path to pairwise JSONL file"
    )
    parser.add_argument(
        "--ground-truth-file", required=True,
        help="Path to ground truth JSON (imobench.json or direct_generation JSON)",
    )
    parser.add_argument(
        "--model-name", required=True, help="Pairwise comparison model name"
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000/v1",
        help="vLLM server URL (default: http://127.0.0.1:8000/v1)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tts_jsonl_eval",
        help="Output directory (default: results/tts_jsonl_eval)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens for generation (default: 16384)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=40,
        help="Max concurrent requests (default: 40)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="HTTP request timeout in seconds (default: 600)",
    )

    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
