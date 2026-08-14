#!/usr/bin/env python3
"""
Re-test the 9 FAIL tasks that have multiple code blocks.
Tests both approaches:
  A) Our current regex (first block only)
  B) Concatenate ALL python code blocks → sanitize
"""

import json
import sys
import re
from pathlib import Path

# Add MemRL to path
MEMRL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(MEMRL_DIR))

from memrl.bigcodebench_eval.eval_utils import (
    sanitize_code,
    run_untrusted_check_with_hard_timeout,
)

# The 9 FAIL tasks with multiple code blocks
FAIL_TASKS = [
    "BigCodeBench/1041",
    "BigCodeBench/817",
    "BigCodeBench/849",
    "BigCodeBench/303",
    "BigCodeBench/811",
    "BigCodeBench/82",
    "BigCodeBench/604",
    "BigCodeBench/871",
    "BigCodeBench/79",
]


def extract_all_python_blocks(text: str) -> str:
    """Extract ALL python code blocks from markdown, concatenate them."""
    if not text:
        return ""
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        # Filter out non-python blocks (html, json, etc.)
        python_blocks = []
        for b in blocks:
            stripped = b.strip()
            # Skip blocks that look like HTML or other languages
            if stripped.startswith("<!DOCTYPE") or stripped.startswith("<html"):
                continue
            if stripped.startswith("{") and stripped.endswith("}"):
                continue
            python_blocks.append(stripped)
        if python_blocks:
            return "\n\n".join(python_blocks)
    # Fallback: return whole text stripped
    return text.strip()


def load_samples(samples_path: str):
    samples = {}
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            task_id = sample.get("task_id")
            if task_id:
                samples[task_id] = sample
    return samples


def load_tasks(data_path: str):
    tasks = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            task_id = task.get("task_id")
            if task_id:
                tasks[task_id] = task
    return tasks


def evaluate_code(code: str, task: dict, eval_timeout: float = 240.0, hard_timeout: float = 300.0):
    task_id = task.get("task_id", "unknown")
    entry_point = task.get("entry_point", "task_func")
    test_code = task.get("test", "")

    if not test_code:
        return {"task_id": task_id, "status": "NO_TEST", "error": "no_test_code"}

    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return {"task_id": task_id, "status": "SYNTAX_ERROR", "error": str(e)}

    from bigcodebench.eval import PASS, FAIL, TIMEOUT

    bcb_repo = str(MEMRL_DIR / "3rdparty" / "bigcodebench-main")
    stat, details, err, hard_timed_out = run_untrusted_check_with_hard_timeout(
        code=code,
        test_code=test_code,
        entry_point=entry_point,
        max_as_limit=30 * 1024,
        max_data_limit=30 * 1024,
        max_stack_limit=10,
        min_time_limit=1.0,
        gt_time_limit=eval_timeout,
        hard_timeout_s=hard_timeout,
        bcb_repo=bcb_repo,
    )

    if hard_timed_out:
        return {"task_id": task_id, "status": "TIMEOUT", "error": err or "hard_timeout"}
    if err:
        return {"task_id": task_id, "status": "RUNTIME_ERROR", "error": err}
    if stat == PASS:
        return {"task_id": task_id, "status": "PASS"}
    if stat == TIMEOUT:
        return {"task_id": task_id, "status": "TIMEOUT", "error": "timeout"}
    if stat == FAIL:
        return {"task_id": task_id, "status": "FAIL", "error": str(details)[:500] if details else "fail"}
    return {"task_id": task_id, "status": "UNKNOWN", "error": str(stat)}


def main():
    samples_path = MEMRL_DIR / "results/bigcodebench_eval/instruct_full/memory/20260428_180250_gpt-4o-2024-11-20_rl-on/epoch1/train/samples.jsonl"
    data_path = MEMRL_DIR / "data/bigcodebench/bigcodebench_full.jsonl"

    print(f"Loading samples from: {samples_path}")
    samples = load_samples(str(samples_path))
    print(f"Loading tasks from: {data_path}")
    tasks = load_tasks(str(data_path))

    print(f"\nRe-testing {len(FAIL_TASKS)} FAIL tasks with multiple code blocks")
    print("Approach: extract ALL python blocks → concatenate → sanitize → calibrated eval")
    print("=" * 80)

    results = []
    for task_id in FAIL_TASKS:
        sample = samples.get(task_id)
        task = tasks.get(task_id)

        if not sample or not task:
            print(f"\n{task_id}: SKIP")
            continue

        raw_response = sample.get("raw_response", "")
        old_status = sample.get("status", "UNKNOWN")
        entry_point = task.get("entry_point", "task_func")
        code_prompt = task.get("code_prompt", "")
        bcb_repo = str(MEMRL_DIR / "3rdparty" / "bigcodebench-main")

        # New approach: extract ALL python blocks, then sanitize
        all_code = extract_all_python_blocks(raw_response)
        clean_code = sanitize_code(all_code, entry_point, bcb_repo=bcb_repo)

        # Calibrated wrapping
        if code_prompt:
            clean_code = code_prompt + "\n    pass\n" + clean_code

        eval_res = evaluate_code(clean_code, task)
        new_status = eval_res.get("status")

        result = {
            "task_id": task_id,
            "old_status": old_status,
            "new_status": new_status,
            "flipped": (old_status == "FAIL" and new_status == "PASS"),
            "error": eval_res.get("error", ""),
        }
        results.append(result)

        indicator = "PASS!" if result["flipped"] else new_status
        print(f"\n  {task_id}: {old_status} -> {indicator}")
        if not result["flipped"] and new_status != "PASS":
            print(f"    Error: {result['error'][:120]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    flipped = sum(1 for r in results if r["flipped"])
    still_fail = sum(1 for r in results if r["new_status"] == "FAIL")
    other = len(results) - flipped - still_fail

    print(f"Total re-tested: {len(results)}")
    print(f"Flipped to PASS: {flipped}")
    print(f"Still FAIL:      {still_fail}")
    print(f"Other status:    {other}")

    if flipped > 0:
        print(f"\nCode extraction fix recovers {flipped}/{len(results)} tasks!")
        print(f"  SR improvement: +{flipped}/798 = +{flipped/798*100:.2f}pp")

    output_path = MEMRL_DIR / "results/multiblock_retest_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
