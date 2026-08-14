#!/usr/bin/env python3
"""
Patch E2 samples: re-evaluate tasks that failed due to missing modules.

Reads E2 samples (partial or final), re-runs evaluation for missing-module tasks
with the now-installed packages, and writes the patched file back.

Also patches epoch_summary.json if it exists.

Usage:
    python scripts/patch_e2_missing_deps.py [--samples PATH]
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memrl.bigcodebench_eval.eval_utils import (
    ensure_bigcodebench_on_path,
    run_untrusted_check_with_hard_timeout,
    sanitize_code,
)
from memrl.bigcodebench_eval.task_wrappers import load_bcb_data

BCB_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "3rdparty", "bigcodebench-main")

DEFAULT_SAMPLES = (
    "/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/"
    "instruct_full/memory/20260427_134515_gpt-4o-2024-11-20_rl-on/epoch2/train/samples.jsonl"
)
DEFAULT_PARTIAL = (
    "/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/"
    "instruct_full/memory/20260427_134515_gpt-4o-2024-11-20_rl-on/epoch2/train/samples_partial.jsonl"
)


def find_samples_path():
    """Use final samples.jsonl if exists, else partial."""
    for flag in sys.argv[1:]:
        if flag.startswith("--samples="):
            return flag.split("=", 1)[1]
    if os.path.isfile(DEFAULT_SAMPLES):
        return DEFAULT_SAMPLES
    if os.path.isfile(DEFAULT_PARTIAL):
        return DEFAULT_PARTIAL
    print("ERROR: No samples file found")
    sys.exit(1)


def retest_task(task_id, solution, bcb_dataset):
    if task_id not in bcb_dataset:
        return None
    task = bcb_dataset[task_id]
    entry_point = task.get("entry_point", "task_func")
    test_code = task.get("test", "")
    if not test_code:
        return None

    clean_code = sanitize_code(solution, entry_point, bcb_repo=BCB_REPO)
    ensure_bigcodebench_on_path(BCB_REPO)
    from bigcodebench.eval import PASS, FAIL, TIMEOUT

    stat, details, err, hard_timed_out = run_untrusted_check_with_hard_timeout(
        code=clean_code, test_code=test_code, entry_point=entry_point,
        max_as_limit=30 * 1024, max_data_limit=30 * 1024, max_stack_limit=10,
        min_time_limit=1.0, gt_time_limit=60.0, hard_timeout_s=120.0,
        bcb_repo=BCB_REPO,
    )

    if hard_timed_out:
        return {"status": "TIMEOUT", "error": "hard_timeout"}
    if err:
        return {"status": "RUNTIME_ERROR", "error": err[:500]}
    if stat == PASS:
        return {"status": "PASS", "error": None}
    if stat == TIMEOUT:
        return {"status": "TIMEOUT", "error": "timeout"}
    if stat == FAIL:
        return {"status": "FAIL", "error": str(details)[:500] if details else "fail"}
    return {"status": "UNKNOWN", "error": str(stat)}


def main():
    samples_path = find_samples_path()
    print(f"Patching: {samples_path}")

    bcb_dataset = load_bcb_data(subset="full")

    # Read all samples
    samples = []
    with open(samples_path) as f:
        for line in f:
            samples.append(json.loads(line))

    # Find missing-module tasks
    to_retest = []
    for i, d in enumerate(samples):
        error = str(d.get("error", "") or "")
        if re.search(r"No module named", error):
            to_retest.append(i)

    print(f"Total samples: {len(samples)}")
    print(f"Missing-module tasks to retest: {len(to_retest)}")

    if not to_retest:
        print("Nothing to patch.")
        return

    # Retest
    old_pass = sum(1 for d in samples if d.get("status") == "PASS")
    flipped = 0
    still_fail = 0

    for idx in to_retest:
        d = samples[idx]
        tid = d["task_id"]
        solution = d["solution"]
        result = retest_task(tid, solution, bcb_dataset)

        if result is None:
            print(f"  [{tid}] SKIP (not in dataset)")
            continue

        old_status = d.get("status")
        new_status = result["status"]

        d["status"] = new_status
        d["error"] = result["error"]
        if new_status == "PASS":
            d["passed"] = True
        else:
            d["passed"] = False
        samples[idx] = d

        if new_status == "PASS" and old_status != "PASS":
            flipped += 1
            print(f"  [{tid}] FLIPPED: {old_status} -> PASS")
        else:
            still_fail += 1
            err_short = (result.get("error") or "")[:80]
            print(f"  [{tid}] {old_status} -> {new_status}: {err_short}")

    new_pass = sum(1 for d in samples if d.get("status") == "PASS")
    total = len(samples)

    print(f"\n{'='*60}")
    print(f"PATCH RESULTS:")
    print(f"  Retested: {len(to_retest)}")
    print(f"  Flipped to PASS: {flipped}")
    print(f"  Still FAIL: {still_fail}")
    print(f"  Old SR: {old_pass}/{total} = {old_pass/total:.1%}")
    print(f"  New SR: {new_pass}/{total} = {new_pass/total:.1%}")

    # Backup and write
    bak = samples_path + ".bak"
    if not os.path.exists(bak):
        import shutil
        shutil.copy2(samples_path, bak)
        print(f"  Backed up: {bak}")

    with open(samples_path, "w") as f:
        for d in samples:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  Written: {samples_path}")

    # Also patch epoch_summary.json if it exists
    epoch_dir = os.path.dirname(os.path.dirname(samples_path))
    summary_path = os.path.join(epoch_dir, "epoch_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        if "train" in summary and summary["train"]:
            summary["train"]["pass"] = new_pass
            summary["train"]["pass@1"] = new_pass / total
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Patched epoch_summary: pass={new_pass}, pass@1={new_pass/total:.4f}")


if __name__ == "__main__":
    main()
