#!/usr/bin/env python3
"""
Re-evaluate E1 tasks that failed due to missing module errors.

Usage:
    pip install statsmodels django openpyxl python-docx xlwt keras sendgrid \
        scikit-image pyquery geopandas geopy xmltodict Flask-Mail flask_login \
        pyfakefs texttable textblob gensim pytesseract holidays folium pycryptodome tensorflow
    python scripts/retest_missing_deps.py
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

E1_SAMPLES = (
    "/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/"
    "instruct_full/memory/20260426_195914_gpt-4o-2024-11-20_rl-on/epoch1/train/samples.jsonl"
)
BCB_REPO = "/storage/openpsi/users/yl/agent-memory/MemRL/3rdparty/bigcodebench-main"


def load_bcb_tasks():
    return load_bcb_data(subset="full")


def find_missing_module_tasks(samples_path):
    tasks = []
    with open(samples_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") != "PASS":
                error = str(d.get("error", "") or "")
                m = re.search(r"No module named '([^']+)'", error)
                if m:
                    tasks.append({
                        "task_id": d["task_id"],
                        "solution": d["solution"],
                        "module": m.group(1).split(".")[0],
                        "original_error": error[:300],
                    })
    return tasks


def retest_task(task_id, solution, bcb_dataset):
    if task_id not in bcb_dataset:
        return {"task_id": task_id, "status": "SKIP", "error": "task not in dataset"}

    task = bcb_dataset[task_id]
    entry_point = task.get("entry_point", "task_func")
    test_code = task.get("test", "")

    if not test_code:
        return {"task_id": task_id, "status": "SKIP", "error": "no test code"}

    clean_code = sanitize_code(solution, entry_point, bcb_repo=BCB_REPO)

    from bigcodebench.eval import PASS, FAIL, TIMEOUT

    stat, details, err, hard_timed_out = run_untrusted_check_with_hard_timeout(
        code=clean_code,
        test_code=test_code,
        entry_point=entry_point,
        max_as_limit=30 * 1024,
        max_data_limit=30 * 1024,
        max_stack_limit=10,
        min_time_limit=1.0,
        gt_time_limit=30.0,
        hard_timeout_s=120.0,
        bcb_repo=BCB_REPO,
    )

    if hard_timed_out:
        return {"task_id": task_id, "status": "TIMEOUT", "error": "hard_timeout"}
    if err:
        return {"task_id": task_id, "status": "RUNTIME_ERROR", "error": err[:300]}
    if stat == PASS:
        return {"task_id": task_id, "status": "PASS"}
    if stat == TIMEOUT:
        return {"task_id": task_id, "status": "TIMEOUT", "error": "timeout"}
    if stat == FAIL:
        return {"task_id": task_id, "status": "FAIL", "error": str(details)[:300]}
    return {"task_id": task_id, "status": "UNKNOWN", "error": str(stat)}


def main():
    print("Loading BCB dataset...")
    bcb_dataset = load_bcb_tasks()

    print(f"Finding missing-module tasks from {E1_SAMPLES}...")
    missing_tasks = find_missing_module_tasks(E1_SAMPLES)
    print(f"Found {len(missing_tasks)} tasks with missing module errors")

    # Check which modules are now available
    from collections import Counter
    module_counts = Counter(t["module"] for t in missing_tasks)
    print("\nModule availability check:")
    for mod, cnt in module_counts.most_common():
        try:
            __import__(mod)
            print(f"  {mod} ({cnt} tasks): INSTALLED")
        except ImportError:
            print(f"  {mod} ({cnt} tasks): STILL MISSING")

    print(f"\nRetesting {len(missing_tasks)} tasks...")
    results = []
    passed = 0
    failed = 0
    still_broken = 0

    for i, task_info in enumerate(missing_tasks):
        tid = task_info["task_id"]
        mod = task_info["module"]
        result = retest_task(tid, task_info["solution"], bcb_dataset)
        results.append(result)

        status = result["status"]
        if status == "PASS":
            passed += 1
            marker = "PASS"
        else:
            failed += 1
            # Check if still a module error
            err = result.get("error", "")
            if "No module named" in err:
                still_broken += 1
                marker = f"STILL_MISSING_MODULE"
            else:
                marker = f"{status}: {err[:80]}"

        print(f"  [{i+1}/{len(missing_tasks)}] {tid} (was missing {mod}): {marker}")

    print(f"\n{'='*60}")
    print(f"RETEST RESULTS:")
    print(f"  Total retested: {len(missing_tasks)}")
    print(f"  Now PASS: {passed}")
    print(f"  Still FAIL: {failed}")
    print(f"    Of which still missing module: {still_broken}")
    print(f"  Recovery rate: {passed}/{len(missing_tasks)} = {passed/len(missing_tasks):.1%}")
    print(f"\n  Original E1 SR: 324/798 = 40.6%")
    print(f"  Adjusted E1 SR: {324+passed}/798 = {(324+passed)/798:.1%}")

    out_path = os.path.join(os.path.dirname(E1_SAMPLES), "retest_missing_deps.json")
    with open(out_path, "w") as f:
        json.dump({"summary": {"total": len(missing_tasks), "passed": passed, "failed": failed, "still_broken": still_broken}, "results": results}, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
