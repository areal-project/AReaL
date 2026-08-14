#!/usr/bin/env python3
"""Re-evaluate only the BCB tasks that FAILED due to missing eval-time packages.

Job 932328's no-mem Phase 1 lost ~41 tasks to 'No module named X' (django, tensorflow,
skimage, Crypto, librosa, ...) because the runner container lacked those packages — NOT
model errors. The model's solutions are already stored in samples.jsonl. This script
re-runs ONLY those tasks' stored solutions through the official BCB untrusted_check in a
full-dependency container. CPU-only; no GPU/vLLM needed.

Usage:
  python scripts/reeval_missing_pkg_tasks.py \
      --run_dir <.../deepseek-ai_DeepSeek-V3_rl-on> \
      --subset full
"""
import argparse, json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="epoch-run dir containing epoch1/{train,val}/samples.jsonl")
    ap.add_argument("--subset", default="full", choices=["full", "hard"])
    ap.add_argument("--bcb_repo", default=str(PROJECT_ROOT / "3rdparty" / "bigcodebench-main"))
    ap.add_argument("--eval_timeout", type=float, default=240.0)
    ap.add_argument("--hard_timeout", type=float, default=300.0)
    args = ap.parse_args()

    from memrl.bigcodebench_eval.task_wrappers import load_bcb_data
    from memrl.bigcodebench_eval.eval_utils import (
        sanitize_code,
        run_untrusted_check_with_hard_timeout,
        ensure_bigcodebench_on_path,
    )
    ensure_bigcodebench_on_path(args.bcb_repo)
    from bigcodebench.eval import PASS, FAIL, TIMEOUT  # type: ignore

    problems = load_bcb_data(subset=args.subset)
    run_dir = Path(args.run_dir)

    missing_re = re.compile(r"No module named .([\w\.]+)")
    results = {"recovered": [], "still_fail": [], "by_split": {}}

    for split in ("train", "val"):
        f = run_dir / "epoch1" / split / "samples.jsonl"
        if not f.exists():
            print(f"[WARN] missing {f}")
            continue
        targets = []
        for line in open(f):
            d = json.loads(line)
            if d.get("status") == "PASS":
                continue
            if missing_re.search(d.get("error") or ""):
                targets.append(d)
        print(f"\n=== {split}: {len(targets)} missing-pkg tasks to re-eval ===")
        rec = 0
        for d in targets:
            tid = d["task_id"]
            task = problems.get(tid)
            if task is None:
                print(f"  {tid}: NOT in dataset, skip")
                continue
            code = d.get("solution") or ""
            entry_point = str(task.get("entry_point", "task_func"))
            test_code = str(task.get("test", "") or "")
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                print(f"  {tid}: SYNTAX_ERROR {e}")
                results["still_fail"].append(tid)
                continue
            clean = sanitize_code(code, entry_point, bcb_repo=args.bcb_repo)
            code_prompt = str(task.get("code_prompt", ""))
            if code_prompt:
                clean = code_prompt + "\n    pass\n" + clean
            stat, details, err, hto = run_untrusted_check_with_hard_timeout(
                code=clean, test_code=test_code, entry_point=entry_point,
                max_as_limit=30 * 1024, max_data_limit=30 * 1024, max_stack_limit=10,
                min_time_limit=1.0, gt_time_limit=args.eval_timeout,
                hard_timeout_s=args.hard_timeout, bcb_repo=args.bcb_repo,
            )
            if stat == PASS and not err and not hto:
                print(f"  {tid}: PASS (recovered)")
                results["recovered"].append(tid); rec += 1
            else:
                short = (err or str(details) or str(stat))[:80]
                print(f"  {tid}: still {stat} | {short}")
                results["still_fail"].append(tid)
        results["by_split"][split] = {"targets": len(targets), "recovered": rec}

    print("\n" + "=" * 50)
    print(f"TOTAL recovered: {len(results['recovered'])}")
    print(f"TOTAL still fail: {len(results['still_fail'])}")

    # Recompute overall SR with recoveries applied.
    orig = json.load(open(run_dir / "summary.json"))["epochs"][0]
    tr, va = orig["train"], orig["val"]
    rec_tr = results["by_split"].get("train", {}).get("recovered", 0)
    rec_va = results["by_split"].get("val", {}).get("recovered", 0)
    new_tr_pass = tr["pass"] + rec_tr
    new_va_pass = va["pass"] + rec_va
    tot_pass = new_tr_pass + new_va_pass
    tot = tr["total"] + va["total"]
    print("\n=== Corrected no-mem pass@1 (missing-pkg tasks recovered) ===")
    print(f"  train: {tr['pass']}->{new_tr_pass}/{tr['total']} = {new_tr_pass/tr['total']*100:.2f}%")
    print(f"  val:   {va['pass']}->{new_va_pass}/{va['total']} = {new_va_pass/va['total']*100:.2f}%")
    print(f"  OVERALL 1140: {tr['pass']+va['pass']}->{tot_pass}/{tot} = {tot_pass/tot*100:.2f}%  (official ~50%)")

    json.dump(results, open(run_dir / "reeval_missing_pkg.json", "w"), indent=2)
    print(f"\nsaved -> {run_dir / 'reeval_missing_pkg.json'}")


if __name__ == "__main__":
    main()
