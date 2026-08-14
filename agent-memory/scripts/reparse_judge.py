#!/usr/bin/env python3
"""
Reparse HLE judge results from llm_calls.jsonl files.

The original judge parse regex `correct\s*:\s*(yes|no)` fails to match
the markdown bold format `**correct**: yes` that gemini-2.5-pro outputs.
This script re-parses with a fixed regex and reports the true accuracy.

Usage:
    python scripts/reparse_judge.py [--dir results/hle] [--verbose]
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path


def reparse_correct(raw_judge: str) -> str:
    """Re-parse correct field from raw judge text, handling markdown bold."""
    if not raw_judge:
        return "no"
    # 1. Try JSON parse first
    try:
        m = re.search(r"\{[\s\S]*\}", raw_judge)
        if m:
            obj = json.loads(m.group(0))
            corr = str(obj.get("correct", "no")).strip().lower()
            return "yes" if "yes" in corr else "no"
    except Exception:
        pass
    # 2. Regex: handle **correct**: yes/no, *correct*: yes/no, correct: yes/no
    m = re.search(r"\*{0,2}correct\*{0,2}\s*:\s*(yes|no)", raw_judge, re.I)
    if m:
        return m.group(1).strip().lower()
    return "no"


def reparse_model_answer(raw_judge: str) -> str:
    """Re-parse extracted_final_answer from raw judge text."""
    if not raw_judge:
        return None
    try:
        m = re.search(r"\{[\s\S]*\}", raw_judge)
        if m:
            obj = json.loads(m.group(0))
            return obj.get("extracted_final_answer") or obj.get("extracted_answer")
    except Exception:
        pass
    m = re.search(
        r"\*{0,2}extracted_final_answer\*{0,2}\s*:\s*(.+?)(?:\n|$)",
        raw_judge, re.I
    )
    if m:
        return m.group(1).strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/hle", help="HLE results directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.dir}/exp_hle_memrl_gemini3_*/local_cache/llm_calls.jsonl"))
    if not files:
        print(f"No llm_calls.jsonl found in {args.dir}")
        return

    grand = defaultdict(int)
    per_exp = []

    for path in files:
        exp_name = Path(path).parent.parent.name
        stats = defaultdict(int)
        misparsed_examples = []

        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue

                if d.get("type") == "solution":
                    stats["total_solutions"] += 1
                    if not d.get("response"):
                        stats["empty_solutions"] += 1
                    if d.get("meta", {}).get("error"):
                        stats["error_solutions"] += 1

                if d.get("type") == "judge":
                    stats["total_judges"] += 1
                    raw = d.get("response", "")
                    old_parsed = d.get("parsed", {}).get("correct", "no")
                    new_parsed = reparse_correct(raw)

                    if old_parsed == "yes":
                        stats["old_yes"] += 1
                    if new_parsed == "yes":
                        stats["new_yes"] += 1
                    if old_parsed != new_parsed:
                        stats["changed"] += 1
                        if new_parsed == "yes":
                            stats["misparsed_yes_as_no"] += 1
                            if len(misparsed_examples) < 3:
                                misparsed_examples.append({
                                    "qid": d.get("meta", {}).get("question_id"),
                                    "gold": d.get("meta", {}).get("gold", "")[:80],
                                    "model_answer": reparse_model_answer(raw),
                                    "raw_snippet": raw[:200],
                                })

        if stats["total_judges"] == 0:
            continue

        tj = stats["total_judges"]
        ts = stats["total_solutions"]
        per_exp.append((exp_name, stats, misparsed_examples))
        for k, v in stats.items():
            grand[k] += v

    # --- Report ---
    print("=" * 80)
    print("HLE Judge Reparse Report")
    print("=" * 80)

    for exp_name, stats, examples in per_exp:
        tj = stats["total_judges"]
        ts = stats["total_solutions"]
        print(f"\n--- {exp_name} ---")
        print(f"  Solutions: {ts} (empty: {stats['empty_solutions']}, error: {stats['error_solutions']})")
        print(f"  Judges:    {tj}")
        print(f"  Old yes:   {stats['old_yes']} ({stats['old_yes']/tj*100:.1f}%)")
        print(f"  New yes:   {stats['new_yes']} ({stats['new_yes']/tj*100:.1f}%)")
        print(f"  Misparsed: {stats['misparsed_yes_as_no']} (yes→no)")

        if args.verbose and examples:
            for ex in examples:
                print(f"    Example: QID={ex['qid']}, gold={ex['gold']}, model={ex['model_answer']}")

    tj = grand["total_judges"]
    ts = grand["total_solutions"]
    if tj:
        print(f"\n{'=' * 80}")
        print(f"GRAND TOTAL across {len(per_exp)} experiments")
        print(f"{'=' * 80}")
        print(f"  Solutions: {ts} (empty: {grand['empty_solutions']} = {grand['empty_solutions']/ts*100:.1f}%, error: {grand['error_solutions']})")
        print(f"  Judges:    {tj}")
        print(f"  Old acc:   {grand['old_yes']}/{tj} = {grand['old_yes']/tj*100:.2f}%")
        print(f"  New acc:   {grand['new_yes']}/{tj} = {grand['new_yes']/tj*100:.2f}%")
        print(f"  Misparsed: {grand['misparsed_yes_as_no']} items (yes wrongly parsed as no)")
        print(f"  Accuracy loss from bug: {grand['misparsed_yes_as_no']/tj*100:.2f}%")
        effective = ts - grand["empty_solutions"]
        if effective:
            print(f"\n  Effective acc (excl empty solutions): {grand['new_yes']}/{effective} = {grand['new_yes']/effective*100:.2f}%")

    # --- Memory pollution analysis ---
    print(f"\n{'=' * 80}")
    print("MEMORY POLLUTION ANALYSIS")
    print(f"{'=' * 80}")
    print("""
The memory system stores trajectories with success=True/False based on
the PARSED judge result (line 1085: successes = [bool(r["correct"])]).

Because of the misparse bug:
  - Items that were actually CORRECT got stored with success=False
  - Their q_value was set to 0.0 instead of 1.0
  - Their metadata has success=False

This means:
  1. "Successful" memories contain only the ~3% that the parser got right
  2. ~8% of memories are mislabeled as failures when they are successes
  3. The RL value updates (update_values) received wrong reward signals
  4. Memory retrieval that filters by success flag will miss good memories

Impact: Memory quality is degraded. The RL loop learned from wrong labels.
Recommendation: Fix the parse bug and RE-RUN from scratch (epoch 0).
Simply resuming will compound the error since existing memories have
wrong labels that influence future retrieval and value estimates.
""")


if __name__ == "__main__":
    main()
