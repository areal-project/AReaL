"""Post-hoc reviewer-defense analysis for val_ablation runs.

Reads each ablation's retrieval_diagnostics.jsonl + ablation_summary.json and
computes:

  (1) Top-K Jaccard vs baseline (per task), summarized.
  (2) Score margin (score@K - score@(K+1)) and how it compares to the
      blend perturbation |Δscore| each ablation could induce.
  (3) Outcome coverage on retrieved candidates (not the global cache).
  (4) McNemar's test + paired bootstrap CI on per-task pass vs baseline.
  (5) Baseline-Q vs outcome correlation (calibration vs new-signal check).

Usage:
    python scripts/analyze_val_ablation.py results/val_ablation_confgate_e10
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

random.seed(42)


def load_jsonl(path):
    out = []
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def mcnemar_pvalue(b, c):
    """Exact binomial McNemar for small counts; chi-square otherwise.
    b = baseline wins (base ok, ablation fail), c = ablation wins (base fail, ablation ok).
    """
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        # Exact two-sided binomial with p=0.5
        from math import comb
        k = min(b, c)
        p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        return min(1.0, 2 * p)
    # Chi-square continuity-corrected
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Approximate p from chi2 with 1 df using survival of normal sqrt(chi2)
    z = math.sqrt(chi2)
    p = math.erfc(z / math.sqrt(2))
    return min(1.0, p)


def bootstrap_ci_diff(base_ok, abl_ok, B=2000, alpha=0.05):
    """Paired bootstrap CI for mean(abl_ok) - mean(base_ok)."""
    n = len(base_ok)
    if n == 0:
        return (0.0, 0.0)
    diffs = []
    idx = list(range(n))
    for _ in range(B):
        sample = [random.choice(idx) for _ in range(n)]
        d = sum(abl_ok[i] - base_ok[i] for i in sample) / n
        diffs.append(d)
    diffs.sort()
    lo = diffs[int(B * alpha / 2)]
    hi = diffs[int(B * (1 - alpha / 2))]
    return (lo, hi)


def parse_outcome(meta_outcome):
    if meta_outcome is None:
        return None
    if isinstance(meta_outcome, bool):
        return "success" if meta_outcome else "failure"
    s = str(meta_outcome).strip().lower()
    if s in ("success", "true", "1"):
        return "success"
    if s in ("failure", "fail", "false", "0"):
        return "failure"
    return None


def summarize_ablation(name, diag_rows, K):
    """Per-ablation summary numbers."""
    n = len(diag_rows)
    ok = sum(1 for r in diag_rows if r.get("ok"))
    margins = []
    succ_in_topk = []
    fail_in_topk = []
    cov_in_topk = []  # fraction of top-K cands with parseable outcome
    for r in diag_rows:
        cands = r.get("candidates_diag") or []
        topk = cands[:K]
        nplus = cands[K] if len(cands) > K else None
        if topk and nplus:
            margins.append(topk[-1]["score"] - nplus["score"])
        if topk:
            s = sum(1 for c in topk if parse_outcome(c.get("outcome")) == "success")
            f = sum(1 for c in topk if parse_outcome(c.get("outcome")) == "failure")
            present = sum(1 for c in topk if parse_outcome(c.get("outcome")) is not None)
            succ_in_topk.append(s)
            fail_in_topk.append(f)
            cov_in_topk.append(present / max(1, len(topk)))
    def avg(xs): return sum(xs) / max(1, len(xs))
    return {
        "n_tasks": n,
        "pass": ok,
        "pass_rate": ok / max(1, n),
        "median_margin": sorted(margins)[len(margins) // 2] if margins else None,
        "mean_margin": avg(margins) if margins else None,
        "avg_succ_in_topk": avg(succ_in_topk),
        "avg_fail_in_topk": avg(fail_in_topk),
        "outcome_coverage_in_topk": avg(cov_in_topk),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="results/val_ablation_confgate_e10 (or similar)")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--K", type=int, default=5, help="Default top-K for analyses (must match retrieve_k)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"Run dir not found: {run_dir}")

    # Discover ablations from subdirectories with retrieval_diagnostics.jsonl
    ablations = {}
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        diag = sub / "retrieval_diagnostics.jsonl"
        if diag.exists():
            rows = load_jsonl(str(diag))
            if rows:
                ablations[sub.name] = rows

    if not ablations:
        sys.exit(f"No retrieval_diagnostics.jsonl found in {run_dir}/<ablation>/")

    if args.baseline not in ablations:
        sys.exit(f"Baseline '{args.baseline}' not in discovered ablations: {sorted(ablations)}")

    base_rows = ablations[args.baseline]
    base_by_task = {r["task_id"]: r for r in base_rows}

    print("=" * 90)
    print(f"Run dir: {run_dir}")
    print(f"Baseline: {args.baseline}")
    print(f"Ablations found: {sorted(ablations)}")
    print("=" * 90)

    # ---- (1) per-ablation summary ----
    print("\n## Per-ablation summary")
    print(f"{'ablation':<24} {'n':>4} {'pass':>5} {'rate':>7} {'med_margin':>11} "
          f"{'avg_succ@K':>11} {'avg_fail@K':>11} {'cov@K':>7}")
    summary_map = {}
    for name in sorted(ablations):
        s = summarize_ablation(name, ablations[name], args.K)
        summary_map[name] = s
        print(f"{name:<24} {s['n_tasks']:>4} {s['pass']:>5} {s['pass_rate']:>7.3f} "
              f"{(s['median_margin'] or 0):>11.4f} "
              f"{s['avg_succ_in_topk']:>11.2f} {s['avg_fail_in_topk']:>11.2f} "
              f"{s['outcome_coverage_in_topk']:>7.2f}")

    # ---- (2) Top-K Jaccard vs baseline ----
    print("\n## Top-K Jaccard vs baseline (per-task, mean over tasks shared with baseline)")
    print(f"{'ablation':<24} {'shared':>7} {'mean_jacc':>11} {'identical_pct':>14}")
    for name in sorted(ablations):
        if name == args.baseline:
            continue
        rows = ablations[name]
        jaccs = []
        identical = 0
        for r in rows:
            tid = r["task_id"]
            br = base_by_task.get(tid)
            if br is None:
                continue
            j = jaccard(r.get("selected_ids", []), br.get("selected_ids", []))
            jaccs.append(j)
            if j == 1.0:
                identical += 1
        mean_j = sum(jaccs) / max(1, len(jaccs))
        ident_pct = 100.0 * identical / max(1, len(jaccs))
        print(f"{name:<24} {len(jaccs):>7} {mean_j:>11.3f} {ident_pct:>13.1f}%")

    # ---- (3) Margin vs likely Δscore from outcome blend ----
    print("\n## Margin vs blend perturbation")
    print("If baseline median margin >> w_q * beta * 1.0, blend cannot flip top-K.")
    print("(weight_q is typically ~0.5; blend Δq ≤ beta on z-normalized q; Δscore ≈ 0.5*beta in z space)")
    base_s = summary_map[args.baseline]
    base_margin = base_s["median_margin"] or 0.0
    print(f"  baseline median_margin = {base_margin:.4f}")
    for beta in (0.1, 0.3, 0.5):
        delta = 0.5 * beta  # rough upper bound on |Δscore| in z space
        verdict = "CAN flip" if delta > base_margin else "cannot flip"
        print(f"  beta={beta}: max |Δscore| ~ {delta:.3f}  vs  margin {base_margin:.4f}  => {verdict}")

    # ---- (4) McNemar + bootstrap CI vs baseline ----
    print("\n## Paired significance vs baseline")
    print(f"{'ablation':<24} {'b':>5} {'c':>5} {'p_mcnemar':>11} "
          f"{'Δpass':>7} {'ci95_low':>9} {'ci95_high':>9}")
    base_ok_by_tid = {r["task_id"]: int(bool(r.get("ok"))) for r in base_rows}
    for name in sorted(ablations):
        if name == args.baseline:
            continue
        rows = ablations[name]
        ok_by_tid = {r["task_id"]: int(bool(r.get("ok"))) for r in rows}
        tids = sorted(set(ok_by_tid) & set(base_ok_by_tid))
        base_ok = [base_ok_by_tid[t] for t in tids]
        abl_ok = [ok_by_tid[t] for t in tids]
        b = sum(1 for i in range(len(tids)) if base_ok[i] == 1 and abl_ok[i] == 0)  # base win
        c = sum(1 for i in range(len(tids)) if base_ok[i] == 0 and abl_ok[i] == 1)  # abl win
        p = mcnemar_pvalue(b, c)
        delta = (sum(abl_ok) - sum(base_ok)) / max(1, len(tids))
        lo, hi = bootstrap_ci_diff(base_ok, abl_ok)
        print(f"{name:<24} {b:>5} {c:>5} {p:>11.4f} "
              f"{delta:>+7.3f} {lo:>+9.3f} {hi:>+9.3f}")

    # ---- (5) Baseline Q vs outcome correlation ----
    print("\n## Baseline Q vs outcome correlation (on retrieved candidates)")
    print("If high: baseline Q already encodes outcome; blend is calibration, not new signal.")
    succ_qs, fail_qs = [], []
    for r in base_rows:
        for c in (r.get("candidates_diag") or []):
            o = parse_outcome(c.get("outcome"))
            q = c.get("q_est")
            if q is None or o is None:
                continue
            if o == "success":
                succ_qs.append(q)
            elif o == "failure":
                fail_qs.append(q)
    if succ_qs and fail_qs:
        ms, mf = sum(succ_qs) / len(succ_qs), sum(fail_qs) / len(fail_qs)
        # Pooled std for crude Cohen's d
        all_qs = succ_qs + fail_qs
        mean = sum(all_qs) / len(all_qs)
        var = sum((x - mean) ** 2 for x in all_qs) / max(1, len(all_qs) - 1)
        sd = math.sqrt(var) if var > 0 else 1.0
        d = (ms - mf) / sd
        print(f"  n_succ={len(succ_qs)}  mean_q={ms:.4f}")
        print(f"  n_fail={len(fail_qs)}  mean_q={mf:.4f}")
        print(f"  diff = {ms - mf:+.4f},  Cohen's d ≈ {d:.3f}")
        if abs(d) > 0.5:
            print("  -> Q already separates outcome decently; blend is largely calibration.")
        else:
            print("  -> Q barely distinguishes outcome; blend adds genuinely new signal.")
    else:
        print("  Insufficient labelled candidates to compute.")

    print("\nDone.")


if __name__ == "__main__":
    main()
