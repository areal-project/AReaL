"""Sanity check ALFWorld query embedding cache hit rate + cross-snapshot diff.

Two angles to quantify resume-time retrieval risk:

  1. **Live cache hit rate** during a run: parses [RETRIEVE DEBUG] log lines
     to compute the fraction of queries that hit the persisted cache vs went
     to a fresh embed API call. High hit rate = resume is safe (most queries
     will hit cache after restore).

  2. **Cross-snapshot embedding stability**: compares query_embeddings.json
     between two snapshots. If the same task_description maps to different
     vectors, retrieval ranking can flip on resume — that's the BCB bug
     (EXPERIMENT_LOG.md:51, BF16 embedding non-determinism across vLLM
     instances). Reports max coordinate delta + cosine similarity per query.

Usage:
    # Live hit rate from a running/completed inner log
    python scripts/sanity_check_embedding_cache.py --hit-rate \\
        logs/alfworld_region_qwen72b_2section_confgate/*.log

    # Cross-snapshot drift
    python scripts/sanity_check_embedding_cache.py --diff \\
        results/alfworld/exp_alfworld_region_qwen72b_2section_confgate_*/local_cache/snapshot/s2_b30/local_cache/query_embeddings.json \\
        results/alfworld/exp_alfworld_region_qwen72b_2section_confgate_*/local_cache/snapshot/s2_b40/local_cache/query_embeddings.json

    # Default: hit rate on the most recent inner log
    python scripts/sanity_check_embedding_cache.py
"""
import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path


HIT_RATE_RE = re.compile(
    r"\[RETRIEVE DEBUG\] query_embeddings cached=(\d+), missing=(\d+)"
)
SINGLE_HIT_RE = re.compile(r"\[RETRIEVE DEBUG\] query_vec.*cached=(True|False)")


def hit_rate(log_path: Path) -> None:
    cached_total = 0
    missing_total = 0
    single_hit = 0
    single_miss = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = HIT_RATE_RE.search(line)
            if m:
                cached_total += int(m.group(1))
                missing_total += int(m.group(2))
                continue
            m = SINGLE_HIT_RE.search(line)
            if m:
                if m.group(1) == "True":
                    single_hit += 1
                else:
                    single_miss += 1

    print(f"Log: {log_path}")
    print(f"  Batch-level cached={cached_total}, missing={missing_total}")
    if cached_total + missing_total > 0:
        rate = cached_total / (cached_total + missing_total)
        print(f"  Batch-level hit rate: {rate:.2%}")
    print(f"  Per-query (single) hit={single_hit}, miss={single_miss}")
    if single_hit + single_miss > 0:
        rate = single_hit / (single_hit + single_miss)
        print(f"  Per-query hit rate: {rate:.2%}")
    if missing_total == 0 and single_miss == 0:
        print("  -> Empty: log has no [RETRIEVE DEBUG] lines (older run?)")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def diff_snapshots(p1: Path, p2: Path) -> None:
    d1 = json.load(open(p1))
    d2 = json.load(open(p2))
    common = set(d1) & set(d2)
    only_1 = set(d1) - set(d2)
    only_2 = set(d2) - set(d1)

    print(f"A: {p1}")
    print(f"   entries={len(d1)}")
    print(f"B: {p2}")
    print(f"   entries={len(d2)}")
    print(f"common={len(common)}, only_A={len(only_1)}, only_B={len(only_2)}")

    if not common:
        print("No overlap — cannot compare drift.")
        return

    identical = 0
    drift = []
    for q in common:
        v1 = d1[q]
        v2 = d2[q]
        if len(v1) != len(v2):
            print(f"  ! dim mismatch: {q[:60]!r} {len(v1)} vs {len(v2)}")
            continue
        if v1 == v2:
            identical += 1
            continue
        max_delta = max(abs(a - b) for a, b in zip(v1, v2))
        sim = cosine(v1, v2)
        drift.append((q, max_delta, sim))

    print(f"identical entries: {identical}/{len(common)} ({identical/len(common):.1%})")
    if drift:
        drift.sort(key=lambda x: -x[1])
        max_d = drift[0][1]
        worst_sim = min(d[2] for d in drift)
        mean_sim = sum(d[2] for d in drift) / len(drift)
        print(f"drifted entries: {len(drift)}")
        print(f"  max |Δcomponent|: {max_d:.6f}")
        print(f"  worst cosine: {worst_sim:.6f}")
        print(f"  mean cosine: {mean_sim:.6f}")
        print(f"  top-5 worst drift (cos < 1, may flip ranking):")
        for q, dlt, sim in sorted(drift, key=lambda x: x[2])[:5]:
            print(f"    cos={sim:.6f} maxΔ={dlt:.4e}  q={q[:80]!r}")


def latest_inner_log() -> Path | None:
    candidates = sorted(
        glob.glob("logs/alfworld_*/*.log"),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    return Path(candidates[0]) if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit-rate", nargs="?", const="__auto__",
                    help="Compute cache hit rate from an inner log. "
                         "If no path given, picks the most recent logs/alfworld_*/*.log.")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"),
                    help="Compare two query_embeddings.json snapshots.")
    args = ap.parse_args()

    if not args.hit_rate and not args.diff:
        # Default: hit rate on the latest log
        log = latest_inner_log()
        if log is None:
            sys.exit("No logs/alfworld_*/*.log found. Use --hit-rate <path> or --diff A B.")
        hit_rate(log)
        return

    if args.hit_rate:
        path = args.hit_rate
        if path == "__auto__":
            log = latest_inner_log()
            if log is None:
                sys.exit("No logs/alfworld_*/*.log found.")
            path = log
        hit_rate(Path(path))

    if args.diff:
        diff_snapshots(Path(args.diff[0]), Path(args.diff[1]))


if __name__ == "__main__":
    main()
