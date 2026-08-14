#!/usr/bin/env python3
"""Compute true per-section SR from Q-DEBUG log lines.

The runner's "Section X Training Stats: Success Rate=Y%" line only reflects
mini-batches sampled AFTER a resume (section_trajectories is not restored from
ckpt). This script bypasses that: it counts every [RL Q-DEBUG] line's
`successes=[...]` from ALL logs of a given experiment (original + all resumes),
groups by section, and prints the true SR.

Usage:
    python scripts/compute_section_sr.py <log_pattern>
    e.g.  python scripts/compute_section_sr.py 'logs/llb_os_memrl_gpt41mini_*.log'
          python scripts/compute_section_sr.py 'logs/llb_os_region_fs_gpt41mini_*.log'
"""
import glob
import re
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("usage: compute_section_sr.py <log_glob> [<log_glob2> ...]")
        sys.exit(1)
    logs = []
    for pat in sys.argv[1:]:
        logs.extend(glob.glob(pat))
    logs = sorted(set(logs), key=lambda p: __import__("os").path.getmtime(p))
    if not logs:
        print(f"No logs match: {sys.argv[1:]}")
        sys.exit(1)

    # Per section: track UNIQUE (mini_batch_idx) -> list of successes
    # We need to dedupe because resume replays the last-completed batch.
    # Key: (section, mini_batch_idx, sample_position) — but Q-DEBUG doesn't
    # print mini_batch_idx directly. Simpler: dedupe by consecutive lines within
    # each log, then concat unique (section, mb_idx) entries across logs by
    # tracking the newest occurrence.
    #
    # Approach: for each log, walk sequentially; track current section from
    # "STARTING SECTION N/10" markers; each Q-DEBUG carries a mini-batch index
    # from the preceding "Processing mini-batch M/70" log line. Key by
    # (section, mb_idx); later occurrences (from resume) overwrite earlier.
    successes_by_key = {}  # (section, mb_idx) -> list of 0/1
    processed_batches_by_section = defaultdict(set)

    section_re = re.compile(r"STARTING SECTION (\d+)/\d+")
    mb_re = re.compile(r"Processing mini-batch (\d+)/70")
    qdbg_re = re.compile(r"successes=\[([01, ]*)\]")

    for logpath in logs:
        cur_section = None
        cur_mb = None
        with open(logpath, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = section_re.search(line)
                if m:
                    cur_section = int(m.group(1)); cur_mb = None; continue
                m = mb_re.search(line)
                if m and cur_section is not None:
                    cur_mb = int(m.group(1)); continue
                m = qdbg_re.search(line)
                if m and cur_section is not None and cur_mb is not None:
                    succ_list = [int(x) for x in m.group(1).replace(" ", "").split(",") if x != ""]
                    successes_by_key[(cur_section, cur_mb)] = succ_list
                    processed_batches_by_section[cur_section].add(cur_mb)

    print("=" * 60)
    print(f"Logs merged: {len(logs)}")
    for L in logs:
        print(f"  - {L}")
    print("=" * 60)
    print(f"{'Section':>8} | {'batches':>7} | {'samples':>7} | {'succ':>5} | {'SR':>6}")
    print("-" * 60)
    total_s = total_n = 0
    for sec in sorted(processed_batches_by_section):
        succ_list = []
        for mb in sorted(processed_batches_by_section[sec]):
            succ_list += successes_by_key[(sec, mb)]
        s = sum(succ_list); n = len(succ_list)
        if n > 0:
            print(f"{sec:>8} | {len(processed_batches_by_section[sec]):>7} | {n:>7} | {s:>5} | {100*s/n:>5.2f}%")
        total_s += s; total_n += n
    print("-" * 60)
    if total_n > 0:
        print(f"{'TOTAL':>8} |         | {total_n:>7} | {total_s:>5} | {100*total_s/total_n:>5.2f}%")


if __name__ == "__main__":
    main()
