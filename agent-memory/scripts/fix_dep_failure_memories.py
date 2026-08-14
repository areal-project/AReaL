#!/usr/bin/env python3
"""
Fix memories for tasks that failed due to missing modules but actually PASS.

Modifies the latest checkpoint to:
1. Fix q_cache: flip negative Q-values to positive for affected memory IDs
2. Fix mem_cache: set success=True, type=procedure for affected entries
3. Fix textual_memory.json: update metadata in qdrant snapshot payload

Usage:
    python scripts/fix_dep_failure_memories.py [--dry-run]
"""

import json
import sys
import os
from pathlib import Path

RETEST_RESULTS = (
    "/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/"
    "instruct_full/memory/20260426_195914_gpt-4o-2024-11-20_rl-on/epoch1/train/retest_missing_deps.json"
)

E1_SAMPLES = (
    "/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/"
    "instruct_full/memory/20260426_195914_gpt-4o-2024-11-20_rl-on/epoch1/train/samples.jsonl"
)

CHECKPOINT = (
    "/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/"
    "instruct_full/memory/20260426_195914_gpt-4o-2024-11-20_rl-on/epoch2/snapshot/step_750"
)


def main():
    dry_run = "--dry-run" in sys.argv

    # Load retest results
    with open(RETEST_RESULTS) as f:
        retest = json.load(f)
    passed_ids = {r["task_id"] for r in retest["results"] if r["status"] == "PASS"}
    print(f"Tasks that now PASS: {len(passed_ids)}")

    # Get prompts for passed tasks
    tid_to_prompt = {}
    with open(E1_SAMPLES) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] in passed_ids:
                tid_to_prompt[d["task_id"]] = d["prompt"]

    # Load checkpoint files
    ckpt = Path(CHECKPOINT)
    q_cache_path = ckpt / "local_cache" / "q_cache.json"
    mem_cache_path = ckpt / "local_cache" / "mem_cache.json"
    tm_path = ckpt / "cube" / "textual_memory.json"

    with open(q_cache_path) as f:
        q_cache = json.load(f)
    with open(mem_cache_path) as f:
        mem_cache = json.load(f)
    with open(tm_path) as f:
        tm = json.load(f)

    # Build index: memory_text -> list of (tm_index, entry_id)
    prompt_to_tm_indices = {tid: [] for tid in passed_ids}
    for idx, entry in enumerate(tm):
        memory_text = entry["payload"].get("memory", "")
        for tid, prompt in tid_to_prompt.items():
            if memory_text and prompt.startswith(memory_text[:100]):
                prompt_to_tm_indices[tid].append((idx, entry["id"]))
                break

    # Collect all affected qdrant IDs
    affected_ids = set()
    for tid, entries in prompt_to_tm_indices.items():
        for _, eid in entries:
            affected_ids.add(eid)

    total_tm = sum(len(v) for v in prompt_to_tm_indices.values())
    in_q = sum(1 for eid in affected_ids if eid in q_cache)
    print(f"Affected textual_memory entries: {total_tm}")
    print(f"Affected entries in q_cache: {in_q}")

    # === Fix 1: q_cache ===
    fixed_q = 0
    for eid in affected_ids:
        if eid in q_cache:
            old_q = q_cache[eid]
            new_q = abs(old_q) if old_q < 0 else old_q
            if not dry_run:
                q_cache[eid] = new_q
            print(f"  [q_cache] {eid[:12]}... q: {old_q:.4f} -> {new_q:.4f}")
            fixed_q += 1

    # === Fix 2: mem_cache ===
    fixed_mc = 0
    for eid in affected_ids:
        if eid in mem_cache:
            meta = mem_cache[eid].get("metadata", {})
            old_success = meta.get("success")
            old_type = meta.get("type")
            if not dry_run:
                meta["success"] = True
                if meta.get("type") == "adjustment":
                    meta["type"] = "procedure"
                meta["q_value"] = q_cache.get(eid, 0.0)
                mem_cache[eid]["metadata"] = meta
            print(f"  [mem_cache] {eid[:12]}... success: {old_success}->True, type: {old_type}->procedure")
            fixed_mc += 1

    # === Fix 3: textual_memory.json (qdrant snapshot) ===
    fixed_tm = 0
    for tid, entries in prompt_to_tm_indices.items():
        for idx, eid in entries:
            entry = tm[idx]
            payload_meta = entry["payload"].get("metadata", {})
            old_type = payload_meta.get("type")
            old_conf = payload_meta.get("confidence")
            if not dry_run:
                if payload_meta.get("type") == "adjustment":
                    payload_meta["type"] = "procedure"
                payload_meta["success"] = "True"
                # Boost confidence back to 100 (was reduced by confidence_factor for adjustments)
                if old_conf and float(old_conf) < 100:
                    payload_meta["confidence"] = "100.0"
                entry["payload"]["metadata"] = payload_meta
                tm[idx] = entry
            print(f"  [textual_memory] {tid} -> {eid[:12]}... type: {old_type}->procedure, conf: {old_conf}->100.0")
            fixed_tm += 1

    print(f"\n{'DRY RUN - ' if dry_run else ''}Summary:")
    print(f"  Fixed q_cache entries: {fixed_q}")
    print(f"  Fixed mem_cache entries: {fixed_mc}")
    print(f"  Fixed textual_memory entries: {fixed_tm}")

    if not dry_run:
        # Backup originals
        for p in [q_cache_path, mem_cache_path, tm_path]:
            bak = p.with_suffix(p.suffix + ".bak")
            if not bak.exists():
                import shutil
                shutil.copy2(p, bak)
                print(f"  Backed up: {bak}")

        with open(q_cache_path, "w") as f:
            json.dump(q_cache, f, ensure_ascii=False)
        with open(mem_cache_path, "w") as f:
            json.dump(mem_cache, f, ensure_ascii=False)
        with open(tm_path, "w") as f:
            json.dump(tm, f, ensure_ascii=False)
        print(f"\n  Written changes to: {CHECKPOINT}")
    else:
        print("\n  Rerun without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
