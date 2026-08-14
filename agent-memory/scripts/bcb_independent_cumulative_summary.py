#!/usr/bin/env python3
"""Build the canonical DeepSeek-V3 BCB independent/cumulative SR audit.

Independent SR: pass rate on the full train/val split at that epoch.
Cumulative SR: fraction of the fixed split passed at least once through that epoch.

The manifest intentionally selects only the validated run chain for each method,
excluding interrupted/stale/known-bad attempts (notably the pre-fix Mem0 E10).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path("/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench")
OUT = Path("scripts/generated/bcb_independent_cumulative_sr.md")

# Method -> epoch -> canonical run directory. The phase paths are appended below.
MANIFEST: Dict[str, Dict[int, Path]] = {
    "No-memory": {
        1: ROOT / "deepseek_v3_local_nomem/bigcodebench_eval/instruct_full/memory/20260618_173854_deepseek-ai_DeepSeek-V3_rl-on",
    },
    "MemRL": {
        **{e: ROOT / "deepseek_v3_local_memrl/bigcodebench_eval/instruct_full/memory/20260618_211416_deepseek-ai_DeepSeek-V3_rl-on" for e in range(1, 4)},
        **{e: ROOT / "deepseek_v3_local_memrl/bigcodebench_eval/instruct_full/memory/20260619_165450_deepseek-ai_DeepSeek-V3_rl-on" for e in range(4, 11)},
    },
    "MemP": {
        **{e: ROOT / "deepseek_v3_memp/bigcodebench_eval/instruct_full/memory/20260709_102932_deepseek-ai_DeepSeek-V3_rl-off" for e in range(1, 6)},
        **{e: ROOT / "deepseek_v3_memp/bigcodebench_eval/instruct_full/memory/20260710_032817_deepseek-ai_DeepSeek-V3_rl-off" for e in range(6, 10)},
    },
    "RAG": {
        **{e: ROOT / "deepseek_v3_rag/bigcodebench_eval/instruct_full/memory/20260712_071437_deepseek-ai_DeepSeek-V3_rl-off" for e in range(1, 9)},
    },
    "Self-RAG": {
        1: ROOT / "deepseek_v3_selfrag/bigcodebench_eval/instruct_full/memory/20260714_061449_deepseek-ai_DeepSeek-V3_rl-off",
        **{e: ROOT / "deepseek_v3_selfrag/bigcodebench_eval/instruct_full/memory/20260714_160058_deepseek-ai_DeepSeek-V3_rl-off" for e in range(2, 4)},
        **{e: ROOT / "deepseek_v3_selfrag/bigcodebench_eval/instruct_full/memory/20260715_063844_deepseek-ai_DeepSeek-V3_rl-off" for e in range(4, 7)},
        **{e: ROOT / "deepseek_v3_selfrag/bigcodebench_eval/instruct_full/memory/20260716_015547_deepseek-ai_DeepSeek-V3_rl-off" for e in range(7, 11)},
    },
    "Mem0": {
        **{e: ROOT / "deepseek_v3_mem0/bigcodebench_eval/instruct_full/memory/20260718_090014_deepseek-ai_DeepSeek-V3_rl-off" for e in range(1, 3)},
        **{e: ROOT / "deepseek_v3_mem0/bigcodebench_eval/instruct_full/memory/20260719_032415_deepseek-ai_DeepSeek-V3_rl-off" for e in range(3, 10)},
        10: ROOT / "deepseek_v3_mem0/bigcodebench_eval/instruct_full/memory/20260720_041329_deepseek-ai_DeepSeek-V3_rl-off",
    },
    "Region+FS": {
        **{e: ROOT / "deepseek_v3_local_region_fs/bigcodebench_eval/instruct_full/region/20260619_153259_deepseek-ai_DeepSeek-V3_region" for e in range(1, 8)},
        **{e: ROOT / "deepseek_v3_local_region_fs/bigcodebench_eval/instruct_full/region/20260623_161817_deepseek-ai_DeepSeek-V3_region" for e in range(8, 11)},
    },
    "Leaf (legacy)": {
        **{e: ROOT / "deepseek_v3_local_region_fs_leaf/bigcodebench_eval/instruct_full/region/20260701_145908_deepseek-ai_DeepSeek-V3_region" for e in range(1, 3)},
        **{e: ROOT / "deepseek_v3_local_region_fs_leaf/bigcodebench_eval/instruct_full/region/20260703_192548_deepseek-ai_DeepSeek-V3_region" for e in range(3, 10)},
        10: ROOT / "deepseek_v3_local_region_fs_leaf/bigcodebench_eval/instruct_full/region/20260705_163036_deepseek-ai_DeepSeek-V3_region",
    },
    "Leaf (split-prior fix; in progress)": {
        **{e: ROOT / "deepseek_v3_local_region_fs_leaf_splitprior_fix/bigcodebench_eval/instruct_full/region/20260722_115359_deepseek-ai_DeepSeek-V3_region" for e in range(1, 8)},
    },
}


def read_samples(path: Path) -> Dict[str, bool]:
    records: Dict[str, bool] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tid = str(r["task_id"])
            records[tid] = r.get("status") == "PASS"
    return records


def fmt(passed: int, total: int) -> str:
    return f"{passed}/{total} ({100 * passed / total:.2f}%)"


def audit_method(name: str, epochs: Dict[int, Path]) -> Tuple[List[str], Dict[str, object]]:
    lines = [f"## {name}", "", "| Epoch | Train independent SR | Train cumulative SR | Val independent SR | Val cumulative SR |", "|---:|---:|---:|---:|---:|"]
    solved: Dict[str, set] = {"train": set(), "val": set()}
    totals: Dict[str, Optional[set]] = {"train": None, "val": None}
    rows = {}
    for ep in range(1, 11):
        run = epochs.get(ep)
        if run is None:
            lines.append(f"| E{ep} | — | — | — | — |")
            continue
        cells = []
        data: Dict[str, Tuple[int, int, int]] = {}
        for phase in ("train", "val"):
            path = run / f"epoch{ep}" / phase / "samples.jsonl"
            if not path.is_file():
                cells.extend(["—", "—"])
                continue
            rec = read_samples(path)
            expected = 798 if phase == "train" else 342
            if len(rec) != expected:
                raise RuntimeError(f"{name} E{ep} {phase}: expected {expected} task results, got {len(rec)}: {path}")
            ids = set(rec)
            if totals[phase] is None:
                totals[phase] = ids
            elif ids != totals[phase]:
                raise RuntimeError(f"{name} E{ep} {phase}: task split differs from earlier epoch")
            passed = {tid for tid, ok in rec.items() if ok}
            solved[phase].update(passed)
            data[phase] = (len(passed), len(ids), len(solved[phase]))
            cells.extend([fmt(len(passed), len(ids)), fmt(len(solved[phase]), len(ids))])
        rows[ep] = data
        lines.append(f"| E{ep} | {' | '.join(cells)} |")
    lines.append("")
    return lines, {"rows": rows, "totals": totals, "solved": solved}


MULTI_EVAL = {
    # Only validated artifacts are listed. Mem0's old E10 is retained below as
    # an excluded diagnostic because the corrected E10 is the canonical result.
    "No-memory": [],
    "Pass@10": [],
    "RAG": [],
    "MemP": [],
    "MemRL": [],
    "Self-RAG": [
        ("E10", ROOT / "deepseek_v3_selfrag/bigcodebench_eval/instruct_full/memory/20260716_015547_deepseek-ai_DeepSeek-V3_rl-off/epoch10/val_multi_summary.json", "canonical"),
    ],
    "Region+FS": [],
    "Leaf (legacy)": [],
    "Leaf (split-prior fix; in progress)": [],
    "Mem0": [
        ("E10", ROOT / "deepseek_v3_mem0/bigcodebench_eval/instruct_full/memory/20260720_041329_deepseek-ai_DeepSeek-V3_rl-off/epoch10/val_multi_summary.json", "canonical; corrected embedding-dimension rerun"),
        ("E10", ROOT / "deepseek_v3_mem0/bigcodebench_eval/instruct_full/memory/20260719_032415_deepseek-ai_DeepSeek-V3_rl-off/epoch10/val_multi_summary.json", "excluded diagnostic; pre-fix E10"),
    ],
}


def multi_eval_section() -> List[str]:
    lines = [
        "## Multi-eval (stochastic val, separate from standard-val SR)",
        "",
        "Multi-eval is **not** folded into the independent/cumulative columns above: those columns use one standard val generation per epoch. `—` means no completed multi-eval artifact exists yet.",
        "",
        "| Method | Checkpoint / epoch | Run 0 | Run 1 | Run 2 | Mean | 95% CI | Status / provenance |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for method in ("No-memory", "Pass@10", "RAG", "Self-RAG", "Mem0", "MemP", "MemRL", "Region+FS", "Leaf (legacy)", "Leaf (split-prior fix; in progress)"):
        entries = MULTI_EVAL[method]
        if not entries:
            lines.append(f"| {method} | — | — | — | — | — | — | not available |")
            continue
        for epoch, path, note in entries:
            d = json.loads(path.read_text())
            runs = d.get("individual_runs", [])
            vals = [f"{100 * float(x):.2f}%" for x in runs]
            vals += ["—"] * (3 - len(vals))
            lines.append(
                f"| {method} | {epoch} | {vals[0]} | {vals[1]} | {vals[2]} | "
                f"{100 * float(d.get('mean_pass@1', 0)):.2f}% | ±{100 * float(d.get('ci_95', 0)):.2f}pp | {note} |"
            )
    lines.extend([
        "",
        "- **Mem0 canonical E10 multi-eval** is the corrected rerun: 49.42% ± 1.45pp. The old pre-fix E10 multi-eval is retained only for auditability and is excluded from main comparisons.",
        "- **Self-RAG E10 multi-eval** is 42.79% ± 0.84pp.",
        "- MemP E10 + multi-eval and RAG E9–E10 + multi-eval remain pending; they are intentionally blank rather than imputed.",
        "",
    ])
    return lines


def passk_section() -> List[str]:
    p = ROOT / "deepseek_v3_passk10/bigcodebench_eval/instruct_full/memory/20260711_001301_deepseek-ai_DeepSeek-V3_rl-off/baseline_passk/results.jsonl"
    by_round: Dict[int, Dict[str, bool]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_round.setdefault(int(r["round"]), {})[str(r["task_id"])] = bool(r["pass"])
    solved = set()
    lines = ["## Pass@10 (train only)", "", "Pass@10 skips tasks once solved, so its later-round independent rate is conditional on the remaining unsolved tasks and is **not** comparable to a full-split independent SR. The cumulative column is the valid pass@k metric.", "", "| Round | Tasks attempted | Newly passed | Conditional SR | Cumulative SR |", "|---:|---:|---:|---:|---:|"]
    for rd in range(1, 11):
        rec = by_round.get(rd, {})
        passed = {tid for tid, ok in rec.items() if ok}
        new = passed - solved
        solved.update(passed)
        attempted = len(rec)
        cond = f"{100*len(new)/attempted:.2f}%" if attempted else "—"
        lines.append(f"| R{rd} | {attempted} | {len(new)} | {cond} | {fmt(len(solved), 798)} |")
    lines.append("")
    return lines


def main() -> None:
    out = [
        "# BCB DeepSeek-V3 — Independent and Cumulative SR Audit",
        "",
        "> **最后更新**: 2026-07-23",
        "> **Generated from per-task `samples.jsonl` / `results.jsonl` artifacts.**",
        "> `independent SR` = that epoch's own pass rate; `cumulative SR` = a task has passed in at least one completed epoch up to and including that row.",
        "> Dataset split: BCB `instruct_full`, seed 42: **798 train / 342 val**.",
        "> A dash denotes a genuinely absent epoch, not zero success. MemP/RAG completion is pending; No-memory was only evaluated once.",
        "",
        "## Scope and provenance",
        "",
        "- **No-memory**: single validated E1 baseline; its one row is both independent and cumulative.",
        "- **MemRL**: E1–E3 + resumed E4–E10 chain.",
        "- **MemP**: E1–E5 + resumed E6–E9; E10 pending.",
        "- **RAG**: E1–E8; E9/E10 pending after the old update hang.",
        "- **Self-RAG**: validated resumed E1–E10 chain.",
        "- **Mem0**: E1–E2 normal run + E3–E9 normal continuation + corrected E10 rerun. The bad old E10 is excluded.",
        "- **Pass@10**: train only; cumulative pass@k is valid, while later independent rounds are conditional because solved tasks are skipped.",
        "- **Region+FS**: E1–E7 + resumed E8–E10 validated chain.",
        "- **Leaf (legacy)**: historical E1–E10 chain retained for reproducibility; its old split semantics are not the current implementation.",
        "- **Leaf (split-prior fix; in progress)**: new clean E1–E10 rerun with `soft_source_conserving`; only completed phase artifacts are shown.",
        "",
    ]
    final = {}
    for name in ("No-memory", "MemRL", "MemP", "RAG", "Self-RAG", "Mem0", "Region+FS", "Leaf (legacy)", "Leaf (split-prior fix; in progress)"):
        lines, meta = audit_method(name, MANIFEST[name])
        out.extend(lines)
        final[name] = meta
    out.extend(passk_section())
    out.extend(multi_eval_section())
    out.extend([
        "## Latest available cumulative coverage",
        "",
        "| Method | Latest completed epoch | Train cumulative SR | Val cumulative SR |",
        "|---|---:|---:|---:|",
    ])
    for name, meta in final.items():
        rows = meta["rows"]
        last = max(rows) if rows else None
        tr = rows[last].get("train") if last else None
        va = rows[last].get("val") if last else None
        tr_s = fmt(tr[2], tr[1]) if tr else "—"
        va_s = fmt(va[2], va[1]) if va else "—"
        out.append(f"| {name} | E{last} | {tr_s} | {va_s} |")
    out.extend([
        "| Pass@10 | R10 | 388/798 (48.62%) | — |",
        "",
        "## Notes",
        "",
        "- These cumulative figures are **within-method, same fixed split**, and should not be mistaken for a single-epoch generalization score.",
        "- Standard E10 multi-eval is a separate stochastic-estimation artifact; it is intentionally not mixed into the standard-val cumulative column.",
        "- Region+FS and both Leaf lineages are included for direct comparison. The corrected Leaf rerun is explicitly marked in progress and should not be summarized as a completed 10-epoch result yet.",
        "",
    ])
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(OUT)

if __name__ == "__main__":
    main()
