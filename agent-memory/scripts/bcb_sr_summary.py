#!/usr/bin/env python3
"""汇总 BCB 实验各 epoch 的 train / eval(val) SR，跨 run 合并。

用法:
    python MemRL/scripts/bcb_sr_summary.py [OUTPUT_DIR]

OUTPUT_DIR 默认 deepseek_v3_memp。会扫描该目录下**所有** run 的
epoch_summary.json，按 epoch 号合并(resume 会产生新 run 目录);同一 epoch
若多个 run 都有,取 mtime 最新的。同时读 val_multi_summary.json(E10 多轮 mean±CI)。
"""
import json
import os
import sys
import glob

DEFAULT = "/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_memp"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    # 收集所有 run 的 epoch_summary,按 epoch 号合并,取最新 mtime
    by_epoch = {}  # ep -> (mtime, epoch_dir)
    for p in glob.glob(os.path.join(output_dir, "**", "epoch*/epoch_summary.json"), recursive=True):
        ed = os.path.dirname(p)
        base = os.path.basename(ed)
        digits = "".join(c for c in base if c.isdigit())
        if not digits:
            continue
        ep = int(digits)
        mt = os.path.getmtime(p)
        if ep not in by_epoch or mt > by_epoch[ep][0]:
            by_epoch[ep] = (mt, ed)
    if not by_epoch:
        print(f"[!] 没找到 epoch_summary.json under {output_dir}")
        return

    # run root = epochN 目录的父目录(形如 .../<run_ts>_.../epochN)
    runs = sorted({os.path.dirname(ed) for _, ed in by_epoch.values()})
    print(f"merged across {len(runs)} run(s):")
    for r in runs:
        print(f"  - {os.path.basename(r)}")
    print()

    hdr = f"{'epoch':>5} | {'train SR':>18} | {'eval SR (temp0.0)':>20} | {'multi-eval (temp0.2)':>26}"
    print(hdr)
    print("-" * len(hdr))
    tr_all, va_all = [], []
    for ep in sorted(by_epoch):
        ed = by_epoch[ep][1]
        summ = load_json(os.path.join(ed, "epoch_summary.json")) or {}
        tr = summ.get("train") or {}
        va = summ.get("val") or {}
        tr_s = f"{tr.get('pass',0)}/{tr.get('total',0)} = {(tr.get('pass@1') or 0)*100:5.1f}%" if tr else "—"
        va_s = f"{va.get('pass',0)}/{va.get('total',0)} = {(va.get('pass@1') or 0)*100:5.1f}%" if va else "—"
        if tr: tr_all.append((tr.get('pass@1') or 0)*100)
        if va: va_all.append((va.get('pass@1') or 0)*100)
        multi = load_json(os.path.join(ed, "val_multi_summary.json"))
        if multi:
            runs_i = multi.get("individual_runs", [])
            m_s = f"{multi.get('mean_pass@1',0)*100:5.1f}% ±{multi.get('ci_95',0)*100:.1f} (n={multi.get('n_runs',len(runs_i))})"
        else:
            m_s = "—"
        print(f"{ep:>5} | {tr_s:>18} | {va_s:>20} | {m_s:>26}")
    if tr_all:
        print("-" * len(hdr))
        print(f"{'mean':>5} | {sum(tr_all)/len(tr_all):>16.1f}% | {sum(va_all)/len(va_all):>18.1f}% |")


if __name__ == "__main__":
    main()
