"""Debug: are learned Q-values actually differentiating OS memories?

If Q-learning works, q_cache.json should show a SPREAD of values away from the
0.5 init. If everything sits near 0.5, value-driven selection == plain RAG,
which explains a flat epoch curve.

Pure JSON stats over checkpoint caches across snapshots 1..7 — no memos import,
no API, no side effects. Run in-container per project rule.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(
    "/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb/"
    "exp_llb_os_memrl_gpt41mini_20260630-165148/snapshot"
)
SEP = "=" * 72


def stats(vals):
    if not vals:
        return "n=0"
    vals = sorted(vals)
    n = len(vals)
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if n > 1 else 0.0
    q = lambda p: vals[min(n - 1, int(p * n))]
    near_init = sum(1 for v in vals if abs(v - 0.5) < 1e-6)
    return (f"n={n} mean={mean:.4f} sd={sd:.4f} min={vals[0]:.3f} "
            f"p25={q(.25):.3f} p50={q(.50):.3f} p75={q(.75):.3f} max={vals[-1]:.3f} "
            f"| exactly-0.5: {near_init} ({100*near_init/n:.1f}%)")


print(SEP)
print("Q-value spread per snapshot (q_cache.json)")
print(SEP)
for sec in range(1, 8):
    qp = ROOT / str(sec) / "local_cache" / "q_cache.json"
    if not qp.exists():
        print(f"snapshot {sec}: (no q_cache.json)")
        continue
    d = json.load(open(qp))
    vals = [float(v) for v in d.values()]
    print(f"snapshot {sec}: {stats(vals)}")

# Also cross-check metadata Q-values in the final mem_cache (the fallback source
# when q_cache misses). If BOTH q_cache and metadata are ~0.5, selection is blind.
print("\n" + SEP)
print("metadata q_value spread in final mem_cache.json (snapshot 7)")
print(SEP)
mc = ROOT / "7" / "local_cache" / "mem_cache.json"
if mc.exists():
    raw = json.load(open(mc))
    qs = []
    succ_q, fail_q = [], []
    for _, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        md = payload.get("metadata", {}) or {}
        q = md.get("q_value")
        s = md.get("success")
        if q is not None:
            q = float(q)
            qs.append(q)
            (succ_q if s else fail_q).append(q)
    print("ALL   :", stats(qs))
    print("success mems:", stats(succ_q))
    print("failed  mems:", stats(fail_q))
    print("\n>>> If success and failed q_value distributions overlap heavily,")
    print(">>> the Q-signal cannot separate good from bad memories at retrieval.")
