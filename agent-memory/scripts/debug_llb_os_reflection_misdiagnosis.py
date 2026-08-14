"""Debug: are OS failure reflections misdiagnosed as DB-style format/tuple problems?

OS is graded by running an evaluation command and checking exit_code==0 on the
resulting SYSTEM STATE (files/perms/users). There is NO "final answer", NO tuples,
NO SQL. But the reflection prompt (_generate_reflection_llb_interactive, shared with
DB via is_llb_interactive) talks about "Action: Answer", "output format", "column
order", "tuples", "Decimal", "JOIN/aggregation".

This reads real stored OS failure memories and counts how many contain DB-only /
answer-format language that cannot apply to OS — i.e. actively-misleading memory.
Pure JSON, read-only.
"""
import json
import re
from pathlib import Path

ROOT = Path(
    "/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb/"
    "exp_llb_os_memrl_gpt41mini_20260630-165148/snapshot/7/local_cache"
)
SEP = "=" * 72

mc = json.load(open(ROOT / "mem_cache.json"))

# DB-only / answer-format vocabulary that is meaningless for OS grading
db_terms = re.compile(
    r"\b(tuple|Decimal|ORDER BY|JOIN|SELECT|GROUP BY|column order|final answer|"
    r"Action: Answer|Action: Operation|comma-separated|round bracket|SQL|"
    r"output format|rounding|dict|JSON|markdown|CSV|table format)\b",
    re.IGNORECASE,
)

fail_mems = []
for mid, payload in mc.items():
    if not isinstance(payload, dict):
        continue
    md = payload.get("metadata", {}) or {}
    if md.get("success"):
        continue
    fc = md.get("full_content", "") or ""
    if not fc:
        continue
    fail_mems.append((mid, fc))

print(f"Total FAILURE memories with content: {len(fail_mems)}")

misleading = []
for mid, fc in fail_mems:
    hits = sorted(set(m.lower() for m in db_terms.findall(fc)))
    if hits:
        misleading.append((mid, hits, fc))

print(f"Contain DB/answer-format vocabulary (misleading for OS): "
      f"{len(misleading)} ({100*len(misleading)/max(1,len(fail_mems)):.1f}%)")

# term frequency
from collections import Counter
cnt = Counter()
for _, hits, _ in misleading:
    for h in hits:
        cnt[h] += 1
print("\nTop misleading terms across OS failure memories:")
for term, c in cnt.most_common(15):
    print(f"  {term:20s} {c}")

# FAILURE_MODE tally
mode_cnt = Counter()
for _, fc in fail_mems:
    m = re.search(r"FAILURE_MODE:\s*(.+)", fc)
    if m:
        mode_cnt[m.group(1).strip()[:60]] += 1
print(f"\n{SEP}\nFAILURE_MODE distribution (OS):\n{SEP}")
for mode, c in mode_cnt.most_common(15):
    print(f"  {c:4d}  {mode}")

# Show 4 concrete misleading examples
print(f"\n{SEP}\nSAMPLE MISLEADING OS FAILURE MEMORIES\n{SEP}")
for mid, hits, fc in misleading[:4]:
    print(f"\n--- {mid}  (DB-terms: {hits}) ---")
    print(fc[:900])
