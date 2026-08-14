"""Debug: inspect real OS trajectories — does the trajectory text contain bash commands?

Read stored memories from the OS MemRL checkpoint, look at their full_content,
and check whether actual bash commands (like useradd, chmod, mkdir, apt-get, etc.)
appear in the procedural memory text vs. only high-level descriptions.

Also read the raw trajectory stored alongside the memory (if available) to confirm
what the proceduralization LLM actually received as input.
"""
import json
from pathlib import Path

ROOT = Path(
    "/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb/"
    "exp_llb_os_memrl_gpt41mini_20260630-165148/snapshot/7/local_cache"
)
SEP = "=" * 72

mc = json.load(open(ROOT / "mem_cache.json"))
print(f"Total memories: {len(mc)}")

# Sample a few success and failure memories, show full_content
print(f"\n{SEP}")
print("SAMPLE STORED OS MEMORIES (full_content from metadata)")
print(SEP)

n_shown = 0
for mid, payload in mc.items():
    if not isinstance(payload, dict):
        continue
    md = payload.get("metadata", {}) or {}
    fc = md.get("full_content", "")
    success = md.get("success")
    if not fc:
        continue

    # Show 3 success and 3 failure
    if n_shown >= 6:
        break
    if n_shown < 3 and not success:
        continue
    if n_shown >= 3 and success:
        continue

    print(f"\n--- Memory {mid} (success={success}) ---")
    print(fc[:2000])
    print("..." if len(fc) > 2000 else "")
    n_shown += 1

# Check whether trajectories contain command-like patterns
print(f"\n{SEP}")
print("STATISTICS: command presence in stored full_content")
print(SEP)

import re
cmd_patterns = [
    r'\b(useradd|userdel|groupadd|groupdel|chmod|chown|chgrp|mkdir|rm |cp |mv |ln |cat |echo |apt-get|apt |pip |bash|sh -c|touch|find |grep )\b',
    r'```(bash|sh)',
]

has_commands = 0
no_commands = 0
total = 0
for mid, payload in mc.items():
    if not isinstance(payload, dict):
        continue
    md = payload.get("metadata", {}) or {}
    fc = md.get("full_content", "")
    if not fc:
        continue
    total += 1
    if any(re.search(p, fc) for p in cmd_patterns):
        has_commands += 1
    else:
        no_commands += 1

print(f"Total memories with content: {total}")
print(f"  Contains command-like text: {has_commands} ({100*has_commands/total:.1f}%)")
print(f"  No command-like text:       {no_commands} ({100*no_commands/total:.1f}%)")
print()
print(">>> If most memories lack commands, the proceduralization LLM isn't seeing")
print(">>> the actual bash commands from tool_calls in the trajectory input.")
