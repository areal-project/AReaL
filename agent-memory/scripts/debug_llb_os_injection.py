"""Debug: is LLB OS MemRL memory injection working?

Runs INSIDE the Singularity container (needs `memos`). Zero API calls, zero
side effects. Loads real stored memories from an OS MemRL checkpoint snapshot
and reproduces the runner's bucketing + formatting path to answer:

  [1] Is `success` a declared field or a Pydantic *extra* on TextualMemoryMetadata?
      -> decides whether the LLB runner's `getattr(meta, "success", False)`
         (llb_rl_runner.py:454) can ever see it.
  [2] On real stored OS memories, how many are bucketed success vs failed by:
         (a) LLB path:      getattr(meta, "success", False)
         (b) ALFWorld path: meta.model_extra.get("success", False)
      A large divergence == the injection bug.
  [3] Does the final injected [Retrieved Memory Context] block contain the
      SUCCESSFUL EXPERIENCES section, or only FAILED?
"""
import sys
import json
from pathlib import Path

PROJECT = Path("/storage/openpsi/users/yl/agent-memory/MemRL")
sys.path.insert(0, str(PROJECT))

SNAP = Path(
    "/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb/"
    "exp_llb_os_memrl_gpt41mini_20260630-165148/snapshot/7"
)
CACHE = SNAP / "local_cache"
SEP = "=" * 80


def section(t):
    print("\n" + SEP + "\n" + t + "\n" + SEP)


# ---------------------------------------------------------------------------
# [1] TextualMemoryMetadata: declared field vs extra
# ---------------------------------------------------------------------------
section("[1] Is `success` a declared field or a Pydantic extra?")
TextualMemoryItem = None
TextualMemoryMetadata = None
for modpath in ("memos.memories.textual.item", "memos.memories.textual.general"):
    try:
        mod = __import__(modpath, fromlist=["TextualMemoryItem", "TextualMemoryMetadata"])
        TextualMemoryItem = getattr(mod, "TextualMemoryItem")
        TextualMemoryMetadata = getattr(mod, "TextualMemoryMetadata")
        print("imported from:", modpath)
        break
    except Exception as e:
        print("  import", modpath, "failed:", e)

if TextualMemoryMetadata is not None:
    fields = list(getattr(TextualMemoryMetadata, "model_fields", {}).keys())
    print("declared model_fields:", fields)
    print("`success` is a DECLARED field?", "success" in fields)
    cfg = getattr(TextualMemoryMetadata, "model_config", {})
    print("model_config.extra =", cfg.get("extra"))

    # Build a metadata exactly like memory_service.py:781 does for a SUCCESS mem
    meta = TextualMemoryMetadata(
        type="procedure",
        source="conversation",
        source_benchmark="lifelongbench",
        success=True,
        full_content="Task: demo\n\nSCRIPT: echo hello",
    )
    got_attr = getattr(meta, "success", "MISSING")
    print("\n--- On a freshly built success=True metadata ---")
    print("getattr(meta, 'success', 'MISSING')  [LLB path] ->", repr(got_attr))
    me = getattr(meta, "model_extra", None)
    print("meta.model_extra.get('success')      [ALF path] ->",
          repr(me.get("success") if isinstance(me, dict) else me))
    print(">>> If LLB path is False/MISSING while ALF path is True: BUG CONFIRMED.")


# ---------------------------------------------------------------------------
# [2] Real stored OS memories: bucketing divergence
# ---------------------------------------------------------------------------
section("[2] Bucketing on REAL stored OS memories (snapshot/7)")
if not CACHE.exists():
    print("cache not found:", CACHE)
    sys.exit(0)

raw_mc = json.load(open(CACHE / "mem_cache.json"))
items = []
n_fail = 0
for mid, payload in raw_mc.items():
    if not isinstance(payload, dict):
        continue
    try:
        items.append(TextualMemoryItem(**payload))
    except Exception:
        try:
            items.append(TextualMemoryItem.model_validate(payload))
        except Exception:
            n_fail += 1
print(f"rehydrated {len(items)} memory items ({n_fail} failed)")


def _llb_success(meta):
    # exact copy of llb_rl_runner.py:454
    return getattr(meta, "success", False)


def _alf_success(meta):
    me = getattr(meta, "model_extra", None)
    if isinstance(me, dict):
        return me.get("success", False)
    return False


llb_succ = alf_succ = both = neither = divergent = 0
for it in items:
    meta = it.metadata
    a = bool(_llb_success(meta))
    b = bool(_alf_success(meta))
    llb_succ += a
    alf_succ += b
    if a and b:
        both += 1
    if (not a) and (not b):
        neither += 1
    if a != b:
        divergent += 1

print(f"LLB path  getattr(meta,'success'):     {llb_succ} counted success")
print(f"ALF path  model_extra.get('success'):  {alf_succ} counted success")
print(f"agree-success={both}  agree-fail={neither}  DIVERGENT={divergent}")
print(">>> DIVERGENT > 0 means LLB mis-buckets memories the ALF path gets right.")


# ---------------------------------------------------------------------------
# [3] Final injected context via the real runner formatting path
# ---------------------------------------------------------------------------
section("[3] Final injected [Retrieved Memory Context] (LLB formatting path)")
from memrl.lifelongbench_eval.memory_context import format_llb_memory_context

# Reproduce process_retrieve_mems (llb_rl_runner.py:437-467) exactly
retrieved = [{"metadata": it.metadata, "content": (getattr(it.metadata, "model_extra", {}) or {}).get("full_content", "")} for it in items[:10]]
success_mems, failed_mems = [], []
for mem in retrieved:
    if getattr(mem["metadata"], "success", False):
        success_mems.append(mem)
    else:
        failed_mems.append(mem)
processed = {}
if success_mems:
    processed["successed"] = success_mems
if failed_mems:
    processed["failed"] = failed_mems

print("buckets present:", list(processed.keys()))
ctx = format_llb_memory_context(processed, task="os")
print("--- injected context (first 1500 chars) ---")
print(ctx[:1500])
print("...")
print("\nHas SUCCESSFUL section?", "SUCCESSFUL EXPERIENCES" in ctx)
print("Has FAILED section?    ", "FAILED EXPERIENCES" in ctx)
