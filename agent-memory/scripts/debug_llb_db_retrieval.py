"""Debug: does LLB DB MemRL retrieval actually produce injectable content?

Runs in the Singularity container (needs `memos` for TextualMemoryItem rehydrate).
Zero API calls, zero side effects: we load the checkpoint's cached objects and
reproduce retrieve_query's core scoring/content-extraction by hand, then run the
exact llb runner formatting path.

Prior run already proved (part [1]) that on a freshly built TextualMemoryMetadata,
both `model_extra['full_content']` and `getattr(meta,'full_content')` return the
content. This run answers: after checkpoint rehydration, is `content` still
populated, and does the final injected context contain the memory text?
"""
import sys
import json
import math
import statistics
from pathlib import Path

PROJECT = Path("/storage/openpsi/users/yl/agent-memory/MemRL")
sys.path.insert(0, str(PROJECT))

SNAP = Path(
    "/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb/"
    "exp_llb_db_memrl_gpt41mini_20260701-185849/snapshot/6"
)
CACHE = SNAP / "local_cache"
SEP = "=" * 80


def section(t):
    print("\n" + SEP + "\n" + t + "\n" + SEP)


# ---- rehydrate _mem_cache exactly like load_checkpoint_snapshot does ----
section("[A] Rehydrate _mem_cache into TextualMemoryItem objects")
TextualMemoryItem = None
for modpath in ("memos.memories.textual.item", "memos.memories.textual.general"):
    try:
        mod = __import__(modpath, fromlist=["TextualMemoryItem"])
        TextualMemoryItem = getattr(mod, "TextualMemoryItem")
        print("imported TextualMemoryItem from:", modpath)
        break
    except Exception as e:
        print("  import", modpath, "failed:", e)

raw_mc = json.load(open(CACHE / "mem_cache.json"))
mem_cache = {}
n_fail = 0
for mid, payload in raw_mc.items():
    if not isinstance(payload, dict):
        continue
    try:
        mem_cache[str(mid)] = TextualMemoryItem(**payload)
    except Exception:
        try:
            mem_cache[str(mid)] = TextualMemoryItem.model_validate(payload)
        except Exception:
            n_fail += 1
print(f"rehydrated {len(mem_cache)} items, {n_fail} failed")

dict_memory = json.load(open(CACHE / "dict_memory.json"))
q_cache = json.load(open(CACHE / "q_cache.json"))
query_embeddings = json.load(open(CACHE / "query_embeddings.json"))
print("dict_memory keys:", len(dict_memory))
print("q_cache:", len(q_cache), "| query_embeddings:", len(query_embeddings))

# ---- inspect content-extraction on a rehydrated object (the real question) ----
section("[B] content extraction on rehydrated objects (retrieve_query path)")
sample_ids = list(mem_cache.keys())[:5]
for mid in sample_ids:
    obj = mem_cache[mid]
    md = getattr(obj, "metadata", None)
    # exact retrieve_query logic:
    content = None
    if hasattr(md, "model_extra"):
        content = (md.model_extra or {}).get("full_content")
    elif isinstance(md, dict):
        content = md.get("full_content")
    print(f"  mem={mid[:8]} md_type={type(md).__name__} "
          f"has_model_extra={hasattr(md,'model_extra')} "
          f"content_is_None={content is None} len={len(content) if content else 0} "
          f"success={getattr(md,'success','?')}")

# ---- reproduce retrieve_query end-to-end (no API: use cached query embedding) ----
section("[C] Reproduce retrieve_query on a self-match task (sim=1.0)")

# pick a task_description that is both a dict_memory key AND has a cached embedding
task_desc = None
for k in dict_memory.keys():
    if k in query_embeddings:
        task_desc = k
        break
if task_desc is None:
    print("no task_desc with cached embedding; abort C")
    sys.exit(0)
print("task_desc:", task_desc[:100])

query_vec = query_embeddings[task_desc]
query_norm = math.sqrt(sum(x * x for x in query_vec)) or 1e-8

SIM_THRESHOLD = 0.369
K = 10
TOPK = 5
W_SIM, W_Q = 0.5, 0.5
SIM_MEAN, SIM_STD = 0.2747681439, 0.1127030626

# sim over all query keys
sim_list = []
for q, qv in query_embeddings.items():
    if q not in dict_memory:
        continue
    qn = math.sqrt(sum(x * x for x in qv)) or 1e-8
    sim = sum(a * b for a, b in zip(query_vec, qv)) / (query_norm * qn)
    if sim >= SIM_THRESHOLD:
        sim_list.append((q, sim))
sim_list.sort(key=lambda x: x[1], reverse=True)
sim_list = sim_list[:K]
print(f"sim_list top-{K}: top_sim={sim_list[0][1]:.4f} n={len(sim_list)}")

# build candidates + content
candidates = []
for q, sim in sim_list:
    for mid in dict_memory.get(q, []):
        obj = mem_cache.get(mid)
        if obj is None:
            continue
        md = getattr(obj, "metadata", None)
        content = None
        if hasattr(md, "model_extra"):
            content = (md.model_extra or {}).get("full_content")
        q_est = q_cache.get(mid)
        if q_est is None:
            q_est = getattr(md, "q_value", 0.0)
        candidates.append({
            "memory_id": mid, "content": content, "similarity": float(sim),
            "metadata": md, "q_estimate": float(q_est or 0.0),
        })
print("candidates:", len(candidates),
      "| content None count:", sum(1 for c in candidates if c["content"] is None))

# hybrid score (z-norm)
qs = [c["q_estimate"] for c in candidates]
mean_q = statistics.fmean(qs) if qs else 0.0
std_q = statistics.pstdev(qs) if len(qs) > 1 else 1.0


def zsim(s):
    return (s - SIM_MEAN) / (SIM_STD or 1.0)


def zq(x):
    z = (x - mean_q) / (std_q or 1.0)
    return max(min(z, 3.0), -3.0)


for c in candidates:
    c["score"] = zsim(c["similarity"]) * W_SIM + zq(c["q_estimate"]) * W_Q
candidates.sort(key=lambda c: c["score"], reverse=True)
selected = candidates[:TOPK]
print("\nselected top-5:")
for i, s in enumerate(selected):
    print(f"  [{i}] sim={s['similarity']:.3f} sim_z={zsim(s['similarity']):.2f} "
          f"q={s['q_estimate']:.3f} q_z={zq(s['q_estimate']):.2f} "
          f"score={s['score']:.3f} content_None={s['content'] is None} "
          f"success={getattr(s['metadata'],'success','?')}")

# ---- run the EXACT llb runner formatting ----
section("[D] llb process_retrieve_mems + format_llb_memory_context")
from memrl.lifelongbench_eval.memory_context import format_llb_memory_context


def process_retrieve_mems(mems):
    succ, fail = [], []
    for m in mems:
        (succ if getattr(m["metadata"], "success", False) else fail).append(m)
    out = {}
    if succ:
        out["successed"] = succ
    if fail:
        out["failed"] = fail
    return out


processed = process_retrieve_mems(selected)
print("buckets:", {k: len(v) for k, v in processed.items()})
ctx = format_llb_memory_context(processed, task="db")
print(f"\n--- FINAL injected context (len={len(ctx)}) ---")
print(ctx[:3000])
print("\n>>> header-only (memory NOT injected)?",
      ctx.strip() == "[Retrieved Memory Context]")
