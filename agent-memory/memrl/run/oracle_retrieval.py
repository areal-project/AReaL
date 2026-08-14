"""Oracle retrieval helper for BCB holdout transfer diagnostic.

Given a target holdout task, score each candidate memory by overlap with
the task's true required libraries. Pick top-K to provide an "oracle ceiling"
for retrieval — i.e., the best memory-augmented pass@1 achievable on this
memory pool, regardless of what the actual retrieval algorithm picks.

Design notes (per Codex review):
- Score uses only `libs` overlap (Jaccard on the BCB ground-truth lib list).
  Earlier versions also used import/call overlap extracted from memory text,
  but memory text is the prompt (not the solution), so those signals were
  largely noise. Lib overlap is the cleanest BCB-native signal.
- No success_bonus: that biased toward easy tasks rather than relevant ones.
- Exclude memories whose `domains` list contains the holdout subtask *anywhere*,
  not just primary — multi-domain memories with holdout-as-secondary still
  carry holdout content (would be partial leakage).
- Dedupe by task_id before taking top-K so 5 results = 5 different source
  tasks, not 5 epoch-snapshots of the same task.
"""
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def _parse_libs(libs_field: Any) -> set:
    """libs field is stringified list (e.g., "['numpy', 'pandas']"). Be defensive."""
    if not libs_field:
        return set()
    if isinstance(libs_field, list):
        return {str(x).lower() for x in libs_field}
    if isinstance(libs_field, str):
        try:
            parsed = ast.literal_eval(libs_field)
            if isinstance(parsed, list):
                return {str(x).lower() for x in parsed}
        except (ValueError, SyntaxError):
            pass
    return set()


def score_memory_for_task(
    target_task: Dict[str, Any],
    memory: Dict[str, Any],
) -> float:
    """Oracle score: lib_jaccard between target's required libs and memory's libs."""
    target_libs = _parse_libs(target_task.get('libs'))
    payload = memory.get('payload', memory) if isinstance(memory, dict) else {}
    metadata = payload.get('metadata', {}) if isinstance(payload, dict) else {}
    mem_libs = _parse_libs(metadata.get('libs'))
    union = target_libs | mem_libs
    if not union:
        return 0.0
    return len(target_libs & mem_libs) / len(union)


def _memory_domains(memory: Dict[str, Any]) -> List[str]:
    payload = memory.get('payload', memory) if isinstance(memory, dict) else {}
    metadata = payload.get('metadata', {}) if isinstance(payload, dict) else {}
    doms = metadata.get('domains') or []
    return [str(d) for d in doms if d]


def _memory_task_id(memory: Dict[str, Any]) -> Optional[str]:
    payload = memory.get('payload', memory) if isinstance(memory, dict) else {}
    metadata = payload.get('metadata', {}) if isinstance(payload, dict) else {}
    return metadata.get('task_id')


def select_oracle_memories(
    target_task: Dict[str, Any],
    memory_pool: List[Dict[str, Any]],
    top_k: int = 5,
    holdout_subtask: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """Return top-k memories ranked by oracle score.

    Strict exclusions to maintain zero-shot integrity:
    1. If `holdout_subtask` is "bcb/X", exclude any memory whose domains list
       contains X anywhere (primary or secondary). Multi-domain memories with
       holdout-as-secondary would still leak holdout knowledge.
    2. Dedupe by task_id: keep only the highest-scoring entry per task_id so
       top-K returns K different source tasks, not K snapshots of the same one.
    """
    holdout_dom = None
    if holdout_subtask and '/' in holdout_subtask:
        holdout_dom = holdout_subtask.split('/', 1)[1].lower()

    # Score all candidates with holdout filter
    candidates: List[Tuple[Dict[str, Any], float]] = []
    for mem in memory_pool:
        if holdout_dom:
            mem_doms_lower = {d.lower() for d in _memory_domains(mem)}
            if holdout_dom in mem_doms_lower:
                continue
        s = score_memory_for_task(target_task, mem)
        candidates.append((mem, s))

    # Dedup by task_id: keep highest-scoring per task
    by_tid: Dict[str, Tuple[Dict[str, Any], float]] = {}
    no_tid_bucket: List[Tuple[Dict[str, Any], float]] = []
    for mem, s in candidates:
        tid = _memory_task_id(mem)
        if not tid:
            no_tid_bucket.append((mem, s))
            continue
        prev = by_tid.get(tid)
        if prev is None or s > prev[1]:
            by_tid[tid] = (mem, s)

    deduped = list(by_tid.values()) + no_tid_bucket
    deduped.sort(key=lambda x: -x[1])
    return deduped[:top_k]


def load_memory_pool(snapshot_dir: str) -> List[Dict[str, Any]]:
    """Load all memories from a snapshot's cube/textual_memory.json.

    Raises FileNotFoundError if path missing, RuntimeError if pool is empty.
    """
    p = Path(snapshot_dir) / 'cube' / 'textual_memory.json'
    if not p.exists():
        raise FileNotFoundError(f"Memory snapshot not found: {p}")
    with open(p) as f:
        pool = json.load(f)
    if not isinstance(pool, list) or not pool:
        raise RuntimeError(f"Memory pool at {p} is empty or not a list")
    return pool


def memory_to_selected_format(mem: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a textual_memory entry into the dict shape that
    _format_memory_context / downstream code expects."""
    payload = mem.get('payload', {}) if isinstance(mem, dict) else {}
    mid = mem.get('id') or payload.get('id') or ''
    content = payload.get('memory', '')
    return {
        'memory_id': str(mid),
        'id': str(mid),
        'content': content,
        'memory': content,
        'metadata': payload.get('metadata', {}),
    }
