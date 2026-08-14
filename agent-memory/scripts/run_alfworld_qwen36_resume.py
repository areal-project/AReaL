#!/usr/bin/env python3
"""Qwen3.6 ALFWorld resume compatibility wrapper.

Keeps the repository-wide services untouched while making the two historical
Qwen3.6 checkpoints resumable and enforcing fail-fast semantics.
"""
from __future__ import annotations

import json
import logging
import os
import re
import runpy
import sys
from pathlib import Path

from memrl.run.alfworld_rl_runner import AlfworldRunner
from memrl.service.memory_service import MemoryService

logger = logging.getLogger(__name__)
_original_memory_load = MemoryService.load_checkpoint_snapshot


def _checkpoint_id(path: Path) -> int:
    if path.name.isdigit():
        return int(path.name)
    meta = path / "snapshot_meta.json"
    if meta.is_file() and meta.stat().st_size:
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            return int(payload.get("checkpoint_id") or payload.get("ckpt_id") or 0)
        except Exception:
            pass
    return 0


def _load_snapshot_compat(self, snapshot_root: str, *args, **kwargs) -> int:
    root = Path(snapshot_root)
    cube_config = root / "cube" / "config.json"
    cache = root / "local_cache"
    required = (
        "dict_memory.json",
        "mem_cache.json",
        "q_cache.json",
        "query_embeddings.json",
    )
    cube_valid = False
    if cube_config.is_file() and cube_config.stat().st_size:
        try:
            cube_valid = isinstance(json.loads(cube_config.read_text(encoding="utf-8")), dict)
        except Exception:
            cube_valid = False
    cache_complete = cache.is_dir() and all(
        (cache / name).is_file() and (cache / name).stat().st_size > 0
        for name in required
    )
    if cube_valid or not cache_complete:
        return _original_memory_load(self, snapshot_root, *args, **kwargs)

    if not self._restore_local_caches(str(cache)):
        raise RuntimeError(f"Local-cache-only restore failed: {cache}")
    mem_cache = getattr(self, "_mem_cache", {}) or {}
    self._mem_cache_max_size = max(
        int(getattr(self, "_mem_cache_max_size", 10000) or 10000),
        len(mem_cache) + 50000,
    )
    available = {str(mid) for mid in mem_cache}
    pruned = {}
    dropped = 0
    for query, ids in (getattr(self, "dict_memory", {}) or {}).items():
        kept = [str(mid) for mid in (ids or []) if str(mid) in available]
        dropped += len(ids or []) - len(kept)
        if kept:
            pruned[str(query)] = kept
    self.dict_memory = pruned
    if not available or not pruned:
        raise RuntimeError("Local-cache-only checkpoint has no usable cached memories")
    ckpt_id = _checkpoint_id(root)
    logger.warning(
        "[RESUME] local-cache-only checkpoint loaded: path=%s checkpoint_id=%d "
        "queries=%d cached_memories=%d dropped_missing_refs=%d active_cube=%s",
        root, ckpt_id, len(pruned), len(available), dropped,
        getattr(self, "default_cube_id", None),
    )
    return ckpt_id


def _resume_fail_fast(self):
    if not self.ckpt_resume_enabled:
        auto_root = Path(self.ck_dir) / "snapshot"
        if auto_root.exists():
            best = self._find_latest_checkpoint(auto_root)
            if best is not None:
                logger.info("[Auto-resume] Found existing snapshot in ck_dir: %s", best)
                return self._load_and_resolve_resume(best)
        return (1, 1)
    snapshot = self._resolve_resume_dir()
    if not snapshot or not snapshot.exists():
        raise RuntimeError(f"Resume enabled but snapshot does not exist: {snapshot}")
    # Numeric snapshot directories are canonical epoch IDs. Explicitly set
    # this before loading because Mem0's loader returns loaded-memory count.
    if snapshot.name.isdigit():
        self.ckpt_resume_epoch = int(snapshot.name)
        logger.info("[RESUME] forcing numeric snapshot epoch=%d from %s", self.ckpt_resume_epoch, snapshot)
    start = self._load_and_resolve_resume(snapshot)
    if start == (1, 1):
        raise RuntimeError(f"Resume enabled but checkpoint load fell back to fresh start: {snapshot}")
    logger.info("[RESUME] resolved start position section=%d batch=%d", *start)
    return start


MemoryService.load_checkpoint_snapshot = _load_snapshot_compat
AlfworldRunner._resume_from_ckpt = _resume_fail_fast

# The historical Mem0 checkpoint is dense-only.  Install the offline BM25
# adapter and live-copy migration before run_alfworld constructs its service.
if "--mem0" in sys.argv:
    from run_alfworld_mem0_bm25 import install_bm25_patches

    install_bm25_patches()

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    target = repo / "run" / "run_alfworld.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
