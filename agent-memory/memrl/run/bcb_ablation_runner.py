"""BCB val-only ablation helpers.

This module contains the ablation table and runner subclass used by
`run/run_bcb_region.py --val_ablations ...`. It was extracted from the
deprecated standalone script `run/run_val_ablation.py` so that a single
entry point owns config parsing and component construction — eliminating
the silent config-drift class of bugs (see docs/CHECKPOINT_STORAGE.md
and `memrl-val-ablation-config-drift` in agent memory).
"""

from __future__ import annotations

import json
import logging
import os

from memrl.run.bcb_region_runner import BCBRegionRunner

logger = logging.getLogger(__name__)


# Ablation registry. Each entry is the *delta* applied on top of the shared
# config (everything else mirrors training run_config exactly via CLI args).
# Keys: desc, k, success_only, success_prerank, w_q, lambda_max, tau_sq,
# sigma_sq, outcome_blend_beta.
ABLATIONS = {
    "baseline":             {"desc": "Standard (k=10)",                                "k": 10, "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "posthoc_succ":         {"desc": "Top-k then drop failures (POST-HOC FILTER)",     "k": 10, "success_only": True,  "success_prerank": False, "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "k3":                   {"desc": "k=3 memories",                                   "k": 3,  "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "k1":                   {"desc": "k=1 memory",                                     "k": 1,  "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "posthoc_succ_k3":      {"desc": "POST-HOC success filter + k=3",                  "k": 3,  "success_only": True,  "success_prerank": False, "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "lmax05":               {"desc": "lambda_max=0.5",                                 "k": 10, "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": 0.5,  "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "lmax03":               {"desc": "lambda_max=0.3",                                 "k": 10, "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": 0.3,  "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "lmax00":               {"desc": "lambda_max=0.0 (pure region utility)",           "k": 10, "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": 0.0,  "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "sigma10":               {"desc": "sigma_sq=1.0 (more noise -> trust region)",      "k": 10, "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": 1.0,   "outcome_blend_beta": None},
    "posthoc_succ_lmax05":  {"desc": "POST-HOC success filter + lambda_max=0.5",       "k": 10, "success_only": True,  "success_prerank": False, "w_q": None, "lambda_max": 0.5,  "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    # Pre-rank baseline: rank ONLY within the success pool (fetches k*3 candidates,
    # drops failures, takes top-k). Fair comparison vs outcome_b* because both
    # intervene at ranking time, not after top-k selection.
    "prerank_succ":         {"desc": "PRE-RANK success only",                          "k": 10, "success_only": False, "success_prerank": True,  "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "prerank_succ_lmax05":  {"desc": "PRE-RANK success + lambda_max=0.5",              "k": 10, "success_only": False, "success_prerank": True,  "w_q": None, "lambda_max": 0.5,  "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": None},
    "outcome_b01":          {"desc": "outcome_blend_beta=0.1 (mild)",                  "k": 10, "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": 0.1},
    "outcome_b03":          {"desc": "outcome_blend_beta=0.3 (recommended)",           "k": 10, "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": 0.3},
    "outcome_b05":          {"desc": "outcome_blend_beta=0.5 (very strong)",           "k": 10, "success_only": False, "success_prerank": False, "w_q": None, "lambda_max": None, "tau_sq": None, "sigma_sq": None,  "outcome_blend_beta": 0.5},
}


def _parse_outcome(raw):
    """Normalize a metadata `outcome` value to 'success'|'failure'|None."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "success" if raw else "failure"
    s = str(raw).strip().lower()
    if s in ("success", "true", "1"):
        return "success"
    if s in ("failure", "fail", "false", "0"):
        return "failure"
    return None


def _read_outcome_from_mem(mo):
    """Best-effort outcome extraction from a mem_cache entry (pydantic or dict)."""
    if mo is None:
        return None
    meta = getattr(mo, "metadata", None)
    if meta is None and isinstance(mo, dict):
        meta = mo.get("metadata")
    if meta is None:
        return None
    if hasattr(meta, "model_extra"):
        return (getattr(meta, "model_extra", {}) or {}).get("outcome")
    if isinstance(meta, dict):
        return meta.get("outcome")
    return None


class AblationRunner(BCBRegionRunner):
    """BCBRegionRunner subclass: success_only filter, success_prerank, and
    per-task retrieval diagnostics dump for paper-grade reviewer defense."""

    def __init__(self, *args, **kwargs):
        self._success_only = kwargs.pop("success_only", False)
        self._success_prerank = kwargs.pop("success_prerank", False)
        self._override_w_q = kwargs.pop("w_q_override", None)
        self._override_lambda_max = kwargs.pop("lambda_max_override", None)
        self._override_tau_sq = kwargs.pop("tau_sq_override", None)
        self._override_sigma_sq = kwargs.pop("sigma_sq_override", None)
        super().__init__(*args, **kwargs)
        # An ablation that explicitly sets lambda_max must NOT be silently
        # overridden by parent's val_lambda_max switching in _run_phase.
        if self._override_lambda_max is not None:
            self.val_lambda_max = None
        if self._override_w_q is not None and self.mem is not None:
            self.mem.weight_q = self._override_w_q
            self.mem.weight_sim = 1.0 - self._override_w_q
        rm = getattr(self.mem, "region_manager", None) if self.mem else None
        if rm is not None:
            if self._override_lambda_max is not None:
                rm.shrinkage_lambda_max = self._override_lambda_max
            if self._override_tau_sq is not None:
                rm.shrinkage_tau_sq = self._override_tau_sq
            if self._override_sigma_sq is not None:
                rm.shrinkage_sigma_sq = self._override_sigma_sq

    def _format_memory_context(self, selected_mems):
        if self._success_only:
            filtered = []
            for m in selected_mems:
                # _read_outcome_from_mem handles BOTH plain dicts and pydantic-like
                # objects, with metadata as dict OR model_extra. Don't reimplement
                # the parsing here — it's the source of subtle dict-vs-pydantic bugs.
                if _parse_outcome(_read_outcome_from_mem(m)) == "success":
                    filtered.append(m)
            if not filtered:
                return ""
            selected_mems = filtered
        return super()._format_memory_context(selected_mems)

    def _retrieve_for_task(self, prompt, task=None, target_subtask=None):
        # PRE-RANK SUCCESS: rank within the success pool, not after top-k.
        if self._success_prerank:
            orig_k = self.retrieve_k
            self.retrieve_k = max(orig_k * 3, orig_k + 5)
            try:
                _selected_ids, _mem_context, retrieval_trace, retrieved_topk_queries = (
                    super()._retrieve_for_task(prompt, task, target_subtask=target_subtask)
                )
            finally:
                self.retrieve_k = orig_k
            mem_cache = getattr(self.mem, "_mem_cache", None) or {}
            # Dedup by mid in stable order (candidates_diag may contain duplicates
            # if dual-source retrieval / region-quota paths emit overlapping rows).
            success_picks = []
            seen_mids = set()
            for c in (retrieval_trace.get("candidates_diag") or []):
                mid = c.get("mid")
                if not mid or mid in seen_mids:
                    continue
                if _parse_outcome(_read_outcome_from_mem(mem_cache.get(mid))) == "success":
                    seen_mids.add(mid)
                    success_picks.append((mid, c))
                if len(success_picks) >= orig_k:
                    break
            kept = success_picks[:orig_k]  # hard cap (defense in depth)
            new_ids = [mid for (mid, _c) in kept]
            assert len(new_ids) <= orig_k, f"prerank_succ produced {len(new_ids)} > orig_k={orig_k}"
            selected_mems = []
            for mid, _c in kept:
                mo = mem_cache.get(mid)
                if mo is None:
                    continue
                if isinstance(mo, dict):
                    selected_mems.append(mo)
                else:
                    selected_mems.append({
                        "memory_id": mid,
                        "content": getattr(mo, "memory", None) or getattr(mo, "content", ""),
                        "metadata": getattr(mo, "metadata", None),
                    })
            new_context = self._format_memory_context(selected_mems) if selected_mems else ""
            retrieval_trace = dict(retrieval_trace)
            retrieval_trace["prerank_success"] = True
            retrieval_trace["prerank_success_kept"] = len(new_ids)
            return new_ids, new_context, retrieval_trace, retrieved_topk_queries

        return super()._retrieve_for_task(prompt, task, target_subtask=target_subtask)

    def _process_single_task(self, task_id, epoch, phase):
        result = super()._process_single_task(task_id, epoch, phase)
        # Reviewer-defense diagnostics: dump per-task retrieval signal.
        try:
            diag_path = os.path.join(self.output_dir, "retrieval_diagnostics.jsonl")
            trace = result.get("retrieval_trace") or {}
            cands = list(trace.get("candidates_diag") or [])
            mem_cache = getattr(self.mem, "_mem_cache", None) or {}
            for c in cands:
                c["outcome"] = _parse_outcome(_read_outcome_from_mem(mem_cache.get(c.get("mid"))))
            payload = {
                "task_id": task_id,
                "ok": bool(result.get("ok")),
                "target_subtask": result.get("target_subtask"),
                "selected_ids": result.get("selected_ids", []),
                "candidates_diag": cands,
            }
            with open(diag_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            logger.warning("Failed to write retrieval_diagnostics.jsonl", exc_info=True)
        return result
