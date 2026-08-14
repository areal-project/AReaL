"""
Value-driven retrieval and RL-style utility update components.

This module introduces a lightweight CRL+RAG layer that can be plugged
into the existing Memp service without breaking current APIs.

Key components:
- RLConfig: configuration for epsilon-greedy, thresholds, and Q-learning.
- ValueAwareSelector: selects a memory from Top-K based on Q with ε-greedy,
  and supports unknown detection (Zero-Shot fallback via null action).
- QValueUpdater: updates per-memory Q in TextualMemoryMetadata, stored in MemOS.
- MemoryCurator: optional novelty/merge helpers (minimal viable ops).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import random

from memos.mem_os.main import MOS


@dataclass
class RLConfig:
    epsilon: float = 0.1           # ε-greedy exploration probability
    tau: float = 0.35              # unknown detection threshold on similarity
    alpha: float = 0.1             # Q-learning step size
    gamma: float = 0.0             # discount factor (single-step default)
    q_init_pos: float = 0        # optimistic initialization
    q_init_neg: float = 0          # negative q init
    q_floor: Optional[float] = None  # optional minimum Q value (disabled by default)
    success_reward: float = 1.0
    failure_reward: float = -1.0
    sim_threshold: float = 0.5     # retrieval filtering threshold (used by some runners)
    topk: int = 5                  # candidate set size for value-aware selection
    novelty_threshold: float = 0.85  # similarity to treat as non-novel (merge)
    recency_boost: float = 0.0     # optional recency weight
    reward_merge_gain: float = 0.1 # gain for attributing success to close memories
    weight_sim: float = 0.5        # weight for similarity in combined score
    weight_q: float = 0.5          # weight for Q-value in combined score


# ----------------- Utilities -----------------

def _meta_to_dict(meta: Any) -> Dict[str, Any]:
    """Convert TextualMemoryMetadata or dict-like to a plain dict safely."""
    if meta is None:
        return {}
    try:
        if hasattr(meta, "model_extra"):
            return dict(meta.model_dump())
    except Exception:
        pass
    try:
        if isinstance(meta, dict):
            return dict(meta)
    except Exception:
        pass
    # best effort fallback
    try:
        return dict(getattr(meta, "__dict__", {}))
    except Exception:
        return {}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _get_similarity(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("similarity", 0.0))
    except Exception:
        return 0.0


# ----------------- Selector -----------------

class ValueAwareSelector:
    """
    Selects the best memory from candidates by Q with ε-greedy, and triggers
    null action when the top similarity is below threshold τ.
    """

    def __init__(self, cfg: RLConfig):
        self.cfg = cfg

    def select(self, candidates: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        if not candidates:
            return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

        simmax = max(_get_similarity(c) for c in candidates)

        # Compute Q for each candidate
        enriched: List[Dict[str, Any]] = []
        for c in candidates:
            md = _meta_to_dict(c.get("metadata"))
            q = md.get("q_value", self.cfg.q_init_pos)
            try:
                q = float(q)
            except Exception:
                q = self.cfg.q_init_pos

            # Optional Q floor (keeps Q from dropping below a minimum).
            if getattr(self.cfg, "q_floor", None) is not None:
                try:
                    q = max(float(self.cfg.q_floor), float(q))
                except Exception:
                    pass

            # optional recency boost
            if self.cfg.recency_boost > 0 and md.get("last_used_at"):
                try:
                    ts = md["last_used_at"]
                    if isinstance(ts, str) and len(ts) >= 10:
                        q = q + self.cfg.recency_boost
                except Exception:
                    pass

            c_local = dict(c)
            c_local["q_estimate"] = q
            enriched.append(c_local)

        q_min = getattr(self.cfg, "q_min_threshold", None)
        if q_min is not None:
            enriched = [c for c in enriched if c.get("q_estimate", 0.0) >= q_min]

        if not enriched:
            return {"actions": [], "selected": [], "candidates": [], "simmax": simmax}

        # sort by Q (desc), tie-breaker by similarity
        enriched_sorted = sorted(
            enriched,
            key=lambda x: (x.get("q_estimate", 0.0), _get_similarity(x)),
            reverse=True,
        )

        # ε-greedy top-k
        if random.random() < self.cfg.epsilon:
            selected = random.sample(enriched_sorted, min(top_k, len(enriched_sorted)))
        else:
            selected = enriched_sorted[:min(top_k, len(enriched_sorted))]


        return {
            "actions": [s.get("memory_id") for s in selected],
            "selected": selected,
            "candidates": enriched_sorted,
            "simmax": simmax,
        }



# ----------------- Q-value updater -----------------

class QValueUpdater:
    """Persist and update memory Q-values in MemOS textual memory metadata."""

    def __init__(self, mos: MOS, user_id: str, cfg: RLConfig, *, default_cube_id: Optional[str] = None) -> None:
        self.mos = mos
        self.user_id = user_id
        self.cfg = cfg
        self.default_cube_id = default_cube_id

    def _get_text_mem(self) -> Any:
        if self.default_cube_id is None or self.default_cube_id not in self.mos.mem_cubes:
            # fallback to first user cube
            cubes = self.mos.user_manager.get_user_cubes(self.user_id)
            if not cubes:
                raise ValueError(f"No mem cube accessible for user {self.user_id}")
            cube_id = cubes[0].cube_id
        else:
            cube_id = self.default_cube_id
        text_mem = self.mos.mem_cubes[cube_id].text_mem
        if text_mem is None:
            raise ValueError("Textual memory is not initialized")
        return text_mem

    def update(self, memory_id: str, reward: float, next_max_q: Optional[float] = None) -> float:
        """Apply single-step Q-learning update on a memory item's metadata.

        Returns new Q value.
        """
        text_mem = self._get_text_mem()
        try:
            item = text_mem.get(memory_id)
        except Exception as e:
            raise RuntimeError(f"Unable to fetch memory {memory_id}: {e}")

        old_meta = _meta_to_dict(getattr(item, "metadata", None))
        old_q = float(old_meta.get("q_value", self.cfg.q_init_pos))
        target = float(reward) + (self.cfg.gamma * float(next_max_q or 0.0))
        new_q = (1.0 - self.cfg.alpha) * old_q + self.cfg.alpha * target

        # Optional Q floor: prevents Q from dropping below configured minimum.
        if getattr(self.cfg, "q_floor", None) is not None:
            try:
                new_q = max(float(self.cfg.q_floor), float(new_q))
            except Exception:
                pass

        visits = int(old_meta.get("q_visits", 0)) + 1
        # simple EMA for reward
        old_ma = float(old_meta.get("reward_ma", 0.0))
        reward_ma = (1.0 - self.cfg.alpha) * old_ma + self.cfg.alpha * float(reward)

        new_meta = old_meta | {
            "q_value": float(new_q),
            "q_visits": visits,
            "last_reward": float(reward),
            "reward_ma": reward_ma,
            "q_updated_at": _now_iso(),
            "last_used_at": _now_iso(),
        }

        # keep retrieval key unchanged; only update metadata
        # Retry on transient failures (429 rate limit from memos internal embedding)
        import time as _time
        _max_retries = 5
        _base_delay = 2.0
        for _attempt in range(1, _max_retries + 1):
            try:
                text_mem.update(memory_id, {"id": memory_id, "memory": getattr(item, "memory", ""), "metadata": new_meta})
                return float(new_q)
            except Exception as _e:
                _err = str(_e)
                _is_transient = (
                    "429" in _err
                    or "rate_limit" in _err.lower()
                    or "NoneType" in _err
                    or "timeout" in _err.lower()
                    or "connection" in _err.lower()
                )
                if _is_transient:
                    if _attempt < _max_retries:
                        _sleep = _base_delay * (2 ** (_attempt - 1))
                        _time.sleep(_sleep)
                        continue
                raise
        return float(new_q)


# ----------------- Curator (novelty/merge) -----------------

class MemoryCurator:
    """
    Minimal curation helper:
    - find_merge_target: for a successful new task, check if an existing memory is highly similar,
      so we can attribute some reward without creating a new entry.
    - attribute_reward: small positive boost to an existing memory's Q.
    """

    def __init__(self, mos: MOS, user_id: str, cfg: RLConfig, *, default_cube_id: Optional[str] = None, q_updater: Optional[QValueUpdater] = None) -> None:
        self.mos = mos
        self.user_id = user_id
        self.cfg = cfg
        self.default_cube_id = default_cube_id
        self.q_updater = q_updater

    def _flatten_text_mem_results(self, result: Dict[str, Any]) -> List[Any]:
        items: List[Any] = []
        for cube in result.get("text_mem", []):
            items.extend(cube.get("memories", []))
        return items

    def find_merge_target(self, task_description: str) -> Optional[Dict[str, Any]]:
        try:
            res = self.mos.search(query=task_description, user_id=self.user_id, top_k=1)
        except Exception:
            return None
        items = self._flatten_text_mem_results(res)
        if not items:
            return None
        itm = items[0]
        # try to read similarity
        md = getattr(itm, "metadata", None)
        try:
            similarity = float(getattr(md, "relativity", 0.0)) if md is not None else 0.0
        except Exception:
            similarity = 0.0
        if similarity >= self.cfg.novelty_threshold:
            return {"memory_id": getattr(itm, "id", None), "item": itm, "similarity": similarity}
        return None

    def attribute_reward(self, memory_id: str, reward: float) -> Optional[float]:
        if not memory_id or self.q_updater is None:
            return None
        try:
            # small positive gain attribution; treat as immediate reward scaled by factor
            gain = float(self.cfg.reward_merge_gain) * float(reward)
            return self.q_updater.update(memory_id, gain)
        except Exception:
            return None
