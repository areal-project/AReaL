from __future__ import annotations
import logging
import json
import hashlib
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
import time
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
import threading
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .base_runner import BaseRunner
from memrl.providers.llm import OpenAILLM
from memrl.service.memory_service import MemoryService

logger = logging.getLogger(__name__)


@dataclass
class HLESelection:
    train_path: Optional[str] = None
    num_valid: Optional[int] = None
    num_train: Optional[int] = None
    categories: Optional[List[str]] = None  # categories to keep (train)
    eval_categories: Optional[List[str]] = None  # categories for eval (cross-category transfer)
    category_ratio: Optional[float] = None  # per-category sampling ratio (0,1]
    text_only: bool = False  # drop rows with images (for text-only LLMs)


class HLERunner(BaseRunner):
    """HLE benchmark runner, mirroring AIME/MATH runners.

    Dataset expected columns: id, question, image (base64 or empty), answer
    """

    def __init__(
        self,
        name: str,
        llm: OpenAILLM,
        llm_judge: Optional[OpenAILLM],
        selection: 'HLESelection',
        output_dir: Path,
        memory_service: Optional[MemoryService] = None,
        run_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        retrieve_k: int = 1,
        num_sections: int = 1,
        batch_size: int = 8,
        dataset_ratio: float = 1.0,
        random_seed: int = 42,
        train_valid_split: float = 0.8,
        ckpt_eval_enabled: bool = False,
        ckpt_eval_path: Optional[str] = None,
        ckpt_resume_enabled: bool = False,
        ckpt_resume_path: Optional[str] = None,
        ckpt_resume_epoch: Optional[int] = None,
        ckpt_resume_prefer_current_run: bool = False,
        ckpt_save_every_n_batches: int = 1,
        ckpt_max_keep: int = 3,
        baseline_mode: Optional[str] = None,
        baseline_k: int = 10,
        mode: str = "train",
        memory_filter_categories: Optional[List[str]] = None,
        self_rag: bool = False,
        self_rag_inject_k: int = 3,
        holdout_categories: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.sel = selection
        self.output_dir = Path(output_dir)
        self.memory_service = memory_service
        self.mode = mode
        self.llm_judge = llm_judge
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieve_k = max(0, int(retrieve_k))
        self.num_sections = num_sections
        self.batch_size = max(1, int(batch_size))
        self.dataset_ratio = float(dataset_ratio)
        self.random_seed = random_seed
        self.train_valid_split = float(train_valid_split)
        self.ckpt_eval_enabled = bool(ckpt_eval_enabled)
        self.ckpt_eval_path = str(ckpt_eval_path) if ckpt_eval_path else None
        self.ckpt_resume_enabled = ckpt_resume_enabled
        self.ckpt_resume_path = ckpt_resume_path
        self.ckpt_resume_epoch = ckpt_resume_epoch
        self.ckpt_resume_prefer_current_run = bool(ckpt_resume_prefer_current_run)
        self.ckpt_save_every_n_batches = max(1, int(ckpt_save_every_n_batches))
        self.ckpt_max_keep = max(1, int(ckpt_max_keep))
        self.baseline_mode = (baseline_mode or "").strip().lower() or None
        self.baseline_k = max(1, int(baseline_k))
        self.memory_filter_categories = memory_filter_categories
        self.self_rag = bool(self_rag)
        self.self_rag_inject_k = max(1, int(self_rag_inject_k))
        self.holdout_categories = holdout_categories

        self.run_id = run_id or time.strftime('%Y%m%d-%H%M%S')
        ts = self.run_id
        tb_dir = self.output_dir / "tensorboard" / f"exp_hle_{self.name}_{ts}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard logs at: {tb_dir}")
        self.ck_dir = self.output_dir / "hle" / f"exp_{self.name}_{self.run_id}"
        self.log_dir = self.ck_dir / "local_cache"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.llm_log_path = self.log_dir / "llm_calls.jsonl"
        self._log_lock = threading.Lock()
        self._image_lock = threading.Lock()
        self._image_store: Dict[str, str] = {}  # image_id -> data_url
        self._image_hash_to_id: Dict[str, str] = {}
        self._image_id_counter = 0
        self._image_store_path = self.log_dir / "image_store.json"
        self._image_index_path = self.log_dir / "image_hash_index.json"
        self._load_image_cache()
        self.df_train: Optional[pd.DataFrame] = None
        self.df_valid: Optional[pd.DataFrame] = None
        self.train_cumulative_correct_map = {}
        self.valid_cumulative_correct_map = {}
        self._cum_state_path = self.log_dir / "cum_state.json"
        self._resume_section_start = 0
        self._resume_batch_start = 0
        self._resume_batch_all_recs = []
        self._resume_pending_task_ids: set[str] = set()
        self._terminal_incorrect_task_ids: set[str] = set()
        self._infrastructure_deferred_task_ids: set[str] = set()
        self._resume_from_ckpt_if_needed()

        self.EXACT_ANSWER_SYSTEM_PROMPT = (
            "Your response should be in the following format:\n"
            "Explanation: {your explanation for your final answer}\n"
            "Exact Answer: {your succinct, final answer}\n"
            "Confidence: {your confidence score between 0% and 100% for your answer}"
        )

        self.MULTIPLE_CHOICE_SYSTEM_PROMPT = (
            "Your response should be in the following format:\n"
            "Explanation: {your explanation for your answer choice}\n"
            "Answer: {your chosen answer}\n"
            "Confidence: {your confidence score between 0% and 100% for your answer}"
        )

        # HLE judge prompt (from hle_eval)
        self.JUDGE_PROMPT = (
            "Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.\n\n"
            "[question]: {question}\n\n"
            "[response]: {response}\n\n"
            "Your judgement must be in the format and criteria specified below:\n\n"
            "extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.\n\n"
            "[correct_answer]: {correct_answer}\n\n"
            "reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.\n\n"
            "correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.\n\n"
            "confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."
        )

    # ------------------------------
    # Resume helpers
    # ------------------------------
    def _load_cum_state(self, state_path: Optional[Path] = None):
        path = state_path or self._cum_state_path
        if path.exists():
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
                persisted_holdout = data.get("holdout_categories")
                current_holdout = getattr(self, "holdout_categories", None)
                if persisted_holdout is not None and current_holdout != persisted_holdout:
                    raise ValueError(
                        f"Resume holdout mismatch: persisted holdout_categories={persisted_holdout!r} "
                        f"but current run started with holdout_categories={current_holdout!r}. "
                        f"Use the same --holdout_categories flag as the original run."
                    )
                self.train_cumulative_correct_map = data.get("train_cum", {})
                self.valid_cumulative_correct_map = data.get("valid_cum", {})
                self._resume_section_start = int(data.get("next_section", 0))
                self._resume_batch_start = int(data.get("next_batch", 0))
                self._resume_batch_all_recs = data.get("batch_all_recs", [])
                self._resume_pending_task_ids = {
                    str(x) for x in data.get("pending_task_ids", []) if x is not None
                }
                loaded_terminal_ids = {
                    str(x) for x in data.get("terminal_incorrect_task_ids", []) if x is not None
                }
                policy_version = int(data.get("infrastructure_error_policy_version", 1) or 1)
                self._infrastructure_deferred_task_ids = {
                    str(x) for x in data.get("infrastructure_deferred_task_ids", []) if x is not None
                }
                if policy_version < 2 and loaded_terminal_ids:
                    # Legacy terminal IDs mixed true 600s in-flight deadlines with
                    # already-returned API overload errors. Fail open for metric
                    # correctness: exclude them from this epoch and retry next epoch.
                    self._infrastructure_deferred_task_ids.update(loaded_terminal_ids)
                    self._terminal_incorrect_task_ids = set()
                    self._resume_batch_all_recs = [
                        rec for rec in self._resume_batch_all_recs
                        if str(rec.get("id")) not in loaded_terminal_ids
                    ]
                    logger.warning(
                        "[Resume] Migrated %d legacy mixed terminal IDs to infrastructure-deferred",
                        len(loaded_terminal_ids),
                    )
                else:
                    self._terminal_incorrect_task_ids = loaded_terminal_ids
                self._reconcile_resume_task_sets()
                logger.info(
                    f"[Resume] Loaded cumulative state from {path}: train {len(self.train_cumulative_correct_map)}, "
                    f"valid {len(self.valid_cumulative_correct_map)}, next_section={self._resume_section_start}, "
                    f"next_batch={self._resume_batch_start}, batch_recs={len(self._resume_batch_all_recs)}, "
                    f"pending={len(self._resume_pending_task_ids)}, "
                    f"terminal_incorrect={len(self._terminal_incorrect_task_ids)}, "
                    f"infrastructure_deferred={len(self._infrastructure_deferred_task_ids)}"
                )
            except ValueError:
                # Holdout mismatch is a hard error — never silently resume into a
                # different experiment. Genuine load failures fall through below.
                raise
            except Exception as e:
                logger.warning(f"[Resume] Failed to load cum_state from {path}: {e}")

    def _reconcile_resume_task_sets(self, completed_task_ids: Optional[set[str]] = None) -> None:
        """Remove scored IDs from pending/deferred sets and de-duplicate both."""
        completed = set(completed_task_ids or set())
        if not completed and getattr(self, "_resume_batch_all_recs", None):
            completed = {
                str(rec.get("id")) for rec in self._resume_batch_all_recs
                if isinstance(rec, dict) and rec.get("id") is not None
            }
        self._resume_pending_task_ids = {
            str(x) for x in getattr(self, "_resume_pending_task_ids", set())
            if str(x) not in completed
        }
        self._infrastructure_deferred_task_ids = {
            str(x) for x in getattr(self, "_infrastructure_deferred_task_ids", set())
            if str(x) not in completed
        }

    def _save_cum_state(
        self,
        next_section: int,
        next_batch: int = 0,
        batch_all_recs: Optional[list] = None,
        pending_task_ids: Optional[list[str]] = None,
    ):
        try:
            completed_ids = {
                str(rec.get("id")) for rec in (batch_all_recs or [])
                if isinstance(rec, dict) and rec.get("id") is not None
            }
            self._reconcile_resume_task_sets(completed_ids)
            data = {
                "train_cum": self.train_cumulative_correct_map,
                "valid_cum": self.valid_cumulative_correct_map,
                "next_section": int(next_section),
                "next_batch": int(next_batch),
                "holdout_categories": getattr(self, "holdout_categories", None),
                "terminal_incorrect_task_ids": sorted(
                    str(x) for x in getattr(self, "_terminal_incorrect_task_ids", set())
                ),
                "infrastructure_deferred_task_ids": sorted(
                    str(x) for x in getattr(self, "_infrastructure_deferred_task_ids", set())
                ),
                "infrastructure_error_policy_version": 2,
            }
            if batch_all_recs is not None:
                data["batch_all_recs"] = batch_all_recs
            if pending_task_ids is not None:
                data["pending_task_ids"] = sorted({str(x) for x in pending_task_ids if x is not None})
            with open(self._cum_state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Resume] Failed to save cum_state: {e}")

    @staticmethod
    def _stable_dedup_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the first scored record per task ID within one epoch.

        Resume/retry may accidentally append the same task multiple times. The
        first completed result is the durable one; later duplicates must not
        change the denominator, SR, memory count, or Region evidence. Legacy
        records without IDs are keyed by question text for compatibility.
        """
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rec in records or []:
            rid = rec.get("id")
            key = f"id:{rid}" if rid is not None else f"question:{str(rec.get('question', ''))}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)
        return deduped

    @staticmethod
    def _normalize_resume_batch_records(records: list) -> list[dict[str, Any]]:
        """Normalize and stably de-duplicate checkpoint records by task ID."""
        normalized: list[dict[str, Any]] = []
        for rec in records or []:
            if isinstance(rec, dict):
                normalized.append({
                    "id": rec.get("id"),
                    "question": str(rec.get("question", "")),
                    "correct": bool(rec.get("correct", False)),
                })
            elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
                normalized.append({"id": None, "question": str(rec[0]), "correct": bool(rec[1])})
        return HLERunner._stable_dedup_records(normalized)

    @staticmethod
    def _slim_batch_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Persist one scored record per task ID for safe partial resume."""
        slim = [
            {
                "id": r.get("id"),
                "question": str(r.get("question", "")),
                "correct": bool(r.get("correct", False)),
            }
            for r in records
        ]
        return HLERunner._stable_dedup_records(slim)

    def _build_train_batch_schedule(
        self,
        df: pd.DataFrame,
        batches: list[list[int]],
        start_batch_idx: int,
        completed_task_ids: set[str],
        legacy_resume_without_ids: bool,
        sec_idx: int,
    ) -> list[tuple[Optional[int], list[int]]]:
        """Build a compact pending prepass followed by batches at the saved cursor.

        ID-aware checkpoints may contain long-tail pending IDs scattered across
        many historical batches. Replaying the original batch layout serializes
        tiny executors. Group those IDs into full batches once, then continue at
        ``start_batch_idx``. ``None`` marks a pending-prepass batch, which must
        not advance the durable original-batch cursor.
        """
        if legacy_resume_without_ids:
            return [(idx, batch) for idx, batch in enumerate(batches)]

        id_to_index = {str(df.iloc[idx].get("id", idx)): idx for idx in range(len(df))}
        pending_indices = [
            id_to_index[task_id]
            for task_id in sorted(self._resume_pending_task_ids)
            if task_id not in completed_task_ids and task_id in id_to_index
        ]
        missing = [
            task_id for task_id in sorted(self._resume_pending_task_ids)
            if task_id not in completed_task_ids and task_id not in id_to_index
        ]
        if missing:
            logger.warning(
                "[train sec %d] %d resume pending IDs are absent from the current dataset",
                sec_idx, len(missing),
            )
        schedule: list[tuple[Optional[int], list[int]]] = []
        if pending_indices:
            pending_batches = [
                pending_indices[i:i + self.batch_size]
                for i in range(0, len(pending_indices), self.batch_size)
            ]
            logger.info(
                "[train sec %d] Compact pending prepass: %d items in %d batch(es), then resume at batch %d",
                sec_idx, len(pending_indices), len(pending_batches), start_batch_idx,
            )
            schedule.extend((None, batch) for batch in pending_batches)
        cursor = max(0, min(int(start_batch_idx), len(batches)))
        schedule.extend((idx, batches[idx]) for idx in range(cursor, len(batches)))
        return schedule

    def _save_cum_state_to_snapshot(self, sec_idx: int):
        """Mirror runner cumulative state into snapshot/<sec>/local_cache for epoch-aligned resume."""
        try:
            if not self._cum_state_path.exists():
                return
            snapshot_dir = self.ck_dir / "snapshot" / str(int(sec_idx))
            if not self._is_valid_snapshot_dir(snapshot_dir):
                return
            snapshot_state = (
                snapshot_dir
                / "local_cache"
                / "cum_state.json"
            )
            snapshot_state.parent.mkdir(parents=True, exist_ok=True)
            snapshot_state.write_text(self._cum_state_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            logger.warning(
                "[Resume] Failed to persist checkpoint cum_state for section %s: %s",
                sec_idx,
                e,
            )

    def _save_cum_state_to_snapshot_batch(self, sec_idx: int, batch_ckpt_id: str):
        """Mirror cum_state into a batch-level snapshot directory."""
        try:
            if not self._cum_state_path.exists():
                return
            snapshot_dir = self.ck_dir / "snapshot" / str(batch_ckpt_id)
            if not self._is_valid_snapshot_dir(snapshot_dir):
                return
            snapshot_state = snapshot_dir / "local_cache" / "cum_state.json"
            snapshot_state.parent.mkdir(parents=True, exist_ok=True)
            snapshot_state.write_text(self._cum_state_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            logger.warning("[Resume] Failed to persist batch cum_state for %s: %s", batch_ckpt_id, e)

    def _save_pending_recovery_checkpoint(
        self,
        sec_idx: int,
        start_batch_idx: int,
        all_recs: list[dict[str, Any]],
        completed_task_ids: set[str],
    ) -> None:
        """Persist pending-prepass progress immediately without advancing cursor."""
        if not self.memory_service:
            return
        try:
            ckpt_id = f"{sec_idx}_pending"
            remaining_pending = sorted(
                task_id for task_id in self._resume_pending_task_ids
                if task_id not in completed_task_ids
            )
            ckpt_meta = self.memory_service.save_checkpoint_snapshot(self.ck_dir, ckpt_id=ckpt_id)
            self._save_cum_state(
                sec_idx,
                next_batch=start_batch_idx,
                batch_all_recs=self._slim_batch_records(all_recs),
                pending_task_ids=remaining_pending,
            )
            self._save_cum_state_to_snapshot_batch(sec_idx, ckpt_id)
            logger.info(
                "[train sec %d] Saved pending recovery ckpt %s: completed=%d remaining_pending=%d cursor=%d meta=%s",
                sec_idx, ckpt_id, len(all_recs), len(remaining_pending), start_batch_idx, ckpt_meta,
            )
        except Exception as e:
            logger.warning("[train sec %d] Failed to save pending recovery ckpt: %s", sec_idx, e)

    def _remove_pending_recovery_checkpoint(self, sec_idx: int) -> None:
        try:
            pending_dir = self.ck_dir / "snapshot" / f"{sec_idx}_pending"
            if pending_dir.exists():
                import shutil
                shutil.rmtree(pending_dir, ignore_errors=True)
                logger.info("[Checkpoint] Removed superseded pending recovery ckpt: %s", pending_dir.name)
        except Exception as e:
            logger.warning("[Checkpoint] Failed to remove pending recovery ckpt for sec %d: %s", sec_idx, e)

    def _cleanup_batch_checkpoints(self, sec_idx: int, keep: int = 3):
        """Remove old batch-level checkpoints for a section, keeping only the latest `keep`."""
        try:
            snapshot_root = self.ck_dir / "snapshot"
            if not snapshot_root.is_dir():
                return
            prefix = f"{sec_idx}_b"
            batch_dirs = [
                p for p in snapshot_root.iterdir()
                if p.is_dir() and p.name.startswith(prefix) and self._is_valid_snapshot_dir(p)
            ]
            if len(batch_dirs) <= keep:
                return
            batch_dirs.sort(key=lambda p: int(p.name.split("_b")[1]))
            for d in batch_dirs[:-keep] if keep > 0 else batch_dirs:
                import shutil
                shutil.rmtree(d, ignore_errors=True)
                logger.info("[Checkpoint] Removed old batch ckpt: %s", d.name)
        except Exception as e:
            logger.warning("[Checkpoint] Cleanup failed for sec %d: %s", sec_idx, e)

    def _is_valid_snapshot_dir(self, snapshot_dir: Path) -> bool:
        """Return True only for real memory snapshots (not placeholder numeric dirs)."""
        try:
            if not snapshot_dir.is_dir():
                return False
            if (snapshot_dir / "snapshot_meta.json").is_file():
                return True
            if (snapshot_dir / "cube").is_dir():
                return True
        except Exception:
            return False
        return False

    def _resume_from_ckpt_if_needed(self):
        # Prefer this run's newest complete snapshot after a platform retry.
        # On the first launch it does not exist, so the configured bootstrap
        # checkpoint remains the source.  Pinning MEMRL_RUN_ID then makes
        # eviction/failure retries resume forward instead of rewinding to the
        # original bootstrap snapshot.
        resume_root = None
        prefer_current = bool(getattr(self, "ckpt_resume_prefer_current_run", False))
        current_snapshot_root = self.ck_dir / "snapshot"
        if prefer_current and current_snapshot_root.exists():
            has_complete = any(
                self._is_valid_snapshot_dir(p)
                for p in current_snapshot_root.iterdir()
                if p.is_dir()
            )
            if has_complete:
                resume_root = self.ck_dir
                logger.info("[AutoResume] preferring latest checkpoint in current run: %s", self.ck_dir)
        if resume_root is None and getattr(self, "ckpt_resume_enabled", False) and getattr(self, "ckpt_resume_path", None):
            resume_root = Path(self.ckpt_resume_path)
            logger.info("[AutoResume] bootstrapping from configured checkpoint: %s", resume_root)
        elif resume_root is None and current_snapshot_root.exists():
            resume_root = self.ck_dir

        if resume_root is None:
            self._load_cum_state()
            return

        snapshot_root = resume_root / "snapshot" if (resume_root / "snapshot").is_dir() else resume_root
        candidates = []
        if snapshot_root.exists():
            for checkpoint_dir in snapshot_root.iterdir():
                if not checkpoint_dir.is_dir() or not self._is_valid_snapshot_dir(checkpoint_dir):
                    continue
                state_path = checkpoint_dir / "local_cache" / "cum_state.json"
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    state = {}
                records = state.get("batch_all_recs", []) or []
                ids = {
                    str(rec.get("id")) for rec in records
                    if isinstance(rec, dict) and rec.get("id") is not None
                }
                legacy_questions = {
                    str(rec.get("question", "")) for rec in records
                    if isinstance(rec, dict) and rec.get("id") is None
                }
                unique_completed = len(ids) + len(legacy_questions)
                section = int(state.get("next_section", 0) or 0)
                batch = int(state.get("next_batch", 0) or 0)
                pending = len({str(x) for x in state.get("pending_task_ids", []) or []})
                score = (section, unique_completed, batch, -pending, checkpoint_dir.stat().st_mtime)
                candidates.append((score, checkpoint_dir, state))

        target = None
        resume_epoch = getattr(self, "ckpt_resume_epoch", None)
        if candidates and resume_epoch is not None and int(resume_epoch) > 0:
            epoch_candidates = [
                item for item in candidates
                if item[1].name == str(int(resume_epoch))
            ]
            if epoch_candidates:
                target = max(epoch_candidates, key=lambda item: item[0])[1]
            else:
                logger.warning("[Resume] ckpt epoch %s not found, falling back to furthest state.", resume_epoch)

        if target is None and candidates:
            best_score, target, best_state = max(candidates, key=lambda item: item[0])
            logger.info(
                "[AutoResume] selected furthest checkpoint %s: section=%s unique=%s batch=%s pending=%s",
                target, best_state.get("next_section", 0), best_score[1],
                best_state.get("next_batch", 0), len(best_state.get("pending_task_ids", []) or []),
            )
        elif (
            target is None and resume_root.is_dir()
            and self._is_valid_snapshot_dir(resume_root)
        ):
            # Concrete checkpoint directory passed as bootstrap.
            target = resume_root

        if target is None:
            # Non-checkpoint resume mode: use experiment-level cumulative state.
            candidate_states = [
                resume_root / "local_cache" / "cum_state.json",
            ]
            if resume_root.parent.name == "snapshot":
                candidate_states.append(resume_root.parent.parent / "local_cache" / "cum_state.json")
            candidate_states.append(self._cum_state_path)

            for state_path in candidate_states:
                if state_path.exists():
                    self._load_cum_state(state_path)
                    return
            self._load_cum_state()
            return

        target_state = target / "local_cache" / "cum_state.json"
        if target_state.exists():
            self._load_cum_state(target_state)
        else:
            import re
            batch_match = re.match(r"^(\d+)_b(\d+)$", target.name)
            fallback_states = [
                resume_root / "local_cache" / "cum_state.json",
                self._cum_state_path,
            ]
            loaded = False
            for sp in fallback_states:
                if sp.exists():
                    self._load_cum_state(sp)
                    loaded = True
                    break
            if not loaded and batch_match:
                self._resume_section_start = int(batch_match.group(1))
                self._resume_batch_start = int(batch_match.group(2)) + 1
                logger.warning(
                    "[Resume] No cum_state for batch ckpt %s; derived sec=%s, batch=%s",
                    target.name, self._resume_section_start, self._resume_batch_start,
                )
            elif not loaded:
                self._load_cum_state()

        try:
            if self.memory_service:
                mem_cube_id = getattr(self.memory_service, "default_cube_id", None)
                self.memory_service.load_checkpoint_snapshot(
                    str(target), mem_cube_id=mem_cube_id
                )
                logger.info(f"[Resume] Loaded memory snapshot from {target}")
        except Exception as e:
            logger.warning(f"[Resume] Failed to load memory snapshot {target}: {e}")

    def _load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset from a single source, apply dataset_ratio, then split into train/valid.

        When eval_categories is set, train uses sel.categories and valid uses eval_categories
        (cross-category transfer mode). Otherwise, stratified split by train_valid_split.
        """

        # 1. Read raw data
        if not Path(self.sel.train_path).exists():
            raise ValueError(f"HLE dataset path does not exist: {self.sel.train_path}")
        df = pd.read_parquet(self.sel.train_path)

        for c in ['id', 'question', 'answer']:
            if c not in df.columns:
                raise ValueError(f"HLE dataset missing required column: {c}")
        if "category" not in df.columns:
            raise ValueError("HLE dataset missing 'category' column")

        # 1b. Optionally drop rows with images (text-only LLMs)
        if getattr(self.sel, "text_only", False) and "image" in df.columns:
            before = len(df)
            has_img = df["image"].apply(lambda x: bool(x) and str(x).strip() not in ("", "None"))
            df = df[~has_img].reset_index(drop=True)
            logger.info("[text_only] Dropped %d image rows; %d text rows remain.", before - len(df), len(df))

        # 2. Cross-category transfer mode
        if self.sel.eval_categories:
            train_cats = self.sel.categories
            eval_cats = self.sel.eval_categories
            if train_cats:
                train = df[df['category'].isin(train_cats)].copy()
            else:
                train = df[~df['category'].isin(eval_cats)].copy()
                logger.info(
                    "[HOLDOUT] Dropped %d/%d items from train (held-out: %s)",
                    len(df) - len(train), len(df), eval_cats,
                )
            valid = df[df['category'].isin(eval_cats)].copy()
            train = self._apply_dataset_ratio(train, "train")
            valid = self._apply_dataset_ratio(valid, "valid")
            if self.sel.num_train:
                train = train.head(int(self.sel.num_train))
            if self.sel.num_valid:
                valid = valid.head(int(self.sel.num_valid))
            self.df_train = train.reset_index(drop=True)
            self.df_valid = valid.reset_index(drop=True)
            logger.info(
                "HLE cross-category transfer: train_cats=%s (%d rows), eval_cats=%s (%d rows)",
                (sorted(self.df_train['category'].unique().tolist()) if not self.df_train.empty else train_cats),
                len(self.df_train), eval_cats, len(self.df_valid),
            )
            return self.df_train, self.df_valid

        # 3. Standard mode: filter categories, ratio, stratified split
        df = self._filter_by_category(df)
        df = self._apply_dataset_ratio(df, "full")

        if len(df) == 0:
            raise ValueError("HLE dataset is empty after category filtering/sampling")

        df = df.reset_index(drop=True)
        n_total = len(df)

        split_ratio = getattr(self, "train_valid_split", 0.8)
        train_parts = []
        valid_parts = []
        for cat, group in df.groupby("category", sort=False):
            shuffled = group.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
            n_train = int(len(shuffled) * split_ratio)
            train_parts.append(shuffled.iloc[:n_train].copy())
            valid_parts.append(shuffled.iloc[n_train:].copy())

        train = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[:0].copy()
        valid = pd.concat(valid_parts, ignore_index=True) if valid_parts else df.iloc[:0].copy()

        if self.sel.num_train:
            train = train.head(int(self.sel.num_train))
        if self.sel.num_valid:
            valid = valid.head(int(self.sel.num_valid))

        self.df_train, self.df_valid = train.reset_index(drop=True), valid.reset_index(drop=True)
        logger.info(
            f"HLE loaded from single dataset: total={n_total}, train={len(self.df_train)}, valid={len(self.df_valid)}"
        )
        return self.df_train, self.df_valid

    def _resolve_legacy_resume_record_ids(self, train_df: pd.DataFrame) -> None:
        """Map uniquely identifiable legacy question-only records to task IDs."""
        if not self._resume_batch_all_recs or train_df is None or train_df.empty:
            return
        question_to_ids: dict[str, list[str]] = {}
        for _, row in train_df.iterrows():
            question_to_ids.setdefault(str(row.get("question", "")), []).append(str(row.get("id")))
        mapped = ambiguous = missing = 0
        migrated: list[dict[str, Any]] = []
        for rec in self._resume_batch_all_recs:
            item = dict(rec) if isinstance(rec, dict) else rec
            if isinstance(item, dict) and item.get("id") is None:
                ids = question_to_ids.get(str(item.get("question", "")), [])
                if len(ids) == 1:
                    item["id"] = ids[0]; mapped += 1
                elif len(ids) > 1:
                    ambiguous += 1
                else:
                    missing += 1
            migrated.append(item)
        before = len(migrated)
        self._resume_batch_all_recs = self._normalize_resume_batch_records(migrated)
        self._reconcile_resume_task_sets()
        logger.info(
            "[Resume] Legacy task-ID migration: mapped=%d ambiguous=%d missing=%d; records %d -> %d",
            mapped, ambiguous, missing, before, len(self._resume_batch_all_recs),
        )

    def _apply_dataset_ratio(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        ratio = getattr(self, "dataset_ratio", 1.0)
        if df is None or df.empty or not (0 < ratio < 1):
            return df
        n_keep = max(1, int(len(df) * ratio))
        if n_keep >= len(df):
            return df
        sampled = df.sample(n=n_keep, random_state=self.random_seed).reset_index(drop=True)
        logger.info(
            "HLE %s split reduced via dataset_ratio %.2f: %d -> %d rows",
            split,
            ratio,
            len(df),
            len(sampled),
        )
        return sampled

    def _filter_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset by categories and optional per-category sampling ratio."""
        if df is None or df.empty:
            return df
        cats = self.sel.categories
        ratio = self.sel.category_ratio
        if cats:
            if 'category' not in df.columns:
                raise ValueError("HLE dataset missing 'category' column for category filtering")
            df = df[df['category'].isin(cats)].reset_index(drop=True)
            logger.info("HLE filtered categories %s -> %d rows", cats, len(df))
        if ratio is not None and 0 < ratio < 1:
            if 'category' not in df.columns:
                raise ValueError("HLE dataset missing 'category' column for category ratio sampling")
            def _sample_group(g: pd.DataFrame) -> pd.DataFrame:
                n_keep = max(1, int(len(g) * ratio))
                return g.sample(n=n_keep, random_state=self.random_seed)
            df = df.groupby('category', group_keys=False).apply(_sample_group).reset_index(drop=True)
            logger.info("HLE applied category_ratio %.2f -> %d rows", ratio, len(df))
        elif ratio is not None:
            logger.warning("category_ratio %.3f is out of (0,1); skip sampling", ratio)
        return df

    def _resolve_ckpt_dirs(self, ckpt_root: Path) -> List[Path]:
        """Resolve snapshot directories (numeric subfolders) from an experiment or snapshot root."""
        if (ckpt_root / "snapshot").is_dir():
            snapshot_root = ckpt_root / "snapshot"
        else:
            snapshot_root = ckpt_root
        if not snapshot_root.is_dir():
            raise ValueError(f"ckpt root does not exist: {snapshot_root}")
        ckpts = [p for p in snapshot_root.iterdir() if p.is_dir() and p.name.isdigit()]
        ckpts.sort(key=lambda p: int(p.name))
        return ckpts

    def _prune_valid_memories(self, valid_questions: Set[str]) -> None:
        """Remove validation questions from local memory indices to avoid leakage."""
        if not self.memory_service or not valid_questions:
            return
        dict_mem = getattr(self.memory_service, "dict_memory", None)
        if isinstance(dict_mem, dict):
            for q in list(dict_mem.keys()):
                if q in valid_questions:
                    dict_mem.pop(q, None)
        query_embeddings = getattr(self.memory_service, "query_embeddings", None)
        if isinstance(query_embeddings, dict):
            for q in list(query_embeddings.keys()):
                if q in valid_questions:
                    query_embeddings.pop(q, None)

    def _eval_ckpt_sequence(self, valid_df: pd.DataFrame) -> None:
        """Load historical checkpoints sequentially and evaluate on valid set."""
        if not self.memory_service:
            raise RuntimeError("memory_service is required for ckpt evaluation")
        if not self.ckpt_eval_path:
            raise ValueError("ckpt_eval_path is not set")
        ckpt_root = Path(self.ckpt_eval_path)
        ckpt_dirs = self._resolve_ckpt_dirs(ckpt_root)
        if not ckpt_dirs:
            raise ValueError(f"No checkpoint folders found under {ckpt_root}")
        valid_questions = set(valid_df["question"].astype(str).tolist())

        for idx, ckpt_dir in enumerate(ckpt_dirs, start=1):
            logger.info("Loading ckpt %s (%d/%d) for eval", ckpt_dir, idx, len(ckpt_dirs))
            mem_cube_id = getattr(self.memory_service, "default_cube_id", None)
            self.memory_service.load_checkpoint_snapshot(
                str(ckpt_dir), mem_cube_id=mem_cube_id
            )
            self._prune_valid_memories(valid_questions)
            self._eval_split(valid_df, tag=f"valid_ckpt_{idx}", step=idx)

    def _baseline_task_key(self, data: Any) -> str:
        """Canonical key for pass@k/reflection loops; prefer id, fallback to question text."""
        candidate_id = None
        question = None
        try:
            candidate_id = data["id"]
        except Exception:
            candidate_id = None
        try:
            question = data["question"]
        except Exception:
            question = None
        if candidate_id is not None:
            try:
                if not pd.isna(candidate_id):
                    cid = str(candidate_id).strip()
                    if cid and cid.lower() != "nan":
                        return cid
            except Exception:
                cid = str(candidate_id).strip()
                if cid:
                    return cid
        return str(question or "")

    def _extract_solution_only(self, trajectory: str) -> str:
        if not trajectory:
            return ""
        if "SOLUTION" in trajectory:
            return trajectory.split("SOLUTION", 1)[1].strip()
        return trajectory.strip()


    def _format_reflection_note(self, question: str, trajectory: str, success: bool) -> str:
        status = "CORRECT" if success else "INCORRECT"

        solution_only = self._extract_solution_only(trajectory)

        note_parts = [
            "You attempted this question before.",
            f"Result: {status}",
            f"Question: {question}",
            "Previous attempt (solution only):",
            solution_only,
            "Reflect on mistakes or gaps, then solve the problem again with a better solution.",
        ]
        return "\n".join([p for p in note_parts if p])


    # ---------- Image helpers ----------
    def _register_image(self, image: Any) -> Optional[Tuple[str, str]]:
        """Convert raw image to data URL, cache in store, and return (image_id, data_url)."""
        if image is None:
            return None
        data_url = None
        if isinstance(image, str) and image.strip():
            data_url = image.strip()
        elif isinstance(image, dict) and 'bytes' in image:
            raw = image.get('bytes')
            if isinstance(raw, bytes):
                b64 = base64.b64encode(raw).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{b64}"
        if not data_url:
            return None
        key = hashlib.md5(data_url.encode('utf-8')).hexdigest()
        with self._image_lock:
            if key in self._image_hash_to_id:
                img_id = self._image_hash_to_id[key]
            else:
                self._image_id_counter += 1
                img_id = f"img_{self._image_id_counter:06d}"
                self._image_hash_to_id[key] = img_id
                self._image_store[img_id] = data_url
                self._persist_image_cache_unlocked()
        return img_id, data_url

    def _fetch_images_by_ids(self, image_ids: List[str]) -> List[Tuple[str, str]]:
        """Return list of (image_id, data_url) for known ids."""
        imgs: List[Tuple[str, str]] = []
        for iid in image_ids or []:
            url = self._image_store.get(str(iid))
            if url:
                imgs.append((str(iid), url))
        return imgs

    def _persist_image_cache_unlocked(self) -> None:
        """Persist image store/index to log_dir (caller holds _image_lock)."""
        try:
            with open(self._image_store_path, "w", encoding="utf-8") as f:
                json.dump(self._image_store, f, ensure_ascii=False)
            with open(self._image_index_path, "w", encoding="utf-8") as f:
                payload = {
                    "hash_index": self._image_hash_to_id,
                    "counter": self._image_id_counter,
                }
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            logger.debug("Failed to persist image cache", exc_info=True)

    def _load_image_cache(self) -> None:
        """Load persisted image cache if available."""
        try:
            if self._image_store_path.exists():
                with open(self._image_store_path, "r", encoding="utf-8") as f:
                    self._image_store = json.load(f)
            if self._image_index_path.exists():
                with open(self._image_index_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    self._image_hash_to_id = payload.get("hash_index", {})
                    self._image_id_counter = int(payload.get("counter", 0))
            if self._image_store:
                logger.info("Loaded %d images from cache", len(self._image_store))
        except Exception:
            logger.debug("Failed to load image cache", exc_info=True)

    def _collect_question_images(self, row: pd.Series) -> List[Any]:
        """Collect raw image objects from the row (supports single image/image_preview)."""
        images: List[Any] = []
        if 'image' in row.index and pd.notna(row['image']) and str(row['image']).strip():
            images.append(row['image'])
        # if 'image_preview' in row.index and pd.notna(row['image_preview']):
        #     images.append(row['image_preview'])
        return images

    def _extract_mem_image_ids(self, mem: Dict[str, Any]) -> List[str]:
        md = mem.get("metadata")
        ids = []
        try:
            if hasattr(md, "model_extra"):
                ids = md.model_extra.get("image_ids") or []
            elif isinstance(md, dict):
                ids = md.get("image_ids") or []
        except Exception:
            pass
        try:
            return [str(x) for x in ids if x]
        except Exception:
            return []

    # ---------- Memory helpers ----------
    def _mem_success_flag(self, m: Dict[str, Any]) -> bool:
        md = m.get("metadata")
        try:
            if hasattr(md, "model_extra"):
                return bool(md.model_extra.get('success'))
            if isinstance(md, dict):
                return bool(md.get('success'))
        except Exception:
            pass
        return False

    def _build_memory_context(self, selected_mems: List[Dict[str, Any]], limit: int) -> Tuple[str, List[str], Set[str]]:
        if not selected_mems:
            return "", [], set()
        retrieved_ids: List[str] = []
        memory_image_ids: Set[str] = set()
        succ_blocks, fail_blocks = [], []
        for m in selected_mems[: max(0, limit) or len(selected_mems)]:
            mid = m.get('memory_id') or m.get('id')
            if mid:
                retrieved_ids.append(str(mid))
            content = m.get('content') or m.get('full_content') or ''
            img_ids = self._extract_mem_image_ids(m)
            if img_ids:
                memory_image_ids.update(img_ids)
                content = f"[Image IDs: {', '.join(img_ids)}]\n{content}"
            (succ_blocks if self._mem_success_flag(m) else fail_blocks).append(content)
        sections = []
        if succ_blocks:
            sections.append("=== Successful Memories ===\n" + "\n\n".join(succ_blocks))
        if fail_blocks:
            sections.append("=== Failed Memories (for caution) ===\n" + "\n\n".join(fail_blocks))
        return "\n\n".join(sections), retrieved_ids, memory_image_ids

    def _self_rag_critique(self, question: str, selected_mems: List[Dict[str, Any]], inject_k: int) -> List[Dict[str, Any]]:
        """Use LLM to judge relevance of each retrieved memory, discard irrelevant ones."""
        if not selected_mems:
            return []
        numbered = []
        for i, m in enumerate(selected_mems):
            content = m.get('content') or m.get('full_content') or ''
            numbered.append(f"[Memory {i+1}]\n{content[:2000]}")
        critique_prompt = (
            "You are a relevance judge. Given a question and a list of retrieved memories from past problem-solving attempts, "
            "decide which memories are RELEVANT and could help solve the current question.\n\n"
            f"Question: {question[:2000]}\n\n"
            "Retrieved memories:\n" + "\n\n".join(numbered) + "\n\n"
            "Return ONLY a JSON list of the relevant memory numbers (1-indexed). "
            "If none are relevant, return an empty list: []\n"
            "Example: [1, 3]"
        )
        try:
            resp = self.llm.generate(
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            import re, json
            match = re.search(r'\[[\d\s,]*\]', resp or "")
            if match:
                indices = json.loads(match.group())
                filtered = []
                for idx in indices:
                    if 1 <= idx <= len(selected_mems):
                        filtered.append(selected_mems[idx - 1])
                logger.info("[Self-RAG] Critique kept %d/%d memories", len(filtered), len(selected_mems))
                return filtered
            logger.info("[Self-RAG] Critique returned no valid indices, using all %d memories", len(selected_mems))
        except Exception as e:
            logger.warning("[Self-RAG] Critique failed (%s), using all %d memories", e, len(selected_mems))
        return selected_mems

    # ---------- Prompt & eval ----------
    def _build_messages(
        self,
        question: str,
        memory_ctx: Optional[str] = None,
        answer_type: Optional[Any] = None,
        question_image_ids: Optional[List[str]] = None,
        images_info: Optional[List[Tuple[str, str, str]]] = None,
        reflection_note: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        answer_type_norm = ""
        if answer_type is not None:
            answer_type_norm = str(answer_type).strip().lower()
        system_prompt = (
            self.EXACT_ANSWER_SYSTEM_PROMPT
            if answer_type_norm == "exactmatch"
            else self.MULTIPLE_CHOICE_SYSTEM_PROMPT
        )
        # Compose multi-modal user content (text + list of images)
        legend = ""
        if images_info:
            lines = [f"{i+1}. [{img_id}] ({source})" for i, (img_id, _, source) in enumerate(images_info)]
            legend = "Attached images:\n" + "\n".join(lines)
        text_block = question if not legend else f"Now solve the following question: \n\n[Image IDs: {question_image_ids}]\n{question}\n\n{legend}"
        content: List[Dict[str, Any]] = [{"type": "text", "text": text_block}]
        if images_info:
            for img_id, url, source in images_info:
                content.append({"type": "text", "text": f"Image [{img_id}] ({source})"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if reflection_note:
            msgs.append({"role": "system", "content": reflection_note})
        if memory_ctx:
            msgs.append({"role": "system", "content": memory_ctx})
        msgs.append({"role": "user", "content": content})
        return msgs

    def _extract_answer(self, text: str) -> str:
        # Extract line starting with 'Answer:' then strip trailing punctuation
        m = re.search(r"^\s*Answer\s*:\s*(.+)$", text or "", flags=re.I|re.M)
        ans = (m.group(1) if m else (text or "")).strip()
        return re.sub(r"[\s\.]$", "", ans)

    def _log_llm_call(
        self,
        call_type: str,
        messages: Any,
        response: Any,
        meta: Optional[Dict[str, Any]] = None,
        parsed: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist each LLM interaction (inputs/outputs) to a local cache JSONL file."""
        entry = {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "type": call_type,
            "meta": meta or {},
            "messages": messages,
            "response": response,
        }
        if parsed is not None:
            entry["parsed"] = parsed
        try:
            payload = json.dumps(entry, ensure_ascii=False, default=str)
        except Exception as e:
            try:
                entry["messages"] = str(messages)
                payload = json.dumps(entry, ensure_ascii=False, default=str)
            except Exception:
                logger.debug("Failed to serialize LLM call log: %s", e)
                return
        try:
            with self._log_lock:
                with open(self.llm_log_path, "a", encoding="utf-8") as f:
                    f.write(payload + "\n")
        except Exception:
            logger.debug("Failed to write LLM call log", exc_info=True)

    def _hle_judge(self, question: str, gold: str, response: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import json as _json
        prompt = self.JUDGE_PROMPT.format(question=question, correct_answer=gold, response=response)
        messages = [{"role": "user", "content": prompt}]
        judge_text = ""
        error_info = None
        try:
            judge_text = self.llm_judge.generate(messages, temperature=0.0, max_tokens=4096)
        except Exception as e:
            logger.warning("HLE judge LLM error: %s", e)
            error_info = str(e)
            judge_text = ""

        result = {
            "correct_answer": gold,
            "model_answer": None,
            "reasoning": None,
            "correct": "no",
            "confidence": 0,
            "raw_judge": judge_text,
            "infrastructure_error": bool(error_info and self._is_infrastructure_error_text(error_info)),
            "infrastructure_error_text": error_info if error_info else None,
        }
        # Try JSON parse
        try:
            # Extract JSON object substring if needed
            m = re.search(r"\{[\s\S]*\}", judge_text)
            jtxt = m.group(0) if m else judge_text
            obj = _json.loads(jtxt)
            # accept keys with slight variations
            result["model_answer"] = obj.get("extracted_final_answer") or obj.get("extracted_answer")
            result["reasoning"] = obj.get("reasoning")
            corr = str(obj.get("correct", "no")).strip().lower()
            result["correct"] = "yes" if "yes" in corr else "no"
            try:
                result["confidence"] = int(obj.get("confidence", 0))
            except Exception:
                result["confidence"] = 0
        except Exception:
            # Fallback regex parsing for 'correct:' and 'extracted_final_answer:'
            try:
                m = re.search(r"\*{0,2}extracted_final_answer\*{0,2}\s*:\s*(.+)", judge_text, flags=re.I)
                if m:
                    result["model_answer"] = m.group(1).strip()
                m = re.search(r"\*{0,2}correct\*{0,2}\s*:\s*(yes|no)", judge_text, flags=re.I)
                if m:
                    result["correct"] = m.group(1).strip().lower()
                m = re.search(r"\*{0,2}confidence\*{0,2}\s*:\s*(\d+)", judge_text, flags=re.I)
                if m:
                    result["confidence"] = int(m.group(1))
            except Exception:
                pass
        try:
            log_meta = {"question": question, "gold": gold}
            if meta:
                log_meta.update(meta)
            if error_info:
                log_meta["error"] = error_info
            self._log_llm_call("judge", messages, judge_text, meta=log_meta, parsed=result)
        except Exception:
            logger.debug("Failed to log judge LLM call", exc_info=True)
        return result

    def _evaluate_row(self, row: pd.Series, reflection_note: Optional[str] = None) -> Dict[str, Any]:
        q = str(row['question'])
        gold = str(row['answer'])
        # Collect question images and register them
        question_imgs_raw = self._collect_question_images(row)
        question_images_info: List[Tuple[str, str, str]] = []
        question_image_ids: List[str] = []
        for img in question_imgs_raw:
            reg = self._register_image(img)
            if reg:
                img_id, url = reg
                if img_id in question_image_ids:
                    continue
                question_image_ids.append(img_id)
                question_images_info.append((img_id, url, "question"))
        memory_ctx = None
        retrieved_ids: List[str] = []
        retrieved_topk_queries = None
        memory_image_ids: Set[str] = set()
        if self.memory_service and self.retrieve_k > 0:
            try:
                # Align retrieval threshold knob across benchmarks: rl_config.sim_threshold (fallback tau).
                rl_cfg = getattr(self.memory_service, "rl_config", None)
                tau = float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
            except Exception:
                tau = 0.0
            try:
                retrieve_kwargs = dict(k=self.retrieve_k, threshold=tau)
                if self.memory_filter_categories:
                    retrieve_kwargs["filter_categories"] = self.memory_filter_categories
                ret = self.memory_service.retrieve_query(q, **retrieve_kwargs)
                if isinstance(ret, tuple):
                    ret_result, retrieved_topk_queries = ret
                else:
                    ret_result, retrieved_topk_queries = ret, None
                selected_mems = ret_result.get('selected', []) if ret_result else []
                if self.self_rag and selected_mems:
                    selected_mems = self._self_rag_critique(q, selected_mems, self.self_rag_inject_k)
                memory_ctx, retrieved_ids, memory_image_ids = self._build_memory_context(selected_mems, self.retrieve_k)
            except Exception as e:
                logger.warning("Memory retrieval failed: %s", e)

        # Resolve memory images from store
        memory_images_info: List[Tuple[str, str, str]] = []
        if memory_image_ids:
            for img_id, url in self._fetch_images_by_ids(list(memory_image_ids)):
                memory_images_info.append((img_id, url, "memory"))

        images_info = question_images_info + memory_images_info

        answer_type = row.get('answer_type', None)
        messages = self._build_messages(
            q,
            memory_ctx=memory_ctx,
            answer_type=answer_type,
            question_image_ids=question_image_ids,
            images_info=images_info,
            reflection_note=reflection_note,
        )
        call_meta = {
            "question_id": row.get('id', None),
            "answer_type": answer_type,
        }
        gen_error = None
        try:
            if not self.llm.model.startswith("gemini-3"):
                kwargs = dict(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    )
            else:
                kwargs = dict(
                    messages=messages,
                    temperature=self.temperature
                    )
                _re = os.environ.get("MEMRL_REASONING_EFFORT", "").strip()
                if _re:
                    kwargs["reasoning_effort"] = _re
                _max_comp = int(os.environ.get("MEMRL_MAX_COMPLETION_TOKENS", "0") or "0")
                if _max_comp > 0:
                    kwargs["max_completion_tokens"] = _max_comp

            if self.llm.model == "gpt-5.2":
                kwargs["reasoning_effort"] = "low"

            output = self.llm.generate(**kwargs)

        except Exception as e:
            logger.error("LLM error: %s", e)
            gen_error = str(e)
            output = ""

        if not output.strip() and not gen_error:
            gen_error = "empty_response"

        _runner_retries = int(os.environ.get("MEMRL_RUNNER_MAX_RETRIES", "3"))
        _retry_max_tokens = int(os.environ.get("MEMRL_RETRY_MAX_COMPLETION_TOKENS", "50000") or "50000")
        for _retry_i in range(_runner_retries):
            if output.strip():
                break
            if self._is_infrastructure_error_text(gen_error):
                logger.info(
                    "Infrastructure error returned; deferring retry to epoch end: %s",
                    str(gen_error)[:240],
                )
                break
            import time as _time
            _wait = 60
            logger.warning(
                "[Runner retry %d/%d] Empty response, waiting %ds, adding max_completion_tokens=%d...",
                _retry_i + 1, _runner_retries, _wait, _retry_max_tokens,
            )
            _time.sleep(_wait)
            try:
                retry_kwargs = dict(kwargs)
                retry_kwargs["max_completion_tokens"] = _retry_max_tokens
                output = self.llm.generate(**retry_kwargs)
                if output.strip():
                    gen_error = None
            except Exception as e:
                logger.error("Runner retry %d failed: %s", _retry_i + 1, e)
        if gen_error:
            call_meta["error"] = gen_error
        self._log_llm_call("solution", messages, output, meta=call_meta)
        judge_res = self._hle_judge(q, gold, output or "", meta={"question_id": row.get('id', None)})
        if judge_res.get("infrastructure_error"):
            gen_error = "judge_infrastructure_error: " + str(judge_res.get("infrastructure_error_text") or "")
        correct = True if str(judge_res.get("correct", "no")).lower() == "yes" else False

        rec: Dict[str, Any] = {
            "id": row.get('id', None),
            "question": q,
            "gold": gold,
            "raw_output": output,
            "correct": bool(correct),
            "judge_response": judge_res,
            "retrieved_ids": retrieved_ids,
            "image_ids": question_image_ids,
            "trajectory": f"QUESTION\n{q}\n\nSOLUTION\n{(output or '').strip()}\n",
            "category": row.get("category", None),
            "raw_subject": row.get("raw_subject", None),
            "gen_error": gen_error,
        }
        if retrieved_topk_queries is not None:
            rec["retrieved_topk_queries"] = retrieved_topk_queries
        return rec

    def _eval_split(self, df: pd.DataFrame, tag: str, step: Optional[int] = None) -> Dict[str, float]:
        total = len(df)
        if total == 0:
            logger.warning("No rows in %s; skip.", tag)
            return {"acc": 0.0}
        results: List[Dict[str, Any]] = []
        correct_so_far = 0
        start = time.time()
        idxs = list(range(total))
        batches = [idxs[i:i + self.batch_size] for i in range(0, total, self.batch_size)]
        processed = 0
        skipped_rows: List[pd.Series] = []
        for b in tqdm(batches, desc=f"Evaluating {tag}"):
            batch_results: List[Optional[Dict[str, Any]]] = [None] * len(b)
            with ThreadPoolExecutor(max_workers=min(len(b), self.batch_size)) as ex:
                fut2pos = {ex.submit(self._evaluate_row, df.iloc[i]): pos for pos, i in enumerate(b)}
                try:
                    for fut in as_completed(fut2pos, timeout=1800):
                        pos = fut2pos[fut]
                        try:
                            batch_results[pos] = fut.result(timeout=60)
                        except Exception as e:
                            logger.warning("[%s] batch eval failed at item #%d: %s", tag, processed + pos + 1, e)
                            batch_results[pos] = None
                except TimeoutError:
                    logger.warning("[%s] batch timed out after 1800s, skipping remaining items", tag)
            for pos_i, (r, idx) in enumerate(zip(batch_results, b)):
                if r is None or r.get("gen_error"):
                    skipped_rows.append(df.iloc[idx])
                    if r is not None:
                        batch_results[pos_i] = None
            batch_valid = [r for r in batch_results if r is not None]
            results.extend(batch_valid)
            processed += len(batch_valid)
            correct_so_far += sum(1 for r in batch_valid if r.get("correct"))
            acc_so_far = correct_so_far / max(1, processed)
            logger.info("[%s] %d/%d | Acc so far: %.2f%%", tag, processed, total, acc_so_far * 100)

        if skipped_rows:
            logger.info("[%s] Retrying %d skipped items...", tag, len(skipped_rows))
            total_skipped = len(skipped_rows)
            recovered = 0
            retry_batches = [skipped_rows[i:i + self.batch_size] for i in range(0, len(skipped_rows), self.batch_size)]
            for rb in tqdm(retry_batches, desc=f"Retry {tag}"):
                retry_results: List[Optional[Dict[str, Any]]] = [None] * len(rb)
                with ThreadPoolExecutor(max_workers=min(len(rb), self.batch_size)) as ex:
                    fut2pos = {ex.submit(self._evaluate_row, row): pos for pos, row in enumerate(rb)}
                    try:
                        for fut in as_completed(fut2pos, timeout=1800):
                            pos = fut2pos[fut]
                            try:
                                retry_results[pos] = fut.result(timeout=60)
                            except Exception as e:
                                logger.warning("[%s] retry failed at pos %d: %s", tag, pos, e)
                    except TimeoutError:
                        logger.warning("[%s] retry batch timed out", tag)
                retry_valid = [r for r in retry_results if r is not None and not r.get("gen_error")]
                still_failed = sum(1 for r in retry_results if r is not None and r.get("gen_error"))
                if still_failed:
                    logger.warning("[%s] %d items still have gen_error after retry", tag, still_failed)
                recovered += len(retry_valid)
                results.extend(retry_valid)
                correct_so_far += sum(1 for r in retry_valid if r.get("correct"))
            logger.info("[%s] Retry done. Recovered %d/%d skipped items.", tag, recovered, total_skipped)

        acc = correct_so_far / max(1, len(results))
        elapsed = time.time() - start
        logger.info("[%s] Eval finished. Acc: %.2f%% | %d items | %.1fs", tag, acc * 100, total, elapsed)
        try:
            if step is None:
                self.writer.add_scalar(f"Evaluation/Acc", acc)
            else:
                self.writer.add_scalar(f"Evaluation/Acc", acc, step)
            self.writer.flush()
        except Exception:
            pass
        out_dir = self.output_dir / "hle"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tag))
        out_path = out_dir / f"hle_{safe_tag}_results_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info("Saved %s results to: %s", tag, out_path)
        per_item = { r["question"]: bool(r["correct"]) for r in results}

        return {
            "acc": float(acc),
            "per_item": per_item,
            "complete": True,
            "pending_task_ids": [],
        }

    @staticmethod
    def _is_infrastructure_error_text(error: Any) -> bool:
        text = str(error or "").lower()
        return any(token in text for token in (
            "stream header timeout", "resource_exhausted", "rate limit",
            "too many requests", "status=429", "error code: 429",
            "connection error", "connecterror", "connect timeout", "connecttimeout",
            "read timeout", "readtimeout", "remoteprotocolerror",
            "server disconnected", "connection reset", "connection aborted",
            "gateway timeout", "bad gateway", "service unavailable",
            "adapter_api_error", "do_request_failed",
        ))

    @staticmethod
    def _infrastructure_deferred_record(row: pd.Series, reason: str) -> dict[str, Any]:
        return {
            "id": row.get("id"), "question": str(row.get("question", "")),
            "infrastructure_deferred": True, "infrastructure_reason": reason,
        }

    @staticmethod
    def _terminal_incorrect_pending_record(row: pd.Series, reason: str) -> dict[str, Any]:
        """Create an auditable wrong result for a resume-pending terminal timeout.

        These records count in the epoch denominator but deliberately carry no
        trajectory/retrieval payload, so they never create or update memory.
        """
        return {
            "id": row.get("id"),
            "question": str(row.get("question", "")),
            "correct": False,
            "trajectory": "",
            "retrieved_ids": [],
            "retrieved_topk_queries": [],
            "terminal_incorrect": True,
            "terminal_reason": reason,
            "category": row.get("category", ""),
            "raw_subject": row.get("raw_subject", ""),
            "image_ids": [],
        }

    def _evaluate_resume_pending_batch(
        self, rows: list[pd.Series], sec_idx: int, *, phase: str = "resume pending"
    ) -> tuple[list[dict[str, Any]], list[pd.Series]]:
        """Evaluate resume pending once with distinct model vs infrastructure outcomes.

        Futures still running at the 600s deadline are true terminal model
        timeouts and count incorrect. Futures that already returned 429/5xx/
        reconnect errors are infrastructure-deferred: no score, no memory, no
        retry in this section.
        """
        if not rows:
            return [], []
        results: list[Optional[dict[str, Any]]] = [None] * len(rows)
        try:
            pending_timeout_s = max(0.01, float(os.environ.get("MEMRL_RESUME_PENDING_TIMEOUT_S", "600") or "600"))
        except (TypeError, ValueError):
            pending_timeout_s = 600.0
        executor = ThreadPoolExecutor(max_workers=min(len(rows), self.batch_size))
        fut2pos = {executor.submit(self._evaluate_row, row): pos for pos, row in enumerate(rows)}
        unresolved_positions: set[int] = set()
        timed_out = False
        try:
            try:
                for fut in as_completed(fut2pos, timeout=pending_timeout_s):
                    pos = fut2pos[fut]
                    try:
                        results[pos] = fut.result(timeout=60)
                    except Exception as e:
                        results[pos] = {"gen_error": str(e), "infrastructure_error": True}
            except FuturesTimeoutError:
                timed_out = True
                unresolved_positions = {pos for fut, pos in fut2pos.items() if not fut.done()}
                logger.warning(
                    "[train sec %d] %s timed out after %.1fs; "
                    "%d still-inflight items become terminal incorrect",
                    sec_idx, phase, pending_timeout_s, len(unresolved_positions),
                )
        finally:
            if timed_out:
                for fut, pos in fut2pos.items():
                    if pos in unresolved_positions:
                        fut.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        finalized: list[dict[str, Any]] = []
        deferred_rows: list[pd.Series] = []
        terminal = 0
        infrastructure = 0
        for pos, (row, result) in enumerate(zip(rows, results)):
            if pos in unresolved_positions:
                record = self._terminal_incorrect_pending_record(row, phase.replace(" ", "_") + "_inflight_600s")
                finalized.append(record)
                terminal += 1
                if record.get("id") is not None:
                    self._terminal_incorrect_task_ids.add(str(record["id"]))
                continue
            if result is None:
                infrastructure += 1
                deferred_rows.append(row)
                continue
            gen_error = result.get("gen_error")
            if result.get("infrastructure_error") or self._is_infrastructure_error_text(gen_error):
                infrastructure += 1
                deferred_rows.append(row)
                if row.get("id") is not None:
                    self._infrastructure_deferred_task_ids.add(str(row.get("id")))
                continue
            if gen_error or not str(result.get("trajectory", "")).strip():
                # A completed response path without infrastructure evidence is a
                # model failure and counts incorrect.
                record = self._terminal_incorrect_pending_record(row, phase.replace(" ", "_") + "_model_error")
                finalized.append(record)
                terminal += 1
                if record.get("id") is not None:
                    self._terminal_incorrect_task_ids.add(str(record["id"]))
            else:
                finalized.append(result)
        logger.info(
            "[train sec %d] %s finalized: %d valid, %d terminal incorrect, "
            "%d infrastructure-deferred",
            sec_idx, phase, len(finalized) - terminal, terminal, infrastructure,
        )
        return finalized, deferred_rows

    def _train_one_section(self, df: pd.DataFrame, sec_idx: int, start_batch_idx: int = 0) -> Dict[str, float]:
        n = len(df)
        if n == 0:
            logger.info("No train data; skip training section %d", sec_idx)
            return {"acc": 0.0}
        idxs = list(range(n))
        batches = [idxs[i:i + self.batch_size] for i in range(0, n, self.batch_size)]
        all_recs: List[Dict[str, Any]] = []
        processed = 0
        correct_so_far = 0

        completed_task_ids: set[str] = set()
        legacy_resume_without_ids = False
        if start_batch_idx > 0 and self._resume_batch_all_recs:
            all_recs = self._normalize_resume_batch_records(self._resume_batch_all_recs)
            processed = len(all_recs)
            correct_so_far = sum(1 for r in all_recs if r.get("correct"))
            completed_task_ids = {str(r["id"]) for r in all_recs if r.get("id") is not None}
            legacy_resume_without_ids = not completed_task_ids
            logger.info(
                "[train sec %d] Resuming from batch %d, preloaded %d recs (acc %.2f%%, id-aware=%s, pending=%d)",
                sec_idx, start_batch_idx, processed, correct_so_far / max(1, processed) * 100,
                not legacy_resume_without_ids, len(self._resume_pending_task_ids),
            )
            self._resume_batch_all_recs = []

        skipped_rows: List[pd.Series] = [
            df.iloc[i] for i in range(len(df))
            if str(df.iloc[i].get("id", i)) in self._infrastructure_deferred_task_ids
            and str(df.iloc[i].get("id", i)) not in completed_task_ids
        ]
        if skipped_rows:
            logger.info(
                "[train sec %d] Queued %d checkpoint infrastructure-deferred item(s) for epoch-end retry",
                sec_idx, len(skipped_rows),
            )
        batch_schedule = self._build_train_batch_schedule(
            df, batches, start_batch_idx, completed_task_ids, legacy_resume_without_ids, sec_idx
        )
        for batch_idx, original_batch in tqdm(batch_schedule, desc=f"Training Section {sec_idx}/{self.num_sections}"):
            if legacy_resume_without_ids and batch_idx is not None and batch_idx < start_batch_idx:
                continue
            # New checkpoints record task IDs. Revisit only unfinished members of
            # a timed-out partial batch; already persisted items are never re-run.
            b = [
                i for i in original_batch
                if str(df.iloc[i].get("id", i)) not in completed_task_ids
                and (batch_idx is None or str(df.iloc[i].get("id", i)) not in self._resume_pending_task_ids)
                and (batch_idx is None or str(df.iloc[i].get("id", i)) not in self._infrastructure_deferred_task_ids)
            ]
            if not b:
                continue
            terminal_indices = [
                i for i in b if str(df.iloc[i].get("id", i)) in self._terminal_incorrect_task_ids
            ]
            terminal_batch_recs = [
                self._terminal_incorrect_pending_record(df.iloc[i], "persisted_terminal_incorrect")
                for i in terminal_indices
            ]
            if terminal_batch_recs:
                logger.info(
                    "[train sec %d] Skipping model for %d persisted terminal-incorrect task(s)",
                    sec_idx, len(terminal_batch_recs),
                )
            b = [i for i in b if i not in set(terminal_indices)]
            infrastructure_deferred_rows: list[pd.Series] = []
            if not b:
                batch_results = []
            elif batch_idx is None:
                batch_results, infrastructure_deferred_rows = self._evaluate_resume_pending_batch(
                    [df.iloc[i] for i in b], sec_idx
                )
                self._resume_pending_task_ids.difference_update(
                    str(df.iloc[i].get("id", i)) for i in b
                )
            else:
                batch_results, infrastructure_deferred_rows = self._evaluate_resume_pending_batch(
                    [df.iloc[i] for i in b], sec_idx, phase=f"batch {batch_idx}"
                )
            if infrastructure_deferred_rows:
                skipped_rows.extend(infrastructure_deferred_rows)
            candidate_batch_recs = terminal_batch_recs + [r for r in batch_results if r is not None]
            batch_recs = [
                r for r in self._stable_dedup_records(candidate_batch_recs)
                if r.get("id") is None or str(r.get("id")) not in completed_task_ids
            ]
            duplicate_count = len(candidate_batch_recs) - len(batch_recs)
            if duplicate_count:
                logger.warning(
                    "[train sec %d] Dropped %d duplicate task result(s) before metrics/memory update",
                    sec_idx, duplicate_count,
                )
            all_recs.extend(batch_recs)
            completed_task_ids.update(str(r.get("id")) for r in batch_recs if r.get("id") is not None)
            self._reconcile_resume_task_sets(completed_task_ids)
            processed += len(batch_recs)
            correct_so_far += sum(1 for r in batch_recs if r.get("correct"))
            acc_so_far = correct_so_far / max(1, processed)
            logger.info("[train sec %d] %d/%d | Acc so far: %.2f%%", sec_idx, processed, n, acc_so_far * 100)
            memory_recs = [r for r in batch_recs if not r.get("terminal_incorrect")]
            if self.memory_service and memory_recs:
                try:
                    task_descriptions = [r["question"] for r in memory_recs]
                    trajectories = [r["trajectory"] for r in memory_recs]
                    successes = [bool(r["correct"]) for r in memory_recs]
                    retrieved_ids_list = [r.get("retrieved_ids") or [] for r in memory_recs]
                    retrieved_queries = [r.get("retrieved_topk_queries") for r in memory_recs]
                    metadatas = []
                    for rec_entry, s in zip(memory_recs, successes):
                        metadatas.append(
                            {
                                "source_benchmark": "HLE",
                                "success": s,
                                "q_value": 1.0 if s else 0.0,
                                "q_visits": 0,
                                "q_updated_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                                "last_used_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                                "reward_ma": 0.0,
                                "image_ids": rec_entry.get("image_ids", []),
                                "category": rec_entry.get("category", ""),
                                "raw_subject": rec_entry.get("raw_subject", ""),
                            }
                        )
                    self.memory_service.add_memories(
                        task_descriptions=task_descriptions,
                        trajectories=trajectories,
                        successes=successes,
                        retrieved_memory_queries=retrieved_queries,
                        retrieved_memory_ids_list=retrieved_ids_list,
                        metadatas=metadatas,
                    )
                    try:
                        self.memory_service.update_values([float(s) for s in successes], retrieved_ids_list)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("[train sec %d] batch memory add/update failed: %s", sec_idx, e)

            if batch_idx is None:
                self._save_pending_recovery_checkpoint(
                    sec_idx, start_batch_idx, all_recs, completed_task_ids
                )

            if self.memory_service and batch_idx is not None and (batch_idx + 1) % self.ckpt_save_every_n_batches == 0:
                try:
                    batch_ckpt_id = f"{sec_idx}_b{batch_idx}"
                    ckpt_meta = self.memory_service.save_checkpoint_snapshot(self.ck_dir, ckpt_id=batch_ckpt_id)
                    batch_all_recs_slim = self._slim_batch_records(all_recs)
                    pending_ids = [row.get("id") for row in skipped_rows if row.get("id") is not None]
                    self._save_cum_state(
                        sec_idx,
                        next_batch=batch_idx + 1,
                        batch_all_recs=batch_all_recs_slim,
                        pending_task_ids=pending_ids,
                    )
                    self._save_cum_state_to_snapshot_batch(sec_idx, batch_ckpt_id)
                    logger.info("[train sec %d] Saved batch ckpt %s: %s", sec_idx, batch_ckpt_id, ckpt_meta)
                    self._remove_pending_recovery_checkpoint(sec_idx)
                    self._cleanup_batch_checkpoints(sec_idx, keep=self.ckpt_max_keep)
                except Exception as e:
                    logger.warning("[train sec %d] Failed to save batch ckpt: %s", sec_idx, e)

        try:
            infrastructure_retry_rounds = max(1, int(os.environ.get("MEMRL_INFRA_RETRY_ROUNDS", "3") or "3"))
        except (TypeError, ValueError):
            infrastructure_retry_rounds = 3
        try:
            infrastructure_retry_wait_s = max(0.0, float(os.environ.get("MEMRL_INFRA_RETRY_WAIT_S", "240") or "240"))
        except (TypeError, ValueError):
            infrastructure_retry_wait_s = 240.0
        remaining_infrastructure_rows: list[pd.Series] = list(skipped_rows)
        for retry_round in range(1, infrastructure_retry_rounds + 1):
            if not remaining_infrastructure_rows:
                break
            if retry_round > 1 and infrastructure_retry_wait_s > 0:
                logger.info(
                    "[train sec %d] Waiting %.0fs before infrastructure retry round %d/%d",
                    sec_idx, infrastructure_retry_wait_s, retry_round, infrastructure_retry_rounds,
                )
                time.sleep(infrastructure_retry_wait_s)
            round_rows = remaining_infrastructure_rows
            remaining_infrastructure_rows = []
            total_skipped = len(round_rows)
            recovered = 0
            terminalized = 0
            logger.info(
                "[train sec %d] Infrastructure retry round %d/%d for %d item(s)...",
                sec_idx, retry_round, infrastructure_retry_rounds, total_skipped,
            )
            retry_batches = [
                round_rows[i:i + self.batch_size]
                for i in range(0, len(round_rows), self.batch_size)
            ]
            for rb in tqdm(retry_batches, desc=f"Retry Infrastructure Section {sec_idx}"):
                retry_recs, deferred_again = self._evaluate_resume_pending_batch(
                    list(rb), sec_idx, phase=f"epoch-end infrastructure retry round {retry_round}"
                )
                remaining_infrastructure_rows.extend(deferred_again)
                recovered += len(retry_recs)
                terminalized += sum(1 for r in retry_recs if r.get("terminal_incorrect"))
                if retry_recs:
                    retry_recs = [
                        r for r in self._stable_dedup_records(retry_recs)
                        if r.get("id") is None or str(r.get("id")) not in completed_task_ids
                    ]
                    all_recs.extend(retry_recs)
                    completed_task_ids.update(
                        str(r.get("id")) for r in retry_recs if r.get("id") is not None
                    )
                    self._reconcile_resume_task_sets(completed_task_ids)
                    processed += len(retry_recs)
                    correct_so_far += sum(1 for r in retry_recs if r.get("correct"))
                    memory_recs = [r for r in retry_recs if not r.get("terminal_incorrect")]
                    if self.memory_service and memory_recs:
                        try:
                            task_descriptions = [r["question"] for r in memory_recs]
                            trajectories = [r["trajectory"] for r in memory_recs]
                            successes = [bool(r["correct"]) for r in memory_recs]
                            retrieved_ids_list = [r.get("retrieved_ids") or [] for r in memory_recs]
                            retrieved_queries = [r.get("retrieved_topk_queries") for r in memory_recs]
                            metadatas = []
                            for rec_entry, success in zip(memory_recs, successes):
                                metadatas.append({
                                    "source_benchmark": "HLE",
                                    "success": success,
                                    "q_value": 1.0 if success else 0.0,
                                    "q_visits": 0,
                                    "q_updated_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                                    "last_used_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                                    "reward_ma": 0.0,
                                    "image_ids": rec_entry.get("image_ids", []),
                                    "category": rec_entry.get("category", ""),
                                    "raw_subject": rec_entry.get("raw_subject", ""),
                                })
                            self.memory_service.add_memories(
                                task_descriptions=task_descriptions,
                                trajectories=trajectories,
                                successes=successes,
                                retrieved_memory_queries=retrieved_queries,
                                retrieved_memory_ids_list=retrieved_ids_list,
                                metadatas=metadatas,
                            )
                            try:
                                self.memory_service.update_values(
                                    [float(success) for success in successes], retrieved_ids_list
                                )
                            except Exception:
                                pass
                        except Exception as e:
                            logger.warning(
                                "[train sec %d] infrastructure retry memory add failed: %s", sec_idx, e
                            )
            logger.info(
                "[train sec %d] Infrastructure retry round %d/%d done: scored=%d "
                "(terminal=%d), still deferred=%d",
                sec_idx, retry_round, infrastructure_retry_rounds, recovered, terminalized,
                len(remaining_infrastructure_rows),
            )
            round_pending_ids = [row.get("id") for row in remaining_infrastructure_rows if row.get("id") is not None]
            self._resume_pending_task_ids = {str(x) for x in round_pending_ids}
            self._infrastructure_deferred_task_ids = {str(x) for x in round_pending_ids}
            self._save_cum_state(
                sec_idx, next_batch=len(batches),
                batch_all_recs=self._slim_batch_records(all_recs),
                pending_task_ids=round_pending_ids,
            )
            if self.memory_service:
                try:
                    retry_ckpt_id = f"{sec_idx}_infra_r{retry_round}"
                    self.memory_service.save_checkpoint_snapshot(self.ck_dir, ckpt_id=retry_ckpt_id)
                    self._save_cum_state_to_snapshot_batch(sec_idx, retry_ckpt_id)
                    logger.info("[train sec %d] Saved infrastructure retry checkpoint %s", sec_idx, retry_ckpt_id)
                except Exception as e:
                    logger.warning("[train sec %d] Failed to save infrastructure retry checkpoint: %s", sec_idx, e)

        pending_ids = [
            row.get("id") for row in remaining_infrastructure_rows
            if row.get("id") is not None
        ]
        self._resume_pending_task_ids = {str(x) for x in pending_ids}
        self._infrastructure_deferred_task_ids = {str(x) for x in pending_ids}
        if pending_ids:
            self._save_cum_state(
                sec_idx,
                next_batch=len(batches),
                batch_all_recs=self._slim_batch_records(all_recs),
                pending_task_ids=pending_ids,
            )
            return {
                "acc": correct_so_far / max(1, len(all_recs)),
                "per_item": {str(r.get("id") or r["question"]): bool(r["correct"]) for r in all_recs},
                "complete": False,
                "pending_task_ids": pending_ids,
            }

        all_recs = self._stable_dedup_records(all_recs)
        processed = len(all_recs)
        correct_so_far = sum(1 for r in all_recs if r.get("correct"))
        if not all_recs:
            return {"acc": 0.0, "per_item": {}, "complete": False, "pending_task_ids": []}
        acc = correct_so_far / len(all_recs)
        logger.info("Section %d Train Acc: %.2f%%", sec_idx, acc * 100)
        try:
            self.writer.add_scalar("Train/Section_Acc", acc, sec_idx)
            self.writer.flush()
        except Exception:
            pass
        ckpt_meta = self.memory_service.save_checkpoint_snapshot(self.ck_dir, ckpt_id=sec_idx)
        logger.info(f" Saved ckpt: {ckpt_meta}")
        self._cleanup_batch_checkpoints(sec_idx, keep=0)
        per_item = {str(r.get("id") or r["question"]): bool(r["correct"]) for r in all_recs}

        return {
            "acc": float(acc),
            "per_item": per_item
        }

    def _baseline_eval_split(
        self,
        df: pd.DataFrame,
        desc: str,
        *,
        reflection_notes: Optional[Dict[str, str]] = None,
        on_result: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate a dataframe with a single continuous thread pool.

        Unlike a per-batch pool that must wait for every item in a batch to
        finish (so one long-tail item stalls the whole batch), this submits all
        items into one pool of `batch_size` workers and collects results as they
        complete. A slow item occupies just one worker slot; the rest keep
        flowing. Result order is not preserved (callers match by task key).

        If on_result is provided, it is called with each successful result dict
        immediately upon completion (for streaming writes / resume support).
        """
        if df is None or len(df) == 0:
            return []
        n = len(df)
        max_workers = max(1, int(self.batch_size))
        results: List[Dict[str, Any]] = []
        skipped_rows: List[Tuple[pd.Series, Optional[str]]] = []

        def _submit(ex, i):
            row = df.iloc[i]
            note = None
            if reflection_notes:
                note = reflection_notes.get(self._baseline_task_key(row))
            return ex.submit(self._evaluate_row, row, note), i

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2i = dict(_submit(ex, i) for i in range(n))
            done = 0
            with tqdm(total=n, desc=desc) as pbar:
                for fut in as_completed(fut2i):
                    i = fut2i[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        logger.warning("[baseline %s] item #%d failed: %s", desc, i, e)
                        r = None
                    if r is None:
                        skipped_rows.append((df.iloc[i], None))
                    else:
                        results.append(r)
                        if on_result:
                            on_result(r)
                    done += 1
                    pbar.update(1)

        if skipped_rows:
            logger.info("[baseline %s] Retrying %d skipped items...", desc, len(skipped_rows))
            retry_results: List[Optional[Dict[str, Any]]] = [None] * len(skipped_rows)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut2pos_r = {ex.submit(self._evaluate_row, row, note): pos for pos, (row, note) in enumerate(skipped_rows)}
                for fut in as_completed(fut2pos_r):
                    pos = fut2pos_r[fut]
                    try:
                        retry_results[pos] = fut.result()
                    except Exception as e:
                        logger.warning("[baseline %s] retry failed at pos %d: %s", desc, pos, e)
            retry_valid = [r for r in retry_results if r is not None]
            for r in retry_valid:
                if on_result:
                    on_result(r)
            results.extend(retry_valid)
            logger.info("[baseline %s] Retry recovered %d/%d items.", desc, len(retry_valid), len(skipped_rows))

        return results

    def _run_passk_baseline(self, train_df: pd.DataFrame) -> None:
        total_tasks = len(train_df)
        if total_tasks == 0:
            logger.warning("No train data for pass@k baseline; aborting.")
            return
        solved: Set[str] = set()
        summary = []
        result_path = self.log_dir / "baseline_passk_results.jsonl"
        summary_path = self.log_dir / "baseline_passk_summary.json"

        # Resume: rebuild solved set and determine start round from existing results
        start_round = 1
        completed_in_round: Dict[int, Set[str]] = {}
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        rd = int(rec.get("round", 0))
                        key = self._baseline_task_key(rec)
                        if rd not in completed_in_round:
                            completed_in_round[rd] = set()
                        if key:
                            completed_in_round[rd].add(key)
                        if rec.get("correct") and key:
                            solved.add(str(key))
                if completed_in_round:
                    max_round = max(completed_in_round.keys())
                    max_round_count = len(completed_in_round[max_round])
                    expected_pending = total_tasks - len(solved - completed_in_round.get(max_round, set()))
                    pending_at_max = total_tasks - len({k for rd in range(1, max_round) for k in completed_in_round.get(rd, set()) if k in solved})
                    if max_round_count >= (total_tasks - sum(1 for k in solved if all(k not in completed_in_round.get(r, set()) for r in range(max_round, max_round + 1)))):
                        start_round = max_round + 1
                    else:
                        start_round = max_round
                    logger.info(
                        "[pass@k resume] Found %d existing results across %d rounds, %d solved. Resuming from round %d.",
                        sum(len(v) for v in completed_in_round.values()), max_round, len(solved), start_round,
                    )
            except Exception as e:
                logger.warning("[pass@k resume] Failed to parse existing results, starting fresh: %s", e)
                solved = set()
                completed_in_round = {}
                start_round = 1

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("Starting pass@k round %d/%d", round_idx, self.baseline_k)
            pending_idx = [
                i for i in range(total_tasks)
                if self._baseline_task_key(train_df.iloc[i]) not in solved
            ]
            if not pending_idx:
                logger.info("All tasks already solved before round %d; skipping inference.", round_idx)
                cum_acc = (len(solved) / total_tasks) if total_tasks > 0 else 0.0
                summary.append({"round": round_idx, "cum_acc": cum_acc, "solved": len(solved), "total": total_tasks})
                continue

            # Determine which tasks in this round were already evaluated (partial round resume)
            already_done_this_round = completed_in_round.get(round_idx, set())
            remaining_idx = [
                i for i in pending_idx
                if self._baseline_task_key(train_df.iloc[i]) not in already_done_this_round
            ]

            if remaining_idx:
                remaining_df = train_df.iloc[remaining_idx].reset_index(drop=True)
                logger.info(
                    "[pass@k round %d] %d pending total, %d already done this round, %d remaining to evaluate.",
                    round_idx, len(pending_idx), len(already_done_this_round), len(remaining_idx),
                )

                _write_lock = threading.Lock()

                def _on_result(traj):
                    key = self._baseline_task_key(traj)
                    if traj.get("correct") and key:
                        solved.add(str(key))
                    payload = {
                        "round": round_idx,
                        "baseline": "passk",
                        **traj,
                    }
                    with _write_lock:
                        with open(result_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

                trajectories = self._baseline_eval_split(
                    remaining_df, desc=f"pass@k round {round_idx}", on_result=_on_result
                )
            else:
                logger.info("[pass@k round %d] All items already evaluated in partial resume.", round_idx)

            cum_acc = (len(solved) / total_tasks) if total_tasks > 0 else 0.0
            summary.append({"round": round_idx, "cum_acc": cum_acc, "solved": len(solved), "total": total_tasks})
            logger.info("[pass@k round %d] Cumulative SR: %.2f%% (%d/%d)", round_idx, cum_acc * 100, len(solved), total_tasks)
            try:
                self.writer.add_scalar("Baseline/PassK_Cumulative_Acc", cum_acc, round_idx)
            except Exception:
                pass

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _run_reflection_baseline(self, train_df: pd.DataFrame) -> None:
        total_tasks = len(train_df)
        if total_tasks == 0:
            logger.warning("No train data for reflection baseline; aborting.")
            return
        solved: Set[str] = set()
        summary = []
        reflection_notes: Dict[str, str] = {}
        result_path = self.log_dir / "baseline_reflection_results.jsonl"
        summary_path = self.log_dir / "baseline_reflection_summary.json"
        state_path = self.log_dir / "baseline_reflection_state.json"

        start_round = 1
        if state_path.exists():
            try:
                state = json.load(open(state_path, "r", encoding="utf-8"))
                solved = {str(x) for x in state.get("solved", [])}
                reflection_notes = {str(k): v for k, v in state.get("reflection_notes", {}).items()}
                last_completed = int(state.get("last_completed_round", 0))
                start_round = max(1, last_completed + 1)
                logger.info("Resuming reflection baseline from round %d", start_round)
            except Exception as e:
                logger.warning("Failed to load reflection baseline state from %s: %s", state_path, e)

        if start_round > self.baseline_k:
            logger.info("Reflection baseline already completed (last round %d).", start_round - 1)
            return

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("Starting reflection round %d/%d", round_idx, self.baseline_k)
            pending_idx = [
                i for i in range(total_tasks)
                if self._baseline_task_key(train_df.iloc[i]) not in solved
            ]
            if pending_idx:
                pending_df = train_df.iloc[pending_idx].reset_index(drop=True)
                trajectories = self._baseline_eval_split(
                    pending_df,
                    desc=f"reflection round {round_idx}",
                    reflection_notes=reflection_notes,
                )
                for traj in trajectories:
                    key = self._baseline_task_key(traj)
                    if key:
                        reflection_notes[key] = self._format_reflection_note(
                            traj.get("question", ""),
                            traj.get("trajectory", ""),
                            bool(traj.get("correct")),
                        )
                        if traj.get("correct"):
                            solved.add(str(key))
                    payload = {
                        "round": round_idx,
                        "baseline": "reflection",
                        **traj,
                    }
                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            else:
                logger.info("All tasks already solved before round %d; skipping inference.", round_idx)
            cum_acc = (len(solved) / total_tasks) if total_tasks > 0 else 0.0
            logger.info("Reflection round %d completed. Cumulative Acc: %.2f%% (%d/%d)", round_idx, cum_acc * 100, len(solved), total_tasks)
            summary.append({"round": round_idx, "cum_acc": cum_acc, "solved": len(solved), "total": total_tasks})
            try:
                self.writer.add_scalar("Baseline/Reflection_Cumulative_Acc", cum_acc, round_idx)
            except Exception:
                pass
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "last_completed_round": round_idx,
                            "solved": sorted(solved),
                            "reflection_notes": reflection_notes,
                            "total": total_tasks,
                            "updated_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                logger.warning("Failed to save reflection baseline state to %s: %s", state_path, e)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _write_epoch_metrics(
        self,
        sec_idx: int,
        *,
        train_result: Optional[Dict[str, Any]],
        valid_result: Optional[Dict[str, Any]],
        complete: bool,
        pending_task_ids: Optional[set[str]] = None,
    ) -> None:
        """Persist finalized epoch metrics without relying on log parsing.

        A section is publishable only when every scheduled train task has a
        result.  Partial sections are recorded as incomplete for audit/recovery,
        but callers must not treat their independent SR as a final metric.
        """
        pending = sorted(str(x) for x in (pending_task_ids or set()))
        train_items = (train_result or {}).get("per_item", {}) or {}
        valid_items = (valid_result or {}).get("per_item", {}) or {}
        train_correct = sum(1 for v in train_items.values() if v)
        valid_correct = sum(1 for v in valid_items.values() if v)
        cum_train_total = len(self.train_cumulative_correct_map)
        cum_train_correct = sum(1 for v in self.train_cumulative_correct_map.values() if v)
        cum_valid_total = len(self.valid_cumulative_correct_map)
        cum_valid_correct = sum(1 for v in self.valid_cumulative_correct_map.values() if v)
        metric = {
            "schema_version": 1,
            "epoch": int(sec_idx),
            "is_complete": bool(complete),
            "pending_task_ids": pending,
            "train": {
                "completed": len(train_items),
                "correct": train_correct,
                "independent_sr": (train_correct / len(train_items)) if train_items else None,
                "cumulative_completed": cum_train_total,
                "cumulative_correct": cum_train_correct,
                "cumulative_sr": (cum_train_correct / cum_train_total) if cum_train_total else None,
            },
            "valid": {
                "completed": len(valid_items),
                "correct": valid_correct,
                "independent_sr": (valid_correct / len(valid_items)) if valid_items else None,
                "cumulative_completed": cum_valid_total,
                "cumulative_correct": cum_valid_correct,
                "cumulative_sr": (cum_valid_correct / cum_valid_total) if cum_valid_total else None,
            },
            "train_valid_split": float(self.train_valid_split),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        try:
            root = self.ck_dir / "epoch_metrics.json"
            history: Dict[str, Any] = {}
            if root.exists():
                try:
                    history = json.loads(root.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("[Metrics] failed to parse %s; rebuilding", root)
            history.setdefault("schema_version", 1)
            history.setdefault("epochs", {})[str(sec_idx)] = metric
            tmp = root.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, root)
            snapshot = self.ck_dir / "snapshot" / str(int(sec_idx)) / "epoch_metrics.json"
            if snapshot.parent.is_dir():
                snapshot.write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("[Metrics] wrote epoch %d (complete=%s, train=%d, pending=%d)", sec_idx, complete, len(train_items), len(pending))
        except Exception as exc:
            logger.warning("[Metrics] failed to write epoch %d metrics: %s", sec_idx, exc)

    def run(self):
        logger.info(
            "[CheckpointV2] ID-aware partial-batch resume enabled: completed IDs are persisted, "
            "pending IDs are replayed, and only complete sections publish epoch_metrics.json."
        )
        train_df, valid_df = self._load()
        self._resolve_legacy_resume_record_ids(train_df)

        if self.baseline_mode in {"passk", "reflection"}:
            if self.baseline_mode == "passk":
                self._run_passk_baseline(train_df)
            else:
                self._run_reflection_baseline(train_df)
            try:
                self.writer.close()
            except Exception:
                pass
            try:
                with self._image_lock:
                    self._persist_image_cache_unlocked()
            except Exception:
                logger.debug("Failed to persist image cache on shutdown", exc_info=True)
            return

        # Test-only mode: evaluate on all data without training
        if self.mode == 'test':
            logger.info("Running in TEST-ONLY mode (no training, no memory updates)")
            # Combine train and valid for full evaluation
            all_df = pd.concat([train_df, valid_df], ignore_index=True) if len(valid_df) > 0 else train_df
            logger.info(f"Evaluating on {len(all_df)} total items")
            eval_res = self._eval_split(all_df, tag="test_eval", step=0)
            eval_per_item = eval_res["per_item"]
            total_items = len(eval_per_item)
            total_correct = sum(1 for x in eval_per_item.values() if x)
            accuracy = total_correct / max(1, total_items)
            logger.info(f"[Test] Final Accuracy: {accuracy * 100:.2f}% ({total_correct}/{total_items})")
            try:
                self.writer.add_scalar("Test/Accuracy", accuracy, 0)
                self.writer.close()
            except Exception:
                pass
            try:
                with self._image_lock:
                    self._persist_image_cache_unlocked()
            except Exception:
                logger.debug("Failed to persist image cache on shutdown", exc_info=True)
            return

        # If enabled, evaluate by loading historical checkpoints sequentially.
        if self.ckpt_eval_enabled:
            if len(valid_df) == 0:
                logger.warning("Valid set is empty; skip ckpt evaluation.")
                return
            self._eval_ckpt_sequence(valid_df)
            return

        # ------------------------------
        if not self.train_cumulative_correct_map:
            self.train_cumulative_correct_map = {}
        if not self.valid_cumulative_correct_map:
            self.valid_cumulative_correct_map = {}

        # ------------------------------
        if len(valid_df) != 0 and self._resume_section_start == 0:
            valid_res = self._eval_split(valid_df, tag="valid_initial", step=0)
            valid_per_item = valid_res["per_item"]

            for qid, correct in valid_per_item.items():
                self.valid_cumulative_correct_map[qid] = bool(correct)

            total_valid_items = len(self.valid_cumulative_correct_map)
            total_valid_correct = sum(1 for x in self.valid_cumulative_correct_map.values() if x)
            cumulative_valid_acc = total_valid_correct / max(1, total_valid_items)

            logger.info(
                f"[Valid] Initial Cumulative Acc: {cumulative_valid_acc * 100:.2f}% "
                f"({total_valid_correct}/{total_valid_items})"
            )
            try:
                self.writer.add_scalar("Valid/Cumulative_Acc", cumulative_valid_acc, 0)
            except Exception:
                pass

        start_section = max(1, int(self._resume_section_start))
        for sec_idx in range(start_section, self.num_sections + 1):
            if sec_idx != start_section and self._infrastructure_deferred_task_ids:
                logger.info(
                    "[train sec %d] Clearing %d infrastructure-deferred task IDs for a fresh epoch retry",
                    sec_idx, len(self._infrastructure_deferred_task_ids),
                )
                self._infrastructure_deferred_task_ids.clear()

            # ------------------------------
            if len(train_df) != 0:
                batch_start = self._resume_batch_start if sec_idx == start_section else 0
                res = self._train_one_section(train_df, sec_idx, start_batch_idx=batch_start)
                train_per_item = res["per_item"]
                train_complete = bool(res.get("complete", True))

                # ------------------------------
                for qid, correct in train_per_item.items():
                    if qid not in self.train_cumulative_correct_map:
                        self.train_cumulative_correct_map[qid] = False
                    if correct:
                        self.train_cumulative_correct_map[qid] = True

                total_items = len(self.train_cumulative_correct_map)
                total_correct = sum(1 for x in self.train_cumulative_correct_map.values() if x)
                cumulative_acc = total_correct / max(1, total_items)

                logger.info(
                    f"[Train] Cumulative Acc after section {sec_idx}: {cumulative_acc * 100:.2f}% "
                    f"({total_correct}/{total_items})"
                )
                try:
                    self.writer.add_scalar("Train/Cumulative_Acc", cumulative_acc, sec_idx)
                except Exception:
                    pass

            if not len(train_df) or train_complete:
                self._infrastructure_deferred_task_ids.clear()
                self._save_cum_state(sec_idx + 1, pending_task_ids=[])
                self._save_cum_state_to_snapshot(sec_idx)
            else:
                deferred_ids = set(res.get("pending_task_ids", []))
                logger.warning(
                    "[Metrics] section %d incomplete after infrastructure retries: %d deferred; "
                    "continuing to section %d without carrying them into the new epoch",
                    sec_idx, len(deferred_ids), sec_idx + 1,
                )
                # Deferred is epoch-local. The next epoch evaluates all task IDs
                # afresh; no unresolved item is scored or written to memory here.
                self._resume_pending_task_ids.clear()
                self._infrastructure_deferred_task_ids.clear()
                self._save_cum_state(sec_idx + 1, pending_task_ids=[])
                if self.memory_service:
                    try:
                        transition_id = f"{sec_idx}_incomplete_final"
                        self.memory_service.save_checkpoint_snapshot(self.ck_dir, ckpt_id=transition_id)
                        self._save_cum_state_to_snapshot_batch(sec_idx, transition_id)
                        logger.info(
                            "[train sec %d] Saved incomplete-epoch transition checkpoint %s -> section %d",
                            sec_idx, transition_id, sec_idx + 1,
                        )
                    except Exception as e:
                        logger.warning(
                            "[train sec %d] Failed to save incomplete-epoch transition checkpoint: %s",
                            sec_idx, e,
                        )
                train_complete = False

            # ------------------------------
            valid_res = None
            if len(valid_df) != 0:
                valid_res = self._eval_split(valid_df, tag=f"valid_sec_{sec_idx}", step=sec_idx)
                valid_per_item = valid_res["per_item"]

                for qid, correct in valid_per_item.items():
                    if qid not in self.valid_cumulative_correct_map:
                        self.valid_cumulative_correct_map[qid] = False
                    if correct:
                        self.valid_cumulative_correct_map[qid] = True

                total_valid_items = len(self.valid_cumulative_correct_map)
                total_valid_correct = sum(1 for x in self.valid_cumulative_correct_map.values() if x)
                cumulative_valid_acc = total_valid_correct / max(1, total_valid_items)

                logger.info(
                    f"[Valid] Cumulative Acc after section {sec_idx}: {cumulative_valid_acc * 100:.2f}% "
                    f"({total_valid_correct}/{total_valid_items})"
                )

                try:
                    self.writer.add_scalar("Valid/Cumulative_Acc", cumulative_valid_acc, sec_idx)
                except Exception:
                    pass

            self._write_epoch_metrics(
                sec_idx,
                train_result=res if len(train_df) else None,
                valid_result=valid_res,
                complete=(train_complete if len(train_df) else True),
                pending_task_ids=(set(res.get("pending_task_ids", [])) if len(train_df) and not train_complete else None),
            )

        try:
            self.writer.close()
        except Exception:
            pass
        try:
            with self._image_lock:
                self._persist_image_cache_unlocked()
        except Exception:
            logger.debug("Failed to persist image cache on shutdown", exc_info=True)
