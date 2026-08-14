"""
Region-aware HLE runner that extends HLERunner with region-based memory.

Overrides:
1. _evaluate_row: passes target_subtask to retrieve_query for region gating
2. _train_one_section: passes target_subtasks to update_values, triggers clustering
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from memrl.run.hle_runner import HLERunner
from memrl.service.region_memory_service import RegionMemoryService
from memrl.configs.task_hierarchy import get_primary_subtask

logger = logging.getLogger(__name__)


class HLERegionRunner(HLERunner):
    """HLE runner with region-aware memory retrieval and utility tracking.

    Key differences from base HLERunner:
    - retrieve_query receives target_subtask for region gating
    - update_values receives target_subtasks for per-subtask Q updates
    - Region clustering maintenance after each batch/section
    """

    def __init__(
        self,
        *args,
        region_cluster_init_step: int = 500,
        region_merge_interval: int = 400,
        val_lambda_max: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.memory_service, RegionMemoryService):
            raise TypeError(
                f"HLERegionRunner requires RegionMemoryService, got {type(self.memory_service)}"
            )
        self._region_cluster_init_step = region_cluster_init_step
        self._region_merge_interval = region_merge_interval
        # Restore clustering progress from the snapshot-local cumulative records.
        # An unclustered pre-init checkpoint must NOT be promoted directly to
        # region_cluster_init_step, or resume would cluster too early.
        if getattr(self, "ckpt_resume_enabled", False):
            rm = getattr(self.memory_service, "region_manager", None)
            if rm is not None and not getattr(rm, "_is_clustered", False):
                self._global_step = len(getattr(self, "_resume_batch_all_recs", []) or [])
                logger.info(
                    "[Resume] Restored pre-cluster region global_step=%d from batch records",
                    self._global_step,
                )
            else:
                self._global_step = int(region_cluster_init_step)
        else:
            self._global_step = 0
        self.val_lambda_max = val_lambda_max
        self._subtasks_registered = False
        # Region failure summary (configured externally via configure_failure_summary).
        self._failure_summary_n_slots = 0
        self._failure_summary_replace = True
        self._region_failure_summaries = None
        self._failure_index = None
        self._failure_index_size = 0
        self._failure_inject_log_counter = 0

    def _get_target_subtask(self, row: pd.Series) -> str:
        category = row.get("category", "other") or "other"
        return get_primary_subtask("hle", {"category": category})

    # ---------- Region failure summary ----------
    def configure_failure_summary(self, n_slots: int = 2, summaries_path: Optional[str] = None,
                                   replace_with_summary: bool = True) -> None:
        """Enable region failure summary injection.

        n_slots: number of top-k slots reserved for failure memories.
        summaries_path: optional JSON (build_region_failure_summaries.py output)
            used as a global fallback; if None only live region.failure_summary is used.
        replace_with_summary: replace failure content with the region pitfall summary.
        """
        import json as _json
        self._failure_summary_n_slots = n_slots
        self._failure_summary_replace = replace_with_summary
        if summaries_path and replace_with_summary:
            data = _json.loads(Path(summaries_path).read_text())
            self._region_failure_summaries = data.get("summaries", {})
            logger.info(
                "[Region Failure Summary] loaded %d region summaries from %s, n_slots=%d, replace=True",
                len(self._region_failure_summaries), summaries_path, n_slots,
            )
        else:
            self._region_failure_summaries = None
            logger.info(
                "[Region Failure Summary] n_slots=%d, replace=%s (live region summaries)",
                n_slots, replace_with_summary,
            )

    def _inject_failure_summary(self, selected_mems: list, query: str) -> list:
        """Reserve n_slots for failure memories and replace their content with the
        region's aggregated failure pitfall summary. No-op until region clustering
        has produced regions (cold start)."""
        n_slots = self._failure_summary_n_slots
        if n_slots <= 0 or not selected_mems:
            return selected_mems

        rm = getattr(self.memory_service, 'region_manager', None)
        if not rm or not getattr(rm, 'regions', None):
            return selected_mems

        success_mems, failure_mems = [], []
        for m in selected_mems:
            (failure_mems if not self._mem_success_flag(m) else success_mems).append(m)

        if len(failure_mems) < n_slots:
            exclude_ids = {m.get("memory_id") or m.get("id") for m in selected_mems}
            extra = self._retrieve_failure_only(query, k=n_slots - len(failure_mems), exclude_ids=exclude_ids)
            failure_mems.extend(extra)

        max_success = max(0, self.retrieve_k - n_slots)
        success_mems = success_mems[:max_success]
        failure_mems = failure_mems[:n_slots]

        n_replaced = 0
        if self._failure_summary_replace:
            n_replaced = self._replace_failure_with_region_summary(failure_mems)

        self._failure_inject_log_counter += 1
        if self._failure_inject_log_counter <= 3 or self._failure_inject_log_counter % 50 == 0:
            logger.info(
                "[Failure Summary] task #%d: %d success + %d failure (%d replaced w/ region summary)",
                self._failure_inject_log_counter, len(success_mems), len(failure_mems), n_replaced,
            )
        return success_mems + failure_mems

    def _retrieve_failure_only(self, query: str, k: int = 2,
                               exclude_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k failure memories by similarity only (no Q rerank), using a
        lazily-built failure index over dict_memory."""
        ms = self.memory_service
        if not getattr(ms, 'dict_memory', None) or not hasattr(ms, '_mem_cache'):
            return []
        try:
            from memrl.service.memory_service import get_embedding_with_retry
            import math
            embed = getattr(getattr(ms, 'embedding_provider', None), 'embed', None)
            if not callable(embed):
                return []
            _qe = getattr(ms, 'query_embeddings', {})
            query_vec = _qe.get(query) or get_embedding_with_retry(embed, [query])[0]
            query_norm = math.sqrt(sum(x * x for x in query_vec)) or 1e-8

            mc = ms._mem_cache
            dict_mem = ms.dict_memory
            current_size = len(dict_mem)

            def _is_success(mem_obj) -> Optional[bool]:
                md = getattr(mem_obj, 'metadata', None)
                if hasattr(md, 'model_extra'):
                    return bool(md.model_extra.get('success', True))
                if isinstance(md, dict):
                    return bool(md.get('success', True))
                return None

            if (self._failure_index is None or self._failure_index_size != current_size):
                self._failure_index = {}
                for query_key, mem_ids in dict_mem.items():
                    fail_ids = []
                    for mid in mem_ids:
                        mem_obj = mc.get(mid)
                        if mem_obj is None:
                            continue
                        s = _is_success(mem_obj)
                        if s is False:
                            fail_ids.append(mid)
                    if fail_ids:
                        self._failure_index[query_key] = fail_ids
                self._failure_index_size = current_size

            candidates = []
            for query_key, fail_ids in self._failure_index.items():
                qv = _qe.get(query_key)
                if qv is None:
                    continue
                q_norm = math.sqrt(sum(x * x for x in qv)) or 1e-8
                sim = sum(a * b for a, b in zip(query_vec, qv)) / (query_norm * q_norm)
                for mid in fail_ids:
                    if exclude_ids and mid in exclude_ids:
                        continue
                    mem_obj = mc.get(mid)
                    if mem_obj is None:
                        continue
                    md = getattr(mem_obj, 'metadata', None)
                    content = None
                    try:
                        if hasattr(md, 'model_extra'):
                            content = md.model_extra.get('full_content')
                        elif isinstance(md, dict):
                            content = md.get('full_content')
                    except Exception:
                        pass
                    candidates.append({
                        'memory_id': mid,
                        'content': content,
                        'similarity': float(sim),
                        'metadata': mem_obj,
                    })
            candidates.sort(key=lambda c: c['similarity'], reverse=True)
            return candidates[:k]
        except Exception as e:
            logger.warning("[failure-only retrieve] failed: %s", e)
            return []

    def _replace_failure_with_region_summary(self, failed_mems: List[Dict[str, Any]]) -> int:
        """Replace each failure memory's content with its region's failure summary
        (live region.failure_summary, falling back to external global summary)."""
        rm = getattr(self.memory_service, 'region_manager', None)
        mem_to_region = {}
        if rm and getattr(rm, 'regions', None):
            for r in rm.regions:
                for mid in r.member_ids:
                    mem_to_region[mid] = r

        external = self._region_failure_summaries or {}
        global_summary = external.get('global', '')

        n_replaced = 0
        for fm in failed_mems:
            region = mem_to_region.get(fm.get('memory_id'))
            summary = (region.failure_summary if region and getattr(region, 'failure_summary', '') else '') or global_summary
            if summary:
                fm['content'] = summary
                fm['_region_failure_summary'] = True
                n_replaced += 1
        return n_replaced

    def _ensure_failure_index_built(self) -> None:
        """Pre-build the failure index single-threaded so parallel
        _retrieve_failure_only threads don't race on the lazy rebuild.
        Cheap no-op when dict_memory size is unchanged since last build."""
        ms = self.memory_service
        if not getattr(ms, 'dict_memory', None) or not hasattr(ms, '_mem_cache'):
            return
        mc = ms._mem_cache
        dict_mem = ms.dict_memory
        current_size = len(dict_mem)
        if self._failure_index is not None and self._failure_index_size == current_size:
            return
        index: Dict[str, List[str]] = {}
        for query_key, mem_ids in dict_mem.items():
            fail_ids = []
            for mid in mem_ids:
                mem_obj = mc.get(mid)
                if mem_obj is None:
                    continue
                md = getattr(mem_obj, 'metadata', None)
                if hasattr(md, 'model_extra'):
                    is_success = md.model_extra.get('success', True)
                elif isinstance(md, dict):
                    is_success = md.get('success', True)
                else:
                    continue
                if not is_success:
                    fail_ids.append(mid)
            if fail_ids:
                index[query_key] = fail_ids
        self._failure_index = index
        self._failure_index_size = current_size
        n_fail = sum(len(v) for v in index.values())
        logger.info("[failure_index] pre-built: %d query_keys with failures, %d total failure mems",
                    len(index), n_fail)

    def _register_subtask_embeddings(self) -> None:
        rm = getattr(self.memory_service, 'region_manager', None)
        if rm is None:
            return

        embed_fn = getattr(getattr(self.memory_service, 'embedding_provider', None), 'embed', None)
        if not callable(embed_fn):
            return

        from memrl.configs.task_hierarchy import (
            HLE_SUBTASK_DESCRIPTIONS, hle_category_to_subtask,
        )
        # Derive subtask names from the SAME function retrieval uses, so the
        # registered embeddings always match _get_target_subtask outputs.
        subtask_descriptions = {
            hle_category_to_subtask(cat): desc
            for cat, desc in HLE_SUBTASK_DESCRIPTIONS.items()
        }

        try:
            texts = list(subtask_descriptions.values())
            embeddings = embed_fn(texts)
            for (subtask, _), emb in zip(subtask_descriptions.items(), embeddings):
                rm.set_subtask_embedding(subtask, np.array(emb))
            logger.info("Registered %d HLE subtask embeddings for region transfer", len(subtask_descriptions))
        except Exception as e:
            logger.warning("Failed to register HLE subtask embeddings: %s", e)

    def _evaluate_row(self, row: pd.Series, reflection_note: Optional[str] = None) -> Dict[str, Any]:
        """Override to inject target_subtask into region-aware retrieval."""
        q = str(row['question'])
        gold = str(row['answer'])
        target_subtask = self._get_target_subtask(row)

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
        memory_image_ids = set()
        if self.memory_service and self.retrieve_k > 0:
            try:
                rl_cfg = getattr(self.memory_service, "rl_config", None)
                tau = float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
            except Exception:
                tau = 0.0
            try:
                ret = self.memory_service.retrieve_query(
                    q,
                    k=self.retrieve_k,
                    threshold=tau,
                    target_subtask=target_subtask,
                )
                if isinstance(ret, tuple):
                    ret_result, retrieved_topk_queries = ret
                else:
                    ret_result, retrieved_topk_queries = ret, None
                selected_mems = ret_result.get('selected', []) if ret_result else []
                if self._failure_summary_n_slots > 0:
                    selected_mems = self._inject_failure_summary(selected_mems, q)
                memory_ctx, retrieved_ids, memory_image_ids = self._build_memory_context(selected_mems, self.retrieve_k)
            except Exception as e:
                logger.warning("Memory retrieval failed: %s", e)

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
                    temperature=self.temperature,
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
            "target_subtask": target_subtask,
        }
        if retrieved_topk_queries is not None:
            rec["retrieved_topk_queries"] = retrieved_topk_queries
        return rec

    def _region_clustering_step(self, n_new: int, sec_idx: int) -> None:
        """Run region clustering maintenance after a batch."""
        rm = getattr(self.memory_service, 'region_manager', None)
        if rm is None:
            return

        self._global_step += n_new

        if not rm._is_clustered and self._global_step >= self._region_cluster_init_step:
            try:
                rm.cluster_by_utility()
                logger.info(
                    "[hle] section %d INITIAL cluster at step %d: %d regions",
                    sec_idx, self._global_step, len(rm.regions),
                )
            except Exception as e:
                logger.warning("Initial clustering failed: %s", e)
        elif rm._is_clustered:
            for mem_id in rm.subtask_q:
                if mem_id not in rm.membership_weights:
                    rm.assign_new_memory(mem_id)
            prev_step = self._global_step - n_new
            if (self._global_step >= self._region_cluster_init_step and
                (self._global_step // self._region_merge_interval) >
                (prev_step // self._region_merge_interval)):
                try:
                    changed = rm.maybe_split_merge()
                    if changed:
                        logger.info(
                            "[hle] section %d split/merge at step %d: %d regions",
                            sec_idx, self._global_step, len(rm.regions),
                        )
                except Exception as e:
                    logger.warning("Split/merge failed: %s", e)

    def _region_end_of_section(self, sec_idx: int) -> None:
        """End-of-section full recluster."""
        rm = getattr(self.memory_service, 'region_manager', None)
        if rm is None:
            return
        try:
            if not rm._is_clustered:
                rm.cluster_by_utility()
            else:
                for mem_id in rm.subtask_q:
                    if mem_id not in rm.membership_weights:
                        rm.assign_new_memory(mem_id)
                rm.maybe_split_merge()
            rm.classify_transfer_patterns()
            logger.info(
                "Section %d: region clustering done. %d regions, %d memories.",
                sec_idx, len(rm.regions), len(rm.membership_weights),
            )
        except Exception as e:
            logger.warning("Region clustering failed at section %d: %s", sec_idx, e)

    def _train_one_section(self, df: pd.DataFrame, sec_idx: int, start_batch_idx: int = 0) -> Dict[str, float]:
        """Override to add region-aware Q updates and clustering maintenance."""
        # Set current epoch for exploration schedule
        if hasattr(self.memory_service, 'set_current_epoch'):
            self.memory_service.set_current_epoch(sec_idx, num_epochs=self.num_sections)

        # Register subtask embeddings once
        if not self._subtasks_registered:
            self._register_subtask_embeddings()
            self._subtasks_registered = True

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
            # Pre-build the failure index single-threaded before the parallel
            # eval pool, so concurrent _retrieve_failure_only threads don't race
            # on the lazy rebuild. Memory grows after each batch's add_memories,
            # so this is refreshed per batch (the size guard makes it cheap).
            if b and self._failure_summary_n_slots > 0:
                self._ensure_failure_index_built()
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
                    target_subtasks = [r.get("target_subtask", "hle/other") for r in memory_recs]

                    metadatas = []
                    for rec_entry, s in zip(memory_recs, successes):
                        metadatas.append({
                            "source_benchmark": "HLE",
                            "success": s,
                            "source_subtask": rec_entry.get("target_subtask", "hle/other"),
                            "q_value": 1.0 if s else 0.0,
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
                    # Region-aware Q update with target_subtasks
                    self.memory_service.update_values(
                        [float(s) for s in successes],
                        retrieved_ids_list,
                        target_subtasks=target_subtasks,
                    )
                except Exception as e:
                    logger.warning("[train sec %d] batch memory add/update failed: %s", sec_idx, e)

                # Region clustering maintenance
                self._region_clustering_step(len(memory_recs), sec_idx)

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

        # Retry only infrastructure-deferred rows at the end of this same epoch.
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
            retry_batches = [round_rows[i:i + self.batch_size] for i in range(0, len(round_rows), self.batch_size)]
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
                            target_subtasks = [r.get("target_subtask", "hle/other") for r in memory_recs]
                            metadatas = []
                            for rec_entry, success in zip(memory_recs, successes):
                                metadatas.append({
                                    "source_benchmark": "HLE",
                                    "success": success,
                                    "source_subtask": rec_entry.get("target_subtask", "hle/other"),
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
                                task_descriptions=task_descriptions, trajectories=trajectories,
                                successes=successes, retrieved_memory_queries=retrieved_queries,
                                retrieved_memory_ids_list=retrieved_ids_list, metadatas=metadatas,
                            )
                            self.memory_service.update_values(
                                [float(success) for success in successes],
                                retrieved_ids_list, target_subtasks=target_subtasks,
                            )
                        except Exception as e:
                            logger.warning(
                                "[train sec %d] infrastructure retry memory add failed: %s", sec_idx, e
                            )
                    self._region_clustering_step(len(memory_recs), sec_idx)
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
                sec_idx, next_batch=len(batches), batch_all_recs=self._slim_batch_records(all_recs),
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
                sec_idx, next_batch=len(batches),
                batch_all_recs=self._slim_batch_records(all_recs),
                pending_task_ids=pending_ids,
            )
            return {
                "acc": correct_so_far / max(1, len(all_recs)),
                "per_item": {str(r.get("id") or r["question"]): bool(r["correct"]) for r in all_recs},
                "complete": False,
                "pending_task_ids": pending_ids,
            }

        # End-of-section region clustering.
        # Guard: only recluster when this section collected new trajectories.
        # On eval-only resume (no new data) reclustering would re-run split/merge
        # with floating-point non-determinism, drifting region state and SR.
        # See docs/RESUME_EVAL_DRIFT.md (same fix as alfworld_rl_runner.py).
        if all_recs:
            self._region_end_of_section(sec_idx)

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
        return {"acc": float(acc), "per_item": per_item, "complete": True, "pending_task_ids": []}

    def _eval_split(self, df: pd.DataFrame, tag: str, step: Optional[int] = None) -> Dict[str, float]:
        """Override to optionally switch shrinkage lambda for eval."""
        rm = getattr(self.memory_service, 'region_manager', None)
        saved_lmax = None
        if self.val_lambda_max is not None and rm is not None:
            saved_lmax = getattr(rm, 'shrinkage_lambda_max', None)
            rm.shrinkage_lambda_max = float(self.val_lambda_max)
            logger.info("[%s] eval: switching shrinkage_lambda_max → %.3f", tag, float(self.val_lambda_max))
        try:
            result = super()._eval_split(df, tag, step)
        finally:
            if saved_lmax is not None and rm is not None:
                rm.shrinkage_lambda_max = saved_lmax
            elif self.val_lambda_max is not None and rm is not None:
                if hasattr(rm, 'shrinkage_lambda_max'):
                    try:
                        del rm.shrinkage_lambda_max
                    except AttributeError:
                        pass
        return result
