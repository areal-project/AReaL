# memrl/run/webshop_rl_runner.py
"""WebShop RL Runner for MemRL with region + failure_summary support."""
import logging
import os
import sys
import json
import time
import math
import random
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from memrl.agent.memp_agent import MempAgent
from memrl.agent.webshop_prompts import (
    WEBSHOP_SYSTEM_PROMPT,
    WEBSHOP_USER_TEMPLATE,
    format_admissible_actions,
    format_webshop_history,
    format_webshop_memories,
)
from memrl.envs.webshop_env import WebShopEnv, get_webshop_sessions
from memrl.service.memory_service import MemoryService
from memrl.service.value_driven import RLConfig

logger = logging.getLogger(__name__)

MAX_LLM_CONCURRENCY = 32


class WebShopRunner:
    """
    WebShop RL runner with region/failure_summary support.
    Architecture mirrors ALFWorld runner: section-based training, parallel batches,
    region-aware retrieval, failure summary injection, periodic eval.
    """

    def __init__(
        self,
        agent: MempAgent,
        root: Path,
        memory_service: MemoryService,
        exp_name: str,
        num_sections: int = 5,
        batch_size: int = 16,
        max_steps: int = 30,
        rl_config: Optional[RLConfig] = None,
        ck_dir: str = None,
        retrieve_k: int = 5,
        mode: str = "train",
        valid_interval: int = 1,
        test_interval: int = 1,
        dataset_ratio: float = 1.0,
        random_seed: int = 42,
        file_path: str = None,
        ood_file_path: str = None,
        split_info_path: str = None,
        human_goals: bool = True,
        skip_initial_eval: bool = False,
        val_lambda_max: float = None,
        task_cluster_k: int = 8,
        ckpt_resume_enabled: bool = False,
        ckpt_resume_path: Optional[str] = None,
        ckpt_resume_epoch: Optional[int] = None,
        **kwargs
    ):
        self.agent = agent
        self.root = Path(root)
        self.memory_service = memory_service
        self.exp_name = exp_name
        self.num_sections = num_sections
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.rl_config = rl_config
        self.retrieve_k = retrieve_k
        self.mode = mode
        self.valid_interval = valid_interval
        self.test_interval = test_interval
        self.dataset_ratio = dataset_ratio
        self.random_seed = random_seed
        self.file_path = file_path
        self.ood_file_path = ood_file_path
        self.split_info_path = split_info_path
        self.human_goals = human_goals
        self.skip_initial_eval = skip_initial_eval
        self.val_lambda_max = val_lambda_max
        self.task_cluster_k = task_cluster_k
        self.ckpt_resume_enabled = ckpt_resume_enabled
        self.ckpt_resume_path = ckpt_resume_path
        self.ckpt_resume_epoch = ckpt_resume_epoch

        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load session splits from split_info.json
        if split_info_path:
            self.train_sessions, self.train_goals = get_webshop_sessions(split_info_path, "train")
            self.valid_sessions, self.valid_goals = get_webshop_sessions(split_info_path, "val")
            self.test_sessions, self.test_goals = get_webshop_sessions(split_info_path, "ood")
        else:
            raise ValueError("split_info_path is required for WebShopRunner")

        if 0 < dataset_ratio < 1.0:
            n = int(len(self.train_sessions) * dataset_ratio)
            self.train_sessions = self.train_sessions[:n]
            self.train_goals = self.train_goals[:n]

        logger.info(
            "WebShop sessions: train=%d, eval_in=%d, eval_ood=%d",
            len(self.train_sessions), len(self.valid_sessions), len(self.test_sessions),
        )

        # Checkpoint dir
        self.ck_dir = Path(ck_dir) if ck_dir else (self.root / "results" / "webshop" / f"exp_{exp_name}")
        # If resuming, write snapshots back into the original experiment dir
        if self.ckpt_resume_enabled and self.ckpt_resume_path:
            resume_root = Path(self.ckpt_resume_path)
            if resume_root.name == "snapshot":
                self.ck_dir = resume_root.parent
            elif resume_root.parent.name == "snapshot":
                self.ck_dir = resume_root.parent.parent
            elif (resume_root / "snapshot").exists():
                self.ck_dir = resume_root
        self.ck_dir.mkdir(parents=True, exist_ok=True)

        # Task clustering (BCB-style)
        self._task_cluster_fitted = False
        self._task_embeddings_buffer = []

        # Failure summary (configured externally via configure_failure_summary)
        self._failure_summary_n_slots = 0
        self._failure_summary_replace = True
        self._region_failure_summaries = None

        # Tracking
        self._global_step = 0
        self._cum_success_ids: Set[int] = set()
        self._cum_total = len(self.train_sessions)
        self._cum_state_path = self.ck_dir / "local_cache" / "cum_state.json"
        self._cum_state_path.parent.mkdir(parents=True, exist_ok=True)

        # Parallel episode execution: thread-local env pool (one WebShopEnv per
        # worker thread, reused across batches). Episode parallelism hides per-step
        # LLM latency. See codex analysis: per-thread independent LuceneSearcher is
        # JNI-safe; env instances share no mutable global state.
        self.episode_workers = int(kwargs.get('episode_workers', 8))
        self._thread_local = threading.local()
        self._env_pool_lock = threading.Lock()
        self._all_pooled_envs: List[WebShopEnv] = []

        # Resume from checkpoint
        self._resume_section_start, self._resume_batch_start = self._resume_from_ckpt()

    # ------------------------------------------------------------------
    # Failure summary (copied from ALFWorld runner)
    # ------------------------------------------------------------------

    def configure_failure_summary(self, n_slots: int = 2, summaries_path: Optional[str] = None,
                                   replace_with_summary: bool = True):
        import json as _json
        self._failure_summary_n_slots = n_slots
        self._failure_summary_replace = replace_with_summary
        if summaries_path and replace_with_summary:
            data = _json.loads(Path(summaries_path).read_text())
            self._region_failure_summaries = data.get("summaries", {})
            logger.info(
                "[Region Failure Summary] loaded %d region summaries from %s, n_slots=%d",
                len(self._region_failure_summaries), summaries_path, n_slots,
            )
        else:
            self._region_failure_summaries = None
            logger.info("[Region Failure Summary] n_slots=%d, replace=%s (raw failure mode)", n_slots, replace_with_summary)

    def process_retrieve_mems(self, retrieved_mems_per_slot, task_descs_per_slot=None):
        processed_mems_per_slot = []
        for i, mems_for_one_slot in enumerate(retrieved_mems_per_slot):
            success_mems = []
            failed_mems = []

            for mem in mems_for_one_slot:
                md = mem.get('metadata')
                is_success = False
                if hasattr(md, 'model_extra'):
                    is_success = md.model_extra.get('success', False)
                elif isinstance(md, dict):
                    is_success = md.get('success', False)
                if is_success:
                    success_mems.append(mem)
                else:
                    failed_mems.append(mem)

            n_failure_slots = self._failure_summary_n_slots
            if n_failure_slots > 0:
                if len(failed_mems) < n_failure_slots and task_descs_per_slot:
                    extra_needed = n_failure_slots - len(failed_mems)
                    extra_failure = self._retrieve_failure_only(
                        task_descs_per_slot[i], k=extra_needed,
                        exclude_ids={m.get('memory_id') for m in mems_for_one_slot},
                    )
                    failed_mems.extend(extra_failure)

                max_success = max(0, self.retrieve_k - n_failure_slots)
                success_mems = success_mems[:max_success]
                failed_mems = failed_mems[:n_failure_slots]

                total = len(success_mems) + len(failed_mems)
                if total < self.retrieve_k:
                    all_remaining = [m for m in mems_for_one_slot
                                     if m.get('memory_id') not in
                                     {x.get('memory_id') for x in success_mems + failed_mems}]
                    for m in all_remaining[:self.retrieve_k - total]:
                        success_mems.append(m)

                if self._failure_summary_replace:
                    self._replace_failure_with_region_summary(failed_mems)

            final_mems = {}
            if success_mems:
                final_mems['successed'] = success_mems
            if failed_mems:
                final_mems['failed'] = failed_mems
            processed_mems_per_slot.append(final_mems)

        return processed_mems_per_slot

    def _retrieve_failure_only(self, task_description: str, k: int = 2,
                               exclude_ids: Optional[set] = None) -> List[Dict]:
        if not hasattr(self.memory_service, 'dict_memory') or not self.memory_service.dict_memory:
            return []
        if not hasattr(self.memory_service, '_mem_cache'):
            return []
        try:
            from memrl.service.memory_service import get_embedding_with_retry
            embed = getattr(self.memory_service.embedding_provider, 'embed', None)
            if not callable(embed):
                return []

            _qe = getattr(self.memory_service, 'query_embeddings', {})
            query_vec = _qe.get(task_description)
            if query_vec is None:
                query_vec = get_embedding_with_retry(embed, [task_description])[0]

            query_norm = math.sqrt(sum(x * x for x in query_vec)) or 1e-8

            candidates = []
            mc = self.memory_service._mem_cache
            for query_key, mem_ids in self.memory_service.dict_memory.items():
                qv = _qe.get(query_key)
                if qv is None:
                    continue
                q_norm = math.sqrt(sum(x * x for x in qv)) or 1e-8
                sim = sum(a * b for a, b in zip(query_vec, qv)) / (query_norm * q_norm)
                for mid in mem_ids:
                    if exclude_ids and mid in exclude_ids:
                        continue
                    mem_obj = mc.get(mid)
                    if mem_obj is None:
                        continue
                    md = getattr(mem_obj, 'metadata', {})
                    if hasattr(md, 'model_extra'):
                        is_success = md.model_extra.get('success', True)
                    elif isinstance(md, dict):
                        is_success = md.get('success', True)
                    else:
                        continue
                    if is_success:
                        continue
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

    def _replace_failure_with_region_summary(self, failed_mems: List[Dict]) -> None:
        rm = getattr(self.memory_service, 'region_manager', None)

        mem_to_region_obj = {}
        if rm and rm.regions:
            for r in rm.regions:
                for mid in r.member_ids:
                    mem_to_region_obj[mid] = r

        external_summaries = getattr(self, '_region_failure_summaries', None)
        global_summary = external_summaries.get('global', '') if external_summaries else ''

        for fm in failed_mems:
            mem_id = fm.get('memory_id')
            region = mem_to_region_obj.get(mem_id)
            summary = region.failure_summary if region and region.failure_summary else ''
            if not summary:
                summary = global_summary
            if summary:
                fm['content'] = summary
                fm['_region_failure_summary'] = True

    # ------------------------------------------------------------------
    # Task clustering (BCB-style: embedding → KMeans)
    # ------------------------------------------------------------------

    def _init_task_clusters(self):
        if self._task_cluster_fitted or self.task_cluster_k <= 0:
            return
        if len(self._task_embeddings_buffer) < 50:
            return
        try:
            from memrl.configs.task_hierarchy import get_task_cluster_manager
            tcm = get_task_cluster_manager(K=self.task_cluster_k)
            embeddings = np.array(self._task_embeddings_buffer)
            tcm.fit(embeddings)
            self._task_cluster_fitted = True
            logger.info("[TaskCluster] Fitted %d clusters from %d task embeddings",
                       self.task_cluster_k, len(self._task_embeddings_buffer))
        except Exception as e:
            logger.warning("[TaskCluster] Failed to fit: %s", e)

    def _get_target_subtask(self, goal: str, goal_embedding=None) -> Optional[str]:
        if not self._task_cluster_fitted:
            return None
        try:
            from memrl.configs.task_hierarchy import get_primary_subtask
            metadata = {"embedding": goal_embedding} if goal_embedding is not None else {}
            return get_primary_subtask("webshop", metadata)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Prompt construction (LaMer-style admissible-actions prompting)
    # ------------------------------------------------------------------

    def _construct_messages(self, goal: str, observation: str, retrieved_memories: dict,
                            history: List[Tuple[str, str]],
                            admissible: dict) -> List[Dict[str, str]]:
        """Render a single user-turn prompt with the legal action list inline.

        Mirrors LaMer's ``WEBSHOP_PLAY_PROMPT`` flow at
        ``LaMer/agent_system/environments/webshop/prompt.py:1-13``: the
        observation history feeds the trajectory block, and the admissible
        actions list is what the LLM must choose from.

        If ``self.agent.few_shot_examples`` is populated (list of
        ``[{role, content}, …]`` dialogues), the FIRST demo is inserted
        right after the system prompt — the ALFWorld idiom — so the LLM
        sees a complete, successful WebShop trajectory before its own task.
        """
        messages = [{"role": "system", "content": WEBSHOP_SYSTEM_PROMPT}]

        # Inject one few-shot demo if available. Cap to 1 demo to keep prompt
        # length bounded (each demo is ~5-10 turns); add a leading user line so
        # the LLM treats it as exemplar, not as the live task.
        few_shot = getattr(self.agent, "few_shot_examples", None)
        if few_shot and isinstance(few_shot, list) and len(few_shot) > 0:
            demo = few_shot[0]
            demo_dialogue = demo.get("example") if isinstance(demo, dict) else None
            if demo_dialogue and isinstance(demo_dialogue, list) and len(demo_dialogue) > 0:
                first = dict(demo_dialogue[0])
                first["content"] = (
                    "Here is one example of how to solve a WebShop task. "
                    "Follow the same reasoning style and the <action>...</action> "
                    "output format.\n\n" + first.get("content", "")
                )
                messages.append(first)
                for turn in demo_dialogue[1:]:
                    messages.append(dict(turn))

        mem_text = format_webshop_memories(retrieved_memories)
        memory_block = f"\n\nRelevant past experiences:\n{mem_text}" if mem_text else ""

        # Always show page context. When there's no history yet, render the
        # initial page as 'Observation 0' so the LLM has something to ground on.
        if history:
            traj_body = format_webshop_history(history)
        else:
            traj_body = f"Observation 0: {(observation or '')[:1500]}"
        trajectory_block = (
            "\n\nBelow are the last few actions and corresponding observations you have:\n"
            f"{traj_body}"
        )

        actions_block = format_admissible_actions(admissible)

        user_content = WEBSHOP_USER_TEMPLATE.format(
            goal=goal,
            memory_block=memory_block,
            trajectory_block=trajectory_block,
            admissible_actions=actions_block,
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------------
    # Core episode execution
    # ------------------------------------------------------------------

    def _run_episode(self, env: WebShopEnv, session_idx: int,
                     retrieved_memories: dict) -> Dict[str, Any]:
        obs, info = env.reset(session_idx=session_idx)
        goal = info.get('goal', '')
        history: List[Tuple[str, str]] = []
        admissible = env.get_available_actions()
        total_reward = 0.0
        done = False
        steps = 0
        actions_taken = []

        # Trace mode: dump full prompt+response for the first N episodes (and a
        # few steps each) so we can see what the LLM is actually saying.
        # Controlled by env var WEBSHOP_TRACE_EPISODES (default 0 = off).
        import os as _os
        trace_n = int(_os.environ.get("WEBSHOP_TRACE_EPISODES", "0") or 0)
        if not hasattr(self.__class__, "_trace_episode_counter"):
            self.__class__._trace_episode_counter = 0
            self.__class__._trace_lock = threading.Lock()
        with self.__class__._trace_lock:
            should_trace = self.__class__._trace_episode_counter < trace_n
            if should_trace:
                self.__class__._trace_episode_counter += 1
                trace_id = self.__class__._trace_episode_counter

        while not done and steps < self.max_steps:
            messages = self._construct_messages(goal, obs, retrieved_memories, history, admissible)

            try:
                # Read temperature from config so reasoning models (V3.2, R1) can
                # use their recommended t=1.0 while non-reasoning Qwen-Instruct
                # stays at t=0.0 for deterministic baseline runs.
                _temp = getattr(self.agent.llm, "default_temperature", 0.0)
                response = self.agent.llm.generate(messages, temperature=_temp, max_tokens=16384)
                action = self._parse_action(response, admissible)
            except Exception as e:
                logger.warning("LLM generation failed: %s", e)
                response = f"<llm-exception: {e}>"
                action = self._fallback_action(admissible)

            if should_trace:
                # Print every step for the first 2 traced episodes, then only
                # steps 0/5/10/15/20/25/29 for the rest, to keep log size sane.
                _verbose = trace_id <= 2 or steps in (0, 5, 10, 15, 20, 25, self.max_steps - 1)
                if _verbose:
                    user_msg = messages[-1]["content"] if messages else ""
                    print(
                        f"\n===== TRACE ep={trace_id} sess={session_idx} step={steps} =====\n"
                        f"[goal] {goal}\n"
                        f"[admissible] has_search_bar={admissible.get('has_search_bar')} "
                        f"clickables={admissible.get('clickables')[:8]}{'...' if len(admissible.get('clickables', [])) > 8 else ''}\n"
                        f"[user-prompt-tail (last 800 chars)]\n{user_msg[-800:]}\n"
                        f"[llm-response (first 1200 chars)]\n{(response or '')[:1200]}\n"
                        f"[parsed-action] {action}\n",
                        flush=True,
                    )

            new_obs, reward, done, step_info = env.step(action)
            history.append((action, (new_obs or "")[:400]))
            actions_taken.append(action)
            total_reward = reward  # WebShop reward is cumulative final
            obs = new_obs
            if not done:
                admissible = env.get_available_actions()
            steps += 1

        if should_trace:
            print(
                f"===== TRACE ep={trace_id} sess={session_idx} END "
                f"steps={steps} reward={total_reward:.3f} success={total_reward >= 0.5} =====\n"
                f"[actions] {actions_taken}\n",
                flush=True,
            )

        success = total_reward >= 0.5

        trajectory = f"Goal: {goal}\nOutcome: {'Success' if success else 'Failure'} (reward={total_reward:.2f})\n"
        trajectory += "Actions:\n" + "\n".join(f"  {a}" for a in actions_taken[-10:])

        return {
            'session_idx': session_idx,
            'goal': goal,
            'success': success,
            'reward': total_reward,
            'steps': steps,
            'trajectory': trajectory,
            'actions': actions_taken,
        }

    def _parse_action(self, response: str, admissible: dict) -> str:
        """Extract `<action>...</action>` and validate against the legal set."""
        import re
        response = response or ""

        m = re.search(r"<action>\s*(.*?)\s*</action>", response, re.IGNORECASE | re.DOTALL)
        candidate = m.group(1).strip() if m else None
        if candidate and self._is_admissible(candidate, admissible):
            return candidate

        # Backwards-compatible fallback: bare search[...] / click[...] anywhere
        # in the response — accept only if the env will accept it.
        if not candidate:
            for pat in (r'(search\[[^\]]*\])', r'(click\[[^\]]+\])'):
                bm = re.search(pat, response, re.IGNORECASE)
                if bm and self._is_admissible(bm.group(1), admissible):
                    return bm.group(1)
                if bm and not candidate:
                    candidate = bm.group(1)

        # If the LLM clearly *wanted* to issue a fresh search but the current
        # page disallows it (search bar only exists on the home page in
        # WebShop), redirect to ``click[back to search]`` so the next step can
        # honour the search intent. This is much better than falling back to
        # an arbitrary content clickable, which discards the LLM's plan.
        if candidate and isinstance(admissible, dict):
            cand_l = candidate.strip().lower()
            clickables = admissible.get("clickables") or []
            if cand_l.startswith("search[") and not admissible.get("has_search_bar") \
                    and "back to search" in clickables:
                self._note_search_redirect()
                return "click[back to search]"

        # Surface the silent-failure mode: response had no parseable admissible
        # action. Print at most once per N to avoid log spam, but log enough to
        # diagnose whether the LLM is being truncated or just hallucinating.
        if not getattr(self, "_parse_fallback_count", None):
            self._parse_fallback_count = 0
        self._parse_fallback_count += 1
        if self._parse_fallback_count <= 5 or self._parse_fallback_count % 50 == 0:
            preview = (response or "")[-200:].replace("\n", " ")
            print(
                f"[parse] WARNING fallback #{self._parse_fallback_count} (no admissible action in response). "
                f"tail=...{preview!r}",
                flush=True,
            )

        return self._fallback_action(admissible)

    def _note_search_redirect(self) -> None:
        """Track how often we auto-rewrite an off-page search into a click[back to search]."""
        if not getattr(self, "_search_redirect_count", None):
            self._search_redirect_count = 0
        self._search_redirect_count += 1
        if self._search_redirect_count <= 5 or self._search_redirect_count % 50 == 0:
            print(
                f"[parse] INFO search-redirect #{self._search_redirect_count}: "
                f"LLM emitted search[...] on a page without a search bar; "
                f"rewriting to click[back to search].",
                flush=True,
            )

    @staticmethod
    def _is_admissible(action: str, admissible: dict) -> bool:
        if not action or not isinstance(admissible, dict):
            return False
        a = action.strip().lower()
        if a.startswith("search["):
            return bool(admissible.get("has_search_bar"))
        if a.startswith("click[") and a.endswith("]"):
            inner = a[len("click["):-1].strip()
            clickables = admissible.get("clickables") or []
            return inner in clickables
        return False

    @staticmethod
    def _fallback_action(admissible: dict) -> str:
        """Safe legal action when the LLM produced nothing usable.

        Skip pure navigation buttons (``back to search``, ``next >``,
        ``< prev``) — choosing them reflexively keeps the episode looping
        between the home page and the same results page without ever entering
        a product page. Prefer a content clickable (product ASIN, option,
        ``buy now``) so the trajectory at least makes forward progress.
        Fall back to an empty search if nothing else is available; the env
        treats an empty arg as a no-op via the ``action_arg != ''`` guard at
        ``web_agent_text_env.py:116``.
        """
        if not isinstance(admissible, dict):
            return ""
        clickables = admissible.get("clickables") or []
        nav = {"back to search", "next >", "< prev"}
        content_clickables = [c for c in clickables if c not in nav]
        if content_clickables:
            return f"click[{content_clickables[0]}]"
        if admissible.get("has_search_bar"):
            return "search[]"
        if clickables:
            return f"click[{clickables[0]}]"
        return ""

    # ------------------------------------------------------------------
    # Batch execution with parallel retrieval
    # ------------------------------------------------------------------

    def _get_thread_env(self, file_path: str, goal_list: List[dict], split_key: str = "train") -> WebShopEnv:
        """Return a thread-local WebShopEnv for the given (file_path, split).

        Each worker thread keeps its own env instance (independent SimServer +
        LuceneSearcher) and reuses it across batches that share the same product
        file + goal split. Rebuilds only when the split changes (e.g. train→ood).

        split_key is a stable semantic identifier ("train"/"eval"/...) used together
        with file_path as the cache key — more robust than id(goal_list).
        """
        cache = getattr(self._thread_local, 'env_cache', None)
        if cache is None:
            cache = {}
            self._thread_local.env_cache = cache

        key = (file_path, split_key)
        env = cache.get(key)
        if env is None:
            env = WebShopEnv(
                file_path=file_path or self.file_path,
                human_goals=self.human_goals,
                seed=self.random_seed,
                max_steps=self.max_steps,
                goal_list=goal_list,
            )
            cache[key] = env
            with self._env_pool_lock:
                self._all_pooled_envs.append(env)
        return env

    def _run_batch(self, sessions: List[int], mode: str = "train",
                   file_path: str = None, goal_list: List[dict] = None) -> List[Dict[str, Any]]:
        results = []
        resolved_fp = file_path or self.file_path

        # Collect goal text for retrieval. Prefer goal_list (avoids a wasted reset pass).
        if goal_list is not None:
            goals = [goal_list[sid]['instruction'] for sid in sessions]
        else:
            # Fallback: use a single env to read goals
            env = self._get_thread_env(resolved_fp, goal_list, split_key=mode)
            goals = []
            for sid in sessions:
                _, info = env.reset(session_idx=sid)
                goals.append(info.get('goal', ''))

        # Parallel retrieval (unchanged)
        retrieved_mems_per_slot = []
        _supports_target_subtask = getattr(self.memory_service, 'region_manager', None) is not None

        if self.memory_service and hasattr(self.memory_service, 'dict_memory') and self.memory_service.dict_memory:
            with ThreadPoolExecutor(max_workers=min(len(sessions), MAX_LLM_CONCURRENCY)) as executor:
                futures = []
                for goal in goals:
                    kw = {
                        'k': self.retrieve_k,
                        'threshold': getattr(self.rl_config, 'sim_threshold', getattr(self.rl_config, 'tau', 0.0)) if self.rl_config else 0.0,
                    }
                    if _supports_target_subtask:
                        kw['target_subtask'] = self._get_target_subtask(goal)
                    futures.append(executor.submit(self.memory_service.retrieve_query, goal, **kw))

                for future in futures:
                    try:
                        result = future.result()
                        if isinstance(result, tuple):
                            ret_dict, _ = result
                            mem = ret_dict.get('selected', []) if isinstance(ret_dict, dict) else []
                        else:
                            mem = []
                    except Exception as e:
                        logger.warning("Retrieval failed: %s", e)
                        mem = []
                    retrieved_mems_per_slot.append(mem)
        else:
            retrieved_mems_per_slot = [[] for _ in sessions]

        # Process retrieved mems (failure summary injection)
        retrieved_mems_per_slot = self.process_retrieve_mems(
            retrieved_mems_per_slot, task_descs_per_slot=goals,
        )

        # Run episodes in PARALLEL across worker threads. Each worker uses its own
        # thread-local env; steps within an episode stay sequential (action depends
        # on prior observation), but different episodes overlap their LLM-call waits.
        results = [None] * len(sessions)

        def _run_one(idx: int):
            sid = sessions[idx]
            try:
                env = self._get_thread_env(resolved_fp, goal_list, split_key=mode)
                res = self._run_episode(env, sid, retrieved_mems_per_slot[idx])
                res['retrieved_mems'] = retrieved_mems_per_slot[idx]
                return idx, res
            except Exception as e:
                logger.warning("Episode %d (session %s) failed: %s", idx, sid, e)
                # Return a failed-episode placeholder so the batch survives
                goal = goal_list[sid]['instruction'] if goal_list else ''
                return idx, {
                    'session_idx': sid, 'goal': goal, 'success': False,
                    'reward': 0.0, 'steps': 0, 'trajectory': '', 'actions': [],
                    'retrieved_mems': retrieved_mems_per_slot[idx],
                }

        n_workers = min(self.episode_workers, len(sessions))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for idx, res in tqdm(executor.map(_run_one, range(len(sessions))),
                                 total=len(sessions), desc=f"[{mode}] batch"):
                results[idx] = res

        return results

    # ------------------------------------------------------------------
    # Memory update (Q-value + store)
    # ------------------------------------------------------------------

    def _update_memory(self, results: List[Dict[str, Any]]):
        if not self.memory_service:
            return

        for result in results:
            if not result.get('trajectory'):
                continue

            goal = result['goal']
            success = result['success']
            reward = result['reward']
            trajectory = result['trajectory']

            metadata = {
                'type': 'webshop',
                'success': success,
                'reward': reward,
                'full_content': trajectory,
            }

            try:
                q_init = self.rl_config.q_init_pos if success else self.rl_config.q_init_neg
                metadata['q_value'] = q_init

                self.memory_service.add_memory(
                    task_description=goal,
                    trajectory=trajectory,
                    success=success,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning("Memory add failed: %s", e)

        # Update Q-values for retrieved memories
        try:
            for result in results:
                retrieved = result.get('retrieved_mems', {})
                success = result['success']
                reward_signal = 1.0 if success else 0.0
                for bucket in ('successed', 'failed'):
                    for mem in retrieved.get(bucket, []):
                        mid = mem.get('memory_id')
                        if mid and hasattr(self.memory_service, '_q_cache'):
                            old_q = self.memory_service._q_cache.get(mid, 0.5)
                            alpha = getattr(self.rl_config, 'alpha', 0.3) if self.rl_config else 0.3
                            new_q = old_q + alpha * (reward_signal - old_q)
                            self.memory_service._q_cache[mid] = new_q
        except Exception as e:
            logger.warning("Q update failed: %s", e)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, sessions: List[int], eval_type: str, after_section: int,
                  file_path: str = None, goal_list: List[dict] = None) -> float:
        msg = f"--- Evaluating on {eval_type} ({len(sessions)} sessions, after section {after_section}) ---"
        logger.info(msg)
        print(msg, flush=True)

        # Override val_lambda_max during eval if configured
        if self.val_lambda_max is not None and hasattr(self.memory_service, '_val_lambda_max_override'):
            self.memory_service._val_lambda_max_override = self.val_lambda_max

        results = self._run_batch(sessions, mode="eval", file_path=file_path, goal_list=goal_list)
        success_rate = sum(r['success'] for r in results) / len(results) if results else 0
        avg_reward = sum(r['reward'] for r in results) / len(results) if results else 0
        avg_steps = sum(r['steps'] for r in results) / len(results) if results else 0

        # Save per-game results for CSR computation (aligned by session order)
        per_game_dir = self.ck_dir / "per_game_eval"
        per_game_dir.mkdir(parents=True, exist_ok=True)
        per_game_file = per_game_dir / f"{eval_type}_s{after_section}.json"
        try:
            per_game_data = []
            for r in results:
                per_game_data.append({
                    "gamefile": self._task_key(r),  # stable task id
                    "success": r.get('success', False),
                    "steps": r.get('steps', 0),
                    "reward": r.get('reward', 0.0),
                })
            with open(per_game_file, 'w', encoding='utf-8') as f:
                json.dump(per_game_data, f, ensure_ascii=False, indent=2)
            logger.info("Saved per-game eval results to: %s", per_game_file)
        except Exception as e:
            logger.warning("Failed to save per-game eval results: %s", e)

        msg1 = f"--- Evaluation Complete on {eval_type} (after training Section {after_section}) ---"
        msg2 = f"Success Rate: {success_rate*100:.2f}% ({sum(r['success'] for r in results)}/{len(results)})"
        msg3 = f"Avg Reward: {avg_reward:.3f}, Avg Steps: {avg_steps:.1f}"
        logger.info(msg1)
        logger.info(msg2)
        logger.info(msg3)
        print(msg1, flush=True)
        print(msg2, flush=True)
        print(msg3, flush=True)

        return success_rate

    def _evaluate_val(self, after_section: int) -> float:
        """Evaluate on in-distribution val split (same product set as train)."""
        return self._evaluate(
            self.valid_sessions, "eval_in_distribution", after_section,
            file_path=self.file_path, goal_list=self.valid_goals,
        )

    def _evaluate_ood(self, after_section: int) -> float:
        """Evaluate on OOD split (different product set)."""
        return self._evaluate(
            self.test_sessions, "eval_out_of_distribution", after_section,
            file_path=self.ood_file_path, goal_list=self.test_goals,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, section: int, batch: int = None):
        tag = f"s{section}_b{batch}" if batch is not None else f"s{section}_end"

        if self.memory_service and hasattr(self.memory_service, 'save_checkpoint_snapshot'):
            try:
                self.memory_service.save_checkpoint_snapshot(str(self.ck_dir), tag)
                logger.info("Checkpoint saved: %s", tag)
            except Exception as e:
                logger.warning("Checkpoint save failed: %s", e)

        # Persist cumulative success state alongside every checkpoint (resume-safe)
        self._persist_cum_state()

    # ------------------------------------------------------------------
    # Cumulative success tracking (per-task success ids → cumulative SR)
    # ------------------------------------------------------------------

    def _task_key(self, result: Dict[str, Any]) -> Optional[str]:
        """Stable per-task identity: prefer asin+instruction, fall back to session_idx."""
        goal = result.get('goal')
        sid = result.get('session_idx')
        if goal:
            return f"goal::{goal}"
        if sid is not None:
            return f"sid::{sid}"
        return None

    def _update_cum_success(self, results: List[Dict[str, Any]]) -> None:
        for r in results:
            if not r.get('success'):
                continue
            key = self._task_key(r)
            if key:
                self._cum_success_ids.add(key)

    def _current_cum_acc(self) -> float:
        if self._cum_total <= 0:
            return 0.0
        return len(self._cum_success_ids) / self._cum_total

    def _persist_cum_state(self, path: Optional[Path] = None) -> None:
        import os as _os
        from datetime import datetime
        path = path or self._cum_state_path
        payload = {
            "success_ids": sorted(self._cum_success_ids),
            "total": self._cum_total,
            "global_step": getattr(self, "_global_step", 0),
            "updated_at": datetime.now().isoformat(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                _os.fsync(f.fileno())
            _os.replace(tmp_path, path)
        except Exception:
            logger.warning("Failed to persist cumulative acc state to %s", path, exc_info=True)

    def _load_cum_state(self, path: Optional[Path] = None) -> None:
        path = path or self._cum_state_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ids = payload.get("success_ids", [])
            self._cum_success_ids = {str(x) for x in ids if x}
            total = payload.get("total")
            if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                self._cum_total = total
            global_step = payload.get("global_step")
            if isinstance(global_step, int) and not isinstance(global_step, bool) and global_step >= 0:
                self._global_step = global_step
            logger.info("Restored cumulative state: %d success ids, total=%d, global_step=%d",
                        len(self._cum_success_ids), self._cum_total, self._global_step)
        except Exception:
            logger.warning("Failed to load cumulative acc state from %s", path, exc_info=True)

    def _resume_from_ckpt(self) -> tuple:
        """Returns (start_section, start_batch) for resuming training."""
        import re
        if not self.ckpt_resume_enabled:
            return (1, 0)

        snapshot_dir = self._resolve_resume_dir()
        if not snapshot_dir or not snapshot_dir.exists():
            logger.warning("Resume enabled but snapshot not found: %s", snapshot_dir)
            return (1, 0)

        # Load memory snapshot (best-effort). Progress resume does NOT depend on this:
        # no_memory runs have no memory state but should still skip completed sections.
        if self.memory_service and hasattr(self.memory_service, 'load_checkpoint_snapshot'):
            try:
                self.memory_service.load_checkpoint_snapshot(str(snapshot_dir))
                logger.info("Loaded memory snapshot from: %s", snapshot_dir)
            except Exception as e:
                logger.warning("Failed to load memory snapshot from %s: %s (continuing with progress resume)",
                               snapshot_dir, e)

        # Restore cumulative success state (per-task ids → cumulative SR)
        self._load_cum_state()

        # Parse section/batch from directory name (independent of memory load)
        batch_match = re.match(r"^s(\d+)_b(\d+)$", snapshot_dir.name)
        if batch_match:
            section = int(batch_match.group(1))
            batch_num = int(batch_match.group(2))
            logger.info("Resuming from batch checkpoint %s → continuing section %d from batch %d",
                       snapshot_dir.name, section, batch_num)
            return (section, batch_num)  # Resume will skip batches <= batch_num

        # Section-end checkpoint: s<N>_end
        section_match = re.match(r"^s(\d+)_end$", snapshot_dir.name)
        if section_match:
            section = int(section_match.group(1))
            logger.info("Resuming from section-end checkpoint %s → section %d complete, start section %d",
                       snapshot_dir.name, section, section + 1)
            return (section + 1, 0)

        # Numeric epoch directory (legacy)
        if snapshot_dir.name.isdigit():
            epoch = int(snapshot_dir.name)
            logger.info("Resuming from epoch checkpoint %d → section %d", epoch, epoch + 1)
            return (epoch + 1, 0)

        logger.warning("Could not parse checkpoint directory name: %s, starting fresh", snapshot_dir.name)
        return (1, 0)

    def _resolve_resume_dir(self) -> Optional[Path]:
        """Find the checkpoint directory to resume from."""
        if not self.ckpt_resume_path:
            return None
        base = Path(self.ckpt_resume_path)

        # Explicit epoch provided
        if self.ckpt_resume_epoch is not None:
            if base.name == "snapshot":
                return base / f"s{self.ckpt_resume_epoch}_end"
            if (base / "snapshot").exists():
                return base / "snapshot" / f"s{self.ckpt_resume_epoch}_end"
            return base / f"s{self.ckpt_resume_epoch}_end"

        # Auto-detect latest
        snapshot_dir = base if base.name == "snapshot" else base / "snapshot"
        if not snapshot_dir.exists():
            return base

        latest = self._find_latest_checkpoint(snapshot_dir)
        return latest if latest else snapshot_dir

    @staticmethod
    def _find_latest_checkpoint(snapshot_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint (section or batch) in snapshot directory."""
        import re
        if not snapshot_dir.exists():
            return None

        batch_re = re.compile(r"^s(\d+)_b(\d+)$")
        section_re = re.compile(r"^s(\d+)_end$")
        best_section = -1
        best_batch = (-1, -1)
        best_batch_path = None

        for d in snapshot_dir.iterdir():
            if not d.is_dir():
                continue

            # Batch checkpoint
            m = batch_re.match(d.name)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key > best_batch:
                    best_batch = key
                    best_batch_path = d
                continue

            # Section-end checkpoint
            m = section_re.match(d.name)
            if m:
                best_section = max(best_section, int(m.group(1)))
                continue

        # Prefer section-end checkpoint when it's at least as advanced as the latest
        # batch checkpoint (a completed section supersedes a mid-section batch of the
        # same section, avoiding re-entry into an already-finished section).
        if best_section >= 0 and best_section >= best_batch[0]:
            return snapshot_dir / f"s{best_section}_end"

        if best_batch_path:
            return best_batch_path

        return None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self):
        msg = f"{'=' * 60}\nWebShop Experiment: {self.exp_name}\n" \
              f"Mode: {self.mode}, Sections: {self.num_sections}, Batch: {self.batch_size}, MaxSteps: {self.max_steps}\n" \
              f"Train: {len(self.train_sessions)}, Eval-In: {len(self.valid_sessions)}, Eval-OOD: {len(self.test_sessions)}\n" \
              f"Retrieve k={self.retrieve_k}, failure_summary_n_slots={self._failure_summary_n_slots}\n{'=' * 60}"
        logger.info(msg)
        print(msg, flush=True)

        if self.mode == "test":
            self._evaluate_val(0)
            self._evaluate_ood(0)
            return

        # Optional initial eval (skip if resuming mid-training)
        resuming = self._resume_section_start > 1 or self._resume_batch_start > 0
        if not self.skip_initial_eval and not resuming:
            self._evaluate_val(0)
            self._evaluate_ood(0)

        if resuming:
            logger.info("Resuming training from section %d, batch %d",
                       self._resume_section_start, self._resume_batch_start)

        # Training loop
        sessions_per_section = len(self.train_sessions) // self.num_sections

        for section in range(1, self.num_sections + 1):
            # Skip already-completed sections when resuming
            if section < self._resume_section_start:
                continue

            msg = f"{'#' * 20} STARTING SECTION {section}/{self.num_sections}{'#' * 20}"
            logger.info(msg)
            print(msg, flush=True)

            # Set explore schedule on memory service
            if hasattr(self.memory_service, 'set_current_epoch'):
                self.memory_service.set_current_epoch(section - 1, self.num_sections)

            # Get sessions for this section
            start_idx = (section - 1) * sessions_per_section
            end_idx = start_idx + sessions_per_section
            section_sessions = self.train_sessions[start_idx:end_idx]

            if not section_sessions:
                logger.warning("Section %d is empty (train_sessions=%d, num_sections=%d); skipping training",
                               section, len(self.train_sessions), self.num_sections)
                continue

            # When resuming into this section, skip already-completed batches.
            # Checkpoints are tagged with 1-indexed batch_num (batch_start//batch_size + 1).
            resume_batch_num = self._resume_batch_start if section == self._resume_section_start else 0

            # Run in batches
            all_results = []
            for batch_start in range(0, len(section_sessions), self.batch_size):
                batch_num = batch_start // self.batch_size + 1
                if batch_num <= resume_batch_num:
                    continue
                batch_sessions = section_sessions[batch_start:batch_start + self.batch_size]
                batch_results = self._run_batch(
                    batch_sessions, mode="train",
                    file_path=self.file_path, goal_list=self.train_goals,
                )
                all_results.extend(batch_results)

                # Update memory
                self._update_memory(batch_results)

                # Update cumulative success tracking (per-task success ids)
                self._update_cum_success(batch_results)

                # Collect embeddings for clustering
                if not self._task_cluster_fitted:
                    for r in batch_results:
                        pass

                self._global_step += 1

                # Periodic checkpoint
                if batch_num % 10 == 0:
                    self._save_checkpoint(section, batch_num)

            # If resume skipped every batch in this section, don't recompute partial
            # stats / re-run eval / re-save the section-end checkpoint (it's already done).
            if not all_results:
                logger.info("Section %d fully completed before resume; skipping section-end processing", section)
                continue

            # Section stats
            success_rate = sum(r['success'] for r in all_results) / len(all_results) if all_results else 0
            avg_reward = sum(r['reward'] for r in all_results) / len(all_results) if all_results else 0
            avg_steps = sum(r['steps'] for r in all_results) / len(all_results) if all_results else 0
            cum_acc = self._current_cum_acc()

            msg = (f"Section {section} Training Stats: Success Rate={success_rate*100:.2f}%, "
                   f"Train Cumulative SR={cum_acc*100:.2f}% ({len(self._cum_success_ids)}/{self._cum_total}), "
                   f"Avg Reward={avg_reward:.3f}, Avg Steps={avg_steps:.2f}")
            logger.info(msg)
            print(msg, flush=True)

            # Trigger clustering after section 1
            if section == 1 and not self._task_cluster_fitted:
                self._init_task_clusters()

            # Region clustering update — only when this section produced new training
            # data. Prevents eval-only / fully-resumed sections from re-clustering and
            # drifting region state (see docs/RESUME_EVAL_DRIFT.md).
            if (all_results and hasattr(self.memory_service, 'region_manager')
                    and self.memory_service.region_manager):
                try:
                    self.memory_service.region_manager.cluster()
                    logger.info("[Region] Clustering updated after section %d", section)
                except Exception as e:
                    logger.warning("[Region] Clustering failed: %s", e)

            # Eval
            if self.valid_interval and section % self.valid_interval == 0:
                self._evaluate_val(section)
            if self.test_interval and section % self.test_interval == 0:
                self._evaluate_ood(section)

            # Section checkpoint
            self._save_checkpoint(section)

        msg = f"{'=' * 60}\nWEBSHOP EXPERIMENT COMPLETED: {self.exp_name}\n{'=' * 60}"
        logger.info(msg)
        print(msg, flush=True)

        self._close_envs()

    def _close_envs(self):
        """Close all pooled WebShopEnv instances created across worker threads."""
        with self._env_pool_lock:
            for env in self._all_pooled_envs:
                try:
                    env.close()
                except Exception:
                    pass
            self._all_pooled_envs.clear()
