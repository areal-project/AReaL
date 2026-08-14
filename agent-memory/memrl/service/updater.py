"""
Updaters for different update strategies in the Memp system.

This module mirrors builders.py and retrievers.py patterns and provides:
- BaseUpdater (abstract)
- VanillaUpdater / ValidationUpdater / AdjustmentUpdater (concrete)
- get_updater factory

Key goals:
- Use MemOS text_mem.add/update to ensure real memory_id is available
- Support Adjustment in append and inplace modes
- Unify metadata fields (task_description, strategies, confidence, updated_at, etc.)
"""

from __future__ import annotations
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from tqdm import tqdm
import time

from memos.mem_os.main import MOS
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata
from memos.vec_dbs.item import VecDBItem

from .strategies import UpdateStrategy, StrategyConfiguration
from .builders import get_builder
from .embedding_rate_limiter import add_text_memory_with_retry

logger = logging.getLogger(__name__)


# ------------ Helper structures ------------

@dataclass
class AdjustmentConfig:
    mode: str = "append"  # "append" | "inplace"
    confidence_factor: float = 0.8  # reduce confidence for adjustment


def _now_iso() -> str:
    return datetime.now().isoformat()


def _run_with_timeout(fn, *args, timeout_s: float, **kwargs):
    """Run fn(*args, **kwargs) in a side thread; raise FuturesTimeoutError if it
    does not finish within timeout_s seconds. Prevents the main training loop
    from hanging forever when an LLM/embed call goes into a half-dead state."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout_s)


def _build_standard_metadata(
    *,
    base: Optional[Dict[str, Any]],
    task_description: str,
    strategies: StrategyConfiguration,
    confidence: float,
    extra: Optional[Dict[str, Any]] = None,
) -> TextualMemoryMetadata:
    """Compose TextualMemoryMetadata with unified fields.

    Note: TextualMemoryMetadata(model_config.extra="allow") permits extra fields.
    """
    meta: Dict[str, Any] = dict(base or {})
    meta.setdefault("updated_at", _now_iso())
    meta.setdefault("memory_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    meta["task_description"] = task_description
    meta["strategy_build"] = strategies.build.value
    meta["strategy_retrieve"] = strategies.retrieve.value
    meta["strategy_update"] = strategies.update.value
    meta["confidence"] = confidence
    if extra:
        meta.update(extra)
    return TextualMemoryMetadata(**meta)


def _get_text_mem(mos: MOS, user_id: str, mem_cube_id: Optional[str]) -> Any:
    if mem_cube_id is None:
        # fallback to user's first accessible cube
        cubes = mos.user_manager.get_user_cubes(user_id)
        if not cubes:
            raise ValueError(f"No mem cube accessible for user {user_id}")
        mem_cube_id = cubes[0].cube_id
    if mem_cube_id not in mos.mem_cubes:
        raise ValueError(f"MemCube '{mem_cube_id}' is not loaded. Please register.")
    text_mem = mos.mem_cubes[mem_cube_id].text_mem
    if text_mem is None:
        raise ValueError("Textual memory is not initialized")
    return text_mem


# ------------ Base class ------------

def mem_add_with_retry(text_mem, item, max_retries=5, base_delay=2.0):
    """Retry and globally rate-limit embedding-backed text memory writes."""
    return add_text_memory_with_retry(
        text_mem, item, max_retries=max_retries, base_delay=base_delay
    )

class BaseUpdater(ABC):
    def __init__(
        self,
        mos: MOS,
        num_workers: int,
        user_id: str,
        strategies: StrategyConfiguration,
        llm: Any,
        *,
        default_cube_id: Optional[str] = None,
        memory_confidence: float = 100.0,
        adjustment_config: Optional[AdjustmentConfig] = None,
        strip_thinking: bool = False,
        max_trajectory_len: int = 0,
    ) -> None:
        self.mos = mos
        self.num_workers = num_workers
        self.user_id = user_id
        self.strategies = strategies
        self.llm = llm
        self.default_cube_id = default_cube_id
        self.memory_confidence = memory_confidence
        self.adjustment_config = adjustment_config or AdjustmentConfig()
        self.strip_thinking = strip_thinking
        self.max_trajectory_len = max_trajectory_len

    @abstractmethod
    def prepare_update_op(
        self,
        task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        [NEW] The "thinking" phase. Prepares the memory operation without writing to the DB.
        This method is thread-safe and can be run in parallel.
        It must return a dictionary representing the write operation.
        The dictionary must include an "op" key ('add', 'update', 'noop') and a "task_description" key.
        """
        ...

    def execute_update_op(self, op: Dict) -> Optional[str]:
        """Execute one prepared operation serially.

        When ``precomputed_vector`` is present, bypass the text-memory embedder
        and commit directly to the vector DB. Network embedding is performed in
        one batch before this serial state-mutation phase.
        """
        if not op or op.get("op") == "noop":
            return None

        text_mem = _get_text_mem(self.mos, self.user_id, self.default_cube_id)
        op_type = op["op"]
        vector = op.get("precomputed_vector")
        if op.get("embedding_precompute_failed"):
            logger.error(
                "Skipping memory write for task %r because batch embedding precompute exhausted its retry budget",
                op.get("task_description", "unknown_task"),
            )
            return None
        if op_type == "add":
            item = op["item"]
            if vector is not None and hasattr(text_mem, "vector_db"):
                text_mem.vector_db.add([VecDBItem(id=item.id, payload=item.model_dump(), vector=vector)])
            else:
                mem_add_with_retry(text_mem, item)
            return str(item.id)
        elif op_type == "update":
            mem_id = op["id"]
            data = op["data"]
            if vector is not None and hasattr(text_mem, "vector_db"):
                item = TextualMemoryItem(**data) if isinstance(data, dict) else data
                item.id = mem_id
                text_mem.vector_db.update(
                    mem_id, VecDBItem(id=mem_id, payload=item.model_dump(), vector=vector)
                )
            else:
                text_mem.update(mem_id, data)
            return str(mem_id)
        
        logger.warning(f"Unknown operation type '{op_type}' in execute_update_op.")
        return None

    def update(
        self,
        task_description: str,
        trajectory: str,
        success: bool,
        retrieved_memory_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """A single, synchronous update operation for convenience and testing."""
        op = self.prepare_update_op(task_description, trajectory, success, retrieved_memory_ids, metadata)
        return self.execute_update_op(op)

    def update_batch(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_ids_list: List[List[str]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Optional[str]]:
        """
        Hybrid parallel/serial batch update using lists of data.
        """
        num_tasks = len(task_descriptions)
        logger.info(f"Starting hybrid parallel update for {num_tasks} memories...")
        
        metadatas = metadatas or [None] * num_tasks
        
        ops_to_execute = [None] * num_tasks

        # Per-op timeout for both phases. If vLLM streaming hangs or a
        # connection goes half-dead, drop that op instead of blocking the
        # whole training loop. 300s is generous for one LLM summarization.
        _OP_TIMEOUT_S = 300.0

        # --- Phase 1: Parallel "Thinking" (prepare_update_op) ---
        # Wall-clock budget for the whole parallel phase. A half-dead streaming
        # LLM connection can hang a worker thread indefinitely (httpx idle-timeout
        # doesn't fire while keepalive bytes trickle in), so we must NOT wait
        # _OP_TIMEOUT_S * num_tasks (~2.7h for 32) — that let one stuck op freeze
        # the whole job for hours. Cap the phase, then abandon stragglers without
        # joining their (un-killable) threads.
        _max_workers = int(os.environ.get("MEMRL_UPDATE_MAX_WORKERS", "0") or "0") or self.num_workers
        import math
        _phase_budget = _OP_TIMEOUT_S + _OP_TIMEOUT_S * math.ceil(num_tasks / max(1, _max_workers))
        executor = ThreadPoolExecutor(max_workers=_max_workers)
        try:
            future_to_index = {
                executor.submit(self.prepare_update_op, td, traj, success, r_ids, meta): i
                for i, (td, traj, success, r_ids, meta) in enumerate(zip(
                    task_descriptions, trajectories, successes, retrieved_ids_list, metadatas
                ))
            }

            pending = set(future_to_index.keys())
            try:
                for future in tqdm(
                    as_completed(future_to_index, timeout=_phase_budget),
                    total=num_tasks,
                    desc="Updating memories (Parallel Processing)",
                ):
                    pending.discard(future)
                    index = future_to_index[future]
                    try:
                        op = future.result(timeout=_OP_TIMEOUT_S)
                        ops_to_execute[index] = op
                    except FuturesTimeoutError:
                        logger.error(
                            f"Timeout ({_OP_TIMEOUT_S}s) preparing update for task "
                            f"'{task_descriptions[index]}'; skipping op."
                        )
                    except Exception as e:
                        logger.error(f"Failed to prepare update for task '{task_descriptions[index]}': {e}", exc_info=True)
            except FuturesTimeoutError:
                logger.error(
                    f"Parallel-phase budget ({_phase_budget:.0f}s) exceeded with {len(pending)} stuck ops; "
                    "cancelling and continuing WITHOUT joining hung threads."
                )
                for f in pending:
                    f.cancel()
        finally:
            # wait=False: do not block on un-killable threads stuck in a hung
            # streaming LLM call; cancel_futures drops any not-yet-started work.
            executor.shutdown(wait=False, cancel_futures=True)

        # --- Phase 1.5: Batch pre-compute vectors for write operations ---
        # GeneralTextMemory.add/update embeds only item.memory. Compute all of
        # those vectors in one request, then keep Qdrant/cache mutation serial.
        embeddable = []
        embeddable_ops = []
        for op in ops_to_execute:
            if not op or op.get("op") not in ("add", "update"):
                continue
            if op.get("op") == "add":
                item = op.get("item")
                text = getattr(item, "memory", None)
            else:
                data = op.get("data") or {}
                text = data.get("memory") if isinstance(data, dict) else getattr(data, "memory", None)
            if text:
                embeddable.append(str(text))
                embeddable_ops.append(op)
        if embeddable:
            try:
                text_mem = _get_text_mem(self.mos, self.user_id, self.default_cube_id)
                embed_fn = getattr(getattr(text_mem, "embedder", None), "embed", None)
                if callable(embed_fn):
                    logger.info("Pre-computing %d memory write embeddings in one batch...", len(embeddable))
                    vectors = embed_fn(embeddable)
                    if len(vectors) != len(embeddable_ops):
                        raise ValueError(
                            f"embedding count mismatch: got {len(vectors)} for {len(embeddable_ops)} ops"
                        )
                    for op, vector in zip(embeddable_ops, vectors):
                        op["precomputed_vector"] = vector
                    logger.info("Pre-computed %d memory write embeddings", len(vectors))
            except Exception as e:
                for op in embeddable_ops:
                    op["embedding_precompute_failed"] = True
                logger.error(
                    "Batch memory-write embedding precompute failed; skipping %d writes to avoid serial retry blocking: %s",
                    len(embeddable_ops), e,
                )

        # --- Phase 2: Serial "Writing" (execute_update_op) ---
        _write_interval = float(os.environ.get("MEMRL_EMBED_MIN_INTERVAL", "0") or "0")
        logger.info(f"Executing {len(ops_to_execute)} prepared memory operations serially (interval={_write_interval}s)...")
        results = []
        for op in tqdm(ops_to_execute, desc="Updating memories (Serial Writing)"):
            if op:
                if _write_interval > 0:
                    time.sleep(_write_interval)
                task_desc = op.get("task_description", "unknown_task")
                try:
                    mem_id = _run_with_timeout(
                        self.execute_update_op, op, timeout_s=_OP_TIMEOUT_S
                    )
                    results.append((task_desc, mem_id, op.get("precomputed_vector")))
                except FuturesTimeoutError:
                    logger.error(
                        f"Timeout ({_OP_TIMEOUT_S}s) writing memory for task '{task_desc}'; skipping op."
                    )
                    results.append((task_desc, None, op.get("precomputed_vector")))
                except Exception as e:
                    logger.error(f"Failed to execute update for task '{task_desc}': {e}", exc_info=True)
                    results.append((task_desc, None, op.get("precomputed_vector")))

        return results


    # utilities usable by subclasses
    def _add_new_memory(self, task_description: str, full_content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Add memory where embedding uses only task_description, and full content is in metadata."""
        text_mem = _get_text_mem(self.mos, self.user_id, self.default_cube_id)
        item = TextualMemoryItem(
            memory=task_description,  # retrieval key only
            metadata=_build_standard_metadata(
                base=metadata,
                task_description=task_description,
                strategies=self.strategies,
                confidence=self.memory_confidence,
                extra={"type": "procedure", "source": "conversation", "full_content": full_content},
            ),
        )
        mem_add_with_retry(text_mem, item)
        return str(item.id)

    def _generate_reflection(
        self,
        task_description: str,
        failed_trajectory: str,
        eval_error: str = "",
        source_benchmark: str = "",
    ) -> str:
        # Head+tail extraction for proceduralized trajectory
        if len(failed_trajectory) > 1600:
            code_context = failed_trajectory[:800] + "\n...[truncated]...\n" + failed_trajectory[-800:]
        else:
            code_context = failed_trajectory

        # LLB DB/OS are interactive-agent tasks, NOT one-shot code generation.
        # Their failures are dominated by protocol/output-format violations and
        # wrong final answers that look fine at the SQL/shell level. The generic
        # "failed code generation" prompt makes the LLM inspect syntax and
        # wrongly conclude "no mistakes". Route those benchmarks to a prompt that
        # (a) states the task is KNOWN to have failed, (b) forbids "no mistake"
        # verdicts, and (c) points at protocol/format/answer correctness.
        sb = (source_benchmark or "").lower()
        is_llb_interactive = sb in ("llb_db", "llb_os", "llb_db_bench", "llb_os_interaction")

        if is_llb_interactive:
            # Pass the FULL trajectory; the LLB dispatcher truncates per-variant so
            # that v2 can preserve the tail (the submitted `Final Answer:`), which
            # is what lets it judge whether the answer was already valid tuple format.
            return self._generate_reflection_llb_interactive(
                task_description, failed_trajectory, eval_error, source_benchmark=sb
            )

        error_context = f"\nTest error: {eval_error[:300]}" if eval_error else ""

        prompt = f"""You are analyzing a failed code generation task. Generate a COMPACT failure analysis.

Task: {task_description[:1000]}

Failed code (for analysis only, do NOT reproduce):
{code_context}{error_context}

Output must begin with FAILURE_MODE: on the first line. Use this EXACT format, plain text only:
FAILURE_MODE: <one line, what category of failure>
MISTAKES:
- <concrete mistake 1>
- <concrete mistake 2>
- <concrete mistake 3>
FIXES:
- <actionable fix 1>
- <actionable fix 2>
- <actionable fix 3>
AVOID:
- <anti-pattern to avoid 1>
- <anti-pattern to avoid 2>
- <anti-pattern to avoid 3>

Rules:
- Each bullet ≤ 120 chars, 3 bullets per section
- No code blocks, no markdown bold, no full task repetition
- Be specific (name conditions, data types, edge cases), not generic
- If uncertain about a point, write UNKNOWN instead of guessing
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.llm.generate(messages, temperature=0.2, max_tokens=1024)
            cleaned = self._clean_reflection(raw)
            if len(cleaned.strip()) < 40:
                return self._fallback_reflection(task_description, eval_error)
            return cleaned
        except Exception as e:
            logger.warning(f"LLM reflection generation failed: {e}")
            return self._fallback_reflection(task_description, eval_error)

    @staticmethod
    def _truncate_llb_headtail(failed_trajectory: str) -> str:
        """Original head+tail truncation (kept identical for legacy byte-compat)."""
        traj = failed_trajectory or ""
        if len(traj) > 1600:
            return traj[:800] + "\n...[truncated]...\n" + traj[-800:]
        return traj

    @staticmethod
    def _truncate_llb_preserve_answer(failed_trajectory: str) -> str:
        """Truncate a long trajectory while GUARANTEEING the submitted final answer
        stays in the window.

        LLB DB submits via `Final Answer: ...`; that line is what the evaluator reads
        and what v2 reflection needs to judge "was the answer already valid tuples?".
        A plain head+tail cut can lose it when the tail is dominated by a large SQL
        result set. We anchor on the LAST `Final Answer:` occurrence and keep a
        generous window around it, plus the trajectory head for context.
        """
        import re as _re
        traj = failed_trajectory or ""
        if len(traj) <= 1600:
            return traj

        head = traj[:700]
        # Find the last submitted answer marker (case-insensitive, tolerate spacing).
        matches = list(_re.finditer(r"(?i)final\s*answer\s*:", traj))
        if not matches:
            # No explicit answer marker (likely incomplete/step-limit); fall back to
            # head+tail so we still see how the interaction ended.
            return traj[:800] + "\n...[truncated]...\n" + traj[-800:]

        m = matches[-1]
        # Keep some lead-in before the answer (the SQL/step that produced it) and
        # everything after it (the answer itself, usually short).
        ans_start = max(0, m.start() - 500)
        answer_block = traj[ans_start:]
        if len(answer_block) > 1100:
            # Answer block itself huge (e.g. answer echoes a big result) — keep the
            # marker plus a bounded tail so the actual submitted value survives.
            answer_block = traj[m.start():][:1100]
            ans_start = m.start()

        if ans_start <= len(head):
            # Head and answer block are adjacent/overlapping: return one continuous
            # prefix that covers both (bounded), no marker needed since nothing is cut
            # between them.
            return traj[: max(len(head), ans_start + len(answer_block))][:1900]
        return head + "\n...[truncated]...\n" + answer_block

    def _generate_reflection_llb_interactive(
        self, task_description: str, trajectory_context: str, eval_error: str = "",
        source_benchmark: str = "",
    ) -> str:
        """Dispatch LLB DB/OS reflection prompt by MEMRL_LLB_REFLECTION_PROMPT.

        Receives the FULL trajectory and truncates per-variant:
        - legacy: original head+tail (byte-identical to previous behavior).
        - v2: answer-preserving truncation so the submitted `Final Answer:` survives.

        Default "legacy" keeps the original prompt untouched. Set "v2" to use the
        corrected prompt. Routing is task-aware under v2:
        - OS (llb_os / llb_os_interaction): OS-specific prompt that describes the real
          OS grading contract (an evaluation command checks exit_code==0 on the
          resulting SYSTEM STATE; there is NO textual answer / tuple / SQL). This stops
          the DB-centric prompt from feeding SQL/tuple/output-format language into OS
          reflections.
        - DB (default / everything else): the DB v2 tuple-contract prompt.
        legacy stays DB-worded for both to preserve byte-for-byte reproducibility.
        """
        import os as _os
        variant = (_os.environ.get("MEMRL_LLB_REFLECTION_PROMPT", "legacy") or "legacy").strip().lower()
        sb = (source_benchmark or "").lower()
        is_os = sb in ("llb_os", "llb_os_interaction")
        if variant in ("v2", "corrected", "new"):
            code_context = self._truncate_llb_preserve_answer(trajectory_context)
            if is_os:
                return self._generate_reflection_llb_os_v2(
                    task_description, code_context, eval_error
                )
            return self._generate_reflection_llb_interactive_v2(
                task_description, code_context, eval_error
            )
        code_context = self._truncate_llb_headtail(trajectory_context)
        return self._generate_reflection_llb_interactive_legacy(
            task_description, code_context, eval_error
        )

    def _generate_reflection_llb_interactive_legacy(
        self, task_description: str, trajectory_context: str, eval_error: str = ""
    ) -> str:
        """Reflection prompt for LLB DB/OS interactive-agent tasks.

        Unlike code generation, these tasks fail mostly on interaction protocol,
        output format, or wrong final answers rather than code syntax. The prompt
        provides the evaluator's failure evidence and explicitly forbids a
        "no mistakes" verdict, since the task is definitionally a failure here.
        """
        if eval_error:
            error_context = f"\n\nEvaluator failure evidence (this task DID fail):\n{eval_error[:400]}"
        else:
            error_context = (
                "\n\n(No structured evaluator detail was captured, but the final "
                "answer was judged INCORRECT — the task DID fail. Infer the most "
                "likely cause from the interaction below.)"
            )

        prompt = f"""You are analyzing a FAILED attempt at an interactive database/OS agent task.
The task was scored INCORRECT by the environment. Your job is to explain WHY it failed and how to succeed next time.

This is NOT one-shot code generation. Failures are usually one of:
- Protocol violation: the agent did not emit the exact required directive (e.g. "Action: Operation" to run a command, then "Action: Answer" to submit the final answer). Missing/misspelled/extra directives cause automatic failure even if the SQL/shell command is correct.
- Output/format mismatch: the final answer's shape, column order, rounding, or types did not match what was asked.
- Wrong answer: the query/command logic returned incorrect values (bad JOIN/filter/aggregation, wrong condition).
- Incomplete: the agent stopped before submitting a final answer, or ran out of interaction steps.

Task: {task_description[:1000]}

Interaction trajectory (for analysis only, do NOT reproduce):
{trajectory_context}{error_context}

CRITICAL: The task failed. Do NOT write "None", "no mistakes", or "correct" — if the SQL/command looks right, the failure is almost certainly a protocol or output-format problem. Identify it.

Output must begin with FAILURE_MODE: on the first line. Use this EXACT format, plain text only:
FAILURE_MODE: <one line: protocol violation | output format mismatch | wrong answer | incomplete/step-limit>
MISTAKES:
- <concrete mistake 1>
- <concrete mistake 2>
- <concrete mistake 3>
FIXES:
- <actionable fix 1>
- <actionable fix 2>
- <actionable fix 3>
AVOID:
- <anti-pattern to avoid 1>
- <anti-pattern to avoid 2>
- <anti-pattern to avoid 3>

Rules:
- Each bullet ≤ 120 chars, 3 bullets per section
- No code blocks, no markdown bold, no full task repetition
- Be specific (name the exact directive, column, or condition), not generic
- Never claim the attempt was correct; it was scored INCORRECT
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.llm.generate(messages, temperature=0.2, max_tokens=1024)
            cleaned = self._clean_reflection(raw)
            if len(cleaned.strip()) < 40:
                return self._fallback_reflection_llb(task_description, eval_error)
            return cleaned
        except Exception as e:
            logger.warning(f"LLM LLB reflection generation failed: {e}")
            return self._fallback_reflection_llb(task_description, eval_error)

    def _generate_reflection_llb_interactive_v2(
        self, task_description: str, trajectory_context: str, eval_error: str = ""
    ) -> str:
        """Corrected LLB DB/OS reflection prompt (MEMRL_LLB_REFLECTION_PROMPT=v2).

        Fixes the legacy prompt's bias of blaming "output format mismatch" whenever
        the SQL looks right, which led it to recommend dict/table formats that BREAK
        the evaluator's required tuple format. Instead we (1) state the evaluator's
        real answer contract (public rule, not the answer), and (2) tell the model to
        prefer logic causes (row count/order/aggregation/filter) when the submitted
        answer is already valid tuple format. Ground truth is intentionally NOT
        available to reflection (consistent with HLE/BCB: no answer peeking).
        """
        if eval_error:
            error_context = f"\n\nEvaluator failure evidence (this task DID fail):\n{eval_error[:400]}"
        else:
            error_context = (
                "\n\n(No structured evaluator detail was captured, but the final "
                "answer was judged INCORRECT — the task DID fail. Infer the most "
                "likely cause from the interaction below.)"
            )

        prompt = f"""You are analyzing a FAILED attempt at an interactive database/OS agent task.
The task was scored INCORRECT by the environment. Explain WHY it failed and how to succeed next time.

You do NOT have the ground-truth answer. Diagnose from the trajectory and the evaluator's contract below — never invent the correct values.

The DB evaluator's answer contract (this is a fixed public rule, NOT the answer):
- The final answer MUST be comma-separated tuples in round brackets: (v1, v2), (v3, v4), ...
- Row COUNT and ORDER must exactly match the query result (honor ORDER BY; do not re-sort or drop rows).
- Numbers are compared numerically and are TOLERANT: Decimal('120000'), 120000, 120000.0 all pass. Do NOT "fix" Decimal.
- Strings must match EXACTLY (case, spacing). Do not add column headers/labels inside the answer.
- Tuple format is REQUIRED. dict/JSON/markdown-table/CSV or extra prose will FAIL to parse.

Because the tuple format + Decimal are already accepted, if the submitted answer is already
comma-separated tuples, the cause is USUALLY a logic problem, not formatting:
- Wrong row count (missing/extra rows from bad WHERE/HAVING/JOIN or missing DISTINCT).
- Wrong row ORDER (ORDER BY direction/columns not matching the request).
- Wrong aggregate/value (SUM/COUNT/AVG/GROUP BY off, wrong column, rounding).
Only diagnose "output format mismatch" if the answer was actually NOT valid tuples (e.g. dict/table/prose).

Failure categories:
- protocol violation: missing/misspelled "Action: Operation" / "Action: Answer" directive.
- wrong answer (logic): row count/order/aggregation/filter wrong (MOST COMMON when tuples were used).
- output format mismatch: answer was not valid comma-separated tuples (dict/table/prose/no brackets).
- incomplete/step-limit: stopped before submitting, or ran out of steps.

Task: {task_description[:1000]}

Interaction trajectory (for analysis only, do NOT reproduce):
{trajectory_context}{error_context}

CRITICAL: The task failed. Do NOT write "None", "no mistakes", or "correct". But do NOT reflexively
blame formatting: if the answer was already valid tuples, look for a logic cause instead.

Output must begin with FAILURE_MODE: on the first line. Use this EXACT format, plain text only:
FAILURE_MODE: <one line: protocol violation | wrong answer | output format mismatch | incomplete/step-limit>
MISTAKES:
- <concrete mistake 1>
- <concrete mistake 2>
- <concrete mistake 3>
FIXES:
- <actionable fix 1>
- <actionable fix 2>
- <actionable fix 3>
AVOID:
- <anti-pattern to avoid 1>
- <anti-pattern to avoid 2>
- <anti-pattern to avoid 3>

Rules:
- Each bullet ≤ 120 chars, 3 bullets per section
- No code blocks, no markdown bold, no full task repetition
- Be specific (name the exact directive, column, condition, or ORDER BY), not generic
- Do NOT recommend dict/JSON/table/CSV formats or "converting Decimal" — those BREAK the evaluator
- Never claim the attempt was correct; it was scored INCORRECT
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.llm.generate(messages, temperature=0.2, max_tokens=1024)
            cleaned = self._clean_reflection(raw)
            if len(cleaned.strip()) < 40:
                return self._fallback_reflection_llb(task_description, eval_error)
            return cleaned
        except Exception as e:
            logger.warning(f"LLM LLB reflection v2 generation failed: {e}")
            return self._fallback_reflection_llb(task_description, eval_error)

    def _generate_reflection_llb_os_v2(
        self, task_description: str, trajectory_context: str, eval_error: str = ""
    ) -> str:
        """OS-specific LLB reflection prompt (MEMRL_LLB_REFLECTION_PROMPT=v2, OS tasks).

        Why separate from the DB prompt: LLB OS is graded completely differently.
        There is NO textual final answer, NO tuples, NO SQL. The evaluator runs a
        hidden checking command and marks the task correct iff its exit_code == 0 —
        i.e. it inspects the resulting SYSTEM STATE (files, permissions, ownership,
        users/groups, symlinks, file contents). The agent's job is to run bash
        commands via `Act: bash` / ```bash ...``` and end with `Act: finish`.

        The DB prompt's talk of tuples/Decimal/ORDER BY/output-format is meaningless
        here and only adds noise. This prompt teaches the real OS contract and steers
        reflection toward the actual OS failure modes: wrong/missing state changes,
        permission/ownership bits (incl. setuid/setgid/sticky), symlink vs target,
        missing verification, and protocol violations.
        """
        if eval_error:
            error_context = f"\n\nEvaluator failure evidence (this task DID fail):\n{eval_error[:400]}"
        else:
            error_context = (
                "\n\n(No structured evaluator detail was captured, but the task was "
                "judged INCORRECT — it DID fail. Infer the most likely cause from the "
                "interaction below.)"
            )

        prompt = f"""You are analyzing a FAILED attempt at an interactive Linux OS agent task.
The task was scored INCORRECT by the environment. Explain WHY it failed and how to succeed next time.

You do NOT have the ground truth. Diagnose from the trajectory and the grading contract below — never invent the expected state.

How OS tasks are graded (fixed public rule, NOT the answer):
- There is NO textual answer to submit. The agent runs bash commands and finishes.
- After the agent finishes, the environment runs a hidden CHECK COMMAND and the task is
  correct ONLY if that command exits 0 — i.e. it verifies the resulting SYSTEM STATE.
- So correctness = did the filesystem/users/groups/permissions/contents actually end up
  as required — NOT whether a command printed something or "looked right".
- There are NO tuples, NO SQL, NO output-format / column-order / Decimal rules here.
  Do NOT mention those; they do not exist for OS tasks.

Common OS failure causes (prefer these):
- Wrong or missing state change: file/dir/user/group/symlink not created, wrong path or name,
  wrong content, edit not actually applied.
- Permission/ownership wrong: chmod bits off (including setuid/setgid/sticky, e.g. 2775 vs 775),
  chown/chgrp on the wrong target, or applied to a symlink instead of its target.
- Symlink vs hard link vs target confused; link points to the wrong path.
- Incomplete verification: agent assumed success from a non-empty output or exit status without
  confirming the actual required end state, then finished prematurely.
- Protocol violation: did not use `Act: bash` with a ```bash``` block, or never sent `Act: finish`.
- Incomplete/step-limit: ran out of interaction steps before achieving the required state.

Task: {task_description[:1000]}

Interaction trajectory (for analysis only, do NOT reproduce):
{trajectory_context}{error_context}

CRITICAL: The task failed. Do NOT write "None", "no mistakes", or "correct". Focus on which required
SYSTEM STATE was wrong or unverified. Name concrete commands/paths/permission bits when you can.

Output must begin with FAILURE_MODE: on the first line. Use this EXACT format, plain text only:
FAILURE_MODE: <one line: wrong/missing state | permission or ownership wrong | symlink/link error | incomplete verification | protocol violation | incomplete/step-limit>
MISTAKES:
- <concrete mistake 1>
- <concrete mistake 2>
- <concrete mistake 3>
FIXES:
- <actionable fix 1>
- <actionable fix 2>
- <actionable fix 3>
AVOID:
- <anti-pattern to avoid 1>
- <anti-pattern to avoid 2>
- <anti-pattern to avoid 3>

Rules:
- Each bullet ≤ 120 chars, 3 bullets per section
- No code blocks, no markdown bold, no full task repetition
- Be specific (name the exact command, path, permission bit, or check), not generic
- Do NOT mention tuples/SQL/Decimal/output-format/column-order — they do not apply to OS tasks
- Prefer verifying end state with commands like ls -l, stat, id, getent, cat, test
- Never claim the attempt was correct; it was scored INCORRECT
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.llm.generate(messages, temperature=0.2, max_tokens=1024)
            cleaned = self._clean_reflection(raw)
            if len(cleaned.strip()) < 40:
                return self._fallback_reflection_llb(task_description, eval_error)
            return cleaned
        except Exception as e:
            logger.warning(f"LLM LLB OS reflection v2 generation failed: {e}")
            return self._fallback_reflection_llb(task_description, eval_error)

    @staticmethod
    def _clean_reflection(text: str) -> str:
        """Strip markdown formatting from reflection output."""
        import re
        text = re.sub(r'^#{1,4}\s+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    @staticmethod
    def _fallback_reflection(task_description: str = "", eval_error: str = "") -> str:
        """Task-conditioned fallback when reflection generation fails."""
        short = task_description[:80].split('\n')[0] if task_description else "the task"
        error_hint = f" Error hint: {eval_error[:100]}." if eval_error else ""
        return (
            f"FAILURE_MODE: Unknown failure\n"
            f"MISTAKES:\n- Solution for '{short}' produced incorrect output{error_hint}\n"
            f"- Likely fails on boundary conditions (empty input, single element, extreme values)\n"
            f"- UNKNOWN\n"
            f"FIXES:\n- Add explicit input validation and boundary guards before core logic\n"
            f"- Test with edge cases: empty, single-element, and large inputs\n"
            f"- UNKNOWN\n"
            f"AVOID:\n- Avoid assuming input is always non-empty or well-formed\n"
            f"- Avoid skipping return type validation\n"
            f"- UNKNOWN"
        )

    @staticmethod
    def _fallback_reflection_llb(task_description: str = "", eval_error: str = "") -> str:
        """LLB DB/OS fallback: never claim correctness; focus on protocol/format."""
        short = task_description[:80].split('\n')[0] if task_description else "the task"
        error_hint = f" Evidence: {eval_error[:120]}." if eval_error else ""
        return (
            f"FAILURE_MODE: Protocol or output-format failure\n"
            f"MISTAKES:\n- Attempt for '{short}' was scored INCORRECT.{error_hint}\n"
            f"- Likely did not emit the exact required directive (Action: Operation / Action: Answer)\n"
            f"- Or the final answer's format/columns/values did not match the requirement\n"
            f"FIXES:\n- Use 'Action: Operation' to run each command, then 'Action: Answer' to submit\n"
            f"- Recheck the requested output shape (column order, rounding, types) before answering\n"
            f"- Verify the query/command logic (joins, filters, aggregation) against the question\n"
            f"AVOID:\n- Avoid submitting without the exact protocol directive\n"
            f"- Avoid returning raw tuples/lists when a specific format is required\n"
            f"- Avoid stopping before submitting a final answer"
        )


# ------------ Concrete updaters ------------

class VanillaUpdater(BaseUpdater):
    def prepare_update_op(
        self, task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Prepare op to always add a new memory."""
        builder = get_builder(self.strategies.build, self.llm,
                              strip_thinking=self.strip_thinking,
                              max_trajectory_len=self.max_trajectory_len)
        raw_body = builder.build(task_description, trajectory)
        from memrl.utils.sanitize import sanitize_llm_output
        memory_body = sanitize_llm_output(raw_body)
        if not memory_body or len(memory_body.strip()) < 20:
            import re
            memory_body = re.sub(r'</?think>', '', raw_body).strip()
        if not memory_body or len(memory_body.strip()) < 20:
            memory_body = trajectory[:2000] if trajectory else task_description
        memory_content = f"Task: {task_description}\n\n{memory_body}"

        item = TextualMemoryItem(
            memory=task_description,
            metadata=_build_standard_metadata(
                base=metadata, task_description=task_description, strategies=self.strategies,
                confidence=self.memory_confidence,
                extra={"type": "procedure", "source": "conversation", "full_content": memory_content},
            ),
        )
        return {"op": "add", "item": item, "task_description": task_description}

class ValidationUpdater(BaseUpdater):
    def prepare_update_op(
        self, task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Prepare op to add memory only if successful."""
        if not success:
            return {"op": "noop", "task_description": task_description}
        
        # If successful, logic is identical to VanillaUpdater
        builder = get_builder(self.strategies.build, self.llm,
                              strip_thinking=self.strip_thinking,
                              max_trajectory_len=self.max_trajectory_len)
        raw_body = builder.build(task_description, trajectory)
        from memrl.utils.sanitize import sanitize_llm_output
        memory_body = sanitize_llm_output(raw_body)
        if not memory_body or len(memory_body.strip()) < 20:
            import re
            memory_body = re.sub(r'</?think>', '', raw_body).strip()
        if not memory_body or len(memory_body.strip()) < 20:
            memory_body = trajectory[:2000] if trajectory else task_description
        memory_content = f"Task: {task_description}\n\n{memory_body}"

        item = TextualMemoryItem(
            memory=task_description,
            metadata=_build_standard_metadata(
                base=metadata, task_description=task_description, strategies=self.strategies,
                confidence=self.memory_confidence,
                extra={"type": "procedure", "source": "conversation", "full_content": memory_content},
            ),
        )
        return {"op": "add", "item": item, "task_description": task_description}

class AdjustmentUpdater(BaseUpdater):
    def prepare_update_op(
        self, task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Prepare op: on success, add new; on failure, prepare an adjustment op."""
        if success:
            builder = get_builder(self.strategies.build, self.llm,
                                  strip_thinking=self.strip_thinking,
                                  max_trajectory_len=self.max_trajectory_len)
            raw_body = builder.build(task_description, trajectory)
            from memrl.utils.sanitize import sanitize_llm_output
            memory_body = sanitize_llm_output(raw_body)
            if not memory_body or len(memory_body.strip()) < 20:
                import re
                memory_body = re.sub(r'</?think>', '', raw_body).strip()
            if not memory_body or len(memory_body.strip()) < 20:
                memory_body = trajectory[:2000] if trajectory else task_description
            memory_content = f"Task: {task_description}\n\n{memory_body}"
            item = TextualMemoryItem(
                memory=task_description,
                metadata=_build_standard_metadata(
                    base=metadata, task_description=task_description, strategies=self.strategies,
                    confidence=self.memory_confidence,
                    extra={"type": "procedure", "source": "conversation", "full_content": memory_content},
                )
            )
            return {"op": "add", "item": item, "task_description": task_description}

        # Failure path — extract eval error for better reflection
        eval_error = ""
        source_benchmark = ""
        if metadata:
            eval_error = str(metadata.get("error", "") or metadata.get("eval_error", "") or "")
            source_benchmark = str(metadata.get("source_benchmark", "") or "")
        reflection = self._generate_reflection(
            task_description, trajectory, eval_error, source_benchmark=source_benchmark
        )
        from memrl.utils.sanitize import sanitize_llm_output
        sanitized = sanitize_llm_output(reflection)
        if len(sanitized.strip()) < 40:
            import re
            sanitized = re.sub(r'</?think>', '', reflection).strip()
        reflection = sanitized
        mode = (self.adjustment_config.mode or "append").lower()

        if mode == "inplace":
            return self._prepare_inplace_adjust(task_description, trajectory, reflection, retrieved_memory_ids or [], metadata)
        
        return self._prepare_append_adjust(task_description, trajectory, reflection, retrieved_memory_ids or [], metadata)

    def _prepare_append_adjust(self, task_description: str, failed_trajectory: str, reflection: str,
                               related_ids: List[str], metadata: Optional[Dict[str, Any]]) -> Dict:
        """Prepares an 'add' operation for a compact failure insight memory."""
        # Truncate at last full line before 2500 chars to avoid cutting mid-section
        if len(reflection) > 2500:
            cut = reflection[:2500].rfind('\n')
            adjustment_content = reflection[:cut] if cut > 100 else reflection[:2500]
        else:
            adjustment_content = reflection
        if len(adjustment_content.strip()) < 40:
            return {"op": "noop", "task_description": task_description}
        meta = _build_standard_metadata(
            base=metadata, task_description=task_description, strategies=self.strategies,
            confidence=self.memory_confidence * self.adjustment_config.confidence_factor,
            extra={
                "type": "adjustment", "source": "conversation", "source_detail": "reflection", 
                "related_memory_ids": related_ids, "full_content": adjustment_content,
            },
        )
        item = TextualMemoryItem(memory=task_description, metadata=meta)
        return {"op": "add", "item": item, "task_description": task_description}

    def _prepare_inplace_adjust(self, task_description: str, failed_trajectory: str, reflection: str,
                                related_ids: List[str], metadata: Optional[Dict[str, Any]]) -> Dict:
        """Prepares an 'update' operation to modify an existing memory."""
        if not related_ids:
            return {"op": "noop", "task_description": task_description}
        
        text_mem = _get_text_mem(self.mos, self.user_id, self.default_cube_id)
        mem_id_to_update = related_ids[0]  # Update the most relevant memory

        try:
            old_item = text_mem.get(mem_id_to_update)
        except Exception as e:
            logger.warning(f"Could not find memory {mem_id_to_update} to update. Skipping. Error: {e}")
            return {"op": "noop", "task_description": task_description}

        old_meta = getattr(old_item, "metadata", None)
        old_meta_dict = old_meta.model_dump() if hasattr(old_meta, "model_dump") else dict(old_meta or {})
        prev_full = old_meta_dict.get("full_content", f"Task: {old_item.memory}\n\n(Original content unavailable)")

        new_full_content = (
            f"{prev_full}\n\n--- ADJUSTMENT NOTE ({_now_iso()}) ---\n"
            f"A similar task failed: {task_description}\n\n"
            f"Reflection:\n{reflection}\n"
        )
        
        new_meta = _build_standard_metadata(
            base={**old_meta_dict, **(metadata or {})}, task_description=old_item.memory, # Keep original task desc in meta
            strategies=self.strategies,
            confidence=(old_meta_dict.get("confidence") or self.memory_confidence) * self.adjustment_config.confidence_factor,
            extra={"full_content": new_full_content},
        )

        return {
            "op": "update",
            "id": mem_id_to_update,
            "data": {"id": mem_id_to_update, "memory": old_item.memory, "metadata": new_meta.model_dump()},
            "task_description": task_description
        }


# ------------ Factory ------------

def get_updater(
    strategy: UpdateStrategy,
    *,
    mos: MOS,
    user_id: str,
    strategies: StrategyConfiguration,
    llm: Any,
    num_workers: int = 32,
    default_cube_id: Optional[str] = None,
    memory_confidence: float = 100.0,
    adjustment_mode: str = "append",
    adjustment_confidence_factor: float = 0.8,
    strip_thinking: bool = False,
    max_trajectory_len: int = 0,
) -> BaseUpdater:
    cfg = AdjustmentConfig(mode=adjustment_mode, confidence_factor=adjustment_confidence_factor)
    kwargs = dict(default_cube_id=default_cube_id, memory_confidence=memory_confidence,
                  adjustment_config=cfg, strip_thinking=strip_thinking, max_trajectory_len=max_trajectory_len)
    if strategy == UpdateStrategy.VANILLA:
        return VanillaUpdater(mos, num_workers, user_id, strategies, llm, **kwargs)
    if strategy == UpdateStrategy.VALIDATION:
        return ValidationUpdater(mos, num_workers, user_id, strategies, llm, **kwargs)
    return AdjustmentUpdater(mos, num_workers, user_id, strategies, llm, **kwargs)

