"""
Mem0 baseline adapter for MemRL.

Drop-in replacement for MemoryService that uses the mem0 library (Chhikara et al., 2025)
for memory management. Implements the same interface consumed by all benchmark runners
(ALFWorld, WebShop, HLE, BCB) without requiring any runner modifications.

Usage:
    from memrl.service.mem0_memory_service import Mem0MemoryService

    memory_service = Mem0MemoryService(
        llm_base_url="http://localhost:8000/v1/",
        llm_model="Qwen2.5-72B-Instruct",
        llm_api_key="EMPTY",
        embed_base_url="http://localhost:8001/v1/",
        embed_model="Qwen/Qwen3-Embedding-8B",
        embed_api_key="EMPTY",
        qdrant_path="/tmp/mem0_baseline_qdrant",
    )
"""

import copy
import json
import logging
import os
import re
import shutil
import threading
import time

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class _Mem0Metadata:
    """Duck-type for Pydantic model metadata with model_extra attribute.

    The runner accesses `mem['metadata'].model_extra.get('success', False)`
    to split memories into success/failure buckets. This wrapper provides
    that interface without requiring actual Pydantic models.
    """

    def __init__(self, extra: dict):
        self.model_extra = extra

    def model_dump(self, *args, **kwargs):
        """Match the Pydantic metadata interface used by benchmark runners."""
        return dict(self.model_extra)

    def get(self, key, default=None):
        return self.model_extra.get(key, default)

    def __getitem__(self, key):
        return self.model_extra[key]

    def __contains__(self, key):
        return key in self.model_extra

    def __getattr__(self, name):
        if name == "model_extra":
            return super().__getattribute__(name)
        return self.model_extra.get(name)


class Mem0MemoryService:
    # mem0ai creates its own OpenAI clients, bypassing the normal MemRL provider
    # gates. Space Mem0 search/add starts when MEMRL_MEM0_MIN_INTERVAL is set.
    _request_gate_lock = threading.Lock()
    _request_gate_next_at = 0.0

    @classmethod
    def _wait_for_request_slot(cls, kind: str = "chat") -> None:
        """Apply a local Mem0 gate plus the shared HLE API gate.

        Mem0 owns private OpenAI clients, so provider-level global gates do not
        intercept its fact extraction and embedding transports.  Reuse the
        exact lock-file protocol used by the normal providers so Mem0 joins the
        same cross-process Chat/embedding queues rather than creating bursts.
        """
        try:
            interval = float(os.environ.get("MEMRL_MEM0_MIN_INTERVAL", "0") or "0")
        except (TypeError, ValueError):
            interval = 0.0
        if interval > 0:
            while True:
                with cls._request_gate_lock:
                    now = time.monotonic()
                    if now >= cls._request_gate_next_at:
                        cls._request_gate_next_at = now + interval
                        break
                    delay = cls._request_gate_next_at - now
                time.sleep(delay)

        if kind not in {"chat", "embed"} or fcntl is None:
            return
        prefix = "LLM" if kind == "chat" else "EMBED"
        try:
            shared_interval = float(
                os.environ.get(f"MEMRL_{prefix}_GLOBAL_MIN_INTERVAL", "0") or "0"
            )
        except (TypeError, ValueError):
            shared_interval = 0.0
        if shared_interval <= 0:
            return
        directory = Path(os.environ.get(
            f"MEMRL_{prefix}_RATE_LIMIT_DIR",
            f"/tmp/memrl_{kind}_rate_limits",
        ))
        key_raw = os.environ.get(
            f"MEMRL_{prefix}_RATE_LIMIT_KEY",
            f"mem0-{kind}",
        )
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key_raw).strip("._")[:80] or "default"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            state_path = directory / f"{key}.state"
            with state_path.open("a+", encoding="utf-8") as state:
                fcntl.flock(state.fileno(), fcntl.LOCK_EX)
                try:
                    state.seek(0)
                    try:
                        next_at = float(state.read().strip() or "0")
                    except ValueError:
                        next_at = 0.0
                    slot = max(time.time(), next_at)
                    state.seek(0)
                    state.truncate()
                    state.write(f"{slot + shared_interval:.6f}\n")
                    state.flush()
                    os.fsync(state.fileno())
                finally:
                    fcntl.flock(state.fileno(), fcntl.LOCK_UN)
            delay = slot - time.time()
            if delay > 0:
                time.sleep(delay)
        except Exception as exc:
            logger.warning("[Mem0MemoryService] shared %s gate unavailable: %s", kind, exc)

    """Drop-in replacement for MemoryService using the mem0 library.

    Delegates memory add/search to mem0's Memory class, which provides:
    - LLM-based atomic fact extraction on add (infer=True)
    - Semantic + BM25 hybrid retrieval on search
    - Automatic memory deduplication and conflict resolution

    The adapter exposes the same method signatures that all MemRL runners
    call (retrieve_query, add_memories, add_memory, update_values,
    save/load_checkpoint_snapshot) so it can be used across benchmarks.
    """

    def __init__(
        self,
        llm_base_url: str,
        llm_model: str,
        llm_api_key: str = "EMPTY",
        embed_base_url: str = "http://localhost:8001/v1/",
        embed_model: str = "text-embedding-3-small",
        embed_api_key: str = "EMPTY",
        embedding_dims: Optional[int] = None,
        qdrant_path: str = "/tmp/mem0_baseline_qdrant",
        collection_name: str = "memrl_mem0_baseline",
        top_k: int = 3,
        infer: bool = True,
        user_id: str = "mem0_baseline",
        custom_instructions: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            llm_base_url: OpenAI-compatible LLM endpoint (e.g. vLLM/sglang)
            llm_model: Model name served at llm_base_url
            llm_api_key: API key for the LLM endpoint
            embed_base_url: OpenAI-compatible embedding endpoint
            embed_model: Embedding model name
            embed_api_key: API key for the embedding endpoint
            embedding_dims: Embedding vector dimensionality (e.g. 4096 for Qwen3-Embedding-8B).
                If None, mem0 uses its default (1536).
            qdrant_path: Local path for qdrant on-disk storage
            collection_name: Qdrant collection name
            top_k: Default number of memories to retrieve
            infer: If True, mem0 uses LLM to extract atomic facts from input.
                   If False, stores raw text directly (faster but not faithful Mem0).
            user_id: User ID for scoping memories in mem0
            custom_instructions: Optional custom instructions for mem0 fact extraction
        """
        from mem0 import Memory

        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.top_k = top_k
        self.infer = infer
        self.user_id = user_id
        self._search_all_scopes = str(
            os.environ.get("MEMRL_MEM0_SEARCH_ALL_SCOPES", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Some OpenAI-compatible embedding servers (notably SGLang serving
        # Qwen3-Embedding) reject the `dimensions` field even when it equals
        # the native output size. mem0ai 1.0.1 always sends that field, so
        # allow the launcher to use the native server output instead.
        self._omit_embedding_dimensions = str(
            os.environ.get("MEMRL_MEM0_OMIT_EMBED_DIMENSIONS", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}

        os.makedirs(qdrant_path, exist_ok=True)

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "openai_base_url": llm_base_url,
                    "api_key": llm_api_key,
                    "temperature": 0,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embed_model,
                    "openai_base_url": embed_base_url,
                    "api_key": embed_api_key,
                    # mem0ai 1.0.1 always forwards `dimensions` to the
                    # OpenAI-compatible endpoint. Use the model-native size.
                    **({"embedding_dims": embedding_dims} if embedding_dims else {}),
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection_name,
                    "path": qdrant_path,
                    "on_disk": True,
                    **({"embedding_model_dims": embedding_dims} if embedding_dims else {}),
                },
            },
            "version": "v1.1",
        }
        if custom_instructions:
            config["custom_instructions"] = custom_instructions

        self.memory = Memory.from_config(config)
        self._configure_embedding_client()
        self._configure_request_gates()
        self._mem0_config = config  # store for checkpoint restore

        # Track success/failure metadata for each memory ID
        # (mem0 doesn't natively track this, we maintain it ourselves)
        self._id_metadata: Dict[str, Dict[str, Any]] = {}

        # Stub attributes that runners may probe via hasattr/getattr
        self._q_cache: Dict[str, float] = {}

        logger.info(
            "[Mem0MemoryService] initialized: model=%s, embed=%s, qdrant=%s, infer=%s, search_all_scopes=%s",
            llm_model, embed_model, qdrant_path, infer, self._search_all_scopes,
        )

    # ------------------------------------------------------------------
    # Retrieval interface (called by all runners)
    # ------------------------------------------------------------------

    def retrieve_query(
        self,
        task_description: str,
        k: Optional[int] = None,
        threshold: float = 0.0,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
        """Retrieve relevant memories for a task description.

        Returns:
            Tuple of (result_dict, sim_list) matching MemoryService contract:
            - result_dict: {"actions": [...], "selected": [...], "candidates": [...], "simmax": float}
            - sim_list: [(content, score), ...]
        """
        k = k if k is not None else self.top_k

        try:
            self._wait_for_request_slot()
            # Mem0 >=2.0 requires an entity scope under ``filters``. A
            # checkpoint must retain the stable user_id it was written with;
            # unscoped queries are rejected and could mix independent runs.
            try:
                search_kwargs = {
                    "top_k": k,
                    "threshold": max(0.0, float(threshold)),
                    "filters": {"user_id": self.user_id},
                }
                results = self.memory.search(task_description, **search_kwargs)
            except TypeError:
                # Legacy Mem0 accepted the entity scope as a top-level arg.
                results = self.memory.search(
                    task_description, limit=k, user_id=self.user_id
                )
        except Exception as e:
            logger.warning("[Mem0MemoryService] search failed: %s", e)
            return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}, []

        selected = []
        sim_list = []
        simmax = 0.0

        result_items = results.get("results", []) if isinstance(results, dict) else results
        for item in result_items or []:
            mem_id = item.get("id", "")
            score = item.get("score", 0.0)
            content = item.get("memory", "")

            if score < threshold:
                continue

            simmax = max(simmax, score)
            meta_dict = self._id_metadata.get(mem_id, {"success": True})
            selected.append({
                "memory_id": mem_id,
                "content": content,
                "similarity": score,
                "metadata": _Mem0Metadata(meta_dict),
                "q_estimate": 0.0,
                "score": score,
            })
            sim_list.append((content, score))

        return {
            "actions": [s["memory_id"] for s in selected],
            "selected": selected,
            "candidates": selected,
            "simmax": simmax,
        }, sim_list

    # ------------------------------------------------------------------
    # Memory addition interface
    # ------------------------------------------------------------------

    def add_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[Any],
        successes: List[bool],
        retrieved_memory_queries: Optional[List] = None,
        retrieved_memory_ids_list: Optional[List] = None,
        metadatas: Optional[List[Dict]] = None,
    ) -> List[Optional[Tuple[str, str]]]:
        """Add memories from batch of episodes.

        Each episode's trajectory is formatted and passed to mem0.add().
        When infer=True, mem0 uses its LLM to extract atomic facts.
        """
        results = []
        event_counts: Dict[str, int] = {}
        for i, (desc, traj, success) in enumerate(zip(task_descriptions, trajectories, successes)):
            try:
                text = self._format_trajectory(desc, traj, success)

                # Preserve benchmark metadata supplied by the runner. In particular,
                # BCB relies on outcome/task_id to distinguish failure reflections from
                # successful procedures when formatting retrieved memories. The old
                # adapter discarded these fields, causing every atomic fact to be
                # injected as an UNKNOWN/SUCCESS_PROCEDURE memory.
                supplied_meta = (
                    dict(metadatas[i])
                    if metadatas is not None and i < len(metadatas) and metadatas[i]
                    else {}
                )
                memory_meta = dict(supplied_meta)
                memory_meta.update({
                    "success": bool(success),
                    "outcome": "success" if success else "failure",
                    "outcome_success": bool(success),
                    "task_description": desc[:2000],
                })

                self._wait_for_request_slot()
                add_result = self.memory.add(
                    text,
                    # ALFWorld trajectories are procedures. The normal factual
                    # path treated them as containing no facts and returned noop.
                    # Keep restored and newly-added ALFWorld memories in
                    # the checkpoint's original searchable partition.
                    user_id=self.user_id,
                    metadata=memory_meta,
                    infer=self.infer,
                    memory_type="procedural_memory",
                )

                # Mem0 may return ADD/UPDATE/DELETE events. Keep metadata for every
                # live ID it reports, not only newly-created memories, so conflict
                # resolution cannot leave stale success/failure labels behind.
                events = add_result.get("results", []) if isinstance(add_result, dict) else []
                live_ids: List[str] = []
                for item in events:
                    event = str(item.get("event") or item.get("action") or "add").lower()
                    event_counts[event] = event_counts.get(event, 0) + 1
                    mid = item.get("id")
                    if not mid:
                        continue
                    mid = str(mid)
                    if event == "delete":
                        self._id_metadata.pop(mid, None)
                        continue
                    self._id_metadata[mid] = dict(memory_meta)
                    live_ids.append(mid)

                first_id = live_ids[0] if live_ids else None
                # Preserve the MemoryService.add_memories return contract:
                # one (task_description, memory_id) entry per input item.  Mem0 may
                # legitimately produce no live ID (noop/delete) or a transient item
                # failure; represent that as (desc, None) instead of a bare None so
                # the batch runner can retain the trajectory and continue safely.
                results.append((desc, first_id))
                if not events:
                    event_counts["noop"] = event_counts.get("noop", 0) + 1
            except Exception as e:
                logger.warning(
                    "[Mem0MemoryService] add_memories failed for item %d: %s", i, e
                )
                event_counts["error"] = event_counts.get("error", 0) + 1
                results.append((desc, None))

        logger.info(
            "[Mem0MemoryService] add_memories: %d items, %d changed, events=%s",
            len(task_descriptions),
            sum(1 for _, memory_id in results if memory_id is not None),
            event_counts,
        )
        return results

    def add_memory(
        self,
        task_description: str,
        trajectory: Any,
        success: bool,
        retrieved_memory_query: Optional[List] = None,
        retrieved_memory_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Add a single memory entry (WebShop compatibility)."""
        results = self.add_memories(
            task_descriptions=[task_description],
            trajectories=[trajectory],
            successes=[success],
            retrieved_memory_queries=[retrieved_memory_query],
            retrieved_memory_ids_list=[retrieved_memory_ids],
            metadatas=[metadata] if metadata else None,
        )
        if results and results[0]:
            return results[0][1]
        return None

    # ------------------------------------------------------------------
    # Q-value interface (no-op for Mem0)
    # ------------------------------------------------------------------

    def update_values(
        self,
        successes: List[float],
        retrieved_ids_list: List[List[str]],
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """No-op: Mem0 does not use Q-values."""
        return {}

    def update_value(self, *args, **kwargs):
        """No-op: Mem0 does not use Q-values."""
        return None

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def save_checkpoint_snapshot(
        self, target_dir: str, ckpt_id: Optional[str] = None
    ) -> str:
        """Persist mem0 state by copying the qdrant directory."""
        target = Path(target_dir) / "snapshot"
        if ckpt_id is not None:
            target = target / str(ckpt_id)
        target = target / "mem0_qdrant"
        target.parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.qdrant_path):
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(self.qdrant_path, target)

        # Also save id_metadata
        meta_path = target.parent / "mem0_id_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._id_metadata, f)

        # Write snapshot_meta.json alongside so the BCB runner's AUTO-RESUME
        # scanner (which looks for epoch<N>/snapshot/<N>/snapshot_meta.json)
        # can find the mem0 checkpoint. Without this, mem0 always restarts
        # from E1 after preemption because the scan misses its layout.
        marker = target.parent / "snapshot_meta.json"
        try:
            with open(marker, "w", encoding="utf-8") as f:
                json.dump({"ckpt_id": ckpt_id, "backend": "mem0"}, f)
        except Exception as e:
            logger.warning("[Mem0MemoryService] failed to write snapshot_meta.json: %s", e)

        logger.info("[Mem0MemoryService] checkpoint saved to %s", target)
        return str(target)

    def load_checkpoint_snapshot(self, path: str, local_cache_dir: Optional[str] = None, **kwargs) -> int:
        """Restore mem0 state from a checkpoint directory.

        Uses atomic swap to avoid data loss if copy fails midway.
        """
        source_qdrant = Path(path) / "mem0_qdrant"
        source_meta = Path(path) / "mem0_id_metadata.json"

        if not source_qdrant.exists():
            for candidate in [
                Path(path),
                Path(path) / "snapshot",
            ]:
                if (candidate / "mem0_qdrant").exists():
                    source_qdrant = candidate / "mem0_qdrant"
                    source_meta = candidate / "mem0_id_metadata.json"
                    break

        if not source_qdrant.exists():
            logger.warning("[Mem0MemoryService] checkpoint not found at %s", path)
            return 0

        # Atomic restore: copy to temp, then swap
        import tempfile

        tmp_restore = tempfile.mkdtemp(prefix="mem0_restore_")
        tmp_qdrant = os.path.join(tmp_restore, "qdrant")
        try:
            shutil.copytree(source_qdrant, tmp_qdrant)
        except Exception as e:
            logger.error("[Mem0MemoryService] copy from checkpoint failed: %s", e)
            shutil.rmtree(tmp_restore, ignore_errors=True)
            return 0

        # Close current local Qdrant clients before replacing their storage.
        # QdrantLocal otherwise retains portalocker handles across re-init.
        for store_name in ("vector_store", "_telemetry_vector_store"):
            store = getattr(self.memory, store_name, None)
            client = getattr(store, "client", None)
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Failed to close %s before restore", store_name, exc_info=True)

        # Swap: remove old, move new into place
        old_path = self.qdrant_path + ".old"
        if os.path.exists(old_path):
            shutil.rmtree(old_path, ignore_errors=True)
        if os.path.exists(self.qdrant_path):
            os.rename(self.qdrant_path, old_path)
        os.rename(tmp_qdrant, self.qdrant_path)
        # Cleanup
        shutil.rmtree(old_path, ignore_errors=True)
        shutil.rmtree(tmp_restore, ignore_errors=True)

        # Restore metadata
        if source_meta.exists():
            try:
                with open(source_meta, "r", encoding="utf-8") as f:
                    self._id_metadata = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("[Mem0MemoryService] failed to load metadata: %s", e)
                self._id_metadata = {}

        # Re-initialize mem0 Memory with restored qdrant using stored config
        from mem0 import Memory

        config = copy.deepcopy(self._mem0_config)
        # Preserve collection/on-disk/dimension settings on restore.
        config["vector_store"]["config"]["path"] = self.qdrant_path
        self.memory = Memory.from_config(config)
        self._configure_embedding_client()
        self._configure_request_gates()

        n_loaded = len(self._id_metadata)
        logger.info(
            "[Mem0MemoryService] checkpoint loaded from %s (%d memories)",
            path, n_loaded,
        )
        # The ALFWorld runner interprets an integer return value as a checkpoint
        # *section ID*, not a memory count.  Returning n_loaded here made an E8
        # snapshot look like section 44718 and skipped the entire continuation.
        # Preserve runner-compatible semantics when the checkpoint directory is
        # named with a numeric epoch; otherwise return 0 so the runner resolves
        # the resume position from the snapshot path itself.
        try:
            return int(Path(path).name) if Path(path).name.isdigit() else 0
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # Stubs for runner compatibility (hasattr/getattr probes)
    # ------------------------------------------------------------------

    def set_current_epoch(self, epoch: int, num_epochs: int = 10):
        """No-op: Mem0 has no exploration schedule."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure_request_gates(self) -> None:
        """Throttle every Mem0-owned LLM/embedding call, not only add/search.

        Mem0 constructs private OpenAI clients, so the normal MemRL provider
        limiters do not see its internal fact-extraction, graph, and embedding
        requests. Wrap the Mem0 adapter entry points with the same process-wide
        gate so MEMRL_MEM0_MIN_INTERVAL applies to each actual sub-request.
        """
        llm = getattr(self.memory, "llm", None)
        generate = getattr(llm, "generate_response", None)
        if callable(generate) and not getattr(generate, "_memrl_rate_limited", False):
            original_generate = generate

            def rate_limited_generate(*args, **kwargs):
                self._wait_for_request_slot("chat")
                return original_generate(*args, **kwargs)

            rate_limited_generate._memrl_rate_limited = True
            llm.generate_response = rate_limited_generate

        # Embeddings are gated at the OpenAI transport below.  That captures
        # embed(), embed_batch(), entity boost, and version-specific internal
        # paths exactly once per outbound request.

        logger.info(
            "[Mem0MemoryService] internal LLM/embedding request gate configured: %ss",
            os.environ.get("MEMRL_MEM0_MIN_INTERVAL", "0"),
        )

    def _configure_embedding_client(self) -> None:
        """Wrap every Mem0 embedding transport call with the shared gate.

        This is intentionally below Mem0's public ``embed`` APIs: mem0 versions
        have internal batch/entity paths that bypass those methods.  Gating the
        OpenAI ``embeddings.create`` transport captures all of them.  Native
        dimensions are also preserved when the compatibility flag is enabled.
        """
        embedder = getattr(self.memory, "embedding_model", None)
        client = getattr(embedder, "client", None)
        embeddings = getattr(client, "embeddings", None)
        create = getattr(embeddings, "create", None)
        if not callable(create):
            logger.warning("[Mem0MemoryService] embedding create() method was not found; shared embed gate unavailable")
            return
        if getattr(create, "_memrl_mem0_transport_gated", False):
            return
        original_create = create

        def create_gated(*args, **kwargs):
            self._wait_for_request_slot("embed")
            if self._omit_embedding_dimensions:
                kwargs.pop("dimensions", None)
            return original_create(*args, **kwargs)

        create_gated._memrl_mem0_transport_gated = True
        create_gated._memrl_original = original_create
        embeddings.create = create_gated
        logger.info(
            "[Mem0MemoryService] all embedding transports use shared cross-process gate%s",
            " and native dimensions" if self._omit_embedding_dimensions else "",
        )

    @staticmethod
    def _format_trajectory(
        task_description: str, trajectory: Any, success: bool
    ) -> str:
        """Format an episode trajectory into text suitable for mem0.add().

        Handles both message-list format (ALFWorld) and string format (other benchmarks).
        """
        outcome = "SUCCESS" if success else "FAILURE"
        header = f"Task: {task_description}\nOutcome: {outcome}\n\n"

        if isinstance(trajectory, list):
            # Message list format (ALFWorld ReAct dialogue)
            parts = []
            for msg in trajectory:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    # Truncate very long system prompts
                    if role == "system" and len(content) > 500:
                        content = content[:500] + "..."
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(msg))
            body = "\n".join(parts)
        elif isinstance(trajectory, str):
            body = trajectory
        else:
            body = str(trajectory)

        # Cap total length to avoid overwhelming mem0's LLM extraction
        max_len = 8000
        if len(body) > max_len:
            body = body[:max_len] + "\n... (truncated)"

        return header + body
