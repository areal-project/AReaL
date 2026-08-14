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

import json
import logging
import os
import re
import shutil
import threading
import time
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
    def _wait_for_request_slot(cls) -> None:
        try:
            interval = float(os.environ.get("MEMRL_MEM0_MIN_INTERVAL", "0") or "0")
        except (TypeError, ValueError):
            interval = 0.0
        if interval <= 0:
            return
        while True:
            with cls._request_gate_lock:
                now = time.monotonic()
                if now >= cls._request_gate_next_at:
                    cls._request_gate_next_at = now + interval
                    return
                delay = cls._request_gate_next_at - now
            time.sleep(delay)

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
        self._mem0_config = config  # store for checkpoint restore

        # Track success/failure metadata for each memory ID
        # (mem0 doesn't natively track this, we maintain it ourselves)
        self._id_metadata: Dict[str, Dict[str, Any]] = {}

        # Stub attributes that runners may probe via hasattr/getattr
        self._q_cache: Dict[str, float] = {}

        logger.info(
            "[Mem0MemoryService] initialized: model=%s, embed=%s, qdrant=%s, infer=%s",
            llm_model, embed_model, qdrant_path, infer,
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
            results = self.memory.search(
                task_description,
                filters={"user_id": self.user_id},
                top_k=k,
            )
        except Exception as e:
            logger.warning("[Mem0MemoryService] search failed: %s", e)
            return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}, []

        selected = []
        sim_list = []
        simmax = 0.0

        for item in results.get("results", []):
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
    ) -> List[Tuple[str, Optional[str]]]:
        """Add memories from batch of episodes.

        Each episode's trajectory is formatted and passed to mem0.add().
        When infer=True, mem0 uses its LLM to extract atomic facts.

        The runner relies on a positional result for every input episode and
        unconditionally unpacks each item as ``(task_description, memory_id)``.
        Keep that contract even when Mem0 returns no live ID or one item fails;
        a failed/no-op item is represented as ``(task_description, None)`` so a
        transient provider error cannot abort the rest of the mini-batch.
        """
        results: List[Tuple[str, Optional[str]]] = []
        event_counts: Dict[str, int] = {}
        for i, desc in enumerate(task_descriptions):
            try:
                traj = trajectories[i]
                success = successes[i]
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
                    user_id=self.user_id,
                    metadata=memory_meta,
                    infer=self.infer,
                )

                # Mem0 may return ADD/UPDATE/DELETE events. Keep metadata for every
                # live ID it reports, not only newly-created memories, so conflict
                # resolution cannot leave stale success/failure labels behind.
                events = add_result.get("results") or []
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
                results.append((desc, first_id))
                if not events:
                    event_counts["noop"] = event_counts.get("noop", 0) + 1
            except Exception as e:
                logger.warning(
                    "[Mem0MemoryService] add_memories failed for item %d; "
                    "preserving positional result with mem_id=None: %s",
                    i,
                    e,
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

    @staticmethod
    def _mem0_checkpoint_id(path: Path, marker: dict) -> int:
        """Return the completed section id; never confuse it with memory count."""
        raw = marker.get("checkpoint_id", marker.get("ckpt_id", path.name))
        text = str(raw or path.name)
        match = re.fullmatch(r"(\d+)(?:_b\d+)?", text)
        if not match:
            raise ValueError(f"invalid Mem0 checkpoint id: {raw!r}")
        return int(match.group(1))

    @staticmethod
    def _validate_mem0_snapshot(path: Path) -> tuple[dict, dict]:
        marker_path = path / "snapshot_meta.json"
        metadata_path = path / "mem0_id_metadata.json"
        qdrant_path = path / "mem0_qdrant"
        if not marker_path.is_file() or marker_path.stat().st_size <= 0:
            raise ValueError(f"missing/empty Mem0 marker: {marker_path}")
        if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
            raise ValueError(f"missing/empty Mem0 metadata: {metadata_path}")
        if not qdrant_path.is_dir() or not any(
            item.is_file() and item.stat().st_size > 0 for item in qdrant_path.rglob("*")
        ):
            raise ValueError(f"missing/empty Mem0 qdrant snapshot: {qdrant_path}")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid Mem0 checkpoint JSON at {path}: {exc}") from exc
        if marker.get("backend") != "mem0":
            raise ValueError(f"not a Mem0 checkpoint: {path}")
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError(f"empty/invalid Mem0 id metadata: {metadata_path}")
        Mem0MemoryService._mem0_checkpoint_id(path, marker)
        return marker, metadata

    def save_checkpoint_snapshot(
        self, target_dir: str, ckpt_id: Optional[str] = None
    ) -> str:
        """Persist Mem0 state atomically without replacing permanent snapshots."""
        snapshot_root = Path(target_dir) / "snapshot"
        target = snapshot_root / str(ckpt_id) if ckpt_id is not None else snapshot_root
        snapshot_root.mkdir(parents=True, exist_ok=True)

        # Never overwrite/remove an existing permanent checkpoint. A healthy existing
        # snapshot is idempotent; an unhealthy collision is surfaced for investigation.
        if target.exists():
            self._validate_mem0_snapshot(target)
            logger.info("[Mem0MemoryService] checkpoint already healthy; keeping %s", target)
            return str(target / "mem0_qdrant")
        if not os.path.isdir(self.qdrant_path):
            raise FileNotFoundError(f"live Mem0 qdrant path missing: {self.qdrant_path}")
        if not self._id_metadata:
            raise ValueError("refusing to save Mem0 checkpoint with empty id metadata")

        import tempfile
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(snapshot_root)))
        try:
            shutil.copytree(self.qdrant_path, staging / "mem0_qdrant")
            with open(staging / "mem0_id_metadata.json", "w", encoding="utf-8") as f:
                json.dump(self._id_metadata, f)
                f.flush()
                os.fsync(f.fileno())
            marker = {
                "ckpt_id": str(ckpt_id) if ckpt_id is not None else target.name,
                "checkpoint_id": str(ckpt_id) if ckpt_id is not None else target.name,
                "backend": "mem0",
                "visible_memories": len(self._id_metadata),
            }
            with open(staging / "snapshot_meta.json", "w", encoding="utf-8") as f:
                json.dump(marker, f)
                f.flush()
                os.fsync(f.fileno())
            self._validate_mem0_snapshot(staging)
            os.rename(staging, target)  # same filesystem; atomic and target must not exist
        except Exception:
            # Staging is job-created scratch inside this experiment only. Never touch
            # any existing checkpoint directory under permanent storage.
            shutil.rmtree(staging, ignore_errors=True)
            raise

        logger.info("[Mem0MemoryService] checkpoint saved atomically to %s", target)
        return str(target / "mem0_qdrant")

    def load_checkpoint_snapshot(self, path: str) -> int:
        """Strictly restore Mem0 and return checkpoint/section id, not memory count."""
        source = Path(path)
        candidates = [source]
        if source.name == "mem0_qdrant":
            candidates.insert(0, source.parent)
        candidates.extend([source / "snapshot"])
        selected = None
        last_error = None
        for candidate in candidates:
            try:
                marker, metadata = self._validate_mem0_snapshot(candidate)
                selected = candidate
                break
            except Exception as exc:
                last_error = exc
        if selected is None:
            raise ValueError(f"no healthy Mem0 checkpoint at {path}: {last_error}")

        source_qdrant = selected / "mem0_qdrant"
        import tempfile
        tmp_restore = Path(tempfile.mkdtemp(prefix="mem0_restore_"))
        tmp_qdrant = tmp_restore / "qdrant"
        old_path = Path(self.qdrant_path + ".old")
        live_path = Path(self.qdrant_path)
        try:
            shutil.copytree(source_qdrant, tmp_qdrant)
            if old_path.exists():
                shutil.rmtree(old_path, ignore_errors=True)
            if live_path.exists():
                os.rename(live_path, old_path)
            os.rename(tmp_qdrant, live_path)
        except Exception:
            if not live_path.exists() and old_path.exists():
                os.rename(old_path, live_path)
            raise
        finally:
            shutil.rmtree(tmp_restore, ignore_errors=True)

        try:
            # Release all QdrantLocal handles owned by the old Memory instance
            # before constructing the restored one. Entity store shares the main
            # client; telemetry is disabled by the launcher, but close it too for
            # compatibility with older Mem0 builds.
            seen_clients = set()
            for store_name in ("vector_store", "_entity_store", "_telemetry_vector_store"):
                store = getattr(self.memory, store_name, None)
                client = getattr(store, "client", None)
                if client is None or id(client) in seen_clients:
                    continue
                seen_clients.add(id(client))
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            close_db = getattr(getattr(self.memory, "db", None), "close", None)
            if callable(close_db):
                close_db()

            self._id_metadata = metadata
            from mem0 import Memory
            import copy
            config = copy.deepcopy(self._mem0_config)
            vector_store = config.setdefault("vector_store", {})
            vector_store["provider"] = "qdrant"
            vector_cfg = vector_store.setdefault("config", {})
            vector_cfg.update({
                "collection_name": self.collection_name,
                "path": self.qdrant_path,
                "on_disk": True,
            })
            # Preserve embedding_model_dims and every other original vector setting.
            self.memory = Memory.from_config(config)
        except Exception:
            if live_path.exists():
                shutil.rmtree(live_path, ignore_errors=True)
            if old_path.exists():
                os.rename(old_path, live_path)
            raise
        else:
            shutil.rmtree(old_path, ignore_errors=True)

        checkpoint_id = self._mem0_checkpoint_id(selected, marker)
        logger.info(
            "[Mem0MemoryService] checkpoint loaded from %s (checkpoint_id=%d, memories=%d)",
            selected, checkpoint_id, len(self._id_metadata),
        )
        return checkpoint_id

    # ------------------------------------------------------------------
    # Stubs for runner compatibility (hasattr/getattr probes)
    # ------------------------------------------------------------------

    def set_current_epoch(self, epoch: int, num_epochs: int = 10):
        """No-op: Mem0 has no exploration schedule."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
