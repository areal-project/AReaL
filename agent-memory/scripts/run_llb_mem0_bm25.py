#!/usr/bin/env python3
"""Run LLB Mem0 with required offline BM25 and safe resume backfill.

The benchmark snapshots remain immutable. run_llb first restores the selected
snapshot into node-local Qdrant storage; only that live copy is backfilled with
sparse BM25 vectors before Section N+1 starts.
"""
from __future__ import annotations

import logging
import os
import runpy
from pathlib import Path

logger = logging.getLogger("llb_mem0_bm25")
CACHE = os.environ.get(
    "FASTEMBED_CACHE_PATH",
    "/storage/openpsi/users/yl/agent-memory/MemRL/scripts/fastembed_cache",
)




def _install_rotated_matrix_credentials() -> None:
    """Inject verified Matrix credentials into MempConfig in memory only.

    No credential is printed, added to generated YAML, or written to a
    checkpoint. The patched JSON serializer redacts both credential fields
    because run_llb logs its resolved config at startup.
    """
    import json
    import yaml
    from memrl.configs.config import MempConfig

    config_path = Path(os.environ.get(
        "MEMRL_MATRIX_CREDENTIAL_CONFIG",
        "/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml",
    ))
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    mappings = {
        str(item.get("model_name")): (item.get("litellm_params") or {})
        for item in payload.get("model_list", [])
        if isinstance(item, dict)
    }

    def resolve(names):
        for name in names:
            value = mappings.get(name, {}).get("api_key")
            if isinstance(value, str) and value.startswith("os.environ/"):
                value = os.environ.get(value.split("/", 1)[1])
            if value:
                return value
        raise RuntimeError(f"No configured Matrix credential for aliases: {names}")

    # This mapping was preflighted against gpt-4.1-mini-2025-04-14 with HTTP 200.
    chat_key = resolve(("gpt-4o-2024-11-20", "gpt-4o", "gpt-5-mini"))
    embed_key = resolve(("text-embedding-3-large", "text-embedding-3-small"))

    original_from_yaml = MempConfig.from_yaml.__func__
    original_dump = MempConfig.model_dump_json

    @classmethod
    def from_yaml_with_rotated_credentials(cls, config_file):
        config = original_from_yaml(cls, config_file)
        config.llm.api_key = chat_key
        config.embedding.api_key = embed_key
        return config

    def redacted_model_dump_json(self, *args, **kwargs):
        # Preserve formatting options accepted by Pydantic while ensuring secrets
        # can never reach the startup log.
        data = self.model_dump(mode="json")
        for section in ("llm", "embedding"):
            if isinstance(data.get(section), dict) and "api_key" in data[section]:
                data[section]["api_key"] = "[REDACTED]"
        indent = kwargs.get("indent")
        return json.dumps(data, ensure_ascii=False, indent=indent)

    MempConfig.from_yaml = from_yaml_with_rotated_credentials
    MempConfig.model_dump_json = redacted_model_dump_json
    # Keep a reference for debuggability without exposing either value.
    MempConfig._memrl_original_model_dump_json = original_dump
    print("[Mem0] rotated Matrix chat+embedding credentials installed (values redacted)", flush=True)


def _patch_mem0_qdrant() -> None:
    from fastembed import SparseTextEmbedding
    from mem0.vector_stores.qdrant import Qdrant

    def get_bm25_encoder(self):
        if self._bm25_encoder is None:
            try:
                self._bm25_encoder = SparseTextEmbedding(
                    model_name="Qdrant/bm25",
                    cache_dir=CACHE,
                    local_files_only=True,
                )
                probe = list(self._bm25_encoder.embed(["llb db mem0 bm25 precheck"]))
                if not probe or not len(probe[0].indices):
                    raise RuntimeError("Qdrant/bm25 probe produced an empty sparse vector")
                logging.getLogger("mem0.vector_stores.qdrant").info(
                    "BM25 encoder loaded (fastembed Qdrant/bm25, offline cache=%s)", CACHE
                )
            except Exception as exc:
                self._bm25_encoder = False
                raise RuntimeError(f"Mem0 BM25 is required but failed to load: {exc}") from exc
        return self._bm25_encoder if self._bm25_encoder is not False else None

    Qdrant._get_bm25_encoder = get_bm25_encoder


def _backfill_store(store) -> int:
    from qdrant_client.models import PointStruct

    if store is None:
        raise RuntimeError("Mem0 vector store is unavailable after checkpoint restore")
    if not getattr(store, "_has_bm25_slot", False):
        raise RuntimeError("Mem0 collection has no 'bm25' sparse vector slot")
    if store._get_bm25_encoder() is None:
        raise RuntimeError("Mem0 BM25 encoder is unavailable")

    updated = 0
    offset = None
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=64,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        updates = []
        for point in points:
            payload = point.payload or {}
            text = (
                payload.get("text_lemmatized")
                or payload.get("data")
                or payload.get("memory")
                or ""
            )
            if not text:
                continue
            sparse = store._encode_bm25(str(text))
            if sparse is None:
                raise RuntimeError(f"BM25 encoding failed for point {point.id}")
            dense = point.vector if isinstance(point.vector, dict) else {"": point.vector}
            dense = dict(dense)
            dense["bm25"] = sparse
            updates.append(PointStruct(id=point.id, vector=dense, payload=payload))
        if updates:
            store.client.upsert(collection_name=store.collection_name, points=updates)
            updated += len(updates)
        if offset is None:
            break
    return updated


def _verify_sparse_query(store) -> int:
    from qdrant_client.models import SparseVector

    points, _ = store.client.scroll(
        collection_name=store.collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return 0
    payload = points[0].payload or {}
    text = payload.get("text_lemmatized") or payload.get("data") or payload.get("memory") or ""
    if not text:
        raise RuntimeError("BM25 verification point has no searchable text")
    encoder = store._get_bm25_encoder()
    query = list(encoder.query_embed([str(text)]))[0]
    result = store.client.query_points(
        collection_name=store.collection_name,
        query=SparseVector(indices=query.indices.tolist(), values=query.values.tolist()),
        using="bm25",
        limit=1,
    )
    hits = len(result.points)
    if hits < 1:
        raise RuntimeError("BM25 sparse query returned no hits after backfill")
    return hits


def _patch_resume_backfill() -> None:
    from memrl.service.mem0_memory_service import Mem0MemoryService

    original = Mem0MemoryService.load_checkpoint_snapshot
    if getattr(original, "_memrl_bm25_backfill", False):
        return

    def load_and_backfill(self, *args, **kwargs):
        checkpoint_id = original(self, *args, **kwargs)
        store = getattr(self.memory, "vector_store", None)
        updated = _backfill_store(store)
        hits = _verify_sparse_query(store)
        logger.info(
            "[Mem0 BM25] resume backfill verified: checkpoint_id=%s sparse_vectors=%d "
            "query_hits=%d collection=%s live_path=%s",
            checkpoint_id,
            updated,
            hits,
            getattr(store, "collection_name", "unknown"),
            self.qdrant_path,
        )
        return checkpoint_id

    load_and_backfill._memrl_bm25_backfill = True
    Mem0MemoryService.load_checkpoint_snapshot = load_and_backfill


_install_rotated_matrix_credentials()
_patch_mem0_qdrant()
_patch_resume_backfill()

# Checkpoint audit mode: validation only. Used only with a node-local copied
# snapshot and a /tmp output directory; skip all training sections.
_eval_only_section = os.environ.get("MEMRL_EVAL_ONLY_SECTION", "").strip()
if _eval_only_section:
    from memrl.run.llb_rl_runner import LLBRunner

    _section = int(_eval_only_section)

    def _eval_only_run(self):
        if not self.valid_dataset:
            raise RuntimeError("eval-only requested but validation dataset is unavailable")
        print(f"[READONLY_EVAL] starting validation only after Section {_section}", flush=True)
        self._evaluate(self.valid_dataset, "Validation", _section)
        try:
            self.writer.close()
        except Exception:
            pass
        print(f"[READONLY_EVAL] complete after Section {_section}", flush=True)

    LLBRunner.run = _eval_only_run

from fastembed import SparseTextEmbedding

_probe_encoder = SparseTextEmbedding(
    model_name="Qdrant/bm25", cache_dir=CACHE, local_files_only=True
)
_probe = list(_probe_encoder.embed(["offline llb db bm25 startup verification"]))
if not _probe or not len(_probe[0].indices):
    raise RuntimeError("BM25 startup verification returned no sparse terms")
print(f"[Mem0] BM25_PRECHECK_OK cache={CACHE} terms={len(_probe[0].indices)}", flush=True)

runpy.run_path(str(Path(__file__).resolve().parents[1] / "run" / "run_llb.py"), run_name="__main__")
