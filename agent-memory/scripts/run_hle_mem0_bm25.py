#!/usr/bin/env python3
"""Launch HLE Mem0 with an offline FastEmbed BM25 encoder and resume backfill.

This wrapper is intentionally Mem0-only. It patches the installed mem0 Qdrant
adapter before importing run/run_hle.py, then backfills sparse vectors after a
checkpoint restore so pre-BM25 points participate in hybrid retrieval.
"""
from __future__ import annotations

import logging
import os
import runpy
from pathlib import Path

logger = logging.getLogger("hle_mem0_bm25")
CACHE = os.environ.get(
    "FASTEMBED_CACHE_PATH",
    "/storage/openpsi/users/yl/agent-memory/MemRL/scripts/fastembed_cache",
)


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
                # Force one real encoding, rather than treating import as proof.
                probe = list(self._bm25_encoder.embed(["mem0 bm25 precheck"]))
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

    if store is None or not getattr(store, "_has_bm25_slot", False):
        raise RuntimeError("Mem0 collection has no 'bm25' sparse vector slot")
    # Fail early if the encoder/cache is unavailable.
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
        vectors = []
        for point in points:
            payload = point.payload or {}
            text = payload.get("text_lemmatized") or payload.get("data") or payload.get("memory") or ""
            if not text:
                continue
            sparse = store._encode_bm25(str(text))
            if sparse is not None:
                dense = point.vector if isinstance(point.vector, dict) else {"": point.vector}
                dense = dict(dense)
                dense["bm25"] = sparse
                vectors.append(PointStruct(id=point.id, vector=dense, payload=payload))
        if vectors:
            store.client.upsert(collection_name=store.collection_name, points=vectors)
            updated += len(vectors)
        if offset is None:
            break
    return updated


def _patch_resume_backfill() -> None:
    from memrl.service.mem0_memory_service import Mem0MemoryService

    original = Mem0MemoryService.load_checkpoint_snapshot
    if getattr(original, "_memrl_bm25_backfill", False):
        return

    def load_and_backfill(self, *args, **kwargs):
        loaded = original(self, *args, **kwargs)
        store = getattr(self.memory, "vector_store", None)
        updated = _backfill_store(store)
        logger.info(
            "[Mem0 BM25] checkpoint backfill complete: loaded=%d sparse_vectors=%d collection=%s",
            loaded,
            updated,
            getattr(store, "collection_name", "unknown"),
        )
        return loaded

    load_and_backfill._memrl_bm25_backfill = True
    Mem0MemoryService.load_checkpoint_snapshot = load_and_backfill


_patch_mem0_qdrant()
_patch_resume_backfill()

# A real offline encoder check before any benchmark/API work.
from fastembed import SparseTextEmbedding
_probe_encoder = SparseTextEmbedding(
    model_name="Qdrant/bm25", cache_dir=CACHE, local_files_only=True
)
_probe = list(_probe_encoder.embed(["offline bm25 startup verification"]))
if not _probe or not len(_probe[0].indices):
    raise RuntimeError("BM25 startup verification returned no sparse terms")
print(f"[Mem0] BM25_PRECHECK_OK cache={CACHE} terms={len(_probe[0].indices)}", flush=True)

runpy.run_path(str(Path(__file__).resolve().parents[1] / "run" / "run_hle.py"), run_name="__main__")
