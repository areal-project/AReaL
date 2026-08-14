#!/usr/bin/env python3
"""Run ALFWorld Mem0 with required offline BM25 and safe legacy-snapshot migration.

Old Qwen72B snapshots contain a dense-only local Qdrant collection.  The normal
Mem0 restore copies that database verbatim, so merely installing fastembed does
not enable keyword retrieval.  This wrapper patches the Mem0 Qdrant adapter to
use the offline Qdrant/bm25 cache and, immediately after restore, migrates the
live node-local database to a fresh collection with a ``bm25`` sparse slot.
The immutable checkpoint under the experiment destination is never modified.
"""
from __future__ import annotations

import copy
import logging
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("alfworld_mem0_bm25")
CACHE = os.environ.get(
    "FASTEMBED_CACHE_PATH",
    "/storage/openpsi/users/yl/agent-memory/MemRL/scripts/fastembed_cache",
)
FASTEMBED_SITE_PACKAGES = os.environ.get("FASTEMBED_SITE_PACKAGES", "")


def _enable_fastembed_path() -> None:
    """Expose target-installed fastembed without overriding runtime packages."""
    if FASTEMBED_SITE_PACKAGES and FASTEMBED_SITE_PACKAGES not in sys.path:
        # Append, never prepend: the base image's transformers/tokenizers pair must
        # remain authoritative.  We only need this fallback path to locate fastembed.
        sys.path.append(FASTEMBED_SITE_PACKAGES)


def _patch_mem0_qdrant() -> None:
    from fastembed import SparseTextEmbedding
    from mem0.vector_stores.qdrant import Qdrant

    original_create_col = Qdrant.create_col

    def get_bm25_encoder(self):
        if getattr(self, "_bm25_encoder", None) is None:
            try:
                self._bm25_encoder = SparseTextEmbedding(
                    model_name="Qdrant/bm25",
                    cache_dir=CACHE,
                    local_files_only=True,
                )
                probe = list(self._bm25_encoder.embed(["alfworld mem0 bm25 precheck"]))
                if not probe or not len(probe[0].indices):
                    raise RuntimeError("Qdrant/bm25 probe produced an empty sparse vector")
                logging.getLogger("mem0.vector_stores.qdrant").info(
                    "BM25 encoder loaded (offline Qdrant/bm25 cache=%s)", CACHE
                )
            except Exception as exc:
                self._bm25_encoder = False
                raise RuntimeError(f"Mem0 BM25 is required but failed to load: {exc}") from exc
        return self._bm25_encoder if self._bm25_encoder is not False else None

    def create_col_with_required_slot(self, *args, **kwargs):
        result = original_create_col(self, *args, **kwargs)
        info = self.client.get_collection(self.collection_name)
        sparse = getattr(info.config.params, "sparse_vectors", None)
        self._has_bm25_slot = bool(sparse and "bm25" in sparse)
        return result

    Qdrant._get_bm25_encoder = get_bm25_encoder
    Qdrant.create_col = create_col_with_required_slot


def _point_text(payload: dict) -> str:
    return str(
        payload.get("text_lemmatized")
        or payload.get("data")
        or payload.get("memory")
        or ""
    )


def _encode_sparse(store, text: str):
    from qdrant_client.models import SparseVector

    encoder = store._get_bm25_encoder()
    encoded = list(encoder.embed([text]))
    if not encoded or not len(encoded[0].indices):
        raise RuntimeError("BM25 encoding returned no terms")
    return SparseVector(
        indices=encoded[0].indices.tolist(),
        values=encoded[0].values.tolist(),
    )


def _close_memory_clients(memory) -> None:
    for name in ("vector_store", "_telemetry_vector_store"):
        store = getattr(memory, name, None)
        close = getattr(getattr(store, "client", None), "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Failed to close %s", name, exc_info=True)


def _migrate_or_backfill(service) -> tuple[int, int, bool]:
    """Return (point_count, sparse_count, migrated_legacy_collection)."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        Modifier,
        PointStruct,
        SparseVectorParams,
        VectorParams,
    )
    from mem0 import Memory

    store = getattr(service.memory, "vector_store", None)
    if store is None:
        raise RuntimeError("Mem0 vector store is unavailable after restore")
    if store._get_bm25_encoder() is None:
        raise RuntimeError("Mem0 BM25 encoder is unavailable")

    collection = store.collection_name
    info = store.client.get_collection(collection)
    sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
    has_slot = bool(sparse_cfg and "bm25" in sparse_cfg)

    points = []
    offset = None
    while True:
        page, offset = store.client.scroll(
            collection_name=collection,
            limit=64,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(page)
        if offset is None:
            break

    if has_slot:
        updates = []
        for point in points:
            payload = point.payload or {}
            text = _point_text(payload)
            if not text:
                continue
            vectors = dict(point.vector) if isinstance(point.vector, dict) else {"": point.vector}
            vectors["bm25"] = _encode_sparse(store, text)
            updates.append(PointStruct(id=point.id, vector=vectors, payload=payload))
        if updates:
            store.client.upsert(collection_name=collection, points=updates, wait=True)
        store._has_bm25_slot = True
        return len(points), len(updates), False

    # Qdrant cannot add a sparse-vector slot to an existing local collection.
    # Rebuild only the live node-local copy; the source snapshot stays immutable.
    live_path = Path(service.qdrant_path)
    tmp_root = Path(tempfile.mkdtemp(prefix="mem0_bm25_migrate_", dir=str(live_path.parent)))
    tmp_db = tmp_root / "qdrant"
    new_client = QdrantClient(path=str(tmp_db))
    dense_params = info.config.params.vectors
    if isinstance(dense_params, dict):
        dense_config = dense_params
    else:
        dense_config = VectorParams(
            size=int(getattr(dense_params, "size", service._mem0_config["vector_store"]["config"].get("embedding_model_dims", 4096))),
            distance=getattr(dense_params, "distance", Distance.COSINE),
            on_disk=getattr(dense_params, "on_disk", True),
        )
    new_client.create_collection(
        collection_name=collection,
        vectors_config=dense_config,
        sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
    )
    migrated = []
    for point in points:
        payload = point.payload or {}
        text = _point_text(payload)
        if not text:
            raise RuntimeError(f"Legacy Mem0 point {point.id} has no searchable text")
        vectors = dict(point.vector) if isinstance(point.vector, dict) else {"": point.vector}
        vectors["bm25"] = _encode_sparse(store, text)
        migrated.append(PointStruct(id=point.id, vector=vectors, payload=payload))
        if len(migrated) >= 64:
            new_client.upsert(collection_name=collection, points=migrated, wait=True)
            migrated.clear()
    if migrated:
        new_client.upsert(collection_name=collection, points=migrated, wait=True)
    new_client.close()

    _close_memory_clients(service.memory)
    backup = Path(str(live_path) + ".dense_legacy")
    if backup.exists():
        shutil.rmtree(backup)
    os.rename(live_path, backup)
    try:
        os.rename(tmp_db, live_path)
    except Exception:
        os.rename(backup, live_path)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(tmp_root, ignore_errors=True)

    config = copy.deepcopy(service._mem0_config)
    config["vector_store"]["config"]["path"] = str(live_path)
    service.memory = Memory.from_config(config)
    service._configure_embedding_client()
    service._configure_request_gates()
    new_store = service.memory.vector_store
    new_info = new_store.client.get_collection(collection)
    new_sparse = getattr(new_info.config.params, "sparse_vectors", None)
    new_store._has_bm25_slot = bool(new_sparse and "bm25" in new_sparse)
    if not new_store._has_bm25_slot:
        raise RuntimeError("Migrated Mem0 collection still has no bm25 sparse slot")
    return len(points), len(points), True


def _verify_sparse_query(service) -> int:
    from qdrant_client.models import SparseVector

    store = service.memory.vector_store
    points, _ = store.client.scroll(
        collection_name=store.collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        logger.info("[Mem0 BM25] empty restored collection; sparse query verification deferred")
        return 0
    text = _point_text(points[0].payload or {})
    if not text:
        raise RuntimeError("BM25 verification point has no searchable text")
    query = list(store._get_bm25_encoder().query_embed([text]))[0]
    result = store.client.query_points(
        collection_name=store.collection_name,
        query=SparseVector(indices=query.indices.tolist(), values=query.values.tolist()),
        using="bm25",
        limit=3,
    )
    hits = len(result.points)
    if hits < 1:
        raise RuntimeError("BM25 sparse query returned no hits after migration/backfill")
    return hits


def _patch_resume_migration() -> None:
    from memrl.service.mem0_memory_service import Mem0MemoryService

    original = Mem0MemoryService.load_checkpoint_snapshot
    if getattr(original, "_memrl_bm25_migration", False):
        return

    def load_and_migrate(self, *args, **kwargs):
        # Mem0 deliberately returns ``0`` for non-numeric snapshot directory
        # names (e.g. batch snapshots ``s9_b10``), because that value is a
        # checkpoint *section* identifier rather than a loaded-memory count.
        # The batch resume position is resolved by AlfworldRunner from the
        # directory name.  Treat a populated restored metadata ledger as the
        # success signal here; returning the original 0 keeps the runner's
        # path-derived batch parsing intact.
        loaded = original(self, *args, **kwargs)
        restored_memories = len(getattr(self, "_id_metadata", {}) or {})
        if not loaded and restored_memories <= 0:
            raise RuntimeError("Mem0 checkpoint restore returned no memories")
        points, sparse, migrated = _migrate_or_backfill(self)
        hits = _verify_sparse_query(self)
        logger.info(
            "[Mem0 BM25] RESUME_MIGRATION_OK loaded=%s points=%d sparse_vectors=%d "
            "query_hits=%d migrated_legacy=%s collection=%s live_path=%s",
            loaded,
            points,
            sparse,
            hits,
            migrated,
            self.memory.vector_store.collection_name,
            self.qdrant_path,
        )
        print(
            f"[Mem0] BM25_RESUME_OK points={points} sparse_vectors={sparse} "
            f"query_hits={hits} migrated_legacy={str(migrated).lower()}",
            flush=True,
        )
        return loaded

    load_and_migrate._memrl_bm25_migration = True
    Mem0MemoryService.load_checkpoint_snapshot = load_and_migrate


def _precheck_bm25() -> None:
    from fastembed import SparseTextEmbedding

    encoder = SparseTextEmbedding(
        model_name="Qdrant/bm25", cache_dir=CACHE, local_files_only=True
    )
    probe = list(encoder.embed(["offline alfworld mem0 bm25 startup verification"]))
    if not probe or not len(probe[0].indices):
        raise RuntimeError("BM25 startup verification returned no sparse terms")
    print(
        f"[Mem0] BM25_PRECHECK_OK cache={CACHE} terms={len(probe[0].indices)}",
        flush=True,
    )


def install_bm25_patches() -> None:
    """Install required Mem0 BM25 patches without starting the benchmark."""
    _enable_fastembed_path()
    _patch_mem0_qdrant()
    _patch_resume_migration()
    _precheck_bm25()


if __name__ == "__main__":
    install_bm25_patches()
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "run" / "run_alfworld.py"),
        run_name="__main__",
    )
