# SPDX-License-Identifier: Apache-2.0

"""Durable episode admission and tensor handoff for native online training."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from areal.elastic.infra.state.batch_queue import (
    BatchConsumer,
    BatchQueue,
    _digest,
    _encode,
)


class AdmissionFull(RuntimeError):
    """The training pipeline has no available episode credit."""


class AdmissionPaused(RuntimeError):
    """Serving weights are being reconciled with committed training state."""


class OnlineInbox:
    """Keep real episode credits and checkpoint-bound batches in one WAL database.

    A credit belongs to one reward-bounded episode, even when it contains many
    model calls or tensor rows. Source acknowledgement does not release it.
    Only confirmed publication or explicit rejection releases a credit. Actor
    workers never own this store; its consumer lock lives with the coordinator.
    """

    def __init__(self, root: Path, *, batch_size: int, max_staleness: int) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(max_staleness) is not int or max_staleness < 0:
            raise ValueError("max_staleness must be a nonnegative integer")
        self.queue = BatchQueue(root)
        self.batch_size = batch_size
        self.max_staleness = max_staleness
        self.capacity = (max_staleness + 1) * batch_size
        with self.queue._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS online_progress (singleton INTEGER PRIMARY KEY "
                "CHECK(singleton=1), format_version INTEGER NOT NULL, batch_size INTEGER "
                "NOT NULL, max_staleness INTEGER NOT NULL, published_version INTEGER NOT NULL)"
            )
            db.execute(
                "INSERT OR IGNORE INTO online_progress "
                "(singleton,format_version,batch_size,max_staleness,published_version) VALUES (1, 1, ?, ?, 0)",
                (batch_size, max_staleness),
            )
            state = db.execute("SELECT * FROM online_progress").fetchone()
            if (
                state["format_version"],
                state["batch_size"],
                state["max_staleness"],
            ) != (1, batch_size, max_staleness):
                raise ValueError(
                    "Online inbox configuration differs from durable identity"
                )
            if "admission_open" not in {
                row["name"] for row in db.execute("PRAGMA table_info(online_progress)")
            }:
                db.execute(
                    "ALTER TABLE online_progress ADD COLUMN admission_open INTEGER NOT NULL DEFAULT 1"
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS online_episodes (sequence INTEGER PRIMARY KEY "
                "AUTOINCREMENT, episode_id TEXT NOT NULL UNIQUE, source_id TEXT NOT NULL, "
                "session_id TEXT NOT NULL, trajectory_id INTEGER NOT NULL, lease TEXT NOT NULL, "
                "status TEXT NOT NULL, payload TEXT, digest TEXT, min_version INTEGER, "
                "max_version INTEGER, source_acked INTEGER NOT NULL DEFAULT 0, "
                "batch_id TEXT, rejection TEXT, UNIQUE(source_id, session_id, trajectory_id))"
            )
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(online_episodes)")
            }
            for column in ("admitted_wall_time", "received_wall_time"):
                if column not in columns:
                    db.execute(f"ALTER TABLE online_episodes ADD COLUMN {column} REAL")

    def _episode_identity(
        self, source_id: str, session_id: str, trajectory_id: int
    ) -> str:
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(session_id, str)
            or not session_id
            or type(trajectory_id) is not int
            or trajectory_id < 0
        ):
            raise ValueError(
                "Admission requires a source, session and nonnegative trajectory ID"
            )
        identity = _encode([self.queue.queue_id, source_id, session_id, trajectory_id])
        return hashlib.sha256(identity.encode()).hexdigest()

    def admit(
        self, source_id: str, session_id: str, trajectory_id: int
    ) -> dict[str, Any]:
        """Reserve once before first generation; retries reuse the same lease."""
        episode_id = self._episode_identity(source_id, session_id, trajectory_id)
        with self.queue._transaction() as db:
            if not db.execute("SELECT admission_open FROM online_progress").fetchone()[
                0
            ]:
                raise AdmissionPaused(
                    "Online training is restoring committed serving weights"
                )
            row = db.execute(
                "SELECT * FROM online_episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
            if row is None:
                active = db.execute(
                    "SELECT COUNT(*) FROM online_episodes WHERE status NOT IN ('published','rejected')"
                ).fetchone()[0]
                if active >= self.capacity:
                    raise AdmissionFull("Online training episode capacity exhausted")
                db.execute(
                    "INSERT INTO online_episodes "
                    "(episode_id,source_id,session_id,trajectory_id,lease,status,admitted_wall_time) "
                    "VALUES (?,?,?,?,?,'admitted',?)",
                    (
                        episode_id,
                        source_id,
                        session_id,
                        trajectory_id,
                        uuid.uuid4().hex,
                        time.time(),
                    ),
                )
                row = db.execute(
                    "SELECT * FROM online_episodes WHERE episode_id=?", (episode_id,)
                ).fetchone()
            return {key: row[key] for key in ("episode_id", "lease", "status")}

    def set_admission_open(self, enabled: bool) -> None:
        with self.queue._transaction() as db:
            db.execute("UPDATE online_progress SET admission_open=?", (int(enabled),))

    @staticmethod
    def _inspect_payload(payload: dict[str, Any]) -> tuple[int, int]:
        import torch

        required = (
            "input_ids",
            "attention_mask",
            "loss_mask",
            "logprobs",
            "versions",
            "rewards",
        )
        if any(not isinstance(payload.get(key), torch.Tensor) for key in required):
            raise ValueError(
                "An online trajectory must contain actual training tensors"
            )
        if any(
            value.device.type != "cpu"
            for value in payload.values()
            if isinstance(value, torch.Tensor)
        ):
            raise ValueError("Persistent trajectories must contain CPU tensors")
        shape = payload["input_ids"].shape
        if (
            len(shape) != 2
            or min(shape) < 1
            or any(payload[key].shape != shape for key in required[1:5])
        ):
            raise ValueError(
                "Online token tensors must have matching nonempty [rows,tokens] shapes"
            )
        if (
            payload["rewards"].numel() != shape[0]
            or not torch.isfinite(payload["rewards"]).all()
        ):
            raise ValueError("Each trajectory row requires a finite reward")
        versions = payload["versions"]
        if versions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Token versions must be integers")
        mask = payload["loss_mask"].bool() & payload["attention_mask"].bool()
        mask[:, 0] = False  # No next-token loss can train the first input token.
        effective = versions[mask]
        if not effective.numel() or (effective < 0).any():
            raise ValueError("Every effective generated loss token requires a version")
        if not torch.isfinite(payload["logprobs"][mask]).all():
            raise ValueError("Generated token logprobs must be finite")
        return int(effective.min()), int(effective.max())

    def receive(self, episode_id: str, lease: str, payload: dict[str, Any]) -> str:
        """Persist self-contained tensors before the caller acknowledges the source."""
        from areal.infra.rpc.serialization import serialize_value

        minimum, maximum = self._inspect_payload(payload)
        encoded = _encode(serialize_value(payload))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.queue._transaction() as db:
            row = db.execute(
                "SELECT * FROM online_episodes WHERE episode_id=? AND lease=?",
                (episode_id, lease),
            ).fetchone()
            if row is None:
                raise ValueError("Unknown episode or obsolete admission lease")
            if row["digest"] is not None:
                if row["digest"] != digest or row["payload"] != encoded:
                    raise ValueError("Episode retry contains different content")
                return digest
            discarded = (
                row["status"] == "rejected" and row["rejection"] == "recovery_rollback"
            )
            if row["status"] != "admitted" and not discarded:
                raise ValueError("Cannot receive a closed episode")
            db.execute(
                "UPDATE online_episodes SET payload=?,digest=?,min_version=?,max_version=?,"
                "status=?,received_wall_time=? WHERE episode_id=?",
                (
                    encoded,
                    digest,
                    minimum,
                    maximum,
                    "rejected" if discarded else "ready",
                    time.time(),
                    episode_id,
                ),
            )
        return digest

    def observation_rows(self) -> tuple[dict[str, Any], ...]:
        """Causal admission/receipt facts; no request-token timing is inferred here."""
        with self.queue._transaction() as db:
            rows = db.execute(
                "SELECT e.episode_id,e.sequence,e.source_id,e.status,e.payload,e.digest,"
                "e.admitted_wall_time,e.received_wall_time,b.sequence AS batch_sequence "
                "FROM online_episodes e LEFT JOIN batches b ON b.batch_id=e.batch_id "
                "WHERE e.status!='rejected' ORDER BY e.sequence"
            ).fetchall()
            return tuple(dict(row) for row in rows)

    def episode(self, episode_id: str) -> dict[str, Any]:
        with self.queue._transaction() as db:
            row = db.execute(
                "SELECT * FROM online_episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
            if row is None:
                raise KeyError(episode_id)
            return dict(row)

    def acknowledge_source(self, episode_id: str, lease: str) -> None:
        """Record the source's idempotent retirement, preserving the training credit."""
        with self.queue._transaction() as db:
            changed = db.execute(
                "UPDATE online_episodes SET source_acked=1 "
                "WHERE episode_id=? AND lease=? AND digest IS NOT NULL",
                (episode_id, lease),
            ).rowcount
            if changed != 1:
                raise ValueError("Cannot acknowledge an episode before durable receipt")

    def reject(self, episode_id: str, reason: str) -> None:
        """Release an untrained episode with a durable, explicit audit reason."""
        if not reason:
            raise ValueError("Rejection requires a reason")
        with self.queue._transaction() as db:
            row = db.execute(
                "SELECT * FROM online_episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
            if row is None or row["status"] not in ("admitted", "ready", "rejected"):
                raise ValueError("Cannot reject a reserved or committed episode")
            db.execute(
                "UPDATE online_episodes SET status='rejected', rejection=? WHERE episode_id=?",
                (reason, episode_id),
            )

    def cancel_admission(
        self,
        source_id: str,
        session_id: str,
        trajectory_id: int,
        *,
        episode_id: str | None,
        lease: str | None,
        reason: str,
    ) -> dict[str, Any]:
        """Retire an unreceived episode after its source proves generation stopped.

        Authenticated source callers fence generation before calling. A missing
        lease means an admission reply was lost before any generation started;
        tombstones also fence admission requests that arrive after cancellation.
        """
        identity = self._episode_identity(source_id, session_id, trajectory_id)
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 256:
            raise ValueError(
                "Cancellation requires a nonempty reason of at most 256 characters"
            )
        if (
            (episode_id is None) != (lease is None)
            or episode_id is not None
            and episode_id != identity
        ):
            raise ValueError("Cancellation identity does not match admission")
        with self.queue._transaction() as db:
            row = db.execute(
                "SELECT * FROM online_episodes WHERE episode_id=?", (identity,)
            ).fetchone()
            if row is None:
                if episode_id is not None:
                    raise ValueError("Unknown cancellation admission lease")
                db.execute(
                    "INSERT INTO online_episodes "
                    "(episode_id,source_id,session_id,trajectory_id,lease,status,rejection) "
                    "VALUES (?,?,?,?,?,'rejected',?)",
                    (
                        identity,
                        source_id,
                        session_id,
                        trajectory_id,
                        uuid.uuid4().hex,
                        "source_abort:" + reason,
                    ),
                )
            else:
                if lease is not None and row["lease"] != lease:
                    raise ValueError("Cancellation lease is obsolete")
                if (
                    row["digest"] is not None
                    or row["batch_id"] is not None
                    or row["status"] not in ("admitted", "rejected")
                ):
                    raise ValueError(
                        "Cannot cancel a received or checkpoint-bound episode"
                    )
                if row["status"] == "rejected":
                    if not (row["rejection"] or "").startswith("source_abort:"):
                        raise ValueError("Episode was rejected for a different reason")
                else:
                    db.execute(
                        "UPDATE online_episodes SET status='rejected',rejection=? WHERE episode_id=?",
                        ("source_abort:" + reason, identity),
                    )
        return {"episode_id": identity, "status": "rejected"}

    def seal_batch(
        self, actor_version: int, *, consumer: BatchConsumer | None = None
    ) -> str | None:
        """Atomically bind B eligible episodes to one checkpoint-consumed batch.

        BatchQueue's existing single consumer and checkpoint cursor remain the
        authority for update replay. At most one uncommitted batch is assembled.
        Stale episodes are rejected whole; future versions require reconciliation.
        """
        if type(actor_version) is not int or actor_version < 0:
            raise ValueError("Actor version must be a nonnegative integer")
        with self.queue._transaction() as db:
            progress = self._progress(db, consumer)
            published = db.execute(
                "SELECT published_version, admission_open FROM online_progress"
            ).fetchone()[0]
            if (
                progress["applied_batches"] != actor_version
                or published != actor_version
            ):
                raise ValueError(
                    "Reconcile applied and serving versions before reserving"
                )
            existing = db.execute(
                "SELECT batch_id FROM batches WHERE sequence>? ORDER BY sequence LIMIT 1",
                (progress["through_sequence"],),
            ).fetchone()
            if existing is not None:
                return existing["batch_id"]
            future = db.execute(
                "SELECT episode_id FROM online_episodes WHERE status='ready' AND max_version>? LIMIT 1",
                (actor_version,),
            ).fetchone()
            if future is not None:
                raise ValueError(
                    "A trajectory was generated by a future serving version"
                )
            db.execute(
                "UPDATE online_episodes SET status='rejected',rejection='staleness' "
                "WHERE status='ready' AND min_version<?",
                (actor_version - self.max_staleness,),
            )
            rows = db.execute(
                "SELECT * FROM online_episodes WHERE status='ready' ORDER BY sequence LIMIT ?",
                (self.batch_size,),
            ).fetchall()
            if len(rows) != self.batch_size:
                return None
            episode_ids = [row["episode_id"] for row in rows]
            batch_id = hashlib.sha256(_encode(episode_ids).encode()).hexdigest()
            payload = _encode(
                {
                    "episode_ids": episode_ids,
                    "trajectories": [json.loads(row["payload"]) for row in rows],
                }
            )
            db.execute(
                "INSERT INTO batches (batch_id,payload,policy_version,digest) VALUES (?,?,?,?)",
                (batch_id, payload, actor_version, _digest(payload, actor_version)),
            )
            db.executemany(
                "UPDATE online_episodes SET status='reserved',batch_id=? WHERE episode_id=?",
                [(batch_id, identity) for identity in episode_ids],
            )
            return batch_id

    def _progress(self, db, consumer):
        if consumer is None:
            return db.execute("SELECT * FROM progress").fetchone()
        if consumer.queue is not self.queue:
            raise ValueError("Publication consumer belongs to another inbox")
        # The fenced consumer owns this live cursor. Claim/checkpoint recovery
        # validate payloads; polling must not rescan the complete retained prefix.
        return consumer.cursor

    def confirm_publication(
        self, version: int, *, consumer: BatchConsumer | None = None
    ) -> None:
        """Release applied episode credits after serving acknowledges the weights.

        The live consumer may be ahead of the last durable checkpoint. Payloads
        remain retained for replay if that GPU state is subsequently lost.
        """
        with self.queue._transaction() as db:
            progress = self._progress(db, consumer)
            committed = progress["applied_batches"]
            published = db.execute(
                "SELECT published_version, admission_open FROM online_progress"
            ).fetchone()[0]
            if type(version) is not int or version != committed or version < published:
                raise ValueError(
                    "Publication must match applied progress (or the committed checkpoint)"
                )
            db.execute(
                "UPDATE online_episodes SET status='published' WHERE status='reserved' "
                "AND batch_id IN (SELECT batch_id FROM batches WHERE sequence<=? "
                "AND status IN ('applied','committed'))",
                (progress["through_sequence"],),
            )
            db.execute("UPDATE online_progress SET published_version=?", (version,))

    def rollback(self, consumer: BatchConsumer) -> None:
        """Fence an abandoned serving branch after restoring its checkpoint cursor.

        Already sealed batches replay in their original order. Other admitted or
        ready episodes are explicitly rejected, including late source replies,
        so reused version numbers cannot disguise data from the abandoned branch.
        The caller pauses serving and stops the old Actor before this operation.
        """
        with self.queue._transaction() as db:
            if db.execute("SELECT admission_open FROM online_progress").fetchone()[0]:
                raise RuntimeError("Rollback requires closed admission")
            cursor = self._progress(db, consumer)
            db.execute(
                "UPDATE online_episodes SET status=CASE WHEN batch_id IN "
                "(SELECT batch_id FROM batches WHERE sequence<=?) THEN 'published' "
                "ELSE 'reserved' END WHERE batch_id IS NOT NULL",
                (cursor["through_sequence"],),
            )
            db.execute(
                "UPDATE online_episodes SET status='rejected', rejection='recovery_rollback' "
                "WHERE status IN ('admitted','ready')"
            )
            db.execute(
                "UPDATE online_progress SET published_version=?",
                (cursor["applied_batches"],),
            )

    def status(self) -> dict[str, Any]:
        """Serving acknowledgement and durable progress; SQLite names stay unchanged."""
        with self.queue._transaction() as db:
            counts = dict(
                db.execute(
                    "SELECT status,COUNT(*) FROM online_episodes GROUP BY status"
                )
            )
            published = db.execute(
                "SELECT published_version, admission_open FROM online_progress"
            ).fetchone()
            checkpoint = db.execute("SELECT applied_batches FROM progress").fetchone()[
                0
            ]
        return {
            "queue_id": self.queue.queue_id,
            "capacity": self.capacity,
            "active_credits": sum(
                count
                for state, count in counts.items()
                if state not in ("published", "rejected")
            ),
            "counts": counts,
            "checkpoint_version": checkpoint,
            "published_version": published[0],
            "admission_open": bool(published[1]),
        }
