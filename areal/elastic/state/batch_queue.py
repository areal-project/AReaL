# SPDX-License-Identifier: Apache-2.0

"""Local durable FIFO; the published training checkpoint owns consumption progress."""

import fcntl
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EMPTY_CHAIN = hashlib.sha256(b"").hexdigest()


def _encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload: str, policy_version: int) -> str:
    return hashlib.sha256(f"{policy_version}\n{payload}".encode()).hexdigest()


def _chain(previous: str, row: sqlite3.Row) -> str:
    value = [previous, row["sequence"], row["batch_id"], row["digest"]]
    return hashlib.sha256(_encode(value).encode()).hexdigest()


@dataclass(frozen=True)
class QueuedBatch:
    sequence: int
    batch_id: str
    payload: dict[str, Any]
    policy_version: int
    digest: str
    lease: str


class BatchQueue:
    """Append-only JSON batches; concurrent producers and one fenced consumer.

    Use a local filesystem with SQLite WAL and flock support, not NFS. Payloads
    remain retained so recovery can validate the checkpoint's complete prefix.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "queue.sqlite3"
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS progress (singleton INTEGER PRIMARY KEY "
                "CHECK(singleton=1), format_version INTEGER NOT NULL, queue_id TEXT NOT NULL, "
                "through_sequence INTEGER NOT NULL, applied_batches INTEGER NOT NULL, "
                "chain_sha256 TEXT NOT NULL)"
            )
            db.execute(
                "INSERT OR IGNORE INTO progress VALUES (1, 1, ?, 0, 0, ?)",
                (uuid.uuid4().hex, _EMPTY_CHAIN),
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS batches (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                "batch_id TEXT NOT NULL UNIQUE, payload TEXT NOT NULL, policy_version INTEGER "
                "NOT NULL, digest TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', "
                "lease TEXT)"
            )
            state = dict(db.execute("SELECT * FROM progress").fetchone())
            if state["format_version"] != 1:
                raise ValueError("Unsupported batch queue format")
            self.queue_id = state["queue_id"]
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def enqueue(
        self, batch_id: str, payload: dict[str, Any], *, policy_version: int
    ) -> int:
        """An ID is idempotent only when both content and policy version match."""
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError("Batch ID must be nonempty")
        if not isinstance(payload, dict):
            raise ValueError("Batch payload must be a JSON object")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("Policy version must be a nonnegative integer")
        encoded = _encode(payload)
        digest = _digest(encoded, policy_version)
        with self._transaction() as db:
            previous = db.execute(
                "SELECT * FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if previous is not None:
                if previous["digest"] != digest or previous["payload"] != encoded:
                    raise ValueError("Batch ID already exists with different content")
                return previous["sequence"]
            row = db.execute(
                "INSERT INTO batches (batch_id, payload, policy_version, digest) "
                "VALUES (?, ?, ?, ?)",
                (batch_id, encoded, policy_version, digest),
            )
            return row.lastrowid

    def status(self) -> dict[str, Any]:
        with self._transaction() as db:
            counts = {key: 0 for key in ("pending", "leased", "applied", "committed")}
            counts.update(
                dict(db.execute("SELECT status, COUNT(*) FROM batches GROUP BY status"))
            )
            state = dict(db.execute("SELECT * FROM progress").fetchone())
        return {"queue_id": self.queue_id, "counts": counts, "cursor": _cursor(state)}

    def consumer(
        self,
        checkpoint: dict[str, Any] | None | Callable[[], dict[str, Any] | None] = None,
    ) -> "BatchConsumer":
        """A callable reads latest only after acquiring the consumer lock."""
        return BatchConsumer(self, checkpoint)


def _cursor(row: dict[str, Any] | sqlite3.Row) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "format_version",
            "queue_id",
            "through_sequence",
            "applied_batches",
            "chain_sha256",
        )
    }


class BatchConsumer:
    """Hold the OS lock until all GPU work has stopped; no expiring leases."""

    def __init__(
        self,
        queue: BatchQueue,
        checkpoint: dict[str, Any] | None | Callable[[], dict[str, Any] | None],
    ):
        self.queue = queue
        self._lock = (queue.root / "consumer.lock").open("a")
        self._closed = True
        self._lease = uuid.uuid4().hex
        self._active: QueuedBatch | None = None
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._closed = False
            if callable(checkpoint):
                checkpoint = checkpoint()
            initial = {
                "format_version": 1,
                "queue_id": queue.queue_id,
                "through_sequence": 0,
                "applied_batches": 0,
                "chain_sha256": _EMPTY_CHAIN,
            }
            if checkpoint is not None:
                initial = checkpoint.get("data_cursor")
                if not isinstance(initial, dict):
                    raise ValueError("Checkpoint has no durable queue cursor")
                if checkpoint["global_step"] != initial.get("applied_batches"):
                    raise ValueError("Checkpoint step disagrees with queue cursor")
            self._recover(initial)
        except BaseException:
            self.close()
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Batch consumer is closed")

    def restore_checkpoint(self, checkpoint: dict[str, Any] | None) -> None:
        """Reconcile after the old GPU generation has stopped, retaining flock."""
        self._require_open()
        if checkpoint is None:
            cursor = {
                "format_version": 1,
                "queue_id": self.queue.queue_id,
                "through_sequence": 0,
                "applied_batches": 0,
                "chain_sha256": _EMPTY_CHAIN,
            }
        else:
            cursor = checkpoint.get("data_cursor")
            if not isinstance(cursor, dict) or checkpoint["global_step"] != cursor.get(
                "applied_batches"
            ):
                raise ValueError("Checkpoint step disagrees with queue cursor")
        self._recover(cursor)
        self._active = None

    def _validate(self, db: sqlite3.Connection, cursor: dict[str, Any]) -> None:
        if (
            cursor.get("format_version") != 1
            or cursor.get("queue_id") != self.queue.queue_id
        ):
            raise ValueError("Checkpoint belongs to a different batch queue")
        sequence, count = cursor.get("through_sequence"), cursor.get("applied_batches")
        if any(type(value) is not int or value < 0 for value in (sequence, count)):
            raise ValueError("Invalid queue cursor")
        chain, seen, last = _EMPTY_CHAIN, 0, 0
        for row in db.execute(
            "SELECT * FROM batches WHERE sequence<=? ORDER BY sequence", (sequence,)
        ):
            if _digest(row["payload"], row["policy_version"]) != row["digest"]:
                raise ValueError("Batch payload digest mismatch")
            chain, seen, last = _chain(chain, row), seen + 1, row["sequence"]
        if (last, seen, chain) != (sequence, count, cursor.get("chain_sha256")):
            raise ValueError("Checkpoint queue prefix is missing or changed")

    def _recover(self, cursor: dict[str, Any]) -> None:
        with self.queue._transaction() as db:
            self._validate(db, cursor)
            progress = db.execute("SELECT * FROM progress").fetchone()
            if progress["through_sequence"] > cursor["through_sequence"]:
                raise ValueError("Queue acknowledgement is ahead of checkpoint")
            db.execute(
                "UPDATE batches SET status=CASE WHEN sequence<=? THEN 'committed' "
                "ELSE 'pending' END, lease=NULL",
                (cursor["through_sequence"],),
            )
            self._write_progress(db, cursor)
        self._current = dict(cursor)

    @staticmethod
    def _write_progress(db: sqlite3.Connection, cursor: dict[str, Any]) -> None:
        db.execute(
            "UPDATE progress SET through_sequence=?, applied_batches=?, chain_sha256=?",
            (
                cursor["through_sequence"],
                cursor["applied_batches"],
                cursor["chain_sha256"],
            ),
        )

    @property
    def cursor(self) -> dict[str, Any]:
        self._require_open()
        if self._active is not None:
            raise RuntimeError("Cannot checkpoint an unapplied batch")
        return dict(self._current)

    def applied_payloads(self):
        """Read the checkpoint-bound prefix, including replicated batches only once."""
        self._require_open()
        with self.queue._transaction() as db:
            self._validate(db, self._current)
            return [
                json.loads(row["payload"])
                for row in db.execute(
                    "SELECT payload FROM batches WHERE sequence<=? ORDER BY sequence",
                    (self._current["through_sequence"],),
                )
            ]

    def claim(self) -> QueuedBatch | None:
        self._require_open()
        if self._active is not None:
            raise RuntimeError("Previous batch has not completed")
        with self.queue._transaction() as db:
            row = db.execute(
                "SELECT * FROM batches WHERE sequence>? ORDER BY sequence LIMIT 1",
                (self._current["through_sequence"],),
            ).fetchone()
            if row is None:
                return None
            if row["status"] != "pending":
                raise RuntimeError("Queue requires checkpoint recovery")
            if _digest(row["payload"], row["policy_version"]) != row["digest"]:
                raise ValueError("Batch payload digest mismatch")
            db.execute(
                "UPDATE batches SET status='leased', lease=? WHERE sequence=?",
                (self._lease, row["sequence"]),
            )
            batch = QueuedBatch(
                row["sequence"],
                row["batch_id"],
                json.loads(row["payload"]),
                row["policy_version"],
                row["digest"],
                self._lease,
            )
        self._active = batch
        return batch

    def peek(self) -> QueuedBatch | None:
        """Validate the next pending envelope without leasing or advancing it."""
        self._require_open()
        if self._active is not None:
            raise RuntimeError("Previous batch has not completed")
        with self.queue._transaction() as db:
            row = db.execute(
                "SELECT * FROM batches WHERE sequence>? ORDER BY sequence LIMIT 1",
                (self._current["through_sequence"],),
            ).fetchone()
            if row is None:
                return None
            if row["status"] != "pending":
                raise RuntimeError("Queue requires checkpoint recovery")
            if _digest(row["payload"], row["policy_version"]) != row["digest"]:
                raise ValueError("Batch payload digest mismatch")
            return QueuedBatch(
                row["sequence"],
                row["batch_id"],
                json.loads(row["payload"]),
                row["policy_version"],
                row["digest"],
                "",
            )

    def mark_applied(self, batch: QueuedBatch) -> None:
        """Volatile training progress, not an acknowledgement or deletion."""
        self._require_open()
        if batch is not self._active:
            raise ValueError("Batch does not hold the active lease")
        with self.queue._transaction() as db:
            row = db.execute(
                "SELECT * FROM batches WHERE sequence=?", (batch.sequence,)
            ).fetchone()
            if row["lease"] != self._lease or row["status"] != "leased":
                raise RuntimeError("Batch lease is stale")
            db.execute(
                "UPDATE batches SET status='applied' WHERE sequence=?",
                (batch.sequence,),
            )
            chain = _chain(self._current["chain_sha256"], row)
        self._current.update(
            through_sequence=batch.sequence,
            applied_batches=self._current["applied_batches"] + 1,
            chain_sha256=chain,
        )
        self._active = None

    def acknowledge(self, checkpoint: dict[str, Any]) -> None:
        """Call only after the checkpoint's latest pointer is durably published."""
        if checkpoint.get("data_cursor") != self.cursor:
            raise ValueError("Acknowledgement differs from applied batches")
        if checkpoint["global_step"] != self._current["applied_batches"]:
            raise ValueError("Checkpoint step disagrees with queue cursor")
        with self.queue._transaction() as db:
            self._validate(db, self._current)
            db.execute(
                "UPDATE batches SET status='committed', lease=NULL WHERE sequence<=?",
                (self._current["through_sequence"],),
            )
            self._write_progress(db, self._current)

    def close(self) -> None:
        if not self._closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
        self._closed = True
        self._lock.close()

    def __enter__(self) -> "BatchConsumer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
