# SPDX-License-Identifier: Apache-2.0

"""Transactional local checkpoints and disk-backed candidate views."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .snapshot import digest, json_bytes
from .types import DataSample, PublicTask, RankedCandidate


def sample_from_dict(value: dict[str, Any]) -> DataSample:
    return DataSample(PublicTask(**value["task"]), value["reference"])


class WorkStore:
    """One coordinator owns this database; worker results commit atomically."""

    def __init__(self, path: Path, *, resume: bool = False):
        self.db = sqlite3.connect(path)
        path.chmod(0o600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        if self.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            self.db.close()
            raise ValueError("Invalid data checkpoint database")
        if resume:
            return
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observed (
                task_id TEXT PRIMARY KEY, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS inputs (
                input_digest TEXT PRIMARY KEY, reference_digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS samples (
                ordinal INTEGER PRIMARY KEY, task_id TEXT UNIQUE NOT NULL,
                batch INTEGER NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS samples_batch ON samples(batch, ordinal);
            CREATE TABLE IF NOT EXISTS probes (
                task_id TEXT NOT NULL, repeat INTEGER NOT NULL,
                payload TEXT NOT NULL, digest TEXT NOT NULL,
                PRIMARY KEY (task_id, repeat));
            CREATE TABLE IF NOT EXISTS ranking (
                task_id TEXT PRIMARY KEY, priority REAL NOT NULL,
                tie_breaker TEXT NOT NULL, group_name TEXT NOT NULL,
                payload TEXT NOT NULL, digest TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS ranking_order ON ranking(priority,tie_breaker,task_id);
            CREATE INDEX IF NOT EXISTS ranking_group ON ranking(group_name,priority,tie_breaker,task_id);
        """)

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def decode(payload: str, expected: str) -> Any:
        if digest(payload.encode()) != expected:
            raise ValueError("Data checkpoint digest mismatch")
        return json.loads(payload)

    def get(self, key: str) -> Any:
        row = self.db.execute(
            "SELECT value, digest FROM checkpoints WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else self.decode(*row)

    def put(self, key: str, value: Any) -> None:
        payload = json_bytes(value)
        self.db.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)",
            (key, payload.decode(), digest(payload)),
        )

    def commit(self, key: str, value: Any) -> None:
        with self.db:
            self.put(key, value)

    def retain(
        self, key: str, batch: int, cleaned: list[DataSample | None], excluded: set[str]
    ) -> None:
        counts = self.get("counts") or {}
        retained = counts.get("clean_candidates", 0)
        with self.db:
            counts["generated"] = counts.get("generated", 0) + len(cleaned)
            for sample in cleaned:
                if sample is None:
                    name = "rejected_by_recipe"
                else:
                    payload = json_bytes(asdict(sample))
                    identity = digest(payload)
                    previous = self.db.execute(
                        "SELECT digest FROM observed WHERE task_id=?",
                        (sample.task.task_id,),
                    ).fetchone()
                    if previous is not None:
                        if previous[0] != identity:
                            raise ValueError(
                                "Same task ID refers to different candidate content"
                            )
                        name = "duplicates"
                    else:
                        self.db.execute(
                            "INSERT INTO observed VALUES (?, ?)",
                            (sample.task.task_id, identity),
                        )
                        input_id = digest(json_bytes(sample.task.input))
                        reference_id = digest(json_bytes(sample.reference))
                        prior = self.db.execute(
                            "SELECT reference_digest FROM inputs WHERE input_digest=?",
                            (input_id,),
                        ).fetchone()
                        if sample.task.task_id in excluded:
                            name = "excluded"
                        elif prior is not None:
                            if prior[0] != reference_id:
                                raise ValueError(
                                    "Duplicate task inputs have conflicting references"
                                )
                            name = "duplicates"
                        else:
                            self.db.execute(
                                "INSERT INTO inputs VALUES (?, ?)",
                                (input_id, reference_id),
                            )
                            self.db.execute(
                                "INSERT INTO samples(task_id,batch,payload,digest) VALUES(?,?,?,?)",
                                (
                                    sample.task.task_id,
                                    batch,
                                    payload.decode(),
                                    identity,
                                ),
                            )
                            retained += 1
                            continue
                counts[name] = counts.get(name, 0) + 1
            counts["clean_candidates"] = retained
            self.put("counts", counts)
            self.put(key, {"raw_count": len(cleaned)})

    def calibrate(
        self, key: str, rows: list[dict[str, Any]], provenance: dict[str, Any]
    ) -> None:
        with self.db:
            for row in rows:
                payload = json_bytes(row)
                self.db.execute(
                    "INSERT INTO probes VALUES (?, ?, ?, ?)",
                    (row["task_id"], row["repeat"], payload.decode(), digest(payload)),
                )
            counts = self.get("counts") or {}
            counts["calibrated_tasks"] = counts.get("calibrated_tasks", 0) + len(
                {row["task_id"] for row in rows}
            )
            self.put("counts", counts)
            self.put(key, provenance)

    def rank(self, key: str, candidates: list[RankedCandidate]) -> None:
        with self.db:
            for candidate in candidates:
                payload = json_bytes(asdict(candidate))
                self.db.execute(
                    "INSERT INTO ranking VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        candidate.task_id,
                        candidate.priority,
                        candidate.tie_breaker,
                        candidate.group,
                        payload.decode(),
                        digest(payload),
                    ),
                )
            counts = self.get("counts") or {}
            counts["eligible"] = counts.get("eligible", 0) + len(candidates)
            self.put("counts", counts)
            self.put(key, {"eligible": len(candidates)})

    def ranked(self, group: str | None = None) -> Iterator[RankedCandidate]:
        where, params = ("", ()) if group is None else (" WHERE group_name=?", (group,))
        cursor = self.db.execute(
            "SELECT payload,digest FROM ranking"
            + where
            + " ORDER BY priority,tie_breaker,task_id",
            params,
        )
        for row in cursor:
            yield RankedCandidate(**self.decode(*row))

    def samples(self, batch: int | None = None) -> SampleSequence:
        return SampleSequence(self, batch)

    def sample(self, task_id: str) -> DataSample:
        row = self.db.execute(
            "SELECT payload, digest FROM samples WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Selection contains an unknown candidate ID")
        return sample_from_dict(self.decode(*row))

    def probe_rows(self) -> Iterator[dict[str, Any]]:
        cursor = self.db.execute("""SELECT p.payload,p.digest FROM probes p
            JOIN samples s ON p.task_id=s.task_id ORDER BY s.ordinal,p.repeat""")
        for row in cursor:
            yield self.decode(*row)


class SampleSequence(Sequence[DataSample]):
    """Detached values on access; recipe mutations cannot change stored tasks."""

    def __init__(self, store: WorkStore, batch: int | None = None):
        self.store = store
        self.where = "" if batch is None else " WHERE batch=?"
        self.params = () if batch is None else (batch,)

    def __len__(self) -> int:
        if not self.where:
            return (self.store.get("counts") or {}).get("clean_candidates", 0)
        return self.store.db.execute(
            "SELECT COUNT(*) FROM samples" + self.where, self.params
        ).fetchone()[0]

    def __iter__(self) -> Iterator[DataSample]:
        cursor = self.store.db.execute(
            "SELECT payload,digest FROM samples" + self.where + " ORDER BY ordinal",
            self.params,
        )
        for row in cursor:
            yield sample_from_dict(self.store.decode(*row))

    def __getitem__(self, index):
        if isinstance(index, slice):
            from itertools import islice

            start, stop, step = index.indices(len(self))
            if step < 0:
                return [self[i] for i in range(start, stop, step)]
            return list(islice(self, start, stop, step))
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        row = self.store.db.execute(
            "SELECT payload,digest FROM samples"
            + self.where
            + " ORDER BY ordinal LIMIT 1 OFFSET ?",
            (*self.params, index),
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return sample_from_dict(self.store.decode(*row))


class ScoreMapping(Mapping[str, tuple[float | None, ...]]):
    def __init__(self, store: WorkStore):
        self.store = store

    def __len__(self) -> int:
        return (self.store.get("counts") or {}).get("calibrated_tasks", 0)

    def __iter__(self) -> Iterator[str]:
        for row in self.store.db.execute(
            "SELECT DISTINCT task_id FROM probes ORDER BY task_id"
        ):
            yield row[0]

    def __getitem__(self, task_id: str) -> tuple[float | None, ...]:
        rows = self.store.db.execute(
            "SELECT payload,digest FROM probes WHERE task_id=? ORDER BY repeat",
            (task_id,),
        ).fetchall()
        if not rows:
            raise KeyError(task_id)
        return tuple(self.store.decode(*row)["score"] for row in rows)
