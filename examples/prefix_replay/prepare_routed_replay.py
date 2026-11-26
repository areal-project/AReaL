# SPDX-License-Identifier: Apache-2.0

"""Merge prefix-replay JSONL sources while attaching an explicit MOPD route."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import tempfile
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO

_DEFAULT_CHUNK_RECORDS = 16
_SUBMISSION_WINDOW_MULTIPLIER = 4
_MP_CONTEXT = "fork"

_RawRoutedRow = tuple[str, int, str, bytes]
_ProcessedChunk = tuple[int, bytes]


def _count_rows(path: Path) -> int:
    with path.open("rb") as input_file:
        count = sum(bool(line.strip()) for line in input_file)
    if count == 0:
        raise ValueError(f"Dataset is empty: {path}")
    return count


def _usable_cpus() -> int:
    """Return CPUs available to this process, respecting scheduler affinity."""
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _iter_nonempty_lines(
    input_file: BinaryIO,
) -> Iterator[tuple[int, bytes]]:
    for line_number, line in enumerate(input_file, 1):
        if not line.strip():
            continue
        yield line_number, line


def _process_chunk(chunk: list[_RawRoutedRow], route_field: str) -> _ProcessedChunk:
    """Parse, route, and serialize one chunk in a worker process."""
    output_lines: list[bytes] = []
    for path, line_number, route, line in chunk:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        existing_route = row.get(route_field)
        if existing_route is not None and str(existing_route) != route:
            raise ValueError(
                f"{path}:{line_number}: {route_field}={existing_route!r} "
                f"conflicts with source route {route!r}"
            )
        row[route_field] = route
        output_lines.append(
            (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        )
    return len(chunk), b"".join(output_lines)


def _iter_interleaved_chunks(
    sources: Sequence[tuple[str, Path, int]],
    *,
    chunk_records: int,
) -> Iterator[list[_RawRoutedRow]]:
    """Yield raw rows in deterministic proportional order, bounded by one chunk."""
    with ExitStack() as stack:
        input_files = [stack.enter_context(path.open("rb")) for _, path, _ in sources]
        iterators = [_iter_nonempty_lines(input_file) for input_file in input_files]
        positions = [0] * len(sources)
        total = sum(count for _, _, count in sources)
        chunk: list[_RawRoutedRow] = []

        for _ in range(total):
            source_index = min(
                (
                    index
                    for index, (_, _, count) in enumerate(sources)
                    if positions[index] < count
                ),
                key=lambda index: (
                    positions[index] / sources[index][2],
                    index,
                ),
            )
            route, path, _ = sources[source_index]
            try:
                line_number, line = next(iterators[source_index])
            except StopIteration as exc:
                raise RuntimeError(
                    f"Replay source changed while preparing output: {path}"
                ) from exc
            positions[source_index] += 1
            chunk.append((str(path), line_number, route, line))
            if len(chunk) >= chunk_records:
                yield chunk
                chunk = []

        if chunk:
            yield chunk

        for iterator, (_, path, _) in zip(iterators, sources, strict=True):
            if next(iterator, None) is not None:
                raise RuntimeError(
                    f"Replay source changed while preparing output: {path}"
                )


def _write_processed_chunks(
    chunks: Iterator[list[_RawRoutedRow]],
    output_file: BinaryIO,
    *,
    route_field: str,
    workers: int,
) -> int:
    """Process chunks through a bounded pool and write them in input order."""
    processed_records = 0
    if workers == 1:
        for chunk in chunks:
            count, output = _process_chunk(chunk, route_field)
            output_file.write(output)
            processed_records += count
        return processed_records

    buffered: dict[int, _ProcessedChunk] = {}
    pending: dict[Future[_ProcessedChunk], int] = {}
    next_sequence = 0
    window = workers * _SUBMISSION_WINDOW_MULTIPLIER

    def flush_ready() -> None:
        nonlocal next_sequence, processed_records
        while next_sequence in buffered:
            count, output = buffered.pop(next_sequence)
            output_file.write(output)
            processed_records += count
            next_sequence += 1

    def drain_one() -> None:
        done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
        for future in done:
            sequence = pending.pop(future)
            buffered[sequence] = future.result()
        flush_ready()

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context(_MP_CONTEXT),
    ) as executor:
        for sequence, chunk in enumerate(chunks):
            # Count completed-but-not-yet-writable chunks against the window as
            # well. If an early chunk is slow, later results cannot accumulate
            # without bound while the parent keeps submitting more work.
            while len(pending) + len(buffered) >= window:
                drain_one()
            pending[executor.submit(_process_chunk, chunk, route_field)] = sequence
        while pending:
            drain_one()
    flush_ready()
    if buffered:
        raise RuntimeError("Out-of-order routed replay chunks remain unflushed")
    return processed_records


def prepare_routed_replay_dataset(
    sources: Sequence[tuple[str, Path]],
    output_path: Path,
    *,
    route_field: str = "replay_teacher",
    workers: int = 1,
    chunk_records: int = _DEFAULT_CHUNK_RECORDS,
) -> dict[str, Any]:
    """Tag, deterministically interleave, and atomically publish replay rows.

    JSON parsing and serialization use a bounded process pool when ``workers`` is
    greater than one. Chunks are always written in submission order, so output is
    byte-identical regardless of worker count.
    """
    if not sources:
        raise ValueError("At least one routed replay source is required")
    if not route_field.strip():
        raise ValueError("route_field must be a non-empty string")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if (
        not isinstance(chunk_records, int)
        or isinstance(chunk_records, bool)
        or chunk_records < 1
    ):
        raise ValueError("chunk_records must be a positive integer")

    routed_sources: list[tuple[str, Path, int]] = []
    seen_routes: set[str] = set()
    seen_paths: set[Path] = set()
    resolved_output_path = output_path.resolve()
    for route, path in sources:
        if not route.strip():
            raise ValueError("Replay source routes must be non-empty strings")
        if route in seen_routes:
            raise ValueError(f"Duplicate replay source route: {route!r}")
        resolved_path = path.resolve()
        if resolved_path == resolved_output_path:
            raise ValueError(f"Replay output cannot overwrite a source: {path}")
        if resolved_path in seen_paths:
            raise ValueError(f"Replay sources must be distinct files: {path}")
        seen_routes.add(route)
        seen_paths.add(resolved_path)
        routed_sources.append((route, path, _count_rows(path)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    counts = {route: count for route, _, count in routed_sources}
    total = sum(counts.values())
    try:
        chunks = _iter_interleaved_chunks(
            routed_sources,
            chunk_records=chunk_records,
        )
        with os.fdopen(file_descriptor, "wb") as output_file:
            processed_records = _write_processed_chunks(
                chunks,
                output_file,
                route_field=route_field,
                workers=workers,
            )
        if processed_records != total:
            raise RuntimeError(
                f"Routed replay count mismatch: {processed_records} != {total}"
            )
        os.replace(temporary_name, output_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    return {
        "output": str(output_path),
        "total": total,
        "route_field": route_field,
        "route_counts": counts,
        "workers": workers,
        "chunk_records": chunk_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("ROUTE", "JSONL"),
        required=True,
        help="Routed source pair; repeat once per teacher dataset.",
    )
    parser.add_argument("--route-field", default="replay_teacher")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        metavar="N",
        help="Parallel JSON worker processes (default 0 = auto, max 16; 1 = serial).",
    )
    parser.add_argument(
        "--chunk-records",
        type=int,
        default=_DEFAULT_CHUNK_RECORDS,
        metavar="N",
        help=f"Records per worker task (default {_DEFAULT_CHUNK_RECORDS}).",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = [(route, Path(path)) for route, path in args.source]
    if args.workers < 0:
        parser.error("--workers must be zero or a positive integer")
    workers = args.workers if args.workers else min(16, _usable_cpus())
    report = prepare_routed_replay_dataset(
        sources,
        args.output,
        route_field=args.route_field,
        workers=workers,
        chunk_records=args.chunk_records,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
