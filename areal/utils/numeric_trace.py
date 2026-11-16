# SPDX-License-Identifier: Apache-2.0
"""Persist opt-in numeric profiling tensors through asynchronous host staging."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
import torch


def tensor_identity(value: torch.Tensor) -> dict:
    stored = value.float() if value.dtype == torch.bfloat16 else value
    array = stored.numpy()
    return {
        "dtype": str(value.dtype),
        "storage_dtype": array.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


class NumericTraceWriter:
    def __init__(self, manifest: str):
        self.manifest = Path(manifest)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="numeric-trace"
        )
        self.pending = []
        self.lock = Lock()
        self.closed = False

    def submit(self, tensors: dict, metadata: dict) -> None:
        """Queue host copies on the producer stream and defer synchronization to I/O."""
        with self.lock:
            if self.closed:
                raise RuntimeError("Numeric trace writer is closed")
            copies, streams = {}, {}
            for name, tensor in tensors.items():
                if isinstance(tensor, (int, float)):
                    tensor = torch.tensor(
                        tensor,
                        dtype=torch.float64
                        if isinstance(tensor, float)
                        else torch.int64,
                    )
                if not isinstance(tensor, torch.Tensor):
                    continue
                value = tensor.detach()
                if value.device.type == "cuda":
                    stream = torch.cuda.current_stream(value.device)
                    host = torch.empty_like(
                        value,
                        device="cpu",
                        pin_memory=True,
                        memory_format=torch.contiguous_format,
                    )
                    host.copy_(value, non_blocking=True)
                    value.record_stream(stream)
                    streams[value.device] = stream
                elif value.device.type == "cpu":
                    host = value.clone().contiguous()
                else:
                    raise ValueError(
                        f"Unsupported numeric trace device: {value.device}"
                    )
                copies[name] = host
            events = []
            for stream in streams.values():
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
            self.pending.append(
                self.executor.submit(self._write, copies, events, metadata)
            )

    def _write(self, tensors: dict, events: list, metadata: dict) -> None:
        for event in events:
            event.synchronize()
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        record_id = uuid4().hex
        filename = f"step-{metadata['global_step']:06d}-{record_id}.npz"
        path = self.manifest.parent / filename
        temporary = path.with_suffix(".tmp")
        arrays = {
            key: (value.float() if value.dtype == torch.bfloat16 else value).numpy()
            for key, value in tensors.items()
        }
        identities = {key: tensor_identity(value) for key, value in tensors.items()}
        record = {
            "schema_version": 1,
            "record_id": record_id,
            **metadata,
            "tensor_file": filename,
            "tensors": identities,
        }
        if "row_indices" in arrays and "token_positions" in arrays:
            rows = arrays["row_indices"].reshape(-1)
            positions = arrays["token_positions"].reshape(-1)
            row_hashes = []
            for row in np.unique(rows):
                selected = np.flatnonzero(rows == row)
                selected = selected[np.argsort(positions[selected], kind="stable")]
                if "attention_mask" in arrays:
                    selected = selected[
                        arrays["attention_mask"].reshape(-1)[selected].astype(bool)
                    ]
                hashes = {}
                for key in ("input_ids", "target_ids", "base_mask", "effective_mask"):
                    if key not in arrays:
                        continue
                    indices = selected
                    if key == "target_ids" and "base_mask" in arrays:
                        indices = indices[
                            arrays["base_mask"].reshape(-1)[indices].astype(bool)
                        ]
                    digest = hashlib.sha256(positions[indices].tobytes())
                    digest.update(arrays[key].reshape(-1)[indices].tobytes())
                    hashes[key] = digest.hexdigest()
                row_hashes.append(
                    {
                        "row_index": int(row),
                        "position_count": len(selected),
                        "hashes": hashes,
                    }
                )
            record["rows"] = row_hashes
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            with path.open("rb") as stream:
                record["tensor_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
            encoded = json.dumps(record, allow_nan=False, sort_keys=True)
            with self.manifest.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            temporary.unlink(missing_ok=True)

    def flush(self) -> None:
        with self.lock:
            pending, self.pending = self.pending, []
        wait(pending)
        for future in pending:
            future.result()

    def close(self) -> None:
        with self.lock:
            self.closed = True
        try:
            self.flush()
        finally:
            self.executor.shutdown(wait=True)
