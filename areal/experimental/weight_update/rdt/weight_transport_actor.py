# SPDX-License-Identifier: Apache-2.0
"""WeightTransportActor for TW tensor transport via RDT.

Created by TW subprocess, shares same GPU via CUDA_VISIBLE_DEVICES.
Receives sliced tensor IPC handles and implements @ray.method(tensor_transport).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from threading import Condition
from typing import Any

import ray
import torch

from areal.utils import logging

logger = logging.getLogger("WeightTransportActor")


@ray.remote
class WeightTransportActor:
    """Actor for weight tensor transport via RDT.

    Key features:
    - max_concurrency set to IW count (multiple IW may pull concurrently)
    - Condition key: {pair_name}/{infer_rank}/{version} for one-to-one TW-IW sync
    - IW calls clear_ipc_handles() after ray.get() to release shared GPU memory
    """

    def __init__(self):
        # Version-based tensor storage: {pair_name}/{infer_rank}/{version}/{param_name}
        self._tensors: dict[str, torch.Tensor] = {}
        self._tensors_lock = threading.Lock()  # Protect _tensors dict access

        # Synchronization: wait for IPC handles ready
        self._tensor_ready_lock = threading.Lock()
        self._tensor_ready: dict[str, Condition] = {}
        self._tensor_ready_flags: dict[str, bool] = defaultdict(bool)

        # Activate cuda:0 for IPC (CUDA_VISIBLE_DEVICES already set to single GPU by TW)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info("WeightTransportActor initialized on cuda:0")

    def store_ipc_handles(
        self,
        pair_name: str,
        infer_rank: int,
        version: int,
        ipc_handles: dict[str, Any],
    ) -> None:
        """Receive all IPC handles, store tensors and notify IW.

        Args:
            pair_name: TW-IW pair identifier
            infer_rank: IW's global rank
            version: Weight version number
            ipc_handles: dict of {param_name: ipc_payload}
        """
        prefix = f"{pair_name}/{infer_rank}/{version}"

        # Build tensors outside lock to reduce lock holding time
        new_tensors: dict[str, torch.Tensor] = {}
        for param_name, ipc_payload in ipc_handles.items():
            rebuild_fn = ipc_payload["rebuild_fn"]
            tensor_meta = ipc_payload["tensor_meta"]
            shared_tensor = rebuild_fn(*tensor_meta)
            key = f"{prefix}/{param_name}"
            new_tensors[key] = shared_tensor

        # Store tensors under lock (brief operation)
        with self._tensors_lock:
            self._tensors.update(new_tensors)

        # Notify waiting IWs
        with self._tensor_ready_lock:
            self._tensor_ready_flags[prefix] = True
            if prefix in self._tensor_ready:
                self._tensor_ready[prefix].notify_all()

        logger.info(f"Stored {len(ipc_handles)} IPC handles for {prefix}")

    def _wait_for_ready(
        self, pair_name: str, infer_rank: int, version: int, timeout: float = 30.0
    ) -> bool:
        """Wait for IPC handles ready (blocking)."""
        prefix = f"{pair_name}/{infer_rank}/{version}"

        with self._tensor_ready_lock:
            if self._tensor_ready_flags.get(prefix, False):
                return True

            if prefix not in self._tensor_ready:
                self._tensor_ready[prefix] = Condition(self._tensor_ready_lock)

            return self._tensor_ready[prefix].wait(timeout=timeout)

    @ray.method(tensor_transport="NIXL")
    def get_weights_tensor_nixl(
        self,
        pair_name: str,
        infer_rank: int,
        version: int,
    ) -> dict[str, Any]:
        """Tensor transport for GPU (NIXL backend).

        IW calls this method, blocks until TW stores IPC handles.
        Returns merged buffer + metadata for efficient single RDMA transfer.
        """
        import time

        t0 = time.monotonic()
        prefix = f"{pair_name}/{infer_rank}/{version}"

        t1 = time.monotonic()
        if not self._wait_for_ready(pair_name, infer_rank, version):
            raise RuntimeError(f"IPC handles not ready for {prefix} after 30s")

        t2 = time.monotonic()
        with self._tensors_lock:
            tensor_items = [
                (k, v) for k, v in self._tensors.items() if k.startswith(prefix)
            ]
            # Sort by key to ensure consistent order
            tensor_items.sort(key=lambda x: x[0])

        t3 = time.monotonic()
        if not tensor_items:
            raise RuntimeError(f"Tensors not found for {prefix}")

        # Merge all tensors into single contiguous buffer
        # This reduces NIXL registration overhead from N times to 1 time
        tensors = [t.clone().detach() for _, t in tensor_items]
        param_names = [k.split("/")[-1] for k, _ in tensor_items]

        t4 = time.monotonic()

        # Flatten and concatenate into single buffer
        flat_tensors = [t.flatten() for t in tensors]
        merged_buffer = torch.cat(flat_tensors)

        # Build metadata for IW to split buffer back
        offsets = []
        current_offset = 0
        shapes = []
        dtypes = []
        for t in tensors:
            numel = t.numel()
            offsets.append(current_offset)
            shapes.append(tuple(t.shape))
            dtypes.append(str(t.dtype))
            current_offset += numel

        t5 = time.monotonic()
        total_bytes = merged_buffer.numel() * merged_buffer.element_size()
        logger.info(
            f"[Actor-Timing] wait_ready={1000 * (t2 - t1):.1f}ms | "
            f"lock_lookup={1000 * (t3 - t2):.1f}ms | "
            f"clone={1000 * (t4 - t3):.1f}ms | "
            f"merge={1000 * (t5 - t4):.1f}ms | "
            f"total={1000 * (t5 - t0):.1f}ms | "
            f"num_tensors={len(tensors)} | "
            f"total_bytes={total_bytes / 1024 / 1024:.1f}MB"
        )

        return {
            "buffer": merged_buffer,
            "metadata": {
                "names": param_names,
                "offsets": offsets,
                "shapes": shapes,
                "dtypes": dtypes,
            },
        }

    @ray.method(tensor_transport="NIXL")
    def warmup_nixl(self) -> dict[str, torch.Tensor]:
        """Warmup NIXL agent by returning a minimal tensor.

        IW calls this during init to trigger NIXL agent initialization
        on both IW (driver) and TW Actor sides before actual weight transfer.
        This moves ~9s initialization overhead from update_weights to connect phase.
        """
        # Create a tiny tensor to trigger NIXL registration
        warmup_tensor = torch.zeros(1, dtype=torch.float32, device="cuda:0")
        logger.info("NIXL warmup tensor created")
        return {"warmup": warmup_tensor}

    # TODO: Implement YR backend for NPU (ray-ascend)
    # @ray.method(tensor_transport="YR")
    # def get_weights_tensor_yr(
    #     self,
    #     pair_name: str,
    #     infer_rank: int,
    #     version: int,
    # ) -> dict[str, torch.Tensor]:
    #     """Tensor transport for NPU (YR backend)."""
    #     ...

    def clear_ipc_handles(self, pair_name: str, infer_rank: int, version: int) -> None:
        """Clean up IPC handles for specific infer_rank and version."""
        prefix = f"{pair_name}/{infer_rank}/{version}"

        # Remove tensors under lock
        with self._tensors_lock:
            for key in list(self._tensors.keys()):
                if key.startswith(prefix):
                    del self._tensors[key]

        # Clear readiness state
        with self._tensor_ready_lock:
            self._tensor_ready_flags[prefix] = False
            if prefix in self._tensor_ready:
                del self._tensor_ready[prefix]
