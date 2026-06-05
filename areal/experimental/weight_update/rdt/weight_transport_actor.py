# SPDX-License-Identifier: Apache-2.0
"""WeightTransportActor for TW tensor transport via RDT.

Created by TW subprocess, shares same GPU via CUDA_VISIBLE_DEVICES.
Receives sliced tensor IPC handles and implements @ray.method(tensor_transport).

With NIXL buffer reuse enabled, pre-allocates merged send buffers and
registers them with register_nixl_memory for lifetime reuse, avoiding
per-step GPU allocation overhead.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from threading import Condition
from typing import Any

import ray
import torch

from areal.experimental.weight_update.rdt import try_register_nixl_memory
from areal.utils import logging

logger = logging.getLogger("WeightTransportActor")


@ray.remote
class WeightTransportActor:
    """Actor for weight tensor transport via RDT.

    Key features:
    - max_concurrency set to IW count (multiple IW may pull concurrently)
    - Condition key: {pair_name}/{infer_rank}/{version} for one-to-one TW-IW sync
    - IW calls clear_ipc_handles() after ray.get() to release shared GPU memory
    - NIXL buffer reuse: pre-allocated send buffers with register_nixl_memory
    """

    def __init__(self):
        # Version-based tensor storage: {pair_name}/{infer_rank}/{version}/{param_name}
        self._tensors: dict[str, torch.Tensor] = {}
        self._tensors_lock = threading.Lock()  # Protect _tensors dict access

        # Synchronization: wait for IPC handles ready
        self._tensor_ready_lock = threading.Lock()
        self._tensor_ready: dict[str, Condition] = {}
        self._tensor_ready_flags: dict[str, bool] = defaultdict(bool)

        # NIXL buffer reuse: pre-allocated merged send buffers
        # Key format: "{pair_name}/{infer_rank}"
        self._send_buffers: dict[str, torch.Tensor] = {}
        self._send_buffer_meta: dict[str, dict] = {}
        self._nixl_reuse_available: bool | None = None  # None = not tested yet

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

    def _init_send_buffer(
        self,
        buffer_key: str,
        tensor_items: list[tuple[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict]:
        """Pre-allocate merged send buffer and register with NIXL.

        Called on first get_weights_tensor_nixl invocation for a given
        (pair_name, infer_rank) combination. The buffer persists for
        lifetime reuse.

        Args:
            buffer_key: "{pair_name}/{infer_rank}" identifier
            tensor_items: Sorted list of (key, tensor) pairs

        Returns:
            tuple of (merged_buffer, metadata_dict)
        """
        tensors = [t.clone().detach() for _, t in tensor_items]
        param_names = [k.split("/")[-1] for k, _ in tensor_items]

        # Determine common dtype for merged buffer
        common_dtype = tensors[0].dtype

        # Flatten and concatenate into single buffer
        flat_tensors = [t.flatten().to(common_dtype) for t in tensors]
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

        metadata = {
            "names": param_names,
            "offsets": offsets,
            "shapes": shapes,
            "dtypes": dtypes,
        }

        # Try registering with NIXL for lifetime reuse
        if self._nixl_reuse_available is None:
            self._nixl_reuse_available = try_register_nixl_memory(merged_buffer)
        elif self._nixl_reuse_available:
            try_register_nixl_memory(merged_buffer)

        if self._nixl_reuse_available:
            self._send_buffers[buffer_key] = merged_buffer
            self._send_buffer_meta[buffer_key] = metadata
            total_bytes = merged_buffer.numel() * merged_buffer.element_size()
            logger.info(
                f"NIXL buffer reuse: pre-allocated send buffer for {buffer_key}, "
                f"size={total_bytes / 1024 / 1024:.1f}MB"
            )

        return merged_buffer, metadata

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

        With NIXL buffer reuse, copies data into pre-allocated buffer
        on subsequent calls, avoiding per-step GPU allocation.
        """
        import time

        t0 = time.monotonic()
        prefix = f"{pair_name}/{infer_rank}/{version}"
        buffer_key = f"{pair_name}/{infer_rank}"

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

        # Check if we have a pre-allocated send buffer
        reuse = self._nixl_reuse_available and buffer_key in self._send_buffers

        if reuse:
            # Subsequent calls: copy data into pre-allocated buffer (in-place)
            send_buffer = self._send_buffers[buffer_key]
            cached_meta = self._send_buffer_meta[buffer_key]
            common_dtype = send_buffer.dtype

            for i, (_, t) in enumerate(tensor_items):
                offset = cached_meta["offsets"][i]
                numel = t.numel()
                flat = t.flatten()
                if t.dtype != common_dtype:
                    send_buffer[offset : offset + numel].copy_(flat.to(common_dtype))
                else:
                    send_buffer[offset : offset + numel].copy_(flat)

            merged_buffer = send_buffer
            metadata = cached_meta
        else:
            # First call (or NIXL reuse unavailable): build buffer from scratch
            merged_buffer, metadata = self._init_send_buffer(buffer_key, tensor_items)

        t4 = time.monotonic()

        total_bytes = merged_buffer.numel() * merged_buffer.element_size()
        reuse_str = "reuse" if reuse else "alloc"
        logger.info(
            f"[Actor-Timing] wait_ready={1000 * (t2 - t1):.1f}ms | "
            f"lock_lookup={1000 * (t3 - t2):.1f}ms | "
            f"build_buffer({reuse_str})={1000 * (t4 - t3):.1f}ms | "
            f"total={1000 * (t4 - t0):.1f}ms | "
            f"num_tensors={len(tensor_items)} | "
            f"total_bytes={total_bytes / 1024 / 1024:.1f}MB"
        )

        return {
            "buffer": merged_buffer,
            "metadata": metadata,
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
        """Clean up IPC handles for specific infer_rank and version.

        Does NOT free pre-allocated NIXL send buffers — those persist for reuse.
        """
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

    def teardown_buffers(self, pair_name: str | None = None) -> None:
        """Free pre-allocated NIXL send buffers.

        Called during disconnect to release GPU memory held by reusable buffers.

        Args:
            pair_name: Specific pair to teardown; all pairs if None
        """
        if pair_name:
            keys_to_remove = [
                k for k in self._send_buffers if k.startswith(f"{pair_name}/")
            ]
        else:
            keys_to_remove = list(self._send_buffers.keys())

        for key in keys_to_remove:
            self._send_buffers.pop(key, None)
            self._send_buffer_meta.pop(key, None)

        if keys_to_remove:
            logger.info(
                f"Teardown NIXL buffers: freed {len(keys_to_remove)} send buffers"
            )
