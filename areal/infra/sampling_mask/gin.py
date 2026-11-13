# SPDX-License-Identifier: Apache-2.0
"""NCCL GIN transport, with bounded registered windows and explicit completion."""

import ctypes
import importlib.util
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def load_extension() -> ModuleType:
    """Build against the installed NCCL headers and library, never a bundled ABI."""
    from torch.utils.cpp_extension import load

    # torch.cuda.nccl.version() reports the wheel's compile-time NCCL version.
    # Query the loaded SONAME instead: LD_LIBRARY_PATH can select a newer ABI.
    library = ctypes.CDLL("libnccl.so.2")
    get_version = library.ncclGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    active_version = ctypes.c_int()
    if get_version(ctypes.byref(active_version)) != 0:
        raise RuntimeError("Cannot query the active NCCL runtime version")
    if active_version.value < 23100:
        raise RuntimeError(
            f"sampling_mask_transfer requires active NCCL >= 2.31; loaded {active_version.value}. "
            "Select the installed NCCL library in LD_LIBRARY_PATH before process startup."
        )
    spec = importlib.util.find_spec("nvidia.nccl")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "sampling_mask_transfer requires installed NCCL >= 2.31 headers"
        )
    root = Path(next(iter(spec.submodule_search_locations)))
    source = Path(__file__).with_suffix(".cu")
    return load(
        name="areal_sampling_mask_gin",
        sources=[str(source)],
        extra_include_paths=[str(root / "include")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[
            str(root / "lib" / "libnccl.so.2"),
            f"-Wl,-rpath,{root / 'lib'}",
        ],
        verbose=False,
    )


class GinEndpoint:
    """Single-owner two-peer session; one physical GPU per communicator rank.

    Reserve source windows before inference KV-cache sizing; register when the
    peer connects. The caller bootstraps the peer concurrently and verifies
    distinct GPU UUIDs before constructing either end.
    Only descriptors travel over the control plane; this API never sends IDs by
    HTTP, pickle, RTensor, or a host NCCL collective.
    """

    def __init__(
        self,
        unique_id: bytes,
        rank: int,
        device: int,
        slot_bytes: int = 8 << 20,
        reserved_buffer: int = 0,
    ):
        self.device = torch.device("cuda", device)
        self.rank = rank
        self.slot_bytes = slot_bytes
        self._lock = Lock()
        self._sequence = [0, 0]
        self._next_slot = 0
        self._endpoint = load_extension().Endpoint(
            unique_id, rank, device, slot_bytes, 2, 30.0, reserved_buffer
        )
        self.payload_bytes = 0
        self.chunks = 0

    def transfer(self, tensor: torch.Tensor) -> None:
        """Transfer one bounded contiguous CUDA byte chunk and wait for completion.

        Explicit completion provides backpressure and permits immediate source
        reuse. A persistent session/window is reused for every chunk and batch.
        """
        if (
            tensor.dtype != torch.uint8
            or tensor.device != self.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "GIN transfer needs contiguous uint8 on the endpoint device"
            )
        if not 0 < tensor.numel() <= self.slot_bytes:
            raise ValueError("GIN chunk exceeds its registered slot capacity")
        with self._lock:
            if self._endpoint is None:
                raise RuntimeError("Sampling-mask GIN endpoint is closed")
            # Producer completion is explicit: the native endpoint uses its own
            # stream and must never race a PyTorch pack/copy on another stream.
            torch.cuda.current_stream(self.device).synchronize()
            slot = self._next_slot
            self._sequence[slot] += 1
            try:
                self._endpoint.transfer(
                    tensor.data_ptr(), tensor.numel(), slot, self._sequence[slot]
                )
                self._endpoint.finish()
            except Exception:
                self._endpoint.abort()
                self._endpoint = None
                raise
            self._next_slot = (slot + 1) % len(self._sequence)
            self.payload_bytes += tensor.numel()
            self.chunks += 1

    def close(self) -> None:
        with self._lock:
            if self._endpoint is not None:
                try:
                    self._endpoint.close()
                except Exception:
                    self._endpoint.abort()
                    raise
                finally:
                    self._endpoint = None
