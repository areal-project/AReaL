# SPDX-License-Identifier: Apache-2.0

"""RDataset — remote dataset proxy, analogous to RTensor for tensors.

Provides a map-style dataset interface backed by remote data workers.
Samples are fetched over HTTP and cached locally via a sampler-aware
prefetch buffer for near-zero latency ``__getitem__`` calls.

Lifecycle
---------
1. Created by ``get_custom_dataset()`` — stores metadata only (unconnected).
2. Trainer calls ``connect(controller, ...)`` — registers with the data
   service and enables fetching.
3. ``create_dataloader()`` wraps the dataset with ``_PrefetchAwareSampler``
   which triggers prefetch on each ``set_epoch`` call.
4. ``__getitem__(idx)`` pops from the prefetch buffer (cache hit) or falls
   back to a blocking HTTP fetch (cache miss).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from torch.utils.data import DistributedSampler

from areal.utils import logging

if TYPE_CHECKING:
    from areal.infra.data_service.controller.controller import DataController

logger = logging.getLogger("RDataset")


class _PrefetchBuffer:
    """Background thread that fetches samples from remote workers.

    Fetches proceed in the exact index order that the
    ``DistributedSampler`` will request, ensuring near-100 % cache hit
    rate.  The buffer is bounded by *max_cached*; when full the
    prefetch thread pauses until space is freed by ``get()`` calls.

    Parameters
    ----------
    fetch_fn : callable
        ``fn(indices: list[int]) -> list[Any]``  — batch-fetch samples
        by index from remote workers.
    chunk_size : int
        Number of indices to fetch in a single HTTP round-trip.
    max_cached : int
        Maximum number of samples to hold in the local cache before
        the prefetch thread pauses.
    """

    def __init__(
        self,
        fetch_fn: Any,
        chunk_size: int = 64,
        max_cached: int = 512,
    ) -> None:
        self._fetch_fn = fetch_fn
        self._chunk_size = chunk_size
        self._max_cached = max_cached

        self._cache: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._indices: list[int] = []
        self._pos: int = 0

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._space_available = threading.Event()
        self._space_available.set()

    # -- Public API --------------------------------------------------------

    def set_index_order(self, indices: list[int]) -> None:
        """Reset the cache and start prefetching in *indices* order.

        Called by ``_PrefetchAwareSampler.set_epoch`` at the beginning
        of each epoch.
        """
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

        with self._lock:
            self._cache.clear()
            self._indices = list(indices)
            self._pos = 0

        self._stop.clear()
        self._space_available.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def get(self, idx: int) -> Any:
        """Return the sample for *idx*.

        Pops from the prefetch cache on hit.  On miss, performs a
        blocking single-index fetch.
        """
        with self._lock:
            if idx in self._cache:
                sample = self._cache.pop(idx)
                self._space_available.set()
                return sample

        logger.debug("Prefetch cache miss for index %d, fetching directly", idx)
        return self._fetch_fn([idx])[0]

    def stop(self) -> None:
        """Signal the prefetch thread to stop."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    # -- Background thread -------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if len(self._cache) >= self._max_cached:
                    self._space_available.clear()

                if self._pos >= len(self._indices):
                    break

                chunk = self._indices[self._pos : self._pos + self._chunk_size]
                self._pos += len(chunk)

            while not self._space_available.wait(timeout=0.1):
                if self._stop.is_set():
                    return

            if self._stop.is_set():
                return

            try:
                samples = self._fetch_fn(chunk)
            except Exception:
                logger.exception(
                    "Prefetch failed for chunk starting at pos %d",
                    self._pos - len(chunk),
                )
                time.sleep(0.5)
                with self._lock:
                    self._pos -= len(chunk)
                continue

            with self._lock:
                for idx, sample in zip(chunk, samples):
                    self._cache[idx] = sample


class RDataset:
    """Remote dataset proxy — map-style dataset backed by remote workers.

    Analogous to :class:`RTensor` for tensors.  Locally stores only
    dataset metadata; actual samples are fetched lazily from the
    distributed data loading service via the :class:`_PrefetchBuffer`.

    Parameters
    ----------
    path : str
        Dataset path (HuggingFace dataset name or local path).
    type : str
        Dataset type (``"sft"``, ``"rl"``, ``"rw"``).
    split : str | None
        Dataset split to load on workers.
    max_length : int | None
        Maximum sequence length for tokenisation on workers.
    dataset_kwargs : dict | None
        Extra keyword arguments forwarded to the dataset loader.
    """

    def __init__(
        self,
        path: str,
        type: str = "rl",
        split: str | None = None,
        max_length: int | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._path = path
        self._type = type
        self._split = split
        self._max_length = max_length
        self._dataset_kwargs = dataset_kwargs or {}

        self._controller: DataController | None = None
        self._api_key: str = ""
        self._dataset_id: str = ""
        self._total_samples: int = 0
        self._connected: bool = False
        self._prefetch_buffer: _PrefetchBuffer | None = None

    # -- Connection --------------------------------------------------------

    def connect(
        self,
        controller: DataController,
        dataset_id: str,
        tokenizer_or_processor_path: str = "",
        shuffle: bool = True,
        drop_last: bool = True,
        prefetch_chunk_size: int = 64,
        prefetch_max_cached: int = 512,
    ) -> None:
        """Register with *controller* and enable data fetching.

        Called by the trainer after the ``DataController`` is
        initialised.  The controller broadcasts dataset loading to all
        workers and returns the total sample count.
        """
        if self._connected:
            raise RuntimeError("RDataset is already connected")

        handle = controller.register_dataset(
            dataset_id=dataset_id,
            dataset_path=self._path,
            dataset_type=self._type,
            split=self._split or "train",
            max_length=self._max_length,
            dataset_kwargs=self._dataset_kwargs,
            tokenizer_or_processor_path=tokenizer_or_processor_path,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        self._controller = controller
        self._api_key = handle["api_key"]
        self._dataset_id = handle["dataset_id"]
        self._total_samples = handle["total_samples"]
        self._connected = True
        self._prefetch_buffer = _PrefetchBuffer(
            fetch_fn=self._fetch_samples,
            chunk_size=prefetch_chunk_size,
            max_cached=prefetch_max_cached,
        )
        logger.info(
            "RDataset connected: id=%s, total_samples=%d",
            self._dataset_id,
            self._total_samples,
        )

    # -- Map-style dataset interface ---------------------------------------

    def __len__(self) -> int:
        if not self._connected:
            raise RuntimeError(
                "RDataset is not connected to a DataController. "
                "Call connect() before using the dataset."
            )
        return self._total_samples

    def __getitem__(self, idx: int) -> Any:
        if not self._connected or self._prefetch_buffer is None:
            raise RuntimeError(
                "RDataset is not connected to a DataController. "
                "Call connect() before using the dataset."
            )
        return self._prefetch_buffer.get(idx)

    # -- Prefetch control (called by _PrefetchAwareSampler) ----------------

    def _start_prefetch(self, indices: list[int]) -> None:
        """Kick off the prefetch buffer in *indices* order."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.set_index_order(indices)

    # -- Remote fetch ------------------------------------------------------

    def _fetch_samples(self, indices: list[int]) -> list[Any]:
        """Batch-fetch samples by index from remote workers via the gateway."""
        assert self._controller is not None
        from areal.infra.rpc.serialization import deserialize_value

        resp = self._controller._gateway_post(
            "/v1/samples/fetch",
            self._api_key,
            {"indices": indices},
        )
        return [deserialize_value(s) for s in resp["samples"]]

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Stop prefetching and unregister from the controller."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.stop()
            self._prefetch_buffer = None
        if self._connected and self._controller is not None:
            try:
                self._controller.unregister_dataset(self._dataset_id)
            except Exception:
                logger.debug(
                    "Failed to unregister dataset %s (expected during teardown)",
                    self._dataset_id,
                )
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected


class _PrefetchAwareSampler(DistributedSampler):
    """``DistributedSampler`` that triggers ``RDataset`` prefetch on epoch change.

    When ``cycle_dataloader`` calls ``sampler.set_epoch(epoch)``, this
    sampler generates the deterministic index order for the new epoch
    and passes it to the ``RDataset``'s prefetch buffer so that
    samples are fetched in the order the ``DataLoader`` will request
    them.
    """

    def __init__(self, dataset: RDataset, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset, *args, **kwargs)
        self._rdataset = dataset
        self._trigger_prefetch()

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._trigger_prefetch()

    def _trigger_prefetch(self) -> None:
        indices = list(super().__iter__())
        self._rdataset._start_prefetch(indices)
