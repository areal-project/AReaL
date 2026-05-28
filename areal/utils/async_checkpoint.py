# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.storage import StorageWriter

from areal.utils.logging import getLogger

logger = getLogger("AsyncCheckpoint")


class AsyncMode(str, enum.Enum):
    AUTO = "auto"
    SYNC = "sync"
    ASYNC = "async"


class AsyncCheckpointManager:
    """Manage async DCP checkpoint saves for a single engine.

    Supports two modes:
    - SYNC: synchronous dcp.save()
    - ASYNC: process-based async with pinned memory staging,
      extra CPU pinned memory proportional to per-rank model shard size
    """

    def __init__(self, async_mode: AsyncMode):
        self.async_mode = async_mode
        self._save_future: Future | None = None
        self._staging_future: Future | None = None
        self._stager: DefaultStager | None = None
        self._bg_future: Future | None = None

        # Sequential executor for post-upload work (consolidation).
        self._executor: ThreadPoolExecutor | None = None
        if async_mode == AsyncMode.ASYNC:
            self._executor = ThreadPoolExecutor(max_workers=1)

        # Two separate gloo PGs: eliminates need for gating.
        # _pg: used by dcp.async_save on the main thread.
        # _consolidation_pg: used by consolidation barriers on the bg thread.
        self._pg: dist.ProcessGroup | None = None
        self._consolidation_pg: dist.ProcessGroup | None = None
        if async_mode == AsyncMode.ASYNC and dist.is_initialized():
            self._pg = dist.new_group(backend="gloo")
            self._consolidation_pg = dist.new_group(backend="gloo")

    @property
    def is_async(self) -> bool:
        return self.async_mode == AsyncMode.ASYNC

    @property
    def process_group(self) -> dist.ProcessGroup | None:
        return self._pg

    @property
    def consolidation_process_group(self) -> dist.ProcessGroup | None:
        return self._consolidation_pg

    def save(
        self,
        state_dict: dict,
        *,
        storage_writer: StorageWriter | None = None,
        checkpoint_id: str | None = None,
        post_fn: Callable | None = None,
    ) -> None:
        """Initiate a save. Sync mode blocks; async mode returns immediately."""
        if self.async_mode == AsyncMode.ASYNC and self._pg is None:
            raise RuntimeError(
                "Async checkpoint requires a process group, but dist was not "
                "initialized when AsyncCheckpointManager was created"
            )

        self._check_bg_error()

        if self._bg_future is not None:
            logger.warning("Background consolidation still running. Blocking.")
            try:
                self._bg_future.result()
            finally:
                self._bg_future = None

        self._wait_for_staging()
        self._wait_for_upload()

        if self.async_mode == AsyncMode.SYNC:
            dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_id,
            )
        elif self.async_mode == AsyncMode.ASYNC:
            if self._stager is None:
                self._stager = DefaultStager(
                    StagingOptions(
                        use_pinned_memory=True,
                        use_shared_memory=True,
                        use_async_staging=True,
                        use_non_blocking_copy=True,
                    )
                )
            result = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_id,
                process_group=self._pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self._stager,
            )
            assert isinstance(result, AsyncSaveResponse)
            self._save_future = result.upload_completion
            self._staging_future = result.staging_completion

        if post_fn is not None:
            self._submit_post_fn(post_fn)

    def _submit_post_fn(self, fn: Callable) -> None:
        """Submit a bg job: wait for upload -> run *fn*."""
        upload_future = self._save_future

        def _upload_then_consolidate():
            if upload_future is not None:
                try:
                    upload_future.result()
                except Exception:
                    logger.error("Async checkpoint upload failed", exc_info=True)
            fn()

        if self._executor is not None:
            self._bg_future = self._executor.submit(_upload_then_consolidate)
        else:
            _upload_then_consolidate()

    def maybe_wait_for_staging(self) -> None:
        """Wait for staging (GPU->CPU) + check for bg errors.

        Called by the training loop before optimizer.step().
        """
        self._check_bg_error()
        self._wait_for_staging()

    def _check_bg_error(self) -> None:
        """Non-blocking: only raises if bg job is done and failed."""
        if self._bg_future is not None and self._bg_future.done():
            try:
                self._bg_future.result()
            finally:
                self._bg_future = None

    def _wait_for_staging(self) -> None:
        """Wait for staging (GPU->CPU) to complete."""
        if self._staging_future is not None:
            try:
                self._staging_future.result()
            except Exception:
                logger.error("Async checkpoint staging failed", exc_info=True)
            finally:
                self._staging_future = None

    def _wait_for_upload(self) -> None:
        """Wait for the previous upload to complete.

        Needed before new staging to ensure shared memory is free.
        """
        if self._save_future is not None:
            try:
                self._save_future.result()
            except Exception:
                pass  # error is logged by the bg thread
            finally:
                self._save_future = None

    def finalize(self) -> None:
        """Call at training end: wait for pending operations and release resources."""
        try:
            self._wait_for_staging()
            if self._bg_future is not None:
                try:
                    self._bg_future.result()
                except Exception:
                    logger.error("Final bg work failed", exc_info=True)
                    raise
                finally:
                    self._bg_future = None
            self._wait_for_upload()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
            if self._stager is not None:
                self._stager.close()
                self._stager = None
            if self._pg is not None:
                dist.destroy_process_group(self._pg)
                self._pg = None
            if self._consolidation_pg is not None:
                dist.destroy_process_group(self._consolidation_pg)
                self._consolidation_pg = None
