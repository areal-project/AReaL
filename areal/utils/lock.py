# SPDX-License-Identifier: Apache-2.0

import time
import uuid

import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_store


class DistributedLock:
    def __init__(self, name, backoff=0.05, namespace=None):
        if not dist.is_initialized():
            raise RuntimeError("Process group not initialized")

        self.store = _get_default_store()
        ns = namespace or f"ws{dist.get_world_size()}"
        self.base = f"lock/{ns}/{name}"
        self.key_counter = f"{self.base}/counter"
        self.key_owner = f"{self.base}/owner"
        self.backoff = backoff
        self.token = None

        try:
            self.store.add(self.key_counter, 0)
        except RuntimeError:
            pass

    def acquire(self, timeout=None):
        start = time.perf_counter()
        sleep = self.backoff
        my_token = f"{dist.get_rank()}:{uuid.uuid4()}".encode()

        while True:
            try:
                current = self.store.add(self.key_counter, 1)
                if current == 1:
                    self._set_owner(my_token)
                    self.token = my_token
                    return True

                self._rollback_counter()
            except RuntimeError:
                # A RuntimeError can occur from store.add under contention.
                # We'll let the loop back off and retry.
                pass

            if timeout is not None and (time.perf_counter() - start) > timeout:
                return False

            time.sleep(sleep)
            sleep = min(sleep * 1.5, 0.5)

    def release(self):
        if self.token is None:
            raise RuntimeError("Lock not held by this process")

        owner = self._get_owner()

        if owner is not None and owner != self.token:
            raise RuntimeError("Lock owner mismatch; refusing to release")

        self._clear_owner()
        self._rollback_counter()
        self.token = None

    def _rollback_counter(self):
        delay = self.backoff
        while True:
            try:
                self.store.add(self.key_counter, -1)
                return
            except RuntimeError:
                time.sleep(delay)
                delay = min(delay * 1.5, 0.5)

    def _set_owner(self, token: bytes):
        try:
            self.store.set(self.key_owner, token)
        except RuntimeError as exc:
            self._rollback_counter()
            raise RuntimeError("Failed to record lock owner") from exc

    def _get_owner(self) -> bytes | None:
        try:
            value = self.store.get(self.key_owner)
            return value if value else None
        except RuntimeError:
            return None

    def _clear_owner(self):
        try:
            self.store.set(self.key_owner, b"")
        except RuntimeError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
