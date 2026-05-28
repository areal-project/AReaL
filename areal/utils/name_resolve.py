# SPDX-License-Identifier: Apache-2.0

# Implements a simple name resolving service, which can be considered as a distributed key-value dict.
import dataclasses
import os
import queue
import random
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

import ray

try:
    import etcd3
except Exception:
    etcd3 = None

from areal.utils import logging, timeutil

if TYPE_CHECKING:
    from areal.api.cli_args import NameResolveConfig

logger = logging.getLogger("NameResolve")


class ArgumentError(Exception):
    pass


class NameEntryExistsError(Exception):
    pass


class NameEntryNotFoundError(Exception):
    pass


class NameRecordRepository:
    def __del__(self):
        try:
            self.reset()
        except Exception as e:
            logger.info(f"Exception ignore when deleting NameResolveRepo {e}")

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        """Creates a name record in the central repository.

        In our semantics, the name record repository is essentially a multimap (i.e. Dict[str, Set[str]]).
        This class keeps a single name->value map, where the name can be non-unique while the value has to be.
        The class also deletes the (name, value) pair on exits (__exit__/__del__) if opted-in. In case of
        preventing unexpected exits (i.e. process crashes without calling graceful exits), an user may also
        want to specify time_to_live and call touch() regularly to allow a more consistent

        Args:
            name: The key of the record. It has to be a valid path-like string; e.g. "a/b/c". If the name
                already exists, the behaviour is defined by the `replace` argument.
            value: The value of the record. This can be any valid string.
            delete_on_exit: If the record shall be deleted when the repository closes.
            keepalive_ttl: If not None, adds a time-to-live in seconds for the record. The repository
                shall keep pinging the backend service with at least this frequency to make sure the name
                entry is alive during the lifetime of the repository. On the other hand, specifying this
                prevents stale keys caused by the scenario that a Python process accidentally crashes before
                calling delete().
            replace: If the name already exists, then replaces the current value with the supplied value if
                `replace` is True, or raises exception if `replace` is False.
        """
        raise NotImplementedError()

    def add_subentry(self, name, value, **kwargs):
        """Adds a sub-entry to the key-root `name`.

        The values is retrievable by get_subtree() given that no other
        entries use the name prefix.
        """
        sub_name = os.path.join(os.path.normpath(name), str(uuid.uuid4())[:8])
        self.add(sub_name, value, **kwargs)
        return sub_name

    def delete(self, name):
        """Deletes an existing record."""
        raise NotImplementedError()

    def clear_subtree(self, name_root):
        """Deletes all records whose names start with the path root name_root;
        specifically, whose name either is `name_root`, or starts with
        `name_root.rstrip("/") + "/"`."""
        raise NotImplementedError()

    def get(self, name):
        """Returns the value of the key.

        Raises NameEntryNotFoundError if not found.
        """
        raise NotImplementedError()

    def get_subtree(self, name_root):
        """Returns all values whose names start with the path root name_root;
        specifically, whose name either is `name_root`, or starts with
        `name_root.rstrip("/") + "/"`."""
        raise NotImplementedError()

    def find_subtree(self, name_root):
        """Returns all KEYS whose names start with the path root name_root."""
        raise NotImplementedError()

    def wait(self, name, timeout=None, poll_frequency=1):
        """Waits until a name appears.

        Raises:
             TimeoutError: if timeout exceeds.
        """
        start = time.monotonic()
        while True:
            try:
                return self.get(name)
            except NameEntryNotFoundError:
                pass
            if timeout is None or timeout > 0:
                time.sleep(
                    poll_frequency + random.random() * 0.1
                )  # To reduce concurrency.
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Timeout waiting for key '{name}' ({self.__class__.__name__})"
                )

    def reset(self):
        """Deletes all entries added via this repository instance's
        add(delete_on_exit=True)."""
        raise NotImplementedError()

    def watch_names(
        self,
        names: list[str],
        call_back: Callable,
        poll_frequency=15,
        wait_timeout=300,
    ):
        """Watch a name, execute call_back when key is deleted."""
        if isinstance(names, str):
            names = [names]

        q = queue.Queue(maxsize=len(names))
        for _ in range(len(names) - 1):
            q.put(0)

        def wrap_call_back():
            try:
                q.get_nowait()
            except queue.Empty:
                logger.info(f"Key {names} is gone. Executing callback {call_back}")
                call_back()

        for name in names:
            t = threading.Thread(
                target=self._watch_thread_run,
                args=(name, wrap_call_back, poll_frequency, wait_timeout),
                daemon=True,
            )
            t.start()

    def _watch_thread_run(self, name, call_back, poll_frequency, wait_timeout):
        self.wait(name, timeout=wait_timeout, poll_frequency=poll_frequency)
        while True:
            try:
                self.get(name)
                time.sleep(poll_frequency + random.random())
            except NameEntryNotFoundError:
                call_back()
                break


class MemoryNameRecordRepository(NameRecordRepository):
    """Stores all the records in a thread-local memory.

    Note that this is most likely for testing purposes:
    any distributed application is impossible to use this.
    """

    def __init__(self, log_events=False):
        self.__store = {}
        self.__to_delete = set()
        self.__log_events = log_events

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        if not name:
            raise ValueError(f"Invalid name: {name}")
        name = os.path.normpath(name)
        if self.__log_events:
            print(f"NameResolve: add {name} {value}")
        if name in self.__store and not replace:
            raise NameEntryExistsError(f"K={name} V={self.__store[name]} V2={value}")
        self.__store[name] = str(value)
        if delete_on_exit:
            self.__to_delete.add(name)

    def touch(self, name, value, new_time_to_live):
        raise NotImplementedError()

    def delete(self, name):
        if self.__log_events:
            print(f"NameResolve: delete {name}")
        if name not in self.__store:
            raise NameEntryNotFoundError(f"K={name}")
        if name in self.__to_delete:
            self.__to_delete.remove(name)
        del self.__store[name]

    def clear_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: clear_subtree {name_root}")
        name_root = os.path.normpath(name_root)
        for name in list(self.__store):
            if (
                name_root == "/"
                or name == name_root
                or name.startswith(name_root + "/")
            ):
                if name in self.__to_delete:
                    self.__to_delete.remove(name)
                del self.__store[name]

    def get(self, name):
        name = os.path.normpath(name)
        if name not in self.__store:
            raise NameEntryNotFoundError(f"K={name}")
        r = self.__store[name]
        if self.__log_events:
            print(f"NameResolve: get {name} -> {r}")
        return r

    def get_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: get_subtree {name_root}")
        name_root = os.path.normpath(name_root)
        rs = []
        for name, value in self.__store.items():
            if (
                name_root == "/"
                or name == name_root
                or name.startswith(name_root + "/")
            ):
                rs.append(value)
        return rs

    def find_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: find_subtree {name_root}")
        rs = []
        for name in self.__store:
            if (
                name_root == "/"
                or name == name_root
                or name.startswith(name_root + "/")
            ):
                rs.append(name)
        rs.sort()
        return rs

    def reset(self):
        for name in self.__to_delete:
            self.__store.pop(name)
        self.__to_delete = set()


class NfsNameRecordRepository(NameRecordRepository):
    def __init__(self, record_root="", **kwargs):
        self.__to_delete = set()
        self.record_root = record_root

    def __dir_path(self, name):
        if not self.record_root:
            raise RuntimeError(
                "The `record_root` of NfsNameRecordRepository is not properly reconfigured."
            )
        return os.path.join(self.record_root, name)

    def __file_path(self, name):
        return os.path.join(self.__dir_path(name), "ENTRY")

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        if not name:
            raise ValueError("Name cannot be empty")
        name = os.path.normpath(name)
        path = self.__file_path(name)
        while True:
            # To avoid concurrency issues when multiple processes
            # call makedirs on the same dirname of CPFS.
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                break
            except (NotADirectoryError, FileNotFoundError):
                pass
        if os.path.isfile(path) and not replace:
            raise NameEntryExistsError(path)
        local_id = str(uuid.uuid4())[:8]
        with open(path + f".tmp.{local_id}", "w") as f:
            f.write(str(value))
        os.rename(path + f".tmp.{local_id}", path)
        if delete_on_exit:
            self.__to_delete.add(name)

    def delete(self, name):
        path = self.__file_path(name)
        if not os.path.isfile(path):
            raise NameEntryNotFoundError(path)
        os.remove(path)
        while True:
            path = os.path.dirname(path)
            if path == self.record_root:
                break
            if len(os.listdir(path)) > 0:
                break
            shutil.rmtree(path, ignore_errors=True)
        if name in self.__to_delete:
            self.__to_delete.remove(name)

    def clear_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        if os.path.isdir(dir_path):
            logger.info("Removing name resolve path: %s", dir_path)
            shutil.rmtree(dir_path)
        else:
            logger.info("No such name resolve path: %s", dir_path)

    def get(self, name):
        name = os.path.normpath(name)
        path = self.__file_path(name)
        if not os.path.isfile(path):
            raise NameEntryNotFoundError(path)
        for _ in range(100):
            # HACK: dealing with the possible OSError: Stale file handle
            try:
                with open(path) as f:
                    return f.read().strip()
            except OSError as e:
                if e.errno == 116:
                    time.sleep(5e-3)
                    continue
                raise e
        raise RuntimeError(f"Failed to read value for {name}")

    def get_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        rs = []
        if os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                try:
                    if len(files) != 1:
                        continue
                    if files[0] != "ENTRY":
                        continue
                    key = root.removeprefix(self.record_root)
                    key = key.removeprefix("/")
                    rs.append(self.get(key))
                except NameEntryNotFoundError:
                    pass
        return rs

    def find_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        rs = []
        if os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                try:
                    if len(files) != 1:
                        continue
                    if files[0] != "ENTRY":
                        continue
                    key = root.removeprefix(self.record_root)
                    key = key.removeprefix("/")
                    rs.append(key)
                except NameEntryNotFoundError:
                    pass
        rs.sort()
        return rs

    def reset(self):
        for name in list(self.__to_delete):
            try:
                self.delete(name)
            except Exception:
                pass
        self.__to_delete = set()


class Etcd3NameRecordRepository(NameRecordRepository):
    """Implements a name record repository using etcd3 as the backend storage.

    This implementation provides distributed key-value storage with support for
    TTL-based expiration, atomic operations, and key watching functionality.
    """

    KEEPALIVE_POLL_FREQUENCY = 1

    @dataclasses.dataclass
    class _Entry:
        value: str
        lease_id: int | None = None
        keepalive_ttl: int | None = None
        keeper: timeutil.FrequencyControl | None = None

    def __init__(self, host=None, port=None, user=None, password=None, **kwargs):
        """Initialize the etcd3 name record repository.

        Args:
            host: etcd server host (defaults to ETCD_HOST)
            port: etcd server port (defaults to ETCD_PORT)
            user: etcd username for authentication (defaults to ETCD_USER)
            password: etcd password for authentication (defaults to ETCD_PASSWORD)
            **kwargs: Additional configuration parameters
        """

        super().__init__()
        self._lock = threading.Lock()

        # Set connection parameters
        self._host = host
        self._port = port
        self._user = user
        self._password = password

        # Connect to etcd
        self._client = etcd3.client(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
        )

        # Keep track of entries for cleanup and keepalive
        self._entries = {}
        self._keepalive_running = True
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_thread_run, daemon=True
        )
        self._keepalive_thread.start()

        self._to_delete = set()

        logger.debug(f"Connected to etcd3 at {self._host}:{self._port}")

    def __del__(self):
        """Clean up resources when the object is deleted."""
        self._keepalive_running = False
        if hasattr(self, "_keepalive_thread"):
            self._keepalive_thread.join(timeout=5)
        self.reset()
        if hasattr(self, "_client"):
            self._client.close()

    def _create_lease(self, ttl_seconds):
        """Create an etcd lease with the specified TTL.

        Args:
            ttl_seconds: Time-to-live in seconds

        Returns:
            The lease ID
        """
        lease = self._client.lease(ttl_seconds)
        return lease.id

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        """Add a key-value pair to etcd with optional TTL.

        Args:
            name: Key name
            value: Value to store
            delete_on_exit: Whether to delete the key when this object is destroyed
            keepalive_ttl: TTL in seconds for the key (default: 10)
            replace: Whether to replace an existing key

        Raises:
            NameEntryExistsError: If the key already exists and replace is False
        """
        if not name:
            raise ValueError(f"Invalid name: {name}")
        name = os.path.normpath(name)
        value = str(value)

        with self._lock:
            # Check if key exists when replace=False
            if not replace:
                existing_value, _ = self._client.get(name)
                if existing_value is not None:
                    raise NameEntryExistsError(
                        f"Key already exists: K={name} V={existing_value.decode()}"
                    )

            # Create lease for TTL if specified
            lease_id = None
            if keepalive_ttl is not None and keepalive_ttl > 0:
                lease_id = self._create_lease(keepalive_ttl)
                # Encode the string value to bytes
                self._client.put(name, value.encode("utf-8"), lease=lease_id)
                self._to_delete.add(name)
            else:
                # Encode the string value to bytes
                self._client.put(name, value.encode("utf-8"))
                if delete_on_exit:
                    self._to_delete.add(name)

            # Store entry information for keepalive management
            self._entries[name] = self._Entry(
                value=value,
                lease_id=lease_id,
                keepalive_ttl=keepalive_ttl,
                keeper=(
                    timeutil.FrequencyControl(frequency_seconds=keepalive_ttl / 3)
                    if keepalive_ttl
                    else None
                ),
            )

    def delete(self, name):
        """Delete a key from etcd.

        Args:
            name: Key to delete

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        with self._lock:
            self._delete_locked(name)
            if name in self._to_delete:
                self._to_delete.remove(name)

    def _delete_locked(self, name):
        """Delete a key from etcd with lock already acquired.

        Args:
            name: Key to delete

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        # First check if the key exists
        value, _ = self._client.get(name)
        if value is None:
            raise NameEntryNotFoundError(f"No such etcd entry to delete: {name}")

        # Clean up entry tracking
        if name in self._entries:
            del self._entries[name]

        # Delete from etcd
        self._client.delete(name)

    def clear_subtree(self, name_root):
        """Delete all keys with the given prefix.

        Args:
            name_root: Prefix to match keys against
        """
        with self._lock:
            count = 0
            name_root = os.path.normpath(name_root)
            # Get all keys with the prefix
            for key_metadata_tuple in self._client.get_prefix(name_root):
                key = key_metadata_tuple[1].key.decode(
                    "utf-8"
                )  # Extract the key from metadata
                # Remove from our tracking
                if key in self._entries:
                    del self._entries[key]
                # Delete from etcd
                self._client.delete(key)
                count += 1

            logger.debug(f"Deleted {count} etcd entries under {name_root}")

    def get_subtree(self, name_root):
        """Get all values with keys having the given prefix.

        Args:
            name_root: Prefix to match keys against

        Returns:
            List of values
        """
        with self._lock:
            rs = []
            name_root = os.path.normpath(name_root)
            for value_metadata_tuple in self._client.get_prefix(name_root):
                value = value_metadata_tuple[0].decode("utf-8")  # Extract the value
                rs.append(value)
            return sorted(rs)

    def find_subtree(self, name_root):
        """Find all keys with the given prefix.

        Args:
            name_root: Prefix to match keys against

        Returns:
            List of keys
        """
        with self._lock:
            rs = []
            for key_metadata_tuple in self._client.get_prefix(name_root):
                key = key_metadata_tuple[1].key.decode(
                    "utf-8"
                )  # Extract the key from metadata
                rs.append(key)
            return sorted(rs)

    def get(self, name):
        """Get the value for a key.

        Args:
            name: Key to retrieve

        Returns:
            The value as a string

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        name = os.path.normpath(name)
        with self._lock:
            return self._get_locked(name)

    def _get_locked(self, name):
        """Get a value with lock already acquired.

        Args:
            name: Key to retrieve

        Returns:
            The value as a string

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        value, _ = self._client.get(name)
        if value is None:
            raise NameEntryNotFoundError(f"No such etcd entry: {name}")
        return value.decode("utf-8")

    def reset(self):
        """Delete all keys added via this repository instance."""
        with self._lock:
            if hasattr(self, "_to_delete"):
                count = 0
                for name in self._to_delete:
                    if name in self._entries:
                        try:
                            self._delete_locked(name)
                            count += 1
                        except NameEntryNotFoundError:
                            pass
                self._to_delete = set()
                logger.info(f"Reset {count} saved etcd entries")

    def _keepalive_thread_run(self):
        """Background thread to keep leases alive."""
        while self._keepalive_running:
            time.sleep(self.KEEPALIVE_POLL_FREQUENCY)
            with self._lock:
                for name, entry in list(self._entries.items()):
                    if (
                        entry.keeper is not None
                        and entry.keepalive_ttl is not None
                        and entry.lease_id is not None
                        and entry.keeper.check()
                    ):
                        try:
                            # Refresh the lease
                            self._client.refresh_lease(entry.lease_id)
                        except Exception as e:
                            logger.error(
                                f"Failed to refresh lease for key: K={name} V={entry.value}. Error: {e}"
                            )

    def watch_names(
        self,
        names: list[str],
        call_back: Callable,
        poll_frequency=15,
        wait_timeout=300,
    ):
        """Watch keys and call back when they are deleted.

        Args:
            names: Keys to watch
            call_back: Function to call when any key is deleted
            poll_frequency: How often to check in seconds
            wait_timeout: Maximum time to wait for keys to exist
        """
        if isinstance(names, str):
            names = [names]

        q = queue.Queue(maxsize=len(names))
        for _ in range(len(names) - 1):
            q.put(0)

        def wrap_call_back():
            try:
                q.get_nowait()
            except queue.Empty:
                logger.info(f"Key {names} is gone. Executing callback {call_back}")
                call_back()

        # Use etcd's native watch capability for more efficient watching
        for name in names:
            # First wait for the key to exist
            self.wait(name, timeout=wait_timeout, poll_frequency=poll_frequency)

            # Start watching for key deletion
            watch_id = self._client.add_watch_callback(
                name, lambda event: self._watch_callback(event, wrap_call_back)
            )

            # Store watch ID for cleanup
            if not hasattr(self, "_watch_ids"):
                self._watch_ids = []
            self._watch_ids.append(watch_id)

    def _watch_callback(self, event, callback):
        """Process watch events and call back on deletion.

        Args:
            event: The etcd watch response (WatchResponse object)
            callback: Function to call when a key is deleted
        """
        # Iterate through the events in the WatchResponse
        for ev in event.events:
            # Check if this is a delete event
            if isinstance(ev, etcd3.events.DeleteEvent):
                logger.debug(f"Key {ev.key.decode()} was deleted. Executing callback.")
                callback()

    def _testonly_drop_cached_entry(self, name):
        """Used by unittest only to simulate the case that the process crashes.

        Args:
            name: Key to drop from local cache
        """
        with self._lock:
            if name in self._entries:
                del self._entries[name]
                logger.debug(f"Testonly: dropped key: {name}")


@ray.remote
class DistributedKVStore:
    """Ray actor implementing a distributed key-value store with TTL support."""

    def __init__(self):
        self.store = {}
        self.ttl_store = {}  # key -> expiry_time
        self.lease_store = {}  # key -> lease_id
        self.lease_counter = 0

    def put(self, key: str, value: str, lease_id: int | None = None):
        """Store a key-value pair with optional lease."""
        self.store[key] = value
        if lease_id is not None:
            self.lease_store[key] = lease_id
        return True

    def get(self, key: str):
        """Get value for a key, checking TTL expiry."""
        self._cleanup_expired()
        if key not in self.store:
            return None
        return self.store[key]

    def delete(self, key: str):
        """Delete a key and its associated metadata."""
        deleted = key in self.store
        self.store.pop(key, None)
        self.ttl_store.pop(key, None)
        self.lease_store.pop(key, None)
        return deleted

    def get_prefix(self, prefix: str):
        """Get all key-value pairs with keys matching the prefix."""
        self._cleanup_expired()
        result = []
        normalized_prefix = os.path.normpath(prefix)

        for key, value in self.store.items():
            normalized_key = os.path.normpath(key)
            # Check if key matches prefix (exact match or starts with prefix/)
            if normalized_key == normalized_prefix or normalized_key.startswith(
                normalized_prefix.rstrip("/") + "/"
            ):
                result.append((key, value))
        return result

    def delete_prefix(self, prefix: str):
        """Delete all keys matching the prefix."""
        self._cleanup_expired()
        normalized_prefix = os.path.normpath(prefix)
        keys_to_delete = []

        for key in self.store.keys():
            normalized_key = os.path.normpath(key)
            if normalized_key == normalized_prefix or normalized_key.startswith(
                normalized_prefix.rstrip("/") + "/"
            ):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self.delete(key)
        return len(keys_to_delete)

    def create_lease(self, ttl_seconds: int):
        """Create a lease with TTL."""
        self.lease_counter += 1
        lease_id = self.lease_counter
        expiry_time = time.time() + ttl_seconds
        return lease_id, expiry_time

    def put_with_lease(self, key: str, value: str, ttl_seconds: int):
        """Store key-value with TTL lease."""
        lease_id, expiry_time = self.create_lease(ttl_seconds)
        self.store[key] = value
        self.ttl_store[key] = expiry_time
        self.lease_store[key] = lease_id
        return lease_id

    def refresh_lease(self, key: str, ttl_seconds: int):
        """Refresh the lease for a key."""
        if key in self.store and key in self.lease_store:
            self.ttl_store[key] = time.time() + ttl_seconds
            return True
        return False

    def _cleanup_expired(self):
        """Remove expired keys."""
        current_time = time.time()
        expired_keys = []

        for key, expiry_time in self.ttl_store.items():
            if current_time > expiry_time:
                expired_keys.append(key)

        for key in expired_keys:
            self.delete(key)

    def get_all_keys(self):
        """Get all keys in the store."""
        self._cleanup_expired()
        return list(self.store.keys())


class RayNameResolveRepository:
    """Ray-based implementation of NameRecordRepository using distributed actors."""

    KEEPALIVE_POLL_FREQUENCY = 1

    @dataclasses.dataclass
    class _Entry:
        value: str
        lease_id: int | None = None
        keepalive_ttl: int | None = None
        keeper: timeutil.FrequencyControl | None = None

    def __init__(self, actor_name: str = "distributed_kv_store", **kwargs):
        """Initialize Ray-based name record repository.

        Args:
            actor_name: Name for the Ray actor (for sharing across processes)
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        self._lock = threading.Lock()
        self._actor_name = actor_name

        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Try to get existing actor or create new one
        try:
            self._kv_store = ray.get_actor(self._actor_name)
            logger.debug(
                f"Connected to existing Ray KV store actor: {self._actor_name}"
            )
        except ValueError:
            # Actor doesn't exist, create it
            self._kv_store = DistributedKVStore.options(
                name=self._actor_name, lifetime="detached"
            ).remote()
            logger.debug(f"Created new Ray KV store actor: {self._actor_name}")

        # Track entries for cleanup and keepalive
        self._entries = {}
        self._keepalive_running = True
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_thread_run, daemon=True
        )
        self._keepalive_thread.start()

        self._to_delete = set()

    def __del__(self):
        """Clean up resources when the object is deleted."""
        try:
            self.reset()
        except Exception as e:
            logger.info(
                f"Exception ignored when deleting RayNameResolveRepository: {e}"
            )

    def add(
        self,
        name: str,
        value: str,
        delete_on_exit: bool = True,
        keepalive_ttl: int | None = None,
        replace: bool = False,
    ):
        """Add a key-value pair to the distributed store.

        Args:
            name: Key name
            value: Value to store
            delete_on_exit: Whether to delete the key when this object is destroyed
            keepalive_ttl: TTL in seconds for the key
            replace: Whether to replace an existing key

        Raises:
            NameEntryExistsError: If the key already exists and replace is False
        """
        if not name:
            raise ValueError(f"Invalid name: {name}")
        name = os.path.normpath(name)
        value = str(value)

        with self._lock:
            # Check if key exists when replace=False
            if not replace:
                existing_value = ray.get(self._kv_store.get.remote(name))
                if existing_value is not None:
                    raise NameEntryExistsError(
                        f"Key already exists: K={name} V={existing_value}"
                    )

            # Store with or without TTL
            lease_id = None
            if keepalive_ttl is not None and keepalive_ttl > 0:
                lease_id = ray.get(
                    self._kv_store.put_with_lease.remote(name, value, keepalive_ttl)
                )
                self._to_delete.add(name)
            else:
                ray.get(self._kv_store.put.remote(name, value))
                if delete_on_exit:
                    self._to_delete.add(name)

            # Store entry information for keepalive management
            self._entries[name] = self._Entry(
                value=value,
                lease_id=lease_id,
                keepalive_ttl=keepalive_ttl,
                keeper=(
                    timeutil.FrequencyControl(frequency_seconds=keepalive_ttl / 3)
                    if keepalive_ttl
                    else None
                ),
            )

    def add_subentry(self, name: str, value: str, **kwargs):
        """Add a sub-entry to the key-root `name`."""
        sub_name = os.path.join(os.path.normpath(name), str(uuid.uuid4())[:8])
        self.add(sub_name, value, **kwargs)
        return sub_name

    def delete(self, name: str):
        """Delete a key from the distributed store.

        Args:
            name: Key to delete

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        with self._lock:
            self._delete_locked(name)
            if name in self._to_delete:
                self._to_delete.remove(name)

    def _delete_locked(self, name: str):
        """Delete a key with lock already acquired."""
        # Check if key exists
        existing_value = ray.get(self._kv_store.get.remote(name))
        if existing_value is None:
            raise NameEntryNotFoundError(f"No such entry to delete: {name}")

        # Clean up entry tracking
        if name in self._entries:
            del self._entries[name]

        # Delete from store
        ray.get(self._kv_store.delete.remote(name))

    def clear_subtree(self, name_root: str):
        """Delete all keys with the given prefix."""
        with self._lock:
            name_root = os.path.normpath(name_root)
            count = ray.get(self._kv_store.delete_prefix.remote(name_root))

            # Clean up local tracking for deleted keys
            keys_to_remove = []
            for key in self._entries.keys():
                normalized_key = os.path.normpath(key)
                if normalized_key == name_root or normalized_key.startswith(
                    name_root.rstrip("/") + "/"
                ):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._entries[key]

            logger.debug(f"Deleted {count} entries under {name_root}")

    def get(self, name: str):
        """Get the value for a key.

        Args:
            name: Key to retrieve

        Returns:
            The value as a string

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        name = os.path.normpath(name)
        with self._lock:
            return self._get_locked(name)

    def _get_locked(self, name: str):
        """Get a value with lock already acquired."""
        value = ray.get(self._kv_store.get.remote(name))
        if value is None:
            raise NameEntryNotFoundError(f"No such entry: {name}")
        return value

    def get_subtree(self, name_root: str):
        """Get all values with keys having the given prefix."""
        with self._lock:
            name_root = os.path.normpath(name_root)
            pairs = ray.get(self._kv_store.get_prefix.remote(name_root))
            values = [value for key, value in pairs]
            return sorted(values)

    def find_subtree(self, name_root: str):
        """Find all keys with the given prefix."""
        with self._lock:
            name_root = os.path.normpath(name_root)
            pairs = ray.get(self._kv_store.get_prefix.remote(name_root))
            keys = [key for key, value in pairs]
            return sorted(keys)

    def wait(self, name: str, timeout: float | None = None, poll_frequency: float = 1):
        """Wait until a name appears.

        Raises:
            TimeoutError: if timeout exceeds.
        """
        start = time.monotonic()
        while True:
            try:
                return self.get(name)
            except NameEntryNotFoundError:
                pass
            if timeout is None or timeout > 0:
                time.sleep(
                    poll_frequency + random.random() * 0.1
                )  # To reduce concurrency.
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Timeout waiting for key '{name}' ({self.__class__.__name__})"
                )

    def reset(self):
        """Delete all keys added via this repository instance."""
        self._keepalive_running = False
        if hasattr(self, "_keepalive_thread"):
            self._keepalive_thread.join(timeout=5)

        with self._lock:
            count = 0
            for name in list(self._to_delete):
                try:
                    self._delete_locked(name)
                    count += 1
                except NameEntryNotFoundError:
                    pass
            self._to_delete = set()
            self._entries = {}
            logger.debug(f"Reset {count} saved entries")

    def watch_names(
        self,
        names: list[str],
        call_back: Callable,
        poll_frequency: float = 15,
        wait_timeout: float = 300,
    ):
        """Watch keys and call back when they are deleted.

        Args:
            names: Keys to watch
            call_back: Function to call when any key is deleted
            poll_frequency: How often to check in seconds
            wait_timeout: Maximum time to wait for keys to exist
        """
        if isinstance(names, str):
            names = [names]

        q = queue.Queue(maxsize=len(names))
        for _ in range(len(names) - 1):
            q.put(0)

        def wrap_call_back():
            try:
                q.get_nowait()
            except queue.Empty:
                logger.info(f"Key {names} is gone. Executing callback {call_back}")
                call_back()

        for name in names:
            t = threading.Thread(
                target=self._watch_thread_run,
                args=(name, wrap_call_back, poll_frequency, wait_timeout),
                daemon=True,
            )
            t.start()

    def _watch_thread_run(
        self, name: str, call_back: Callable, poll_frequency: float, wait_timeout: float
    ):
        """Background thread to watch a key for deletion."""
        self.wait(name, timeout=wait_timeout, poll_frequency=poll_frequency)
        while True:
            try:
                self.get(name)
                time.sleep(poll_frequency + random.random())
            except NameEntryNotFoundError:
                call_back()
                break

    def _keepalive_thread_run(self):
        """Background thread to keep leases alive."""
        while self._keepalive_running:
            time.sleep(self.KEEPALIVE_POLL_FREQUENCY)
            with self._lock:
                for name, entry in list(self._entries.items()):
                    if (
                        entry.keeper is not None
                        and entry.keepalive_ttl is not None
                        and entry.lease_id is not None
                        and entry.keeper.check()
                    ):
                        try:
                            # Refresh the lease
                            success = ray.get(
                                self._kv_store.refresh_lease.remote(
                                    name, entry.keepalive_ttl
                                )
                            )
                            if not success:
                                logger.warning(
                                    f"Failed to refresh lease for key: {name}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to refresh lease for key: K={name} V={entry.value}. Error: {e}"
                            )


def make_repository(args: "NameResolveConfig"):
    if args.type == "nfs":
        repo = NfsNameRecordRepository(args.nfs_record_root)
        os.makedirs(repo.record_root, exist_ok=True)
        return repo
    elif args.type == "etcd3":
        from areal.utils.network import split_hostport

        host, port = split_hostport(args.etcd3_addr)
        return Etcd3NameRecordRepository(host=host, port=int(port))
    elif args.type == "ray":
        return RayNameResolveRepository(actor_name=args.ray_actor_name)
    else:
        raise NotImplementedError(f"No such name resolver: {args.type}")


DEFAULT_REPOSITORY = NfsNameRecordRepository()
add = DEFAULT_REPOSITORY.add
add_subentry = DEFAULT_REPOSITORY.add_subentry
delete = DEFAULT_REPOSITORY.delete
clear_subtree = DEFAULT_REPOSITORY.clear_subtree
get = DEFAULT_REPOSITORY.get
get_subtree = DEFAULT_REPOSITORY.get_subtree
find_subtree = DEFAULT_REPOSITORY.find_subtree
wait = DEFAULT_REPOSITORY.wait
reset = DEFAULT_REPOSITORY.reset
watch_names = DEFAULT_REPOSITORY.watch_names


def reconfigure(config: "NameResolveConfig"):
    global DEFAULT_REPOSITORY
    global \
        add, \
        add_subentry, \
        delete, \
        clear_subtree, \
        get, \
        get_subtree, \
        find_subtree, \
        wait, \
        reset, \
        watch_names
    DEFAULT_REPOSITORY = make_repository(config)
    add = DEFAULT_REPOSITORY.add
    add_subentry = DEFAULT_REPOSITORY.add_subentry
    delete = DEFAULT_REPOSITORY.delete
    clear_subtree = DEFAULT_REPOSITORY.clear_subtree
    get = DEFAULT_REPOSITORY.get
    get_subtree = DEFAULT_REPOSITORY.get_subtree
    find_subtree = DEFAULT_REPOSITORY.find_subtree
    wait = DEFAULT_REPOSITORY.wait
    reset = DEFAULT_REPOSITORY.reset
    watch_names = DEFAULT_REPOSITORY.watch_names
