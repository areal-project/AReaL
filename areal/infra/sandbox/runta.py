# SPDX-License-Identifier: Apache-2.0

"""Runta sandbox runners for executing Python code in cloud runtimes.

`RuntaRunner` is the synchronous entry point for non-async user code (reward
functions, evaluation scripts); `AsyncRuntaRunner` is its asyncio twin for
rollout workflows. Both create one runtime lazily, keep it warm across
`run()` calls, and delete it on close. A runtime that stopped running (it
reached `max_lifetime`, or crashed) is replaced by a fresh one on the next
call, so files written to the old one are gone.

A Runta runtime is a full machine with Python and Docker preinstalled. By
default code runs on the runtime's own Python, which has only the standard
library: `import numpy`, or any other third-party package, fails there with
`ModuleNotFoundError`, and the runners install nothing. For code that needs
packages, pass `image` (a Docker image that contains them and has `python3`
and `sh`) to run it inside a container started from that image, or
`checkpoint_id` to restore the runtime from a Runta checkpoint of a runtime
you prepared with the packages installed. A restored runtime keeps the
checkpoint's CPU and memory: `vcpus` and `memory_mib` only apply to fresh
runtimes.

A runtime is held on a renewable lease, so one whose runner process was
killed does not keep running: it is created to shut down after
`lease_seconds` (default ten minutes) and, while its runner is alive, that
clock is restarted about every `lease_seconds / 3`. `max_lifetime` stays the
hard cap on a runtime's age. `lease_seconds` must be longer than the runtime
takes to prepare (the image pull), and `close()` is still the way to release
a runtime promptly.

Credentials come from `RUNTA_TOKEN` (or the config file written by
`runta login`); `RUNTA_ENDPOINT` selects a non-default API endpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from areal.utils import logging

logger = logging.getLogger("RuntaRunner")

_CONTAINER = "areal-sandbox"
_TIMEOUT_EXIT_CODE = 124
_DEFAULT_VCPUS = 1
_DEFAULT_MEMORY_MIB = 1024
_DEFAULT_LEASE_SECONDS = 600
_KILL_TIMEOUT = 30

# Runta's exec timeout kills only the `docker exec` client on the runtime; the
# program inside the container keeps running. In image mode the program
# therefore runs under `_TRACK`, which records its process group in a PID file
# (removed again when the program ends in time), and `_KILL` stops a timed-out
# program through that file. `setsid -w` gives the program a group of its own
# where the exec'd shell does not already lead one. Neither script reads
# stdin, which carries the program to `python3 -`.
_TRACKED = '{ echo $$ > "$1"; } 2>/dev/null; python3 -; rc=$?; rm -f "$1"; exit $rc'
_TRACK = (
    "if setsid -w true >/dev/null 2>&1; then "
    f'exec setsid -w sh -c {shlex.quote(_TRACKED)} areal-exec "$@"; fi; {_TRACKED}'
)
_KILL = (
    'pid=$(cat "$1" 2>/dev/null) || exit 0; rm -f "$1"; '
    'kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; exit 0'
)


def _load_runta_sdk() -> Any:
    try:
        import runta
    except ImportError as exc:
        raise ImportError(
            "RuntaRunner requires the optional 'runta-sdk' dependency. Install it with `uv pip install 'runta-sdk>=0.2.6,<0.3'`."
        ) from exc
    return runta


def _last_non_empty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


@dataclass
class RuntaRunResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str | None = None


class _RuntaRunnerBase:
    """Configuration and command construction shared by the sync and async runners."""

    def __init__(
        self,
        *,
        image: str | None = None,
        checkpoint_id: str | None = None,
        vcpus: int = _DEFAULT_VCPUS,
        memory_mib: int = _DEFAULT_MEMORY_MIB,
        env_vars: dict[str, str] | None = None,
        network_block_all: bool = False,
        create_timeout: float = 240.0,
        default_timeout: int = 30,
        max_lifetime: int = 3600,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
        endpoint: str | None = None,
    ):
        self._sdk = _load_runta_sdk()
        if image is not None and checkpoint_id is not None:
            raise ValueError("Pass either checkpoint_id or image, not both")
        if lease_seconds < 1:
            # Zero would disable Runta's shutdown policy instead of shortening it.
            raise ValueError("lease_seconds must be at least 1")
        if checkpoint_id is not None and (
            vcpus != _DEFAULT_VCPUS or memory_mib != _DEFAULT_MEMORY_MIB
        ):
            logger.warning(
                "vcpus and memory_mib are ignored with checkpoint_id: the runtime "
                "keeps the resources of checkpoint %s.",
                checkpoint_id,
            )

        self.image = image
        self.checkpoint_id = checkpoint_id
        self.vcpus = vcpus
        self.memory_mib = memory_mib
        self.env_vars = dict(env_vars or {})
        self.network_block_all = network_block_all
        self.create_timeout = create_timeout
        self.default_timeout = default_timeout
        self.max_lifetime = max_lifetime
        self.lease_seconds = lease_seconds
        self.endpoint = endpoint
        self._lease_interval = lease_seconds / 3
        self._client = None
        self._runtime = None
        self._heartbeat = None
        self._closed = False

    def _client_kwargs(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint} if self.endpoint else {}

    def _create_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            # Server-side backstop: a runner that dies without close(), even by
            # SIGKILL, stops renewing this lease, so Runta shuts its runtime
            # down within lease_seconds and then deletes it.
            "auto_shutdown_after_secs": min(self.lease_seconds, self.max_lifetime),
            "auto_delete_after_secs": 600,
            # Readiness is awaited separately: a create that waits and fails
            # never returns the handle needed to delete the runtime.
            "wait": False,
        }
        if self.checkpoint_id is not None:
            # A restore keeps the checkpoint's resources; the SDK ignores
            # vcpus and memory_mib here.
            kwargs["checkpoint_id"] = self.checkpoint_id
        else:
            kwargs["vcpus"] = self.vcpus
            kwargs["memory_mib"] = self.memory_mib
        return kwargs

    def _runtime_name(self) -> str:
        return f"areal-{uuid.uuid4().hex[:12]}"

    def _lifetime_deadline(self) -> float:
        """Monotonic time at which a runtime created now has used up max_lifetime."""
        return time.monotonic() + self.max_lifetime

    def _lease_needs_renewal(self) -> bool:
        # A runtime whose whole lifetime fits in the first lease just expires.
        return self.max_lifetime > self.lease_seconds

    def _lease_renewal(self, deadline: float) -> int | None:
        """Seconds to restart the shutdown clock with, or None once max_lifetime is used up."""
        remaining = int(deadline - time.monotonic())
        if remaining < 1:
            return None
        return min(self.lease_seconds, remaining)

    def _lease_is_final(self, seconds: int) -> bool:
        # A renewal cut short by max_lifetime already ends at the hard cap.
        return seconds < self.lease_seconds

    def _log_renewal_failure(self, runtime_id: str, exc: Exception) -> None:
        logger.warning(
            "Could not renew the lease of Runta runtime %s (%s); retrying in %.0fs. "
            "Runta shuts the runtime down if no renewal succeeds within "
            "lease_seconds=%d.",
            runtime_id,
            exc,
            self._lease_interval,
            self.lease_seconds,
        )

    def _is_conflict(self, exc: Exception) -> bool:
        return getattr(exc, "status_code", None) == 409

    def _container_argv(self) -> list[str]:
        assert self.image is not None
        return [
            "docker",
            "run",
            "-d",
            "--name",
            _CONTAINER,
            "--network",
            "host",
            self.image,
            "sleep",
            "infinity",
        ]

    def _blocked_egress_policy(self):
        # An allowlist with no hosts denies all outbound traffic. It is applied
        # after the image pull, which needs the network.
        return self._sdk.EgressPolicy(
            mode=self._sdk.EgressMode.ALLOWLIST, allowed_hosts=[]
        )

    def _is_gone_status(self, status: Any) -> bool:
        statuses = self._sdk.RuntimeStatus
        return status in (
            statuses.SHUTDOWN,
            statuses.DELETING,
            statuses.ERROR,
            statuses.CRASHED,
        )

    def _new_pidfile(self) -> str | None:
        """Per-call PID file inside the container; None when Runta kills the program itself."""
        if self.image is None:
            return None
        return f"/tmp/areal-exec-{uuid.uuid4().hex[:12]}.pid"

    def _python_argv(self, pidfile: str | None) -> list[str]:
        """Command that runs a program read from stdin, never shell-interpolated."""
        if pidfile is None:
            return ["python3", "-"]
        argv = ["docker", "exec", "-i"]
        for key, value in self.env_vars.items():
            argv += ["-e", f"{key}={value}"]
        return [*argv, _CONTAINER, "sh", "-c", _TRACK, "areal-exec", pidfile]

    def _kill_argv(self, pidfile: str) -> list[str]:
        return [
            "docker",
            "exec",
            "-u",
            "0",
            _CONTAINER,
            "sh",
            "-c",
            _KILL,
            "areal-kill",
            pidfile,
        ]

    def _timed_out_in_container(self, response: Any, pidfile: str | None) -> bool:
        return pidfile is not None and response.exit_code == _TIMEOUT_EXIT_CODE

    def _log_kill_failure(self, runtime_id: str, exc: Exception) -> None:
        logger.warning(
            "Could not stop the timed-out program in Runta runtime %s (%s); "
            "it may still be running in the container.",
            runtime_id,
            exc,
        )

    def _exec_env(self) -> dict[str, str] | None:
        # With an image the variables are passed to `docker exec` instead.
        if self.image is not None:
            return None
        return self.env_vars or None

    def _check_container_started(self, result: Any) -> None:
        if result.exit_code != 0:
            raise RuntimeError(
                f"Runta could not start image {self.image}: {result.stderr_text[-400:]}"
            )

    def _to_result(
        self,
        response: Any,
        on_stdout: Callable[[str], None] | None,
        on_stderr: Callable[[str], None] | None,
    ) -> RuntaRunResult:
        stdout, stderr = response.stdout_text, response.stderr_text
        if on_stdout is not None and stdout:
            on_stdout(stdout)
        if on_stderr is not None and stderr:
            on_stderr(stderr)

        if response.exit_code == 0:
            return RuntaRunResult(stdout=stdout, stderr=stderr, exit_code=0)

        error = (
            _last_non_empty_line(stderr)
            or f"Process exited with code {response.exit_code}"
        )
        return RuntaRunResult(
            stdout=stdout, stderr=stderr, exit_code=response.exit_code, error=error
        )

    def _api_error_result(
        self, exc: Exception, on_stderr: Callable[[str], None] | None
    ) -> RuntaRunResult:
        # Returned like a failing program, so the log is what tells them apart.
        logger.warning("Runta sandbox failure, not a program error: %s", exc)
        error = str(exc)
        if on_stderr is not None:
            on_stderr(error)
        exit_code = _TIMEOUT_EXIT_CODE if isinstance(exc, TimeoutError) else 1
        return RuntaRunResult(stderr=error, exit_code=exit_code, error=error)


class _LeaseHeartbeat:
    """Daemon thread that keeps renewing one runtime's shutdown lease.

    It never takes the runner's lock, so a slow renewal cannot block `run()`,
    and it talks to Runta through a client of its own: the SDK does not
    document its synchronous client as safe to share between threads.
    """

    def __init__(self, runner: RuntaRunner, runtime_id: str, deadline: float):
        self.deadline = deadline
        self._runner = runner
        self._runtime_id = runtime_id
        self._stopped = threading.Event()
        self._client = None
        self._runtime = None
        self._thread = threading.Thread(
            target=self._loop, name="RuntaLease", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to end; it is not joined, a renewal may be in flight."""
        self._stopped.set()

    def beat(self) -> bool:
        """Renew the lease once; False when there is nothing left to renew."""
        runner = self._runner
        seconds = runner._lease_renewal(self.deadline)
        if seconds is None:
            return False
        try:
            if self._client is None:
                self._client = runner._sdk.Runta(**runner._client_kwargs())
            if self._runtime is None:
                self._runtime = self._client.runtimes.get(self._runtime_id)
            self._runtime.update_timeout_policies(auto_shutdown_after_secs=seconds)
        except Exception as exc:
            # After stop() the runtime is being deleted: failing is expected.
            if not self._stopped.is_set():
                runner._log_renewal_failure(self._runtime_id, exc)
            return True
        return not runner._lease_is_final(seconds)

    def _loop(self) -> None:
        try:
            while not self._stopped.wait(self._runner._lease_interval):
                if not self.beat():
                    break
        finally:
            if self._client is not None:
                with contextlib.suppress(Exception):
                    self._client.close()


class RuntaRunner(_RuntaRunnerBase):
    """Run Python code in one Runta runtime from synchronous code.

    A program that exceeds its timeout is terminated and reported with exit
    code 124. Interpreter state does not carry over between `run()` calls;
    files written inside the sandbox do, until the runtime is replaced.

    While the runner holds a runtime, a daemon thread renews the runtime's
    shutdown lease (see the module docstring); `close()` stops it.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._lock = threading.Lock()

    def _get_runtime(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("RuntaRunner is closed")
            if self._runtime is None:
                self._create()
            return self._runtime

    def _create(self) -> None:
        client = self._sdk.Runta(**self._client_kwargs())
        runtime = None
        deadline = self._lifetime_deadline()
        try:
            runtime = client.runtimes.create(
                self._runtime_name(), **self._create_kwargs()
            )
            runtime.wait_until_ready(timeout=self.create_timeout)
            logger.debug("Created Runta runtime %s", runtime.id)
            if self.image is not None:
                self._check_container_started(
                    runtime.exec(self._container_argv(), timeout=self.create_timeout)
                )
            if self.network_block_all:
                runtime.egress.set_policy(self._blocked_egress_policy())
                if self.checkpoint_id is not None:
                    # A restore inherits the checkpoint's raw-TCP allowlist.
                    runtime.egress.set_tcp_allowlist([])
            # Started last: a renewal changes the runtime's revision, which
            # would make a concurrent egress update above fail.
            heartbeat = self._start_heartbeat(runtime, deadline)
        except BaseException:
            # Never keep a half-prepared runtime: the next run() must not land
            # on a machine without its container or network policy.
            self._release(runtime, client)
            raise
        self._client, self._runtime, self._heartbeat = client, runtime, heartbeat

    def _start_heartbeat(self, runtime, deadline: float) -> _LeaseHeartbeat | None:
        if not self._lease_needs_renewal():
            return None
        heartbeat = _LeaseHeartbeat(self, runtime.id, deadline)
        heartbeat.start()
        return heartbeat

    def _delete(self, runtime) -> None:
        # A lease renewal still in flight changes the runtime's revision under
        # the delete; the SDK reports that conflict instead of retrying it.
        for retry in (True, True, False):
            try:
                runtime.delete()
                return
            except self._sdk.ApiError as exc:
                if not (retry and self._is_conflict(exc)):
                    raise

    def _release(self, runtime, client, heartbeat=None) -> None:
        # Best effort: cleanup must not replace the caller's result or error.
        if heartbeat is not None:
            heartbeat.stop()
        try:
            if runtime is not None:
                self._delete(runtime)
        except Exception as exc:
            logger.warning("RuntaRunner could not delete its runtime: %s", exc)
        try:
            if client is not None:
                client.close()
        except Exception as exc:
            logger.warning("RuntaRunner could not close its client: %s", exc)

    def _is_gone(self, runtime) -> bool:
        try:
            return self._is_gone_status(runtime.status())
        except (self._sdk.ApiError, TimeoutError) as exc:
            return getattr(exc, "status_code", None) == 404

    def _discard(self, runtime, exc: Exception) -> None:
        with self._lock:
            if self._runtime is not runtime:
                return  # another caller already replaced it
            client, heartbeat = self._client, self._heartbeat
            self._runtime = self._client = self._heartbeat = None
        logger.warning(
            "Runta runtime %s is no longer running (%s); recreating it.",
            runtime.id,
            exc,
        )
        self._release(runtime, client, heartbeat)

    def _kill_timed_out(self, runtime, pidfile: str) -> None:
        try:
            runtime.exec(self._kill_argv(pidfile), timeout=_KILL_TIMEOUT)
        except Exception as exc:
            self._log_kill_failure(runtime.id, exc)

    def start(self) -> RuntaRunner:
        try:
            self._get_runtime()
            return self
        except BaseException:
            self.close()
            raise

    def run(
        self,
        code: str,
        *,
        timeout: int | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> RuntaRunResult:
        timeout = timeout if timeout is not None else self.default_timeout
        pidfile = self._new_pidfile()
        try:
            # A runtime that is gone cannot have finished the program, so it is
            # retried once on a fresh one.
            for retry in (True, False):
                runtime = self._get_runtime()
                try:
                    response = runtime.exec(
                        self._python_argv(pidfile),
                        timeout=timeout,
                        stdin=code,
                        env=self._exec_env(),
                    )
                    break
                except self._sdk.ApiError as exc:
                    if not (retry and self._is_gone(runtime)):
                        raise
                    self._discard(runtime, exc)
        except (self._sdk.ApiError, TimeoutError) as exc:
            return self._api_error_result(exc, on_stderr)
        if self._timed_out_in_container(response, pidfile):
            self._kill_timed_out(runtime, pidfile)
        return self._to_result(response, on_stdout, on_stderr)

    def close(self) -> None:
        with self._lock:
            runtime, client, heartbeat = self._runtime, self._client, self._heartbeat
            self._runtime = self._client = self._heartbeat = None
            self._closed = True
        self._release(runtime, client, heartbeat)

    def __enter__(self) -> RuntaRunner:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()


class AsyncRuntaRunner(_RuntaRunnerBase):
    """Asyncio twin of `RuntaRunner`.

    The underlying client is bound to the event loop that first uses the
    runner, so start, run and close it from the same loop. The lease is
    renewed by a task on that loop, so a loop that stays blocked for longer
    than `lease_seconds` loses the runtime.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._lock = asyncio.Lock()

    async def _aget_runtime(self):
        async with self._lock:
            if self._closed:
                raise RuntimeError("AsyncRuntaRunner is closed")
            if self._runtime is None:
                await self._acreate()
            return self._runtime

    async def _acreate(self) -> None:
        client = self._sdk.AsyncRunta(**self._client_kwargs())
        runtime = None
        deadline = self._lifetime_deadline()
        try:
            # Shielded until the handle is captured: a cancel that aborts the
            # request mid-flight would leave a runtime that nothing can delete.
            creating = asyncio.ensure_future(
                client.runtimes.create(self._runtime_name(), **self._create_kwargs())
            )
            try:
                runtime = await asyncio.shield(creating)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    runtime = await creating
                raise
            await runtime.wait_until_ready(timeout=self.create_timeout)
            logger.debug("Created Runta runtime %s", runtime.id)
            if self.image is not None:
                self._check_container_started(
                    await runtime.exec(
                        self._container_argv(), timeout=self.create_timeout
                    )
                )
            if self.network_block_all:
                await runtime.egress.set_policy(self._blocked_egress_policy())
                if self.checkpoint_id is not None:
                    # A restore inherits the checkpoint's raw-TCP allowlist.
                    await runtime.egress.set_tcp_allowlist([])
            # Started last: a renewal changes the runtime's revision, which
            # would make a concurrent egress update above fail.
            heartbeat = self._astart_heartbeat(runtime, deadline)
        except BaseException:
            # Never keep a half-prepared runtime: the next arun() must not land
            # on a machine without its container or network policy.
            await self._arelease(runtime, client)
            raise
        self._client, self._runtime, self._heartbeat = client, runtime, heartbeat

    def _astart_heartbeat(self, runtime, deadline: float) -> asyncio.Task | None:
        if not self._lease_needs_renewal():
            return None
        return asyncio.create_task(self._aheartbeat(runtime, deadline))

    async def _aheartbeat(self, runtime, deadline: float) -> None:
        """Task that keeps renewing one runtime's shutdown lease until cancelled."""
        while True:
            await asyncio.sleep(self._lease_interval)
            if not await self._abeat(runtime, deadline):
                return

    async def _abeat(self, runtime, deadline: float) -> bool:
        """Renew the lease once; False when there is nothing left to renew."""
        seconds = self._lease_renewal(deadline)
        if seconds is None:
            return False
        try:
            await runtime.update_timeout_policies(auto_shutdown_after_secs=seconds)
        except Exception as exc:
            self._log_renewal_failure(runtime.id, exc)
            return True
        return not self._lease_is_final(seconds)

    async def _adelete(self, runtime) -> None:
        # A lease renewal cancelled mid-request can still change the runtime's
        # revision under the delete; the SDK reports that conflict instead of
        # retrying it.
        for retry in (True, True, False):
            try:
                await runtime.delete()
                return
            except self._sdk.ApiError as exc:
                if not (retry and self._is_conflict(exc)):
                    raise

    async def _arelease(self, runtime, client, heartbeat=None) -> None:
        # Best effort: cleanup must not replace the caller's result or error.
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        try:
            if runtime is not None:
                await self._adelete(runtime)
        except Exception as exc:
            logger.warning("AsyncRuntaRunner could not delete its runtime: %s", exc)
        try:
            if client is not None:
                await client.close()
        except Exception as exc:
            logger.warning("AsyncRuntaRunner could not close its client: %s", exc)

    async def _ais_gone(self, runtime) -> bool:
        try:
            return self._is_gone_status(await runtime.status())
        except (self._sdk.ApiError, TimeoutError) as exc:
            return getattr(exc, "status_code", None) == 404

    async def _adiscard(self, runtime, exc: Exception) -> None:
        async with self._lock:
            if self._runtime is not runtime:
                return  # another caller already replaced it
            client, heartbeat = self._client, self._heartbeat
            self._runtime = self._client = self._heartbeat = None
        logger.warning(
            "Runta runtime %s is no longer running (%s); recreating it.",
            runtime.id,
            exc,
        )
        await self._arelease(runtime, client, heartbeat)

    async def _akill_timed_out(self, runtime, pidfile: str) -> None:
        try:
            await runtime.exec(self._kill_argv(pidfile), timeout=_KILL_TIMEOUT)
        except Exception as exc:
            self._log_kill_failure(runtime.id, exc)

    async def astart(self) -> AsyncRuntaRunner:
        try:
            await self._aget_runtime()
            return self
        except BaseException:
            await self.aclose()
            raise

    async def arun(
        self,
        code: str,
        *,
        timeout: int | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> RuntaRunResult:
        timeout = timeout if timeout is not None else self.default_timeout
        pidfile = self._new_pidfile()
        try:
            # A runtime that is gone cannot have finished the program, so it is
            # retried once on a fresh one.
            for retry in (True, False):
                runtime = await self._aget_runtime()
                try:
                    response = await runtime.exec(
                        self._python_argv(pidfile),
                        timeout=timeout,
                        stdin=code,
                        env=self._exec_env(),
                    )
                    break
                except self._sdk.ApiError as exc:
                    if not (retry and await self._ais_gone(runtime)):
                        raise
                    await self._adiscard(runtime, exc)
        except (self._sdk.ApiError, TimeoutError) as exc:
            return self._api_error_result(exc, on_stderr)
        if self._timed_out_in_container(response, pidfile):
            await self._akill_timed_out(runtime, pidfile)
        return self._to_result(response, on_stdout, on_stderr)

    async def aclose(self) -> None:
        # The lock makes a create that is still in flight finish first, so its
        # runtime is deleted here instead of landing on a closed runner.
        async with self._lock:
            runtime, client, heartbeat = self._runtime, self._client, self._heartbeat
            self._runtime = self._client = self._heartbeat = None
            self._closed = True
        await self._arelease(runtime, client, heartbeat)

    async def __aenter__(self) -> AsyncRuntaRunner:
        return await self.astart()

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
