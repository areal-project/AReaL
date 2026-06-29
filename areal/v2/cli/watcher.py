# SPDX-License-Identifier: Apache-2.0

"""Foreground watcher with signal-aware teardown.

Subcommand CLIs use this in their ``run`` (non-detach) path to block
until the service exits, with a clear distinction between:

  - SIGINT / SIGTERM → user-requested stop → run the teardown callback
    (kill workers, remove state file) → return 0
  - SIGHUP → terminal disconnect → exit quietly without teardown;
    workers spawned with ``start_new_session=True`` survive

Subclasses or constructor kwargs can swap the SIGHUP semantics for
``"teardown"`` if a particular CLI wants the older "all signals tear
down" behavior.
"""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from typing import Literal

Disposition = Literal["teardown", "detach"]


class ForegroundWatcher:
    """Block the calling thread until the service exits.

    Construct with an ``is_alive`` callback (typically
    ``lambda: lifecycle.gateway_alive(state)``) and a ``teardown``
    callback (kills workers + removes state file). Call ``watch()`` —
    it returns the desired CLI exit code.
    """

    default_signal_dispositions: dict[int, Disposition] = {
        signal.SIGTERM: "teardown",
    }
    if hasattr(signal, "SIGHUP"):
        default_signal_dispositions[signal.SIGHUP] = "detach"

    def __init__(
        self,
        *,
        is_alive: Callable[[], bool],
        teardown: Callable[[], None],
        idle_poll: float = 1.0,
        service_name: str = "",
        signal_dispositions: dict[int, Disposition] | None = None,
    ) -> None:
        self.is_alive = is_alive
        self.teardown = teardown
        self.idle_poll = idle_poll
        self.service_name = service_name
        self.dispositions = (
            dict(signal_dispositions)
            if signal_dispositions is not None
            else dict(self.default_signal_dispositions)
        )

    def watch(self) -> int:
        previous = self._install_handlers()
        try:
            while self.is_alive():
                time.sleep(self.idle_poll)
        except KeyboardInterrupt:
            # SIGINT (Ctrl+C) plus any SIGTERM-dispatched teardown lands here.
            self.teardown()
            return 0
        except SystemExit as exc:
            # SIGHUP-dispatched detach path lands here.
            return exc.code if isinstance(exc.code, int) else 0
        except BaseException:
            self.teardown()
            raise
        finally:
            self._restore_handlers(previous)
        # is_alive() flipped to False on its own — gateway died externally.
        # We do not call teardown; the children are already gone, but the
        # caller may want to clean up state (handled outside this class).
        return 0

    # ------------------------------------------------------------------
    # Signal plumbing

    def _install_handlers(self) -> dict[int, object]:
        previous: dict[int, object] = {}
        for sig, disposition in self.dispositions.items():
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handler_for(disposition))
        return previous

    def _restore_handlers(self, previous: dict[int, object]) -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    def _handler_for(self, disposition: Disposition):
        if disposition == "teardown":

            def teardown_handler(signum, frame):  # noqa: ARG001
                raise KeyboardInterrupt

            return teardown_handler

        def detach_handler(signum, frame):  # noqa: ARG001
            raise SystemExit(0)

        return detach_handler
