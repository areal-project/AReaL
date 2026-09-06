# SPDX-License-Identifier: Apache-2.0
"""Tests for worker teardown and long-op robustness."""

import os
import signal
from types import SimpleNamespace
from unittest import mock

import pytest

from areal.infra.launcher import sglang_server
from areal.trainer.rl_trainer import PPOTrainer


class TestSglangLauncherReapsItsChildren:
    def test_sigterm_is_converted_into_systemexit(self):
        with (
            mock.patch.object(sglang_server, "launch_sglang_server") as launch,
            mock.patch.object(sglang_server, "kill_process_tree") as reap,
        ):
            launch.side_effect = lambda argv: os.kill(os.getpid(), signal.SIGTERM)

            with pytest.raises(SystemExit) as exc:
                sglang_server.main([])

        assert exc.value.code == 128 + signal.SIGTERM
        reap.assert_called_once()

    def test_unexpected_errors_still_reap_the_tree(self):
        with (
            mock.patch.object(sglang_server, "launch_sglang_server") as launch,
            mock.patch.object(sglang_server, "kill_process_tree") as reap,
        ):
            launch.side_effect = RuntimeError("boom")

            with pytest.raises(SystemExit):
                sglang_server.main([])

        reap.assert_called_once()


class TestTrainerCloseToleratesPartialConstruction:
    def test_close_on_a_bare_trainer_does_not_raise(self):
        trainer = object.__new__(PPOTrainer)

        PPOTrainer.close(trainer)

    def test_one_failing_component_does_not_skip_the_others(self):
        trainer = object.__new__(PPOTrainer)
        closed = []
        trainer.saver = SimpleNamespace(
            finalize=lambda: (_ for _ in ()).throw(RuntimeError("saver down"))
        )
        trainer.stats_logger = SimpleNamespace(close=lambda: closed.append("stats"))

        PPOTrainer.close(trainer)

        assert "stats" in closed, (
            "a failing saver prevented the remaining components from closing"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
