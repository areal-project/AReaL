# SPDX-License-Identifier: Apache-2.0

import errno
from unittest.mock import Mock, patch

import psutil
import pytest

from areal.infra.utils.proc import kill_process_tree

_PSUTIL_PROCESS = psutil.Process


def _process(*, running: bool, status: str = psutil.STATUS_RUNNING) -> Mock:
    proc = Mock(spec=_PSUTIL_PROCESS)
    proc.is_running.return_value = running
    proc.status.return_value = status
    return proc


@patch("areal.infra.utils.proc.psutil.Process")
@patch("areal.infra.utils.proc.psutil.wait_procs")
def test_kill_process_tree_pidfd_einval_all_gone_succeeds(
    mock_wait_procs,
    mock_process_class,
):
    """Treat pidfd EINVAL as success when every process has exited."""
    parent = _process(running=False)
    child = _process(running=False)
    parent.children.return_value = [child]
    mock_process_class.return_value = parent
    mock_wait_procs.side_effect = OSError(errno.EINVAL, "Invalid argument")

    kill_process_tree(1234, timeout=3, graceful=True)

    child.kill.assert_not_called()
    parent.kill.assert_not_called()


@patch("areal.infra.utils.proc.psutil.Process")
@patch("areal.infra.utils.proc.psutil.wait_procs")
def test_kill_process_tree_pidfd_einval_kills_only_live_processes(
    mock_wait_procs,
    mock_process_class,
):
    """Escalate only processes that remain alive after pidfd EINVAL."""
    parent = _process(running=False)
    child = _process(running=True)
    parent.children.return_value = [child]
    mock_process_class.return_value = parent
    mock_wait_procs.side_effect = [
        OSError(errno.EINVAL, "Invalid argument"),
        ([], []),
    ]

    kill_process_tree(1234, timeout=3, graceful=True)

    child.kill.assert_called_once()
    parent.kill.assert_not_called()
    assert mock_wait_procs.call_count == 2


@patch("areal.infra.utils.proc.psutil.Process")
@patch("areal.infra.utils.proc.psutil.wait_procs")
def test_kill_process_tree_pidfd_einval_zombie_is_gone(
    mock_wait_procs,
    mock_process_class,
):
    """Do not signal zombies even though psutil reports them as running."""
    parent = _process(running=False)
    child = _process(running=True, status=psutil.STATUS_ZOMBIE)
    parent.children.return_value = [child]
    mock_process_class.return_value = parent
    mock_wait_procs.side_effect = OSError(errno.EINVAL, "Invalid argument")

    kill_process_tree(1234, timeout=3, graceful=True)

    child.kill.assert_not_called()


@patch("areal.infra.utils.proc.psutil.Process")
@patch("areal.infra.utils.proc.psutil.wait_procs")
def test_kill_process_tree_wait_reports_zombie_alive_succeeds(
    mock_wait_procs,
    mock_process_class,
):
    """Treat a non-child zombie returned by wait_procs as terminated."""
    parent = _process(running=False)
    child = _process(running=True, status=psutil.STATUS_ZOMBIE)
    parent.children.return_value = [child]
    mock_process_class.return_value = parent
    mock_wait_procs.return_value = ([], [child])

    kill_process_tree(1234, timeout=3, graceful=True)

    child.kill.assert_not_called()


@patch("areal.infra.utils.proc.psutil.Process")
@patch("areal.infra.utils.proc.psutil.wait_procs")
def test_kill_process_tree_final_wait_pidfd_einval_gone_succeeds(
    mock_wait_procs,
    mock_process_class,
):
    """Re-check process state when pidfd EINVAL follows SIGKILL."""
    parent = _process(running=False)
    child = _process(running=True)
    child.is_running.side_effect = [True, False]
    parent.children.return_value = [child]
    mock_process_class.return_value = parent
    mock_wait_procs.side_effect = [
        ([], [child]),
        OSError(errno.EINVAL, "Invalid argument"),
    ]

    kill_process_tree(1234, timeout=3, graceful=True)

    child.kill.assert_called_once()


@patch("areal.infra.utils.proc.psutil.Process")
@patch("areal.infra.utils.proc.psutil.wait_procs")
def test_kill_process_tree_final_wait_pidfd_einval_live_raises(
    mock_wait_procs,
    mock_process_class,
):
    """Report a real cleanup failure when a process survives SIGKILL."""
    parent = _process(running=False)
    child = _process(running=True)
    parent.children.return_value = [child]
    mock_process_class.return_value = parent
    mock_wait_procs.side_effect = [
        ([], [child]),
        OSError(errno.EINVAL, "Invalid argument"),
    ]

    with pytest.raises(RuntimeError, match="still alive"):
        kill_process_tree(1234, timeout=3, graceful=True)

    child.kill.assert_called_once()


@patch("areal.infra.utils.proc.psutil.Process")
@patch("areal.infra.utils.proc.psutil.wait_procs")
def test_kill_process_tree_non_einval_wait_error_propagates(
    mock_wait_procs,
    mock_process_class,
):
    """Preserve unexpected wait errors instead of masking cleanup failures."""
    parent = _process(running=True)
    parent.children.return_value = []
    mock_process_class.return_value = parent
    mock_wait_procs.side_effect = OSError(errno.EBADF, "Bad file descriptor")

    with pytest.raises(OSError) as exc_info:
        kill_process_tree(1234, timeout=3, graceful=True)

    assert exc_info.value.errno == errno.EBADF


@patch("areal.infra.utils.proc.psutil.Process")
@patch("areal.infra.utils.proc.psutil.wait_procs")
def test_kill_process_tree_pidfd_einval_liveness_error_propagates(
    mock_wait_procs,
    mock_process_class,
):
    """Do not hide permission errors encountered during the state re-check."""
    parent = _process(running=True)
    parent.children.return_value = []
    parent.is_running.side_effect = psutil.AccessDenied(1234)
    mock_process_class.return_value = parent
    mock_wait_procs.side_effect = OSError(errno.EINVAL, "Invalid argument")

    with pytest.raises(psutil.AccessDenied):
        kill_process_tree(1234, timeout=3, graceful=True)
