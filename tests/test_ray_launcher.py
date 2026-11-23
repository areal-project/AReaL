# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from unittest.mock import Mock, call

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ray") is None,
    reason="ray is required for importing the Ray launcher",
)


@pytest.fixture(
    params=[
        "RayTaskError",
        "WorkerCrashedError",
        "NodeDiedError",
        "OutOfMemoryError",
        "RayError",
    ]
)
def ray_failure(request):
    from ray import exceptions

    if request.param == "RayTaskError":
        return exceptions.RayTaskError("train", "traceback", RuntimeError("boom"))
    return getattr(exceptions, request.param)("simulated worker failure")


def _select_trainer_node_count(*args, **kwargs):
    from areal.infra.launcher.ray import _select_trainer_node_count as select

    return select(*args, **kwargs)


def test_select_trainer_node_count_uses_train_world_size():
    assert (
        _select_trainer_node_count(
            train_world_size=4,
            available_nodes=2,
            n_gpus_per_node=8,
        )
        == 1
    )
    assert (
        _select_trainer_node_count(
            train_world_size=64,
            available_nodes=8,
            n_gpus_per_node=8,
        )
        == 8
    )


def test_select_trainer_node_count_can_span_partial_nodes_evenly():
    assert (
        _select_trainer_node_count(
            train_world_size=9,
            available_nodes=3,
            n_gpus_per_node=8,
        )
        == 3
    )


def test_select_trainer_node_count_rejects_insufficient_gpus():
    with pytest.raises(ValueError, match="requires 17 GPUs"):
        _select_trainer_node_count(
            train_world_size=17,
            available_nodes=2,
            n_gpus_per_node=8,
        )


def test_select_trainer_node_count_rejects_uneven_placement():
    with pytest.raises(ValueError, match="Cannot evenly place 17 trainer processes"):
        _select_trainer_node_count(
            train_world_size=17,
            available_nodes=3,
            n_gpus_per_node=8,
        )


def test_wait_requires_all_trainer_tasks_to_complete(monkeypatch, tmp_path):
    import areal.infra.launcher.ray as ray_launcher
    from areal.infra.utils.launcher import JobException, JobState

    launcher = ray_launcher.RayLauncher("exp", "trial", str(tmp_path))
    trainer_0 = object()
    trainer_1 = object()
    launcher.jobs = {
        "trainer:0": trainer_0,
        "trainer:1": trainer_1,
    }
    trainer_1_polls = 0

    def fake_get(future, timeout):
        nonlocal trainer_1_polls
        assert timeout == 0.1
        if future is trainer_0:
            return None
        trainer_1_polls += 1
        if trainer_1_polls == 1:
            raise ray_launcher.ray.exceptions.GetTimeoutError()
        return None

    monkeypatch.setattr(ray_launcher.ray, "get", fake_get)
    monkeypatch.setattr(ray_launcher.time, "sleep", lambda _: None)

    with pytest.raises(JobException) as exc_info:
        launcher.wait(
            check_status=(JobState.COMPLETED, JobState.FAILED),
            complete_all_worker_types=("trainer",),
        )

    assert trainer_1_polls == 2
    assert exc_info.value.reason == JobState.COMPLETED
    assert exc_info.value.worker_type == "trainer"


def test_wait_prioritizes_failure_over_trainer_completion(
    monkeypatch, tmp_path, ray_failure
):
    import areal.infra.launcher.ray as ray_launcher
    from areal.infra.utils.launcher import JobException, JobState

    launcher = ray_launcher.RayLauncher("exp", "trial", str(tmp_path))
    completed = object()
    failed = object()
    launcher.jobs = {
        "trainer:0": completed,
        "trainer:1": failed,
    }

    def fake_get(future, timeout):
        assert timeout == 0.1
        if future is completed:
            return None
        raise ray_failure

    monkeypatch.setattr(ray_launcher.ray, "get", fake_get)

    with pytest.raises(JobException) as exc_info:
        launcher.wait(
            check_status=(JobState.COMPLETED, JobState.FAILED),
            complete_all_worker_types=("trainer",),
        )

    assert exc_info.value.reason == JobState.FAILED
    assert exc_info.value.worker_type == "trainer"


@pytest.mark.parametrize("worker_type", ["trainer", "llm_server"])
def test_wait_ray_error_reports_failed_job(
    monkeypatch, tmp_path, ray_failure, worker_type
):
    """Ray task and infrastructure failures share the failed-job contract."""
    import areal.infra.launcher.ray as ray_launcher
    from areal.infra.utils.launcher import JobException, JobState

    launcher = ray_launcher.RayLauncher("exp", "trial", str(tmp_path))
    future = object()
    launcher.jobs = {f"{worker_type}:0": future}
    get = Mock(side_effect=ray_failure)
    monkeypatch.setattr(ray_launcher.ray, "get", get)

    with pytest.raises(JobException) as exc_info:
        launcher.wait()

    assert exc_info.value.reason == JobState.FAILED
    assert exc_info.value.worker_type == worker_type
    assert exc_info.value.run_name == "exp_trial"
    assert exc_info.value.host == "ray"
    assert launcher.jobs == {f"{worker_type}:0": future}
    get.assert_called_once_with(future, timeout=0.1)


def test_wait_polling_timeouts_keep_job_until_completion(monkeypatch, tmp_path):
    """A polling timeout must neither fail nor remove a running job."""
    import areal.infra.launcher.ray as ray_launcher

    launcher = ray_launcher.RayLauncher("exp", "trial", str(tmp_path))
    future = object()
    launcher.jobs = {"trainer:0": future}
    timeout = ray_launcher.ray.exceptions.GetTimeoutError
    get = Mock(side_effect=[timeout(), timeout(), None])
    pending_polls = []

    def fake_sleep(interval):
        assert interval == ray_launcher.RAY_WAIT_CHECK_TIME_INTERVAL
        if launcher.jobs:
            assert launcher.jobs == {"trainer:0": future}
            pending_polls.append(get.call_count)

    monkeypatch.setattr(ray_launcher.ray, "get", get)
    monkeypatch.setattr(ray_launcher.time, "sleep", fake_sleep)

    launcher.wait()

    assert pending_polls == [1, 2]
    assert get.call_args_list == [call(future, timeout=0.1)] * 3
    assert launcher.jobs == {}


def test_ray_main_ray_error_cancels_jobs_and_runs_post_exit_hook(
    monkeypatch, tmp_path, ray_failure
):
    """Normalized failures reach the existing cancellation and post-exit path."""
    from omegaconf import OmegaConf

    import areal.infra.launcher.ray as ray_launcher
    from areal.infra.utils.launcher import JobException, JobState

    config = OmegaConf.create(
        {
            "experiment_name": "exp",
            "trial_name": "trial",
            "allocation_mode": "fsdp:d1",
            "cluster": {"n_nodes": 1, "n_gpus_per_node": 1},
            "recover": {"mode": "off", "retries": 0},
            "actor": {"scheduling_spec": [{"cpu": 1, "mem": 1}]},
        }
    )
    launcher = ray_launcher.RayLauncher("exp", "trial", str(tmp_path))
    server = object()
    trainer = object()
    launcher.jobs = {"llm_server:0": server, "trainer:0": trainer}
    cleanup = Mock()

    def fake_get(future, timeout):
        assert timeout == 0.1
        if future is server:
            raise ray_launcher.ray.exceptions.GetTimeoutError()
        raise ray_failure

    monkeypatch.setattr(ray_launcher, "RAY_LAUNCHER", launcher)
    monkeypatch.setattr(ray_launcher, "check_if_recover", lambda *_: True)
    monkeypatch.setattr(
        ray_launcher, "validate_config_for_distributed_launcher", lambda _: None
    )
    monkeypatch.setattr(ray_launcher.name_resolve, "reconfigure", lambda _: None)
    monkeypatch.setattr(ray_launcher.name_resolve, "clear_subtree", lambda _: None)
    monkeypatch.setattr(launcher, "submit_array", Mock())
    monkeypatch.setattr(ray_launcher.sys, "argv", ["ray.py", "entry.py"])
    monkeypatch.setattr(ray_launcher.ray, "get", fake_get)
    monkeypatch.setattr(ray_launcher.ray, "cancel", cleanup.cancel)
    monkeypatch.setattr(ray_launcher, "run_post_exit_hook", cleanup.post_exit)

    with pytest.raises(JobException) as exc_info:
        ray_launcher.ray_main(config)

    assert exc_info.value.reason == JobState.FAILED
    assert exc_info.value.worker_type == "trainer"
    assert cleanup.mock_calls == [
        call.cancel(server, force=False),
        call.cancel(trainer, force=True),
        call.post_exit(config),
    ]
    assert launcher.jobs == {}


def test_wait_reports_single_llm_server_completion(monkeypatch, tmp_path):
    import areal.infra.launcher.ray as ray_launcher
    from areal.infra.utils.launcher import JobException, JobState

    launcher = ray_launcher.RayLauncher("exp", "trial", str(tmp_path))
    launcher.jobs = {"llm_server:0": object()}
    monkeypatch.setattr(ray_launcher.ray, "get", lambda future, timeout: None)

    with pytest.raises(JobException) as exc_info:
        launcher.wait(
            check_status=(JobState.COMPLETED, JobState.FAILED),
            complete_all_worker_types=("trainer",),
        )

    assert exc_info.value.reason == JobState.COMPLETED
    assert exc_info.value.worker_type == "llm_server"


def test_run_func_waits_for_merged_log_drain(monkeypatch):
    import areal.infra.launcher.ray as ray_launcher

    calls = []
    waited_refs = []

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self, *args):
            ref = object()
            calls.append((self.name, args, ref))
            return ref

    class LogWriter:
        write = RemoteMethod("write")
        drain = RemoteMethod("drain")

    def fake_run_func(*args, **kwargs):
        os.write(1, b"tail-log-line\n")
        return "done"

    def fake_get(ref, timeout):
        waited_refs.append((ref, timeout))
        return None

    monkeypatch.setattr(ray_launcher, "run_func", fake_run_func)
    monkeypatch.setattr(ray_launcher.ray, "get", fake_get)

    result = ray_launcher.run_func_with_file_log(
        LogWriter(),
        "trainer:0",
        "entry.py",
        "main",
    )

    assert result == "done"
    assert calls[-1][0] == "drain"
    assert any(
        name == "write" and args == (b"tail-log-line\n",) for name, args, _ in calls
    )
    assert waited_refs == [(calls[-1][2], ray_launcher.LOG_WRITER_DRAIN_TIMEOUT)]


def test_submit_array_resolves_log_writer_once(monkeypatch, tmp_path):
    """All ranks in one array submission reuse a single writer probe."""
    # Arrange
    import areal.infra.launcher.ray as ray_launcher

    launcher = ray_launcher.RayLauncher("exp", "trial", str(tmp_path))
    launcher.placement_groups["trainer"] = object()
    log_writer = object()
    writer_lookups = []
    submitted_writers = []

    def fake_log_writer_of(job_name):
        writer_lookups.append(job_name)
        return log_writer

    def fake_submit(**kwargs):
        submitted_writers.append(kwargs["log_writer"])
        return object()

    monkeypatch.setattr(launcher, "_log_writer_of", fake_log_writer_of)
    monkeypatch.setattr(launcher, "submit", fake_submit)

    # Act
    futures = launcher.submit_array(
        job_name="trainer",
        file_path="entry.py",
        func_name="main",
        count=4,
        nodes=2,
        list_args=[[] for _ in range(4)],
        gpus_per_task=1,
        cpus_per_task=1,
        mem_per_task=1,
    )

    # Assert
    assert writer_lookups == ["trainer"]
    assert submitted_writers == [log_writer] * 4
    assert len(futures) == 4
