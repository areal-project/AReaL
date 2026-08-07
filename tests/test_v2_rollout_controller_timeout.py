from dataclasses import dataclass

import pytest

from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec
from areal.v2.inference_service.controller.controller import RolloutControllerV2


class _StopAfterGetWorkers(Exception):
    pass


@dataclass
class _FakeScheduler:
    n_gpus_per_node: int = 8
    get_workers_call: tuple[str, float | None] | None = None

    def create_workers(self, job):
        return [f"{job.role}/{i}" for i in range(job.replicas)]

    def get_workers(self, role, timeout=None):
        self.get_workers_call = (role, timeout)
        raise _StopAfterGetWorkers


def test_v2_rollout_controller_passes_worker_ready_timeout_to_scheduler():
    scheduler = _FakeScheduler()
    config = InferenceEngineConfig(
        model="/tmp/model",
        backend="sglang:d2t4p1",
        admin_api_key="test-admin-key",
        workers_ready_timeout=1234.0,
        scheduling_spec=(SchedulingSpec(cpu=1, gpu=1, mem=1),),
    )
    controller = RolloutControllerV2(config=config, scheduler=scheduler)
    controller._worker_role = "rollout"

    with pytest.raises(_StopAfterGetWorkers):
        controller._bg_initialize(server_args={})

    assert scheduler.get_workers_call == ("rollout-inf", 1234.0)
