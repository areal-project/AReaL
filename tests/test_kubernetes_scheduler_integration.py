import os

import pytest

if not os.environ.get("AREAL_RUN_K8S_TESTS"):
    pytest.skip(
        "Kubernetes integration tests require AREAL_RUN_K8S_TESTS=1",
        allow_module_level=True,
    )

from areal.api import Job
from areal.api.cli_args import BaseExperimentConfig, SchedulingSpec
from areal.infra.scheduler.kubernetes import KubernetesScheduler


def test_kubernetes_scheduler_create_delete_lifecycle(tmp_path):
    image = os.environ.get("AREAL_K8S_TEST_IMAGE")
    if not image:
        pytest.skip("Set AREAL_K8S_TEST_IMAGE to run the live Kubernetes test")

    config = BaseExperimentConfig(
        experiment_name="test-k8s-scheduler",
        trial_name="integration",
    )
    config.cluster.fileroot = str(tmp_path / "fileroot")
    config.cluster.name_resolve.nfs_record_root = str(tmp_path / "name_resolve")
    os.makedirs(config.cluster.fileroot, exist_ok=True)
    os.makedirs(config.cluster.name_resolve.nfs_record_root, exist_ok=True)

    scheduler = KubernetesScheduler(exp_config=config, startup_timeout=120.0)
    try:
        worker_ids = scheduler.create_workers(
            Job(
                role="worker",
                replicas=1,
                tasks=[
                    SchedulingSpec(
                        cpu=1,
                        gpu=0,
                        mem=2,
                        image=image,
                        port_count=1,
                    )
                ],
            )
        )
        assert worker_ids == ["worker/0"]
        workers = scheduler.get_workers("worker", timeout=120.0)
        assert len(workers) == 1
        assert workers[0].worker_ports
    finally:
        scheduler.delete_workers()
