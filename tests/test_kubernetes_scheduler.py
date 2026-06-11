# ruff: noqa: E402
import os
import sys
import time
import types
import logging
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

os.environ.setdefault("AREAL_CACHE_DIR", os.path.join(os.getcwd(), ".areal-test-cache"))

colorlog_mod = types.ModuleType("colorlog")
escape_codes_mod = types.ModuleType("colorlog.escape_codes")
formatter_mod = types.ModuleType("colorlog.formatter")


class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        if "format" in kwargs:
            kwargs["fmt"] = kwargs.pop("format")
        kwargs.pop("log_colors", None)
        super().__init__(*args, **kwargs)

    def _escape_code_map(self, levelname):
        return {}

    def _append_reset(self, message, escapes):
        return message


def parse_colors(color):
    return ""


def colored_record(record, escapes):
    record.log_color = ""
    return record


colorlog_mod.ColoredFormatter = ColoredFormatter
escape_codes_mod.parse_colors = parse_colors
formatter_mod.ColoredRecord = colored_record
colorlog_mod.escape_codes = escape_codes_mod
colorlog_mod.formatter = formatter_mod
sys.modules["colorlog"] = colorlog_mod
sys.modules["colorlog.escape_codes"] = escape_codes_mod
sys.modules["colorlog.formatter"] = formatter_mod

mock_modules = [
    "torch",
    "torch.distributed",
    "torch.distributed.fsdp",
    "torch.distributed.fsdp.wrap",
    "torch.distributed.checkpoint",
    "torch.distributed.checkpoint.staging",
    "torch.distributed.checkpoint.state_dict_saver",
    "torch.distributed.checkpoint.storage",
    "torch.distributed.tensor",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.optim.adam",
    "torch.utils",
    "torchdata",
    "torchdata.stateful_dataloader",
    "huggingface_hub",
    "transformers",
    "transformers.utils",
    "transformers.utils.import_utils",
    "transformers.integrations",
    "transformers.integrations.hub_kernels",
    "peft",
    "wandb",
    "flask",
    "uvloop",
    "setproctitle",
    "nvidia-ml-py",
    "math_verify",
    "mathruler",
    "lark",
    "hydra",
    "hydra.core",
    "hydra.core.config_store",
    "hydra.core.global_hydra",
    "omegaconf",
    "ray",
    "ray.exceptions",
    "ray.runtime_env",
    "ray.util",
    "ray.util.placement_group",
    "ray.util.scheduling_strategies",
    "numba",
    "einops",
    "aiofiles",
    "aiofiles.os",
    "trackio",
    "awex",
    "swanlab",
    "swanboard",
]
for mod in mock_modules:
    sys.modules.setdefault(mod, MagicMock())
sys.modules["torch"].Tensor = type("Tensor", (), {})

from areal.api import Job, Worker
from areal.api.cli_args import (
    BaseExperimentConfig,
    SchedulingSpec,
    SchedulingStrategy,
    SchedulingStrategyType,
)
from areal.infra.scheduler.exceptions import WorkerCreationError, WorkerFailedError
from areal.infra.scheduler.kubernetes import K8sWorkerInfo, KubernetesScheduler


class FakeK8sClient:
    def __init__(self):
        self.services = {}
        self.statefulsets = {}
        self.deleted_statefulsets = []
        self.deleted_services = []
        self.pods = []
        self.logs = {}
        self.events = {}

    def apply_service(self, namespace, body):
        self.services[(namespace, body["metadata"]["name"])] = body

    def apply_statefulset(self, namespace, body):
        self.statefulsets[(namespace, body["metadata"]["name"])] = body

    def delete_statefulset(self, namespace, name):
        self.deleted_statefulsets.append((namespace, name))

    def delete_service(self, namespace, name):
        self.deleted_services.append((namespace, name))

    def list_pods(self, namespace, selector):
        self.last_selector = selector
        return self.pods

    def pod_logs(self, namespace, pod_name, tail_lines=80):
        return self.logs.get(pod_name, "")

    def pod_events(self, namespace, pod_name):
        return self.events.get(pod_name, "")


@pytest.fixture
def config(tmp_path):
    config = BaseExperimentConfig(
        experiment_name="test_k8s",
        trial_name=f"test_{int(time.time())}",
    )
    config.cluster.n_gpus_per_node = 8
    config.cluster.fileroot = str(tmp_path / "fileroot")
    os.makedirs(config.cluster.fileroot, exist_ok=True)
    config.cluster.name_resolve.nfs_record_root = os.path.join(
        config.cluster.fileroot, "name_resolve"
    )
    os.makedirs(config.cluster.name_resolve.nfs_record_root, exist_ok=True)
    return config


@pytest.fixture
def fake_k8s():
    return FakeK8sClient()


@pytest.fixture
def scheduler(config, fake_k8s):
    with patch("areal.utils.name_resolve.reconfigure"), patch(
        "areal.utils.name_resolve.clear_subtree"
    ):
        scheduler = KubernetesScheduler(exp_config=config, k8s_client=fake_k8s)
        yield scheduler
        scheduler._workers.clear()
        scheduler._colocated_roles.clear()
        scheduler._statefulsets.clear()


def test_initialization(scheduler, config):
    assert scheduler.experiment_name == config.experiment_name
    assert scheduler.trial_name == config.trial_name
    assert scheduler.namespace == "default"
    assert scheduler.n_gpus_per_node == 8


def test_sanitize_k8s_name():
    assert (
        KubernetesScheduler._sanitize_k8s_name("My-Experiment_123")
        == "my-experiment-123"
    )
    assert len(KubernetesScheduler._sanitize_k8s_name("a" * 100)) <= 63


def test_resource_name(scheduler):
    scheduler.experiment_name = "exp"
    scheduler.trial_name = "trial"
    assert scheduler._resource_name("actor") == "areal-exp-trial-actor"


def test_render_statefulset_yaml_honors_command_and_production_defaults(scheduler):
    spec = SchedulingSpec(
        cpu=2,
        mem=4,
        gpu=1,
        image="ghcr.io/areal/worker:latest",
        cmd="python -m custom.guard",
        additional_bash_cmds=["export FOO=bar"],
    )
    yaml_out = scheduler._render_statefulset_yaml(
        role="actor", replicas=2, spec=spec
    )

    svc, sts = list(yaml.safe_load_all(yaml_out))
    assert svc["kind"] == "Service"
    assert sts["kind"] == "StatefulSet"
    assert sts["spec"]["replicas"] == 2
    assert sts["spec"]["podManagementPolicy"] == "Parallel"

    labels = sts["metadata"]["labels"]
    assert labels["app.kubernetes.io/name"] == "areal"
    assert labels["app.kubernetes.io/instance"] == scheduler._resource_name("actor")
    assert "areal.openpsi.io/role" in labels

    container = sts["spec"]["template"]["spec"]["containers"][0]
    command = container["command"][-1]
    assert "python -m custom.guard" in command
    assert "export FOO=bar" in command
    assert "--port 8000" in command
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["resources"]["requests"]["cpu"] == "2"
    assert container["resources"]["requests"]["memory"] == "4Gi"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"


def test_create_workers_separation_uses_python_client(scheduler, fake_k8s):
    job = Job(
        role="actor",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=1, mem=4, image="ghcr.io/areal/img:tag")],
    )
    worker_ids = scheduler.create_workers(job)
    assert worker_ids == ["actor/0", "actor/1"]
    assert "actor" in scheduler._workers
    assert ("default", scheduler._resource_name("actor")) in fake_k8s.services
    assert ("default", scheduler._resource_name("actor")) in fake_k8s.statefulsets


def test_create_workers_rejects_missing_or_sif_image(scheduler):
    with pytest.raises(WorkerCreationError, match="image is required"):
        scheduler.create_workers(
            Job(role="actor", replicas=1, tasks=[SchedulingSpec(image="")])
        )

    with pytest.raises(WorkerCreationError, match="must be a container image"):
        scheduler.create_workers(
            Job(role="actor", replicas=1, tasks=[SchedulingSpec(image="model.sif")])
        )


def test_create_workers_rejects_per_replica_specs(scheduler):
    job = Job(
        role="actor",
        replicas=2,
        tasks=[
            SchedulingSpec(image="ghcr.io/areal/img:tag"),
            SchedulingSpec(image="ghcr.io/areal/img:tag"),
        ],
    )
    with pytest.raises(WorkerCreationError, match="exactly one SchedulingSpec"):
        scheduler.create_workers(job)


def test_create_workers_rejects_slurm_node_fields(scheduler):
    job = Job(
        role="actor",
        replicas=1,
        tasks=[SchedulingSpec(image="ghcr.io/areal/img:tag", nodelist="node-a")],
    )
    with pytest.raises(WorkerCreationError, match="nodelist/exclude"):
        scheduler.create_workers(job)


def test_pod_health_uses_scoped_selector_and_reports_diagnostics(scheduler, fake_k8s):
    pod = {
        "metadata": {"name": "actor-0"},
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "state": {
                        "waiting": {
                            "reason": "CrashLoopBackOff",
                            "message": "boom",
                        }
                    }
                }
            ],
        },
    }
    fake_k8s.pods = [pod]
    fake_k8s.events["actor-0"] = "Warning BackOff: restarting"
    fake_k8s.logs["actor-0"] = "traceback"
    scheduler._statefulsets["actor"] = scheduler._resource_name("actor")

    with pytest.raises(WorkerFailedError, match="CrashLoopBackOff") as exc:
        scheduler._check_pods_health("actor")

    assert "app.kubernetes.io/instance=" in fake_k8s.last_selector
    assert "Warning BackOff" in str(exc.value)
    assert "traceback" in str(exc.value)


def test_delete_regular_role_deletes_statefulset_and_service(scheduler, fake_k8s):
    job = Job(
        role="actor",
        replicas=1,
        tasks=[SchedulingSpec(image="ghcr.io/areal/img:tag")],
    )
    scheduler.create_workers(job)
    scheduler.delete_workers("actor")

    name = scheduler._resource_name("actor")
    assert ("default", name) in fake_k8s.deleted_statefulsets
    assert ("default", name) in fake_k8s.deleted_services
    assert "actor" not in scheduler._workers


def test_delete_forked_role_kills_children(scheduler):
    target = K8sWorkerInfo(
        worker=Worker(id="actor/0", ip="10.0.0.1", worker_ports=["8000"]),
        role="actor",
        task_index=0,
        discovered=True,
    )
    child = K8sWorkerInfo(
        worker=Worker(id="proxy/0", ip="10.0.0.1", worker_ports=["8001"]),
        role="proxy",
        task_index=0,
        discovered=True,
    )
    scheduler._workers["actor"] = [target]
    scheduler._workers["proxy"] = [child]
    scheduler._colocated_roles["proxy"] = "actor"

    with patch.object(
        scheduler, "_cleanup_forked_workers_async", new_callable=AsyncMock
    ) as cleanup:
        scheduler.delete_workers("proxy")

    cleanup.assert_awaited_once()
    assert "proxy" not in scheduler._workers
    assert "proxy" not in scheduler._colocated_roles


def test_partial_fork_failure_rolls_back_successful_workers(scheduler):
    target_workers = [
        K8sWorkerInfo(
            worker=Worker(id="actor/0", ip="10.0.0.1", worker_ports=["8000"]),
            role="actor",
            task_index=0,
            discovered=True,
        ),
        K8sWorkerInfo(
            worker=Worker(id="actor/1", ip="10.0.0.2", worker_ports=["8000"]),
            role="actor",
            task_index=1,
            discovered=True,
        ),
    ]
    scheduler._workers["actor"] = target_workers
    successful_child = K8sWorkerInfo(
        worker=Worker(id="proxy/0", ip="10.0.0.1", worker_ports=["8001"]),
        role="proxy",
        task_index=0,
        discovered=True,
    )

    async def fork_one(session, role, idx, target_wi, target_role, command=None):
        if idx == 0:
            return successful_child
        raise WorkerCreationError(role, "fork failed")

    scheduler._fork_single_worker = fork_one
    scheduler._cleanup_forked_workers_async = AsyncMock()

    async def run_test():
        with pytest.raises(WorkerCreationError, match="Failed to fork 1 out of 2"):
            await scheduler._create_forked_workers_async(
                "proxy", "actor", target_workers
            )

        scheduler._cleanup_forked_workers_async.assert_awaited_once_with(
            "proxy", "actor", [successful_child]
        )

    asyncio.run(run_test())
    assert "proxy" not in scheduler._workers
