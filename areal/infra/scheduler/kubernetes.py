# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp
import orjson
import requests
import yaml

from areal.api import Job, Scheduler, Worker
from areal.api.cli_args import (
    BaseExperimentConfig,
    NameResolveConfig,
    SchedulingSpec,
    SchedulingStrategyType,
)
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.scheduler.exceptions import (
    EngineCallError,
    EngineCreationError,
    EngineImportError,
    RPCConnectionError,
    SchedulerError,
    WorkerConfigurationError,
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.http import get_default_connector
from areal.infra.utils.launcher import get_env_vars, get_thread_env_vars
from areal.utils import logging, name_resolve, names
from areal.utils.fs import validate_shared_path
from areal.utils.network import format_hostport, split_hostport
from areal.utils.offload import get_tms_env_vars

logger = logging.getLogger("KubernetesScheduler")

_K8S_RPC_PORT = 8000


@dataclass
class K8sWorkerInfo:
    worker: Worker
    role: str
    task_index: int
    discovered: bool = False
    spec: SchedulingSpec | None = None


class KubernetesClient(Protocol):
    def apply_service(self, namespace: str, body: dict[str, Any]) -> None: ...

    def apply_statefulset(self, namespace: str, body: dict[str, Any]) -> None: ...

    def delete_statefulset(self, namespace: str, name: str) -> None: ...

    def delete_service(self, namespace: str, name: str) -> None: ...

    def list_pods(self, namespace: str, selector: str) -> list[Any]: ...

    def pod_logs(self, namespace: str, pod_name: str, tail_lines: int = 80) -> str: ...

    def pod_events(self, namespace: str, pod_name: str) -> str: ...


class KubernetesPythonClient:
    """Small adapter around the official Kubernetes Python client.

    The scheduler depends on this protocol instead of shelling out to `kubectl`.
    Tests can inject a fake client, while real runs load in-cluster config first
    and fall back to the user's kubeconfig.
    """

    def __init__(self, kube_context: str | None = None):
        try:
            from kubernetes import client, config
            from kubernetes.client import ApiException
        except ImportError as e:
            raise WorkerCreationError(
                "kubernetes",
                "Kubernetes Python client not installed",
                "Install the `kubernetes` package in the controller environment.",
            ) from e

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config(context=kube_context)

        self._api_exception = ApiException
        self.core = client.CoreV1Api()
        self.apps = client.AppsV1Api()

    def apply_service(self, namespace: str, body: dict[str, Any]) -> None:
        name = body["metadata"]["name"]
        try:
            self.core.read_namespaced_service(name=name, namespace=namespace)
            self.core.patch_namespaced_service(
                name=name, namespace=namespace, body=body
            )
        except self._api_exception as e:
            if e.status != 404:
                raise
            self.core.create_namespaced_service(namespace=namespace, body=body)

    def apply_statefulset(self, namespace: str, body: dict[str, Any]) -> None:
        name = body["metadata"]["name"]
        try:
            self.apps.read_namespaced_stateful_set(name=name, namespace=namespace)
            self.apps.patch_namespaced_stateful_set(
                name=name, namespace=namespace, body=body
            )
        except self._api_exception as e:
            if e.status != 404:
                raise
            self.apps.create_namespaced_stateful_set(namespace=namespace, body=body)

    def delete_statefulset(self, namespace: str, name: str) -> None:
        try:
            self.apps.delete_namespaced_stateful_set(name=name, namespace=namespace)
        except self._api_exception as e:
            if e.status != 404:
                raise

    def delete_service(self, namespace: str, name: str) -> None:
        try:
            self.core.delete_namespaced_service(name=name, namespace=namespace)
        except self._api_exception as e:
            if e.status != 404:
                raise

    def list_pods(self, namespace: str, selector: str) -> list[Any]:
        return list(
            self.core.list_namespaced_pod(
                namespace=namespace, label_selector=selector
            ).items
        )

    def pod_logs(self, namespace: str, pod_name: str, tail_lines: int = 80) -> str:
        try:
            return self.core.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail_lines,
                timestamps=True,
            )
        except Exception as e:
            return f"[Could not read logs for pod {pod_name}: {e}]"

    def pod_events(self, namespace: str, pod_name: str) -> str:
        try:
            events = self.core.list_namespaced_event(
                namespace=namespace,
                field_selector=f"involvedObject.name={pod_name}",
            ).items
        except Exception as e:
            return f"[Could not read events for pod {pod_name}: {e}]"

        lines = []
        for event in events[-10:]:
            reason = getattr(event, "reason", "") or ""
            message = getattr(event, "message", "") or ""
            event_type = getattr(event, "type", "") or ""
            lines.append(f"{event_type} {reason}: {message}".strip())
        return "\n".join(lines)


def _obj_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
        if cur is default:
            return default
    return cur


class KubernetesScheduler(Scheduler):
    """Kubernetes-backed scheduler using StatefulSets and worker guard HTTP APIs."""

    def __init__(
        self,
        *,
        n_gpus_per_node: int = 8,
        namespace: str | None = None,
        kube_context: str | None = None,
        startup_timeout: float = 300.0,
        health_check_interval: float = 5.0,
        enable_tms_offload: bool | None = None,
        name_resolve_type: str = "nfs",
        nfs_record_root: str = "/tmp/areal/name_resolve",
        etcd3_addr: str = "localhost:2379",
        exp_config: BaseExperimentConfig | None = None,
        k8s_client: KubernetesClient | None = None,
    ):
        self.exp_config = exp_config
        self._n_gpus_per_node = n_gpus_per_node
        self.enable_tms_offload = bool(enable_tms_offload)

        self.experiment_name: str | None = None
        self.trial_name: str | None = None
        self.fileroot: str | None = None
        if exp_config is not None:
            self._n_gpus_per_node = exp_config.cluster.n_gpus_per_node
            self.enable_tms_offload = exp_config.enable_offload
            self.experiment_name = exp_config.experiment_name
            self.trial_name = exp_config.trial_name
            self.fileroot = exp_config.cluster.fileroot

        if self.experiment_name is None or self.trial_name is None:
            raise ValueError("experiment_name and trial_name must be provided")

        self.name_resolve_config = NameResolveConfig(
            type=name_resolve_type,
            nfs_record_root=nfs_record_root,
            etcd3_addr=etcd3_addr,
        )
        if exp_config is not None:
            self.name_resolve_config = exp_config.cluster.name_resolve

        if self.fileroot:
            validate_shared_path(self.fileroot, "cluster.fileroot")
        if self.name_resolve_config.type == "nfs":
            validate_shared_path(
                self.name_resolve_config.nfs_record_root,
                "name_resolve.nfs_record_root",
            )

        name_resolve.reconfigure(self.name_resolve_config)
        name_resolve.clear_subtree(
            names.trial_root(self.experiment_name, self.trial_name)
        )

        self.namespace = (
            namespace
            or os.environ.get("AREAL_K8S_NAMESPACE")
            or os.environ.get("KUBERNETES_NAMESPACE")
            or "default"
        )
        self.kube_context = kube_context or os.environ.get("AREAL_K8S_CONTEXT")
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self._k8s_client = k8s_client

        self._workers: dict[str, list[K8sWorkerInfo]] = {}
        self._statefulsets: dict[str, str] = {}
        self._colocated_roles: dict[str, str] = {}

        logger.info(
            "Initialized KubernetesScheduler: exp=%s trial=%s ns=%s ctx=%s "
            "n_gpus_per_node=%s",
            self.experiment_name,
            self.trial_name,
            self.namespace,
            self.kube_context,
            self.n_gpus_per_node,
        )

    @property
    def n_gpus_per_node(self) -> int:
        return self._n_gpus_per_node

    @property
    def k8s(self) -> KubernetesClient:
        if self._k8s_client is None:
            self._k8s_client = KubernetesPythonClient(self.kube_context)
        return self._k8s_client

    @staticmethod
    def _sanitize_k8s_name(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9-]+", "-", s)
        s = re.sub(r"-+", "-", s).strip("-")
        if not s:
            s = "areal"
        if len(s) > 63:
            suffix = hashlib.sha1(s.encode()).hexdigest()[:8]
            s = f"{s[: 62 - len(suffix)]}-{suffix}"
        return s

    @staticmethod
    def _sanitize_label_value(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-.")
        if not s:
            s = "unknown"
        if len(s) > 63:
            suffix = hashlib.sha1(s.encode()).hexdigest()[:8]
            s = f"{s[: 62 - len(suffix)]}-{suffix}".strip("-.")
        return s

    def _resource_name(self, role: str) -> str:
        base = f"areal-{self.experiment_name}-{self.trial_name}-{role}"
        return self._sanitize_k8s_name(base)

    def _labels(self, role: str) -> dict[str, str]:
        return {
            "app.kubernetes.io/name": "areal",
            "app.kubernetes.io/managed-by": "areal-scheduler",
            "app.kubernetes.io/instance": self._resource_name(role),
            "areal.openpsi.io/experiment": self._sanitize_label_value(
                str(self.experiment_name)
            ),
            "areal.openpsi.io/trial": self._sanitize_label_value(str(self.trial_name)),
            "areal.openpsi.io/role": self._sanitize_label_value(role),
        }

    def _selector(self, role: str) -> str:
        return ",".join(f"{k}={v}" for k, v in self._labels(role).items())

    def _prepare_worker_specs(
        self, role: str, num_workers: int, schedulings: list[SchedulingSpec] | None
    ) -> list[SchedulingSpec]:
        if not schedulings:
            raise WorkerCreationError(
                role, "Invalid configuration", "Tasks SchedulingSpec must be provided"
            )
        if len(schedulings) != 1:
            raise WorkerCreationError(
                role,
                "Unsupported Kubernetes scheduling",
                "KubernetesScheduler currently supports exactly one SchedulingSpec "
                "per role.",
            )

        spec = schedulings[0]
        if spec.nodelist or spec.exclude:
            raise WorkerCreationError(
                role,
                "Unsupported Kubernetes SchedulingSpec fields",
                "nodelist/exclude are Slurm-specific; use Kubernetes node "
                "selectors, affinity, or taints instead.",
            )
        if spec.task_type != "worker":
            raise WorkerCreationError(
                role,
                "Unsupported Kubernetes SchedulingSpec fields",
                "Only task_type='worker' is supported by KubernetesScheduler.",
            )

        spec.env_vars.update(get_env_vars())
        if self.enable_tms_offload:
            spec.env_vars.update(get_tms_env_vars())
        spec.env_vars.update(
            get_thread_env_vars(cpus_per_task=spec.cpu, existing_env_vars=spec.env_vars)
        )
        return [spec] * num_workers

    def _manifest_objects(
        self, *, role: str, replicas: int, spec: SchedulingSpec
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not spec.image:
            raise WorkerCreationError(
                role, "Invalid configuration", "SchedulingSpec.image is required"
            )
        if spec.image.endswith(".sif"):
            raise WorkerCreationError(
                role,
                "Invalid Kubernetes image",
                "SchedulingSpec.image must be a container image reference, "
                "not an Apptainer/Singularity .sif path.",
            )

        sts_name = self._resource_name(role)
        labels = self._labels(role)
        env_list = [
            {"name": k, "value": str(v)} for k, v in (spec.env_vars or {}).items()
        ]
        rpc_cmd = spec.cmd or "python -m areal.infra.rpc.rpc_server"
        setup_cmds = spec.additional_bash_cmds or []
        bash = "\n".join(
            [
                "set -euo pipefail",
                'IDX="${HOSTNAME##*-}"',
                *setup_cmds,
                (
                    f"exec {rpc_cmd} "
                    "--host 0.0.0.0 "
                    f"--port {_K8S_RPC_PORT} "
                    '--experiment-name "${AREAL_EXPERIMENT_NAME}" '
                    '--trial-name "${AREAL_TRIAL_NAME}" '
                    '--role "${AREAL_ROLE}" '
                    '--worker-index "${IDX}" '
                    '--name-resolve-type "${AREAL_NAME_RESOLVE_TYPE}" '
                    '--nfs-record-root "${AREAL_NFS_RECORD_ROOT}" '
                    '--etcd3-addr "${AREAL_ETCD3_ADDR}" '
                    '--fileroot "${AREAL_FILEROOT}"'
                ),
            ]
        )

        base_env = [
            {"name": "AREAL_EXPERIMENT_NAME", "value": str(self.experiment_name)},
            {"name": "AREAL_TRIAL_NAME", "value": str(self.trial_name)},
            {"name": "AREAL_ROLE", "value": role},
            {
                "name": "AREAL_NAME_RESOLVE_TYPE",
                "value": str(self.name_resolve_config.type),
            },
            {
                "name": "AREAL_NFS_RECORD_ROOT",
                "value": str(self.name_resolve_config.nfs_record_root),
            },
            {
                "name": "AREAL_ETCD3_ADDR",
                "value": str(self.name_resolve_config.etcd3_addr),
            },
            {"name": "AREAL_FILEROOT", "value": str(self.fileroot or "")},
        ]

        resources: dict[str, Any] = {
            "requests": {"cpu": str(spec.cpu), "memory": f"{int(spec.mem)}Gi"},
            "limits": {"cpu": str(spec.cpu), "memory": f"{int(spec.mem)}Gi"},
        }
        if spec.gpu > 0:
            resources["requests"]["nvidia.com/gpu"] = str(spec.gpu)
            resources["limits"]["nvidia.com/gpu"] = str(spec.gpu)

        pod_spec: dict[str, Any] = {
            "serviceAccountName": os.environ.get(
                "AREAL_K8S_SERVICE_ACCOUNT", "default"
            ),
            "terminationGracePeriodSeconds": int(
                os.environ.get("AREAL_K8S_TERMINATION_GRACE_SECONDS", "60")
            ),
            "restartPolicy": "Always",
            "containers": [
                {
                    "name": "worker",
                    "image": spec.image,
                    "imagePullPolicy": os.environ.get(
                        "AREAL_K8S_IMAGE_PULL_POLICY", "IfNotPresent"
                    ),
                    "command": ["bash", "-lc", bash],
                    "env": [*base_env, *env_list],
                    "ports": [{"name": "rpc", "containerPort": _K8S_RPC_PORT}],
                    "readinessProbe": {
                        "httpGet": {"path": "/health", "port": "rpc"},
                        "periodSeconds": 5,
                        "failureThreshold": 24,
                    },
                    "resources": resources,
                }
            ],
        }

        node_selector = os.environ.get("AREAL_K8S_NODE_SELECTOR")
        if node_selector:
            pod_spec["nodeSelector"] = dict(
                item.split("=", 1) for item in node_selector.split(",") if "=" in item
            )

        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": sts_name, "labels": labels},
            "spec": {
                "clusterIP": "None",
                "selector": labels,
                "ports": [{"name": "rpc", "port": _K8S_RPC_PORT, "targetPort": "rpc"}],
            },
        }
        statefulset = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {"name": sts_name, "labels": labels},
            "spec": {
                "serviceName": sts_name,
                "podManagementPolicy": "Parallel",
                "replicas": int(replicas),
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": pod_spec,
                },
            },
        }
        return service, statefulset

    def _render_statefulset_yaml(
        self, *, role: str, replicas: int, spec: SchedulingSpec
    ) -> str:
        service, statefulset = self._manifest_objects(
            role=role, replicas=replicas, spec=spec
        )
        return "\n---\n".join(
            [
                yaml.safe_dump(service, sort_keys=False),
                yaml.safe_dump(statefulset, sort_keys=False),
            ]
        )

    def _apply_statefulset(self, role: str, replicas: int, spec: SchedulingSpec) -> str:
        service, statefulset = self._manifest_objects(
            role=role, replicas=replicas, spec=spec
        )
        try:
            self.k8s.apply_service(self.namespace, service)
            self.k8s.apply_statefulset(self.namespace, statefulset)
        except Exception as e:
            raise WorkerCreationError(
                role, "Failed to apply Kubernetes resources", str(e)
            ) from e
        sts_name = self._resource_name(role)
        self._statefulsets[role] = sts_name
        return sts_name

    def _delete_statefulset(self, role: str) -> None:
        sts = self._statefulsets.get(role) or self._resource_name(role)
        try:
            self.k8s.delete_statefulset(self.namespace, sts)
        except Exception as e:
            logger.warning(
                "Failed to delete StatefulSet %s for role %s: %s", sts, role, e
            )
        try:
            self.k8s.delete_service(self.namespace, sts)
        except Exception as e:
            logger.warning("Failed to delete Service %s for role %s: %s", sts, role, e)

    def _pod_diagnostics(self, role: str, tail_lines: int = 80) -> str:
        try:
            pods = self.k8s.list_pods(self.namespace, self._selector(role))
        except Exception as e:
            return f"[Could not list pods for role {role}: {e}]"

        chunks = []
        for pod in pods:
            name = _obj_get(pod, "metadata.name", "<unknown>")
            phase = _obj_get(pod, "status.phase", "<unknown>")
            chunks.append(f"Pod {name}: phase={phase}")
            events = self.k8s.pod_events(self.namespace, name)
            if events:
                chunks.append(f"Events:\n{events}")
            logs = self.k8s.pod_logs(self.namespace, name, tail_lines=tail_lines)
            if logs:
                chunks.append(f"Logs:\n{logs}")
        if chunks:
            return "\n".join(chunks)
        return f"No pods found for selector {self._selector(role)}"

    def _check_pods_health(self, role: str) -> None:
        sts = self._statefulsets.get(role)
        if not sts:
            return
        try:
            pods = self.k8s.list_pods(self.namespace, self._selector(role))
        except Exception as e:
            raise WorkerFailedError(
                f"{role}/*",
                -1,
                f"Failed to query Kubernetes pods: {e}",
            ) from e

        for pod in pods:
            phase = _obj_get(pod, "status.phase")
            name = _obj_get(pod, "metadata.name", "<unknown>")
            container_statuses = _obj_get(pod, "status.container_statuses", None)
            if container_statuses is None:
                container_statuses = _obj_get(pod, "status.containerStatuses", []) or []
            for cs in container_statuses:
                state = _obj_get(cs, "state", {}) or {}
                waiting = _obj_get(state, "waiting")
                terminated = _obj_get(state, "terminated")
                waiting_reason = _obj_get(waiting, "reason", "") if waiting else ""
                if waiting_reason in {
                    "CrashLoopBackOff",
                    "ImagePullBackOff",
                    "ErrImagePull",
                    "CreateContainerConfigError",
                }:
                    raise WorkerFailedError(
                        f"{role}/*",
                        -1,
                        f"Pod {name} {waiting_reason}\n{self._pod_diagnostics(role)}",
                    )
                if terminated:
                    exit_code = int(
                        _obj_get(
                            terminated,
                            "exit_code",
                            _obj_get(terminated, "exitCode", -1),
                        )
                    )
                    if exit_code != 0:
                        raise WorkerFailedError(
                            f"{role}/*",
                            exit_code,
                            f"Pod {name} exited\n{self._pod_diagnostics(role)}",
                        )
            if phase == "Failed":
                raise WorkerFailedError(
                    f"{role}/*",
                    -1,
                    f"Pod {name} phase=Failed\n{self._pod_diagnostics(role)}",
                )

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        role = job.role
        replicas = job.replicas
        if role in self._workers:
            raise WorkerCreationError(
                role,
                "Worker group already exists",
                f"Use delete_workers('{role}') first to remove existing workers.",
            )
        if replicas <= 0:
            raise WorkerCreationError(
                role, "Invalid configuration", "replicas must be greater than 0"
            )

        schedulings = self._prepare_worker_specs(role, replicas, job.tasks)
        strategy = job.scheduling_strategy
        strategy_type = SchedulingStrategyType(strategy.type)
        colocate_role = strategy.target

        if strategy_type == SchedulingStrategyType.colocation:
            if not colocate_role:
                raise WorkerCreationError(
                    role,
                    "Invalid strategy",
                    "Colocation strategy requires target role to be specified",
                )
            if colocate_role not in self._workers:
                raise WorkerNotFoundError(
                    f"Cannot colocate with role '{colocate_role}' - role not found"
                )
            target_workers = self._workers[colocate_role]
            if replicas != len(target_workers):
                raise WorkerCreationError(
                    role,
                    "Replica count mismatch",
                    "Colocated role must have same replica count as target "
                    f"({replicas} != {len(target_workers)})",
                )
            if strategy.fork:
                return self.fork_workers(role, colocate_role)
            worker_ids = [w.worker.id for w in target_workers]
            self._colocated_roles[role] = colocate_role
            return worker_ids

        if strategy_type != SchedulingStrategyType.separation:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")

        spec = schedulings[0]
        self._apply_statefulset(role, replicas, spec)

        workers: list[K8sWorkerInfo] = []
        worker_ids: list[str] = []
        for idx in range(replicas):
            worker_id = f"{role}/{idx}"
            workers.append(
                K8sWorkerInfo(
                    worker=Worker(
                        id=worker_id,
                        ip="",
                        worker_ports=[],
                        engine_ports=[],
                    ),
                    role=role,
                    task_index=idx,
                    discovered=False,
                    spec=spec,
                )
            )
            worker_ids.append(worker_id)

        self._workers[role] = workers
        return worker_ids

    def _discover_worker_network(self, role: str) -> None:
        if role not in self._workers:
            raise WorkerNotFoundError(f"Role '{role}' is not created yet")

        for wi in self._workers[role]:
            if wi.discovered:
                continue
            key = names.worker_discovery(
                self.experiment_name, self.trial_name, role, str(wi.task_index)
            )
            try:
                addr = name_resolve.get(key)
            except name_resolve.NameEntryNotFoundError:
                continue
            ip, port = split_hostport(addr)
            wi.worker.ip = ip
            wi.worker.worker_ports = [str(port)]
            wi.discovered = True

            if wi.spec is not None and wi.spec.port_count > 1:
                try:
                    resp = requests.post(
                        f"http://{format_hostport(ip, int(port))}/alloc_ports",
                        json={"count": int(wi.spec.port_count - 1)},
                        timeout=10.0,
                    )
                    resp.raise_for_status()
                    wi.worker.worker_ports += list(map(str, resp.json()["ports"]))
                except requests.RequestException as e:
                    raise WorkerFailedError(
                        wi.worker.id,
                        -1,
                        "Failed to allocate worker ports: "
                        f"{e}\n{self._pod_diagnostics(role)}",
                    ) from e

            logger.debug("Discovered %s at %s", wi.worker.id, addr)

    def _is_worker_ready(self, wi: K8sWorkerInfo) -> bool:
        if not wi.discovered:
            return False
        port = int(wi.worker.worker_ports[0])
        url = f"http://{format_hostport(wi.worker.ip, port)}/health"
        try:
            return requests.get(url, timeout=2.0).status_code == 200
        except Exception:
            return False

    def _configure_worker(self, wi: K8sWorkerInfo, worker_rank: int) -> None:
        if self.exp_config is None:
            return
        while not self._is_worker_ready(wi):
            time.sleep(0.1)

        worker_id = wi.worker.id
        port = int(wi.worker.worker_ports[0])
        url = f"http://{format_hostport(wi.worker.ip, port)}/configure"
        try:
            resp = requests.post(
                url,
                data=orjson.dumps(
                    serialize_value(
                        dict(config=self.exp_config, role=wi.role, rank=worker_rank)
                    )
                ),
                headers={"Content-Type": "application/json"},
                timeout=300.0,
            )
            if resp.status_code == 200:
                return
            detail = resp.json().get("error", "Unknown error")
            raise WorkerConfigurationError(worker_id, detail, str(resp.status_code))
        except requests.exceptions.ConnectionError as e:
            raise RPCConnectionError(worker_id, wi.worker.ip, port, str(e)) from e

    def get_workers(self, role: str, timeout: float | None = None) -> list[Worker]:
        if role in self._colocated_roles:
            if role in self._workers:
                workers = self._workers[role]
                for wi in workers:
                    if not self._is_worker_ready(wi):
                        raise WorkerFailedError(
                            wi.worker.id,
                            -1,
                            "Forked worker not responding\n"
                            f"{self._pod_diagnostics(self._colocated_roles[role])}",
                        )
                return [w.worker for w in workers]
            return self.get_workers(self._colocated_roles[role], timeout)

        if role not in self._workers:
            raise WorkerNotFoundError(role)

        workers = self._workers[role]
        timeout = timeout if timeout is not None else self.startup_timeout
        start = time.time()

        while time.time() - start < timeout:
            self._check_pods_health(role)
            if any(not w.discovered for w in workers):
                self._discover_worker_network(role)
            ready = [w for w in workers if self._is_worker_ready(w)]
            if len(ready) == len(workers):
                if self.exp_config is not None:
                    for rank, wi in enumerate(workers):
                        self._configure_worker(wi, rank)
                return [w.worker for w in workers]
            time.sleep(self.health_check_interval)

        raise WorkerTimeoutError(
            f"{role}. Diagnostics:\n{self._pod_diagnostics(role)}", timeout
        )

    @staticmethod
    async def _wait_for_fork_ready(
        session: aiohttp.ClientSession, host: str, port: int, timeout: float = 60
    ) -> bool:
        url = f"http://{format_hostport(host, port)}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        return True
            except (TimeoutError, aiohttp.ClientError):
                pass
            await asyncio.sleep(0.5)
        return False

    async def _fork_single_worker(
        self,
        session: aiohttp.ClientSession,
        role: str,
        idx: int,
        target_wi: K8sWorkerInfo,
        target_role: str,
        command: str | None = None,
    ) -> K8sWorkerInfo:
        worker_id = f"{role}/{idx}"
        guard_url = (
            "http://"
            f"{format_hostport(
                target_wi.worker.ip, int(target_wi.worker.worker_ports[0])
            )}"
        )

        async with session.post(
            f"{guard_url}/alloc_ports", json={"count": 1}
        ) as alloc_resp:
            if alloc_resp.status != 200:
                raise WorkerCreationError(
                    role,
                    f"Port allocation failed for worker {idx}",
                    await alloc_resp.text(),
                )
            alloc_data = await alloc_resp.json()
            forked_host = alloc_data["host"]
            forked_port = alloc_data["ports"][0]

        module_path = command or "areal.infra.rpc.rpc_server"
        raw_cmd = [
            "python",
            "-m",
            module_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(forked_port),
            "--experiment-name",
            str(self.experiment_name),
            "--trial-name",
            str(self.trial_name),
            "--role",
            role,
            "--worker-index",
            str(idx),
            "--name-resolve-type",
            self.name_resolve_config.type,
            "--nfs-record-root",
            self.name_resolve_config.nfs_record_root,
            "--etcd3-addr",
            self.name_resolve_config.etcd3_addr,
            "--fileroot",
            str(self.fileroot or ""),
        ]
        async with session.post(
            f"{guard_url}/fork",
            json={"role": role, "worker_index": idx, "raw_cmd": raw_cmd},
        ) as response:
            if response.status != 200:
                raise WorkerCreationError(
                    role, f"Fork failed for worker {idx}", await response.text()
                )
            result = await response.json()
            if result.get("status") != "success":
                raise WorkerCreationError(
                    role,
                    f"Fork failed for worker {idx}",
                    result.get("error", "Unknown error"),
                )

        if not await self._wait_for_fork_ready(session, forked_host, int(forked_port)):
            await self._kill_forked_worker(session, role, idx, target_wi)
            raise WorkerCreationError(
                role,
                f"Forked worker {idx} failed to become ready",
                f"{forked_host}:{forked_port}",
            )

        worker = Worker(
            id=worker_id,
            ip=forked_host,
            worker_ports=[str(forked_port)],
            engine_ports=[],
        )
        port_cnt = len(self._workers[target_role][0].worker.worker_ports)
        if port_cnt > 1:
            async with session.post(
                f"http://{format_hostport(forked_host, int(forked_port))}/alloc_ports",
                json={"count": int(port_cnt - 1)},
            ) as resp:
                if resp.status != 200:
                    await self._kill_forked_worker(session, role, idx, target_wi)
                    raise WorkerCreationError(
                        role,
                        f"alloc_ports failed for forked worker {idx}",
                        await resp.text(),
                    )
                worker.worker_ports += list(map(str, (await resp.json())["ports"]))

        return K8sWorkerInfo(
            worker=worker,
            role=role,
            task_index=idx,
            discovered=True,
            spec=target_wi.spec,
        )

    async def _kill_forked_worker(
        self,
        session: aiohttp.ClientSession,
        role: str,
        idx: int,
        target_wi: K8sWorkerInfo,
    ) -> None:
        url = (
            "http://"
            f"{format_hostport(
                target_wi.worker.ip, int(target_wi.worker.worker_ports[0])
            )}"
            "/kill_forked_worker"
        )
        try:
            async with session.post(
                url, json={"role": role, "worker_index": idx}
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Failed to kill forked worker %s/%s: HTTP %s: %s",
                        role,
                        idx,
                        resp.status,
                        await resp.text(),
                    )
        except Exception as e:
            logger.warning("Exception killing forked worker %s/%s: %s", role, idx, e)

    async def _cleanup_forked_workers_async(
        self,
        role: str,
        target_role: str,
        workers: list[K8sWorkerInfo],
    ) -> None:
        target_workers = self._workers.get(target_role, [])
        if not target_workers:
            logger.warning(
                "Cannot cleanup forked workers: target role '%s' not found",
                target_role,
            )
            return
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(
            timeout=timeout, connector=get_default_connector()
        ) as session:
            tasks = []
            for wi in workers:
                idx = int(wi.worker.id.rsplit("/", 1)[-1])
                if idx < len(target_workers):
                    tasks.append(
                        self._kill_forked_worker(
                            session, role, idx, target_workers[idx]
                        )
                    )
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _create_forked_workers_async(
        self,
        role: str,
        target_role: str,
        target_workers: list[K8sWorkerInfo],
        command: str | None = None,
    ) -> list[str]:
        timeout = aiohttp.ClientTimeout(total=120.0)
        async with aiohttp.ClientSession(
            timeout=timeout, connector=get_default_connector()
        ) as session:
            tasks = [
                self._fork_single_worker(
                    session, role, idx, target_wi, target_role, command
                )
                for idx, target_wi in enumerate(target_workers)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        workers: list[K8sWorkerInfo] = []
        failures: list[int] = []
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                failures.append(idx)
                logger.error("Failed to fork worker %s/%s: %s", role, idx, res)
            else:
                workers.append(res)

        if failures:
            if workers:
                await self._cleanup_forked_workers_async(role, target_role, workers)
            raise WorkerCreationError(
                role,
                f"Failed to fork {len(failures)} out of {len(target_workers)} workers",
                f"Failed indices: {failures}",
            )

        self._workers[role] = workers
        self._colocated_roles[role] = target_role
        if self.exp_config is not None:
            for rank, wi in enumerate(workers):
                self._configure_worker(wi, rank)
        return [w.worker.id for w in workers]

    def fork_workers(
        self, role: str, target_role: str, command: str | None = None
    ) -> list[str]:
        if target_role not in self._workers:
            raise WorkerNotFoundError(f"Target role '{target_role}' not found for fork")
        target_workers = self._workers[target_role]
        try:
            return run_async_task(
                self._create_forked_workers_async,
                role,
                target_role,
                target_workers,
                command,
            )
        except Exception:
            self._workers.pop(role, None)
            self._colocated_roles.pop(role, None)
            raise

    def _find_worker_by_id(self, worker_id: str) -> K8sWorkerInfo | None:
        for workers in self._workers.values():
            for wi in workers:
                if wi.worker.id == worker_id:
                    return wi
        return None

    def _verify_worker_alive(self, worker_id: str) -> K8sWorkerInfo:
        wi = self._find_worker_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)
        self._check_pods_health(self._colocated_roles.get(wi.role, wi.role))
        return wi

    async def set_worker_env(self, worker_id: str, env: dict[str, str]) -> None:
        wi = self._verify_worker_alive(worker_id)
        if not env:
            return

        port = int(wi.worker.worker_ports[0])
        url = f"http://{format_hostport(wi.worker.ip, port)}/set_env"
        try:
            timeout = aiohttp.ClientTimeout(total=30.0)
            async with aiohttp.ClientSession(
                timeout=timeout, connector=get_default_connector()
            ) as session:
                async with session.post(
                    url,
                    data=orjson.dumps({"env": env}),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        return
                    detail = (await response.json()).get("error", "Unknown error")
                    raise SchedulerError(
                        f"Failed to set env on worker '{worker_id}' "
                        f"(status={response.status}): {detail}"
                    )
        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            raise RPCConnectionError(worker_id, wi.worker.ip, port, str(e)) from e
        except TimeoutError as e:
            raise SchedulerError(
                f"set_env timed out for worker '{worker_id}': {e}"
            ) from e

    async def create_engine(
        self,
        worker_id: str,
        engine: str,
        engine_name: str | None = None,
        *args,
        **kwargs,
    ) -> Any:
        wi = self._verify_worker_alive(worker_id)
        health_role = self._colocated_roles.get(wi.role, wi.role)
        if engine_name is None:
            engine_name = worker_id
        if not isinstance(engine, str):
            raise EngineCreationError(
                worker_id, f"Engine must be a string import path, got {type(engine)}"
            )

        payload = {
            "engine": engine,
            "engine_name": engine_name,
            "init_args": serialize_value(list(args)),
            "init_kwargs": serialize_value(kwargs),
        }
        port = int(wi.worker.worker_ports[0])
        url = f"http://{format_hostport(wi.worker.ip, port)}/create_engine"

        self._check_pods_health(health_role)
        try:
            timeout = aiohttp.ClientTimeout(total=300.0)
            async with aiohttp.ClientSession(
                timeout=timeout,
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                async with session.post(
                    url,
                    data=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        return (await response.json()).get("result")
                    detail = (await response.json()).get("error", "Unknown error")
                    if response.status == 400 and "Failed to import" in detail:
                        raise EngineImportError(engine, detail)
                    if response.status in (400, 500):
                        raise EngineCreationError(worker_id, detail, response.status)
                    raise EngineCreationError(
                        worker_id,
                        f"Unexpected status code: {response.status}",
                        response.status,
                    )
        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            raise RPCConnectionError(worker_id, wi.worker.ip, port, str(e)) from e
        except TimeoutError as e:
            raise EngineCreationError(
                worker_id, f"Engine creation timed out: {e}"
            ) from e

    def call_engine(
        self,
        worker_id: str,
        method: str,
        engine_name: str | None = None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        wi = self._verify_worker_alive(worker_id)
        health_role = self._colocated_roles.get(wi.role, wi.role)
        if engine_name is None:
            engine_name = worker_id

        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
            "rpc_meta": rpc_meta,
        }
        port = int(wi.worker.worker_ports[0])
        url = f"http://{format_hostport(wi.worker.ip, port)}/call"
        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            self._check_pods_health(health_role)
            try:
                resp = requests.post(url, json=payload, timeout=http_timeout)
                if resp.status_code == 200:
                    return deserialize_value(resp.json().get("result"))
                if resp.status_code in (400, 500):
                    detail = resp.json().get("error", "Unknown error")
                    raise EngineCallError(worker_id, method, detail, attempt=attempt)
                last_error = (
                    "Service unavailable (503)"
                    if resp.status_code == 503
                    else f"HTTP {resp.status_code}: {resp.text}"
                )
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout: {e}"
            except requests.exceptions.ConnectionError as e:
                self._check_pods_health(health_role)
                last_error = f"Connection error: {e}"
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Unexpected error: {e}"

            if attempt < max_retries:
                time.sleep(retry_delay * (2 ** (attempt - 1)))

        raise EngineCallError(
            worker_id,
            method,
            f"{last_error or 'Max retries exceeded'}\n"
            f"{self._pod_diagnostics(health_role)}",
            attempt=max_retries,
        )

    async def async_call_engine(
        self,
        worker_id: str,
        method: str,
        engine_name: str | None = None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        wi = self._verify_worker_alive(worker_id)
        health_role = self._colocated_roles.get(wi.role, wi.role)
        if engine_name is None:
            engine_name = worker_id

        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
            "rpc_meta": rpc_meta,
        }
        port = int(wi.worker.worker_ports[0])
        url = f"http://{format_hostport(wi.worker.ip, port)}/call"
        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            self._check_pods_health(health_role)
            try:
                timeout = aiohttp.ClientTimeout(total=http_timeout)
                async with aiohttp.ClientSession(
                    timeout=timeout,
                    read_bufsize=1024 * 1024 * 10,
                    connector=get_default_connector(),
                ) as session:
                    async with session.post(
                        url,
                        data=orjson.dumps(payload),
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        if resp.status == 200:
                            return deserialize_value((await resp.json()).get("result"))
                        if resp.status in (400, 500):
                            detail = (await resp.json()).get("error", "Unknown error")
                            raise EngineCallError(
                                worker_id, method, detail, attempt=attempt
                            )
                        last_error = (
                            "Service unavailable (503)"
                            if resp.status == 503
                            else f"HTTP {resp.status}: {await resp.text()}"
                        )
            except TimeoutError as e:
                last_error = f"Timeout: {e}"
            except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
                self._check_pods_health(health_role)
                last_error = f"Connection error: {e}"
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Unexpected error: {e}"

            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))

        raise EngineCallError(
            worker_id,
            method,
            f"{last_error or 'Max retries exceeded'}\n"
            f"{self._pod_diagnostics(health_role)}",
            attempt=max_retries,
        )

    def delete_workers(self, role: str | None = None, reverse_order: bool = False):
        del reverse_order
        if role is None:
            for r in list(self._colocated_roles.keys()):
                self.delete_workers(r)
            for r in list(self._workers.keys()):
                self.delete_workers(r)
            return

        if role in self._colocated_roles:
            target_role = self._colocated_roles[role]
            workers = self._workers.get(role)
            if workers:
                run_async_task(
                    self._cleanup_forked_workers_async, role, target_role, workers
                )
                self._workers.pop(role, None)
            self._colocated_roles.pop(role, None)
            return

        if role not in self._workers:
            logger.warning("Role '%s' not found, skipping deletion", role)
            return

        workers = self._workers[role]
        try:
            self._destroy_engines_on_workers(workers)
        except Exception as e:
            logger.warning("Failed to destroy engines before k8s delete: %s", e)

        self._delete_statefulset(role)
        del self._workers[role]
        self._statefulsets.pop(role, None)

    def _destroy_engines_on_workers(
        self, workers: list[K8sWorkerInfo], timeout: float = 30.0
    ) -> None:
        if not workers:
            return

        async def _destroy_all():
            destroy_timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(
                timeout=destroy_timeout, connector=get_default_connector()
            ) as session:
                reqs = []
                for wi in workers:
                    if not wi.worker.worker_ports:
                        continue
                    port = int(wi.worker.worker_ports[0])
                    payload = {
                        "method": "destroy",
                        "engine_name": wi.worker.id,
                        "args": serialize_value([]),
                        "kwargs": serialize_value({}),
                        "rpc_meta": None,
                    }
                    reqs.append(
                        session.post(
                            f"http://{format_hostport(wi.worker.ip, port)}/call",
                            data=orjson.dumps(payload),
                            headers={"Content-Type": "application/json"},
                        )
                    )
                await asyncio.gather(
                    *[self._safe_destroy_request(r) for r in reqs],
                    return_exceptions=True,
                )

        run_async_task(_destroy_all)

    @staticmethod
    async def _safe_destroy_request(coro):
        try:
            async with coro as resp:
                await resp.read()
        except Exception as e:
            raise RuntimeError(str(e)) from e

    def __del__(self):
        try:
            self.delete_workers()
        except Exception:
            pass
