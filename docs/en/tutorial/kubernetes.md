# Run AReaL on Kubernetes

This page describes how to run AReaL in **single-controller mode** with the built-in
`KubernetesScheduler`.

## What the Scheduler Creates

For each AReaL role, such as `actor`, `rollout`, or `critic`, the scheduler creates:

- one headless `Service`
- one `StatefulSet`
- one worker pod per replica

Each pod runs an AReaL guard/RPC process. By default this is
`python -m areal.infra.rpc.rpc_server`, but `SchedulingSpec.cmd` is honored so
specialized guards can provide their own module.

Workers still use the same discovery and HTTP guard flow as the local and Slurm
schedulers:

- workers publish `host:port` through `name_resolve` (NFS or etcd)
- the controller waits for `/health`
- the controller uses `/alloc_ports`, `/configure`, `/set_env`, `/create_engine`,
  `/call`, `/fork`, and `/kill_forked_worker` for lifecycle and RPC operations

## Prerequisites

- Run the AReaL controller inside the cluster, or from a network that can directly
  reach worker pod IPs.
- Install the Kubernetes Python client in the controller environment:

  ```bash
  pip install kubernetes
  ```

- Configure `cluster.fileroot` and `cluster.name_resolve.*` so the controller and
  worker pods share the same name-resolution backend.
- Install and configure the NVIDIA device plugin if workers request GPUs with
  `SchedulingSpec.gpu > 0`.
- Provide a service account with permission to create, patch, list, and delete
  `Services`, `StatefulSets`, `Pods`, pod logs, and pod events in the target namespace.

## RBAC Permissions

If you are running the AReaL controller with a service account other than a cluster admin, you must provide a `Role` (or `ClusterRole`) with sufficient permissions.

Below is a minimal `Role` and `RoleBinding` example for a namespace named `areal`:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: areal
  name: areal-scheduler
rules:
- apiGroups: [""]
  resources: ["services", "pods"]
  verbs: ["get", "list", "watch", "create", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["statefulsets"]
  verbs: ["get", "list", "watch", "create", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  namespace: areal
  name: areal-scheduler-binding
subjects:
- kind: ServiceAccount
  name: default
  namespace: areal
roleRef:
  kind: Role
  name: areal-scheduler
  apiGroup: rbac.authorization.k8s.io
```

## Minimal Launch

Use the normal training entrypoint and override the scheduler type:

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=kubernetes
```

The short alias is also accepted:

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=k8s
```

## Container Images

Kubernetes requires a normal container image reference in `SchedulingSpec.image`.
Many Slurm examples use an Apptainer/Singularity `.sif` image by default; those are
rejected by the Kubernetes scheduler.

Example:

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=kubernetes \
  actor.scheduling_spec[0].image=ghcr.io/<org>/<image>:<tag> \
  rollout.scheduling_spec[0].image=ghcr.io/<org>/<image>:<tag>
```

The image must contain AReaL, Python dependencies, model runtime dependencies, and any
custom workflow/reward modules imported by workers.

## Namespace and Cluster Selection

The scheduler first tries in-cluster Kubernetes config, then falls back to kubeconfig.
You can select the namespace and kubeconfig context with:

```bash
export AREAL_K8S_NAMESPACE=areal
export AREAL_K8S_CONTEXT=my-cluster

python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=kubernetes
```

Optional environment knobs:

| Variable | Purpose |
| --- | --- |
| `AREAL_K8S_SERVICE_ACCOUNT` | Service account for worker pods. Defaults to `default`. |
| `AREAL_K8S_IMAGE_PULL_POLICY` | Worker image pull policy. Defaults to `IfNotPresent`. |
| `AREAL_K8S_TERMINATION_GRACE_SECONDS` | Pod termination grace period. Defaults to `60`. |
| `AREAL_K8S_NODE_SELECTOR` | Comma-separated `key=value` node selector entries. |

## SchedulingSpec Support

Supported fields:

- `cpu`
- `gpu`
- `mem`
- `port_count`
- `image`
- `env_vars`
- `cmd`
- `additional_bash_cmds`

Rejected fields:

- multiple per-replica `SchedulingSpec` entries
- `nodelist`
- `exclude`
- `task_type != "worker"`

Use Kubernetes-native node selectors, affinity, taints, tolerations, and admission
policies for placement instead of Slurm-specific fields.

## Storage and Name Resolution

The scheduler does not automatically create volumes. Configure shared storage through
your cluster policy, namespace defaults, image conventions, or a future project-specific
pod template mechanism.

For NFS name resolution, both controller and pods must see the same
`cluster.name_resolve.nfs_record_root`. For etcd name resolution, both must reach the
same `cluster.name_resolve.etcd3_addr`.

## Networking

The controller must be able to connect to worker pod IPs on the discovered RPC port.
This is simplest when the controller runs as a pod in the same cluster. Running outside
the cluster usually requires routed pod CIDRs, a VPN, or another cluster networking
setup that exposes pod IPs directly.

## Failure Diagnostics

When workers fail or time out, the scheduler includes pod phase, recent pod events, and
tail logs in scheduler exceptions. This is intended to mirror the Slurm scheduler's
log-tail behavior and make image-pull, crash-loop, and startup failures actionable.

## Limitations

- The scheduler creates one `StatefulSet` per role and currently uses a single
  `SchedulingSpec` as the pod template for all replicas in that role.
- Rich pod customization, such as volumes, tolerations, affinity, image pull secrets,
  and security context, should be provided by namespace policy or added to the
  scheduler before relying on this backend for production clusters.
- Integration tests require a real Kubernetes cluster and are skipped unless explicitly
  enabled.
