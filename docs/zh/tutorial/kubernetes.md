# 在 Kubernetes 上运行 AReaL

本文介绍如何在 **single-controller mode** 下使用内置的
`KubernetesScheduler`。

## 调度器会创建什么

每个 AReaL role（例如 `actor`、`rollout`、`critic`）会对应：

- 一个 headless `Service`
- 一个 `StatefulSet`
- 每个 replica 一个 worker pod

worker 默认运行 `python -m areal.infra.rpc.rpc_server`。如果
`SchedulingSpec.cmd` 被设置，调度器会使用该命令。

worker 的发现与 RPC 流程和 local / Slurm scheduler 保持一致：

- worker 通过 `name_resolve`（NFS 或 etcd）发布 `host:port`
- controller 等待 `/health`
- controller 使用 `/alloc_ports`、`/configure`、`/set_env`、`/create_engine`、
  `/call`、`/fork`、`/kill_forked_worker` 管理生命周期和 RPC 调用

## 前置条件

- 建议将 AReaL controller 运行在集群内；如果在集群外运行，需要能直连 worker
  pod IP。
- controller 环境中需要安装 Kubernetes Python client：

  ```bash
  pip install kubernetes
  ```

- `cluster.fileroot` 和 `cluster.name_resolve.*` 必须让 controller 与 worker pod
  使用同一套 name-resolution 后端。
- 如果 `SchedulingSpec.gpu > 0`，集群需要安装并配置 NVIDIA device plugin。
- service account 需要有创建、更新、列出和删除 `Services`、`StatefulSets`、
  `Pods`，以及读取 pod logs/events 的权限。

## RBAC 权限

如果你在一个没有集群管理员权限的 service account 下运行 AReaL controller，你需要为其配置一个具有足够权限的 `Role`（或 `ClusterRole`）。

以下是在名为 `areal` 的 namespace 下配置 `Role` 和 `RoleBinding` 的示例：

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

## 最小启动命令

使用正常的训练入口，并覆盖 scheduler 类型：

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=kubernetes
```

也可以使用短别名：

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=k8s
```

## 镜像

Kubernetes 需要普通容器镜像引用。Slurm 示例里常见的 `.sif`
Apptainer/Singularity 镜像会被 Kubernetes scheduler 拒绝。

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=kubernetes \
  actor.scheduling_spec[0].image=ghcr.io/<org>/<image>:<tag> \
  rollout.scheduling_spec[0].image=ghcr.io/<org>/<image>:<tag>
```

镜像中需要包含 AReaL、Python 依赖、模型运行时依赖，以及 worker 会动态导入的
自定义 workflow / reward 模块。

## Namespace 与集群选择

调度器会优先使用 in-cluster config，失败后回退到 kubeconfig。可通过环境变量选择
namespace 和 kubeconfig context：

```bash
export AREAL_K8S_NAMESPACE=areal
export AREAL_K8S_CONTEXT=my-cluster

python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=kubernetes
```

可选环境变量：

| 变量 | 作用 |
| --- | --- |
| `AREAL_K8S_SERVICE_ACCOUNT` | worker pod 使用的 service account，默认 `default`。 |
| `AREAL_K8S_IMAGE_PULL_POLICY` | worker 镜像拉取策略，默认 `IfNotPresent`。 |
| `AREAL_K8S_TERMINATION_GRACE_SECONDS` | pod 终止宽限时间，默认 `60`。 |
| `AREAL_K8S_NODE_SELECTOR` | 逗号分隔的 `key=value` node selector。 |

## SchedulingSpec 支持范围

支持：

- `cpu`
- `gpu`
- `mem`
- `port_count`
- `image`
- `env_vars`
- `cmd`
- `additional_bash_cmds`

当前会拒绝：

- 每个 replica 使用不同 `SchedulingSpec`
- `nodelist`
- `exclude`
- `task_type != "worker"`

节点选择请使用 Kubernetes 原生 node selector、affinity、taints、tolerations 或集群
准入策略，而不是 Slurm 专用字段。

## 存储与网络

调度器不会自动创建 volume。共享存储应通过集群策略、namespace 默认配置、镜像约定，
或后续新增的 pod template 机制配置。

controller 必须能连接到 worker pod IP 和发现到的 RPC 端口。最简单的方式是把
controller 也运行在同一个 Kubernetes 集群内。

## 失败诊断

worker 失败或超时时，调度器会在异常中包含 pod phase、近期 pod events 和日志尾部，
以接近 Slurm scheduler 的 log-tail 诊断体验。

## 限制

- 每个 role 目前只支持一个 `SchedulingSpec`，并用它作为整个 StatefulSet 的 pod
  template。
- volumes、tolerations、affinity、image pull secrets、security context 等更丰富的
  pod 配置需要依赖集群策略，或在生产使用前扩展 scheduler。
- Kubernetes 集成测试需要真实集群，默认跳过。
