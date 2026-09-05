# 快速入门

欢迎阅读 **AReaL** 快速入门指南！本指南将演示如何使用 AReaL 运行一个使用 GRPO 算法和基于函数的奖励来训练 LLM 的实验，数据集为
GSM8K。在继续之前，请确保已完成[安装和环境设置](installation.md)。

## 运行实验（单节点）

要运行实验，您需要：

- 训练脚本：
  [examples/math/gsm8k_rl.py](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_rl.py)
- 配置文件 YAML：
  [examples/math/gsm8k_grpo.yaml](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_grpo.yaml)

我们的训练脚本会自动下载数据集（openai/gsm8k）和模型（Qwen/Qwen2.5-1.5B-Instruct）。要使用默认配置运行示例，请从仓库目录执行：

```
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local experiment_name=<您的实验名称> trial_name=<您的试验名称>
```

> **注意**：如需在多节点上运行分布式实验，请参阅[使用 Ray 或 Slurm 的分布式实验](#distributed-experiments-with-ray-or-slurm)。

### 单 GPU 快速验证

AReaL 可以在同一张 GPU 上运行单 rank 的 SGLang rollout 和单 rank 的 FSDP 训练。下面的命令使用 Qwen3-0.6B，并在两个训练
step 后停止，适合在 32 GB 显存的工作站上验证环境。这是快速验证配置，不是针对吞吐量优化的训练配置。

```bash
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    experiment_name=gsm8k-grpo-single-gpu \
    trial_name=smoke \
    actor.path=Qwen/Qwen3-0.6B \
    actor.backend=fsdp:d1 \
    +actor.attn_impl=sdpa \
    rollout.backend=sglang:d1 \
    +rollout.setup_timeout=900 \
    actor.weight_update_mode=disk \
    enable_offload=false \
    +total_train_steps=2 \
    rollout.max_concurrent_rollouts=2 \
    +reward_max_workers=1 \
    gconfig.n_samples=2 \
    gconfig.max_new_tokens=64 \
    gconfig.max_tokens=512 \
    actor.mb_spec.max_tokens_per_mb=512 \
    actor.optimizer.type=adam_bf16 \
    +actor.optimizer_dtype=bfloat16 \
    train_dataset.batch_size=2 \
    train_dataset.num_workers=0 \
    train_dataset.pin_memory=false \
    +train_dataset.scheduling_spec=null \
    valid_dataset=null \
    sglang.context_length=1024 \
    sglang.mem_fraction_static=0.2 \
    +sglang.max_prefill_tokens=1024 \
    +sglang.attention_backend=torch_native \
    +sglang.sampling_backend=pytorch \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=1
```

关键配置如下：

- `actor.backend=fsdp:d1` 和 `rollout.backend=sglang:d1` 会在唯一配置的 GPU 上分别启动单 rank
  的训练与推理服务。
- `valid_dataset=null` 在这次训练快速验证中跳过独立的评估 rollout。
- `reward_max_workers=1` 将奖励计算限制为一个工作进程，降低工作站的系统内存占用。
- `actor.weight_update_mode=disk` 通过磁盘传递更新后的权重，避免在同一张 GPU 上建立 NCCL 通信组。
- BF16 优化器以及较小的 batch、序列长度和 SGLang 内存配置可使快速验证控制在 32 GB 显存以内。
- `rollout.setup_timeout=900` 为新机器上 SGLang 的冷启动预留了足够时间。
- PyTorch attention 和 sampling 后端优先保证快速验证的兼容性，而非吞吐量。
- 当运行接近显存上限时，优先降低 `sglang.mem_fraction_static` 和 `actor.mb_spec.max_tokens_per_mb`。

该快速验证成功后，再增加 `total_train_steps`、序列长度、batch size 和模型规模。更多显存调优方法请参阅
{doc}`../best_practices/handling_oom`。

## 修改配置

所有可用的配置选项都列在
[areal/api/cli_args.py](https://github.com/areal-project/AReaL/blob/main/areal/api/cli_args.py)中。要自定义实验（模型、资源、算法选项），您可以：

1. 直接编辑 YAML 文件
   [examples/math/gsm8k_grpo.yaml](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_grpo.yaml)。
1. 添加命令行选项：
   - 对于 YAML 文件中已存在的选项，直接添加： `actor.path=Qwen/Qwen3-1.7B`。
   - 对于 `cli_args.py` 中存在但不在 YAML 文件中的选项，使用前缀 "+" 添加：
     `+sglang.attention_backend=triton`。

例如，以下是基于我们的 GSM8K GRPO 示例启动自定义配置的命令：

```
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    experiment_name=<您的实验名称> \
    trial_name=<您的试验名称> \
    rollout.backend=sglang:d2p1t1 actor.backend=fsdp:d2p1t1 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=4 \
    gconfig.max_new_tokens=2048 \
    train_dataset.batch_size=1024 \
    +sglang.attention_backend=triton
```

如果您想在训练中启用 [Hugging Face Kernels](https://github.com/huggingface/kernels)，请显式添加训练引擎覆盖项：

```bash
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    experiment_name=<您的实验名称> \
    trial_name=<您的试验名称> \
    +actor.attn_impl=kernels-community/flash-attn \
    +actor.use_kernels=true
```

如果 `critic` 或 `teacher` 也需要使用 kernels，请为这些引擎添加相同的覆盖项。

(distributed-experiments-with-ray-or-slurm)=

## 使用 Ray 或 Slurm 的分布式实验

对于跨多节点的分布式实验，您可以使用 Ray 或 Slurm 调度器。在设置好 Ray 或 Slurm 集群后，通过指定适当的调度器类型启动实验：

```
# 使用 Ray 调度器启动。4 个节点（每个 4 GPU），3 个节点用于生成，1 个节点用于训练。
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=ray \
    experiment_name=<您的实验名称> \
    trial_name=<您的试验名称> \
    rollout.backend=sglang:d12p1t1 actor.backend=fsdp:d4p1t1 \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=4

# 使用 Slurm 调度器启动。16 个节点（每个 8 GPU），12 个节点用于生成，4 个节点用于训练
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=slurm \
    experiment_name=<您的实验名称> \
    trial_name=<您的试验名称> \
    rollout.backend=sglang:d96p1t1 actor.backend=fsdp:d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8
```

其他参考：

- 有关调度器的更多选项，请查看
  [areal/api/cli_args.py](https://github.com/areal-project/AReaL/blob/main/areal/api/cli_args.py)中的
  `SchedulerConfig`。
- 有关如何设置 Ray 集群的指南，请参阅安装文档中的分布式设置部分。

> **重要提示**：确保 `rollout.backend` 和 `actor.backend` 的 GPU 总数与您的集群配置匹配
> （`#GPU == cluster.n_nodes * cluster.n_gpus_per_node`）

<!--
> **Notes**: Before launching distributed experiments, please check if your per-engine `backend` fields match your cluster configuration. Make sure the total GPUs allocated by `rollout.backend` and `actor.backend` equals `cluster.n_nodes * cluster.n_gpus_per_node`.
> **Note**: Ray and Slurm launchers only work for distributed experiments with more than 1 node (`cluster.n_nodes > 1`). They allocate GPUs for training and generation at the granularity of **nodes**, which means the number of GPUs allocated for generation and training must be integer multiples of `cluster.n_gpus_per_node`.
-->

## 传统模式：使用专用启动器的 SPMD 模式

AReaL 还支持通过专用启动器使用 SPMD（单程序多数据）模式。此模式是为向后兼容性而维护的，但现在推荐使用单控制器模式（直接使用 `scheduler.type`
执行脚本）来处理大多数用例。

在 SPMD 模式下，启动器通过 `torchrun` 管理进程生成，并设置 `AREAL_SPMD_MODE=1`。每个 GPU worker 独立运行完整的训练脚本，通过
PyTorch 分布式原语进行协调。

```bash
# 使用本地启动器的 SPMD 模式（传统）
python3 -m areal.infra.launcher.local examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml

# 使用 Ray 启动器的 SPMD 模式（传统）
python3 -m areal.infra.launcher.ray examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml

# 使用 Slurm 启动器的 SPMD 模式（传统）
python3 -m areal.infra.launcher.slurm examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml
```

## 使用 SkyPilot 在云或 K8s 上运行分布式实验

如果您想直接在云或自己的 Kubernetes 基础设施上运行实验，我们建议您使用 SkyPilot。安装和设置 SkyPilot 后（请参阅
{ref}`安装 SkyPilot <install-skypilot>`），您可以基于我们的 SkyPilot 示例（两个 8xA100 GPU
节点）使用一行命令启动分布式实验：

```bash
# 在 GCP 上启动
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml --infra gcp
# 在 AWS 上启动
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml --infra aws
# 在您的 K8s 集群上启动
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml --infra k8s
```

查看
[使用 SkyPilot 运行 AReaL](https://github.com/areal-project/AReaL/blob/main/examples/skypilot/README.md)
了解更多示例详情。查看 [SkyPilot 文档](https://docs.skypilot.co/en/latest/docs/index.html) 了解更多关于
SkyPilot 的信息。

## 下一步

查看 [AReaL 入门](gsm8k_grpo.md) 了解 GRPO GSM8K 示例的完整代码详解。

定制指南：

- [自定义数据集](../customization/dataset.md)
- [自定义 agentic/RVLR rollout 工作流](../customization/agent.md)
