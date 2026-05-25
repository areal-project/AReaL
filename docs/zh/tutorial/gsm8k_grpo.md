# 在 GSM8K 数据集上运行 GRPO

本指南将逐步介绍 AReaL 如何在 GSM8K 数据集上运行 GRPO 算法。我们将使用示例训练脚本
[`examples/math/gsm8k_rl.py`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_rl.py)
和配置文件
[`examples/math/gsm8k_grpo.yaml`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_grpo.yaml)
逐步解释关键概念。

## 概述：AReaL 如何工作

### 单控制器架构

AReaL 使用**单控制器架构**，训练脚本通过 RPC 协调远程工作进程：

```
控制器进程（您的脚本）
    │
    ├─> RolloutController
    │   ├─> 管理 rollout 工作进程（SGLang/vLLM）
    │   ├─> 向推理工作进程提交 prompts
    │   ├─> 收集轨迹
    │   └─> 返回：RTensor（分布式批次）
    │
    └─> TrainController
        ├─> 管理训练工作进程（FSDP/Megatron）
        ├─> 通过 data_parallel_dispatch() 分发 RTensor
        ├─> 工作进程计算前向/反向传播
        ├─> 通过 data_parallel_merge() 合并结果
        └─> 返回：loss、metrics
```

**训练步骤流程**：

1. **Rollout 阶段**：控制器加载数据并将其传递给 RolloutController，后者调度并将 rollout 请求路由到 rollout
   工作进程（GPU）。

   - 每个 rollout 工作进程运行一个完整模型（可能占用多个 GPU）
   - 返回：RTensor，其中分片存储在 rollout 工作进程上（控制器仅持有元数据）

1. **分发阶段**：TrainController 通过 `data_parallel_dispatch()` 分发工作

   - 使用 FFD（First Fit Decreasing）平衡各工作进程间的序列长度
   - 工作进程直接从 rollout 工作进程获取分配的分片

1. **训练阶段**：每个训练工作进程独立处理其分片

   - 支持 5D 并行（数据、张量流水线、上下文、专家）

1. **权重同步**：将更新后的权重传输到推理工作进程

   - 通过 NCCL（快速，GPU 到 GPU）或磁盘（后备）

### 使用 RTensor 的数据流

```
Rollout 工作进程（GPU 0-3）        控制器              训练工作进程（GPU 4-7）
─────────────────────────────    ───────────         ─────────────────────────────
Worker 0: 生成 16 个样本
          ├─> 分片 0 存储 ────────────┐
Worker 1: 生成 16 个样本              │
          ├─> 分片 1 存储 ──────────┐ │
Worker 2: 生成 16 个样本              │ │
          ├─> 分片 2 存储 ────────┐ │ │
Worker 3: 生成 16 个样本              │ │ │
          └─> 分片 3 存储 ──────┐ │ │ │
                                 │ │ │ │
                                 │ │ │ │    RTensor 元数据
                                 │ │ │ └─> 控制器 ─> data_parallel_dispatch()
                                 │ │ └───────────┼────────────┬────────────┐
                                 │ └─────────────┼────────────┼────────────┤
                                 └───────────────┼────────────┼────────────┤
                                                 │            │            │
                                                 ▼            ▼            ▼
                                             Worker 4:    Worker 5:    Worker 6:
                                             获取        获取        获取
                                             分片 0,1    分片 2      分片 3
                                                 │            │            │
                                             ├─> 前向    ├─> 前向    ├─> 前向
                                             ├─> 反向    ├─> 反向    ├─> 反向
                                             └─> 梯度    └─> 梯度    └─> 梯度
                                                          │
                                                   NCCL AllReduce
                                                          │
                                             Worker 4:    Worker 5:    Worker 6:
                                             返回        返回        返回
                                             RTensor    RTensor    RTensor
                                                 │            │            │
                                                 └────────────┴────────────┘
                                                              │
                                                     data_parallel_merge()
                                                              │
                                                              ▼
                                                      控制器接收：
                                                      • loss（标量）
                                                      • metrics（字典）
```

在接下来的章节中，我们将逐步阅读代码来详细解释每个组件。

## 启动实验

AReaL 支持为不同环境使用不同的调度器后端启动实验。如[快速入门指南](../tutorial/quickstart.md)所示，您可以通过以下方式启动实验：

```bash
# 本地机器（使用子进程）
python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local

# Ray 集群
python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=ray

# Slurm 集群
python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=slurm
```

### 单控制器模式如何工作

**训练脚本**：您的实验入口点（例如 `examples/math/gsm8k_rl.py`），运行在控制器节点上。

**控制器职责**：

1. 控制器创建工作进程（HTTP 或 Ray 服务器） `scheduler.create_workers()`
1. 创建工作进程后，控制器通过 `scheduler.create_engine()` 创建引擎（例如 `RemoteSGLangEngine`、`FSDPEngine`）
1. 控制器通过 RPC 分发工作，并通过 PyTorch 分布式原语协调

**关键配置**：

- `scheduler.type`：确定使用哪个后端（`local`、`ray` 或 `slurm`）
- 各引擎的 `backend` 字段（如 `rollout.backend`、`actor.backend`）：确定各引擎的 GPU 分配和并行策略
- 调度器自动处理工作进程放置、资源分配和生命周期管理

### 配置文件

配置文件是 YAML 文件，指定来自
[`areal/api/cli_args.py`](https://github.com/areal-project/AReaL/blob/main/areal/api/cli_args.py)
的选项。您可以通过 CLI 覆盖设置：

```bash
# 示例：更改模型和注意力后端
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    actor.path=Qwen/Qwen3-1.7B \
    +sglang.attention_backend=triton
```

在训练脚本中解析配置：

```python
config, _ = load_expr_config(args, GRPOConfig)
config: GRPOConfig
```

请参阅 [CLI 参考](../cli_reference.md) 获取所有可用选项。

## 训练脚本：入口点

训练脚本
（[`examples/math/gsm8k_rl.py`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_rl.py)）
遵循以下模式：

```python
def main(args):
    # 1. 加载配置（YAML + CLI 覆盖）
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # 2. 准备数据集（在控制器上加载）
    train_dataset = get_custom_dataset(split="train", dataset_config=config.train_dataset, tokenizer=tokenizer)
    valid_dataset = get_custom_dataset(split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer)

    # 3. 定义工作流配置（在工作进程上导入）
    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
    )

    # 4. 使用 PPOTrainer 进行训练
    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
        )
```

**关键点**：

- 数据集在控制器上加载，然后由控制器分发到工作进程
- 工作流指定为导入字符串，以便在远程工作进程上动态实例化
- `PPOTrainer` 处理所有基础设施（调度器、控制器、工作进程）

请参阅 [CLI 参考](../cli_reference.md)
获取配置选项，以及[自定义数据集](../customization/dataset.md)了解自定义数据集。

## PPOTrainer：基于控制器的训练

[`PPOTrainer`](https://github.com/areal-project/AReaL/blob/main/areal/trainer/rl_trainer.py)
通过初始化调度器并为 actor（策略/评论家）和 rollout 工作进程创建控制器来编排分布式训练。

### 控制器架构

```
PPOTrainer（控制器进程）
    │
    ├── actor: PPOActorController（TrainController）
    │   ├── scheduler.create_workers() → 训练工作进程
    │   ├── 远程引擎：FSDPPPOActor 实例
    │   └── API：compute_logp()、compute_advantages()、ppo_update()
    │
    ├── rollout: RolloutController
    │   ├── scheduler.create_engine() → 推理工作进程（SGLang/vLLM）
    │   ├── BatchTaskDispatcher → 异步工作流执行
    │   └── API：prepare_batch() → 返回批次张量
    │
    └── ref: PPOActorController（可选）
        └── 用于 KL 惩罚的冻结参考模型
```

**关键模式**：引擎使用 `as_controller(config, scheduler)` 将自己包装在控制器中。控制器处理工作进程创建、RPC 分发和结果合并。

## Rollout：生成训练数据

### 工作流规范

在 `examples/math/gsm8k_rl.py` 中，工作流指定为字符串，以便在远程工作进程上动态导入：

```python
trainer.train(
    workflow="areal.workflow.rlvr.RLVRWorkflow",
    workflow_kwargs={
        "reward_fn": "areal.reward.gsm8k.gsm8k_reward_fn",
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
    },
)
```

### RLVRWorkflow：单轮奖励学习

[`RLVRWorkflow`](https://github.com/areal-project/AReaL/blob/main/areal/workflow/rlvr.py)
定义了 prompts 如何转化为训练样本。每个轨迹经历以下步骤：

1. **Tokenize 输入**：将聊天模板应用于消息
1. **生成响应**：调用推理引擎（SGLang/vLLM）
1. **计算奖励**：根据标准答案评估补全
1. **构建训练样本**：构造包含以下内容的张量字典：
   - `input_ids`：完整序列（prompt + completion）
   - `loss_mask`：prompt token 为 0，completion token 为 1
   - `logprobs`：生成时的对数概率
   - `versions`：每个 token 的模型版本（prompt 为 -1）
   - `rewards`：标量奖励

**GSM8K 奖励**：二元奖励（正确答案 1.0，否则 0.0）。请参阅
[`gsm8k_reward_fn`](https://github.com/areal-project/AReaL/blob/main/areal/reward/gsm8k.py)。

**注意**：此工作流采用推理引擎的低级 API —— `agenerate` API。如果您想更细粒度地控制 token IDs，这是更好的选择。`agenerate` 将
token IDs 输入推理服务器，并输出 token IDs 供用户处理。我们还提供了高级 API 用于便捷的 agent 工作流编排。请参阅
[agentic RL 指南](../tutorial/agentic_rl.md)。

### 异步 Rollout 收集

AReaL 中的 Rollout 完全异步，具有三个级别的并发，实现了生成和训练的重叠。

#### 三进程架构

```
控制器进程              工作进程（RPC 服务器）         GPU 进程
──────────────────      ───────────────────────────    ───────────
RolloutController       Flask HTTP 服务器（CPU）        SGLang/vLLM
    │                          │                          │
    └─> BatchTaskDispatcher /call 端点                  推理
       （后台线程）               │                          引擎
            │                   └─> 引擎线程                │
            ├─ 提交任务 1           └─> RemoteInfEngine    │
            │  （HTTP POST）            └─> submit() ──────>│
            │                                              生成
            ├─ 提交任务 2                                   token
            │  （HTTP POST）
            │
            ├─ 提交任务 3              HTTP 回调  <──────────┘
            │                        （轨迹）
            │              ┌─────────────┘
            └─ 收集  <──────┘

与此同时（在不同的 GPU 上）...
TrainController           训练工作进程
    │                          │
    └─> ppo_update(batch) ────> 前向/反向

关键：生成和训练在不同 GPU 上同时进行
```

#### 三个并发级别

**级别 1 - 控制器线程**：
[`BatchTaskDispatcher`](https://github.com/areal-project/AReaL/blob/main/areal/core/workflow_executor.py)
在后台线程中运行，通过 HTTP 持续向工作进程提交 rollout 请求：

- 轮流向 rollout 工作进程提交任务
- 维护 2 个或更多批次inflight 请求以隐藏延迟
- 非阻塞：立即返回 task_id

因此，**在 AReaL 中 rollout 和训练同时进行**，尽管代码看起来像是同步编排。

**级别 2 - 工作进程 RPC 服务器**：每个 rollout 工作进程在 **CPU** 上运行 Flask HTTP 服务器
（[`rpc_server.py`](https://github.com/areal-project/AReaL/blob/main/areal/infra/rpc/rpc_server.py)）：

- 接受并发 HTTP 请求（多线程 Flask）
- **引擎线程**：串行处理引擎操作（NCCL 兼容性）
- 将请求路由到 `RemoteInfEngine`，后者将工作排队到 SGLang/vLLM

**级别 3 - GPU 子进程**：SGLang/vLLM 作为 **独立子进程在 GPU** 上运行：

- 通过 `backend.launch_server()` 启动（与 RPC 服务器分开）
- 维护自己的请求队列
- 通过连续批处理处理多个并发生成
- 轨迹完成时发送 HTTP 回调

#### 请求流程

```python
# 1. 控制器调用 prepare_batch
batch = rollout.prepare_batch(
    dataloader,
    workflow="areal.workflow.rlvr.RLVRWorkflow",
    workflow_kwargs=workflow_kwargs,
)

# 2. RolloutController 委托给 BatchTaskDispatcher
# 后台线程提交任务：
for data in dataloader:
    task = _RemoteRolloutTaskInput(data, workflow, workflow_kwargs, task_id)
    dispatcher.submit_task_input(task)  # 非阻塞 HTTP POST

# 3. 工作进程 RPC 服务器接收 HTTP POST /call (method="submit")
# 引擎线程执行：
workflow_instance = import_from_string(workflow)(**workflow_kwargs)
task_id = workflow_executor.submit(data, workflow_instance)
# 立即返回（非阻塞）

# 4. WorkflowExecutor（在工作进程上）在后台运行：
result = await workflow_instance.arun_episode(engine, data)
# 发送 HTTP 回调给控制器，包含轨迹

# 5. 控制器收集结果
# BatchTaskDispatcher 等待 batch_size 个已接受的轨迹
results = dispatcher.wait_results(batch_size)
return concat_padded_tensors(results)  # 形状：[batch_size, seq_len]
```

**过期管理**：
[`StalenessManager`](https://github.com/areal-project/AReaL/blob/main/areal/infra/staleness_manager.py)
限制并发 inflight 请求：

- `max_concurrent_rollouts`：最大 inflight 轨迹数
- `max_head_offpolicyness`：拒绝使用太旧权重生成的样本
- 版本跟踪：每个 token 标记生成时使用的模型版本

## 训练：控制器-工作进程模式

训练遵循标准的控制器-工作进程模式。控制器通过 RPC 将算法操作分发到训练工作进程，工作进程处理其数据分片，然后结果被合并回来。

### TrainController：分发机制

[`TrainController`](https://github.com/areal-project/AReaL/blob/main/areal/infra/controller/train_controller.py)
提供核心 RPC 分发：

1. `_dispatch_inputs()`：使用 FFD 负载平衡跨工作进程分割批次
1. RPC 调用：每个工作进程接收其分片，处理后返回结果
1. `_merge_results()`：从数据并行工作进程重构结果

**使用 RTensor 的数据流**：

```
控制器                  Worker 0                  Worker 1
    │                         │                         │
    ├─ RTensor（元数据） ──────┼─────────────────────────┤
    │  • 分片 0,1,2,3         │                         │
    │                         │                         │
    ├─ dispatch() ──────────> │                         │
    │  • Worker 0: 分片 0,1   │                         │
    │  • Worker 1: 分片 2,3   │                         │
    │                         │                         │
    │                         ├─> 获取分片 0,1           │
    │                         │   从 rollout 工作进程   │
    │                         │                         ├─> 获取分片 2,3
    │                         │                         │   从 rollout 工作进程
    │                         │                         │
    │                         ├─> compute_logp()       ├─> compute_logp()
    │                         │                         │
    │                         ├─> RTensor（结果）       ├─> RTensor（结果）
    │<─ merge() ──────────────┴─────────────────────────┘
    │  • 重构排序
    │  • 返回统一的 RTensor
    └─> batch["logp"] = result
```

**关键设计**：控制器仅处理元数据（RTensor）。工作进程直接从 rollout 工作进程获取实际张量，避免控制器内存开销。

### 训练工作进程：算法实现

在每个训练工作进程上，
[`FSDPPPOActor`](https://github.com/areal-project/AReaL/blob/main/areal/trainer/ppo/actor.py)
实现了 GRPO/PPO 算法：

**算法方法**：

- `compute_logp(batch)`：通过模型前向传播计算对数概率
- `compute_advantages(batch)`：应用奖励/优势归一化（组或批次级别）
- `ppo_update(batch)`：使用小批量训练和梯度累积进行策略更新
  - 将批次分割成小批量
  - 计算 PPO 损失（裁剪的替代目标 + 可选的 KL 惩罚）
  - 执行反向传播和优化器步骤

**并行性**：各引擎的 `backend` 字段决定 GPU 分配：

```
rollout.backend=sglang:d4 actor.backend=fsdp:d4, n_gpus=8

Rollout 工作进程：      训练工作进程：
GPU 0: SGLang         GPU 4: FSDP rank 0  ─┐
GPU 1: SGLang         GPU 5: FSDP rank 1   ├─ 数据并行
GPU 2: SGLang         GPU 6: FSDP rank 2   │  (DP 大小 = 4)
GPU 3: SGLang         GPU 7: FSDP rank 3  ─┘
                           │
                     NCCL AllReduce 用于梯度
```

每个工作进程处理其分片，然后通过 NCCL 同步梯度。

### 训练循环

`trainer.train()` 方法编排完整循环。请参阅
[`PPOTrainer.train()`](https://github.com/areal-project/AReaL/blob/main/areal/trainer/rl_trainer.py)
获取完整实现：

```python
for global_step in range(start_step, max_steps):
    # 1. Rollout
    rollout_batch = self.actor.prepare_batch(train_dataloader, workflow, workflow_kwargs)

    # 2. 计算对数概率和优势
    if config.actor.should_compute_prox_logp():
        rollout_batch["prox_logp"] = self.actor.compute_logp(rollout_batch)
    if self.ref:
        rollout_frame["ref_logp"] = self.ref.compute_logp(rollout_batch)
    adv_batch = self.actor.compute_advantages(rollout_batch)

    # 3. PPO 更新
    self.actor.ppo_update(adv_batch)
    self.actor.step_lr_scheduler()

    # 4. 权重同步
    self.rollout.pause()
    self.actor.update_weights(weight_update_meta)
    self.actor.set_version(global_step + 1)
    self.rollout.set_version(global_step + 1)
    self.rollout.resume()
```

所有算法操作都是控制器方法调用，分发到远程工作进程。

## 权重同步

每个训练步骤后，更新后的权重必须同步到推理工作进程。AReaL 支持两种传输方式：

### 传输方式

**基于 NCCL 的传输**（推荐）：

- 基于 NCCL broadcast 的直接 GPU 到 GPU 通信
- 更快但使用更多 GPU 内存
- 需要在同一通信后端上非重叠的训练和推理 GPU

**基于磁盘的传输**：

- 保存到共享存储，然后在推理服务器上加载
- 当 NCCL 不可用或机器不共享后端时使用

### 权重更新过程

`PPOTrainer.train()` 中的权重同步过程遵循此模式：

1. 暂停 rollout 服务器以中断所有 inflight 生成并返回到 rollout 客户端（例如 `RemoteSGLangEngine`）
1. 通过配置的方式（NCCL 或磁盘）传输权重
1. 更新版本跟踪以进行过期管理
1. 使用重新计算的 KV 缓存恢复 rollout

请参阅
[`PPOTrainer.train()`](https://github.com/areal-project/AReaL/blob/main/areal/trainer/rl_trainer.py)
第 861-874 行获取完整实现。

## 监控和工具

AReaL 提供由 `PPOTrainer` 管理的工具，用于检查点保存、评估和指标跟踪。这些在训练期间自动编排。

### 检查点保存

AReaL 提供两种检查点机制：

| 组件                                                                                        | 用途              | 格式        | 配置             |
| ------------------------------------------------------------------------------------------- | ----------------- | ----------- | ---------------- |
| [`Saver`](https://github.com/areal-project/AReaL/blob/main/areal/utils/saver.py)            | 导出用于评估/部署 | HuggingFace | `config.saver`   |
| [`RecoverHandler`](https://github.com/areal-project/AReaL/blob/main/areal/utils/recover.py) | 故障后恢复        | DCP（分片） | `config.recover` |

**Saver** 创建与 HuggingFace 兼容的检查点，可以使用 `transformers` 加载或发布到 HuggingFace Hub。每次保存创建一个新目录。

**RecoverHandler** 保存完整训练状态（模型、优化器、数据加载器、RNG）以实现容错。检查点是后端特定的，需要相同的并行配置才能加载。每次保存覆盖之前的检查点。

两者都在 `trainer.train()` 中自动调用。详情请参阅[检查点保存参考](../reference/checkpointing.md)。

### 评估

[`Evaluator`](https://github.com/areal-project/AReaL/blob/main/areal/utils/evaluator.py)
在验证集上运行定期评估。通过 `config.evaluation` 配置。在 `trainer.train()` 中自动调用。

### 指标跟踪

AReaL 使用两组件指标系统：

**`stats_tracker`**
（[源码](https://github.com/areal-project/AReaL/blob/main/areal/utils/stats_tracker.py)）：
以两种针对不同用例优化的范式收集统计信息：

- **流式指标**用于 rollout 工作进程：每个工作流单独记录标量（例如 `reward`），由控制器跨工作进程聚合
- **批次指标**用于训练：带布尔掩码的张量统计按批次记录，然后跨数据并行等级 all-reduced

```python
# Rollout 指标（流式）- 在工作流中
stats_tracker.get("rollout").scalar(reward=0.8, num_turns=3)

# 训练指标（批次）- 在 PPO actor 中
stats_tracker.denominator(n_valid_tokens=loss_mask.bool())
stats_tracker.stat(advantages=tensor, denominator="n_valid_tokens")
```

**`StatsLogger`**
（[源码](https://github.com/areal-project/AReaL/blob/main/areal/utils/stats_logger.py)）：
将聚合指标从 rank 0 发送到日志后端（Weights & Biases、SwanLab、TensorBoard）。在每个训练步骤中，`PPOTrainer`
从所有组件收集指标并提交：

```python
# areal/trainer/rl_trainer.py
stats = self.actor.export_stats()         # 训练指标
stats.update(self.rollout.export_stats()) # Rollout 指标
self.stats_logger.commit(epoch, step, global_step, stats)  # → wandb/tensorboard
```

完整 API 参考请参阅[指标跟踪参考](../reference/metrics_tracking.md)。

## 下一步

现在您已了解基础知识，请探索以下高级主题：

**教程**：

- [评估](../tutorial/eval.md) - 评估您训练的模型
- [训练大型 MoE 模型](../tutorial/megatron.md) - 通过 Megatron 集成扩展到大规模模型
- [Agentic RL](../tutorial/agentic_rl.md) - 构建使用工具和任何 agent 框架的 agents

**自定义指南**：

- [自定义数据集](../customization/dataset.md) - 使用您自己的数据源
- [自定义工作流](../customization/agent.md) - 使用自定义奖励函数构建 agentic/RLVR 工作流
