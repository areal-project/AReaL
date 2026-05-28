# 在线 RL 训练

本指南介绍如何使用在线模式训练语言模型。在该模式下，用户首先启动一个 AReaL RL 服务并暴露代理网关，外部应用程序（智能体运行时、人类评估者或任何 OpenAI
兼容客户端）通过此网关与模型交互。每次交互都会自动收集为 RL 训练数据。

**免责声明：** 此 API 为实验性质，可能会发生变更。

## 概述

AReaL 支持三种智能体工作流执行模式：

| 模式         | 描述                               | 适用场景                   |
| ------------ | ---------------------------------- | -------------------------- |
| `inline`     | 智能体在 rollout worker 进程内运行 | 大多数智能体框架           |
| `subproc`    | 智能体在子进程池中运行             | 非异步或需要进程隔离的代码 |
| **`online`** | 外部用户通过 HTTP API 驱动交互     | 人类反馈、外部运行时       |

本指南重点介绍 **online 模式**。该模式的独特之处在于，智能体代码运行在 AReaL _外部_。AReaL 暴露一个 OpenAI 兼容的 HTTP
API，任何支持聊天补全协议的应用程序都可以连接。

离线训练指南请参阅[智能体 RL 指南](./agentic_rl.md)。

## 架构

```
                          外部应用程序
                     (ZeroClaw、脚本等)
                                  |
                      POST /chat/completions
                      POST /rl/set_reward
                                  |
                                  v
                      +-------------------+
                      |   代理网关         |  (FastAPI，无状态路由)
                      |  - 会话管理        |
                      |  - 密钥认证        |
                      |  - 负载均衡        |
                      +-------------------+
                         /        |        \
                        v         v         v
                  +---------+ +---------+ +---------+
                  |  代理   | |  代理    | |  代理   |
                  | Worker  | | Worker  | | Worker  |  (每个 rollout worker 一个)
                  +---------+ +---------+ +---------+
                      |           |           |
                      v           v           v
                  +---------+ +---------+ +---------+
                  | SGLang/ | | SGLang/ | | SGLang/ |
                  | vLLM    | | vLLM    | | vLLM    |  (推理服务器)
                  +---------+ +---------+ +---------+
                                  |
                      自动收集 token 级别数据
                                  |
                                  v
                      +-------------------+
                      |    RL 训练器       |
                      |   (PPOTrainer)    |
                      +-------------------+
```

**核心组件：**

- **代理网关（Proxy Gateway）**：轻量级 FastAPI 服务器，将外部应用程序的请求路由到后端代理 worker。它管理会话生命周期、认证和负载均衡。
- **代理 Worker（Proxy Workers）**：与 rollout worker 共置的后端服务器。每个 worker 管理会话、记录 token
  级别数据（token ID、对数概率），并导出轨迹用于训练。
- **推理服务器**：执行实际 LLM 推理的 SGLang 或 vLLM 服务器。

## 快速开始

### 步骤 1：配置在线模式

在配置 YAML 中将 `rollout.agent.mode` 设置为 `online`：

```yaml
# config.yaml
rollout:
  agent:
    mode: online
    admin_api_key: "my-secret-admin-key"  # 保护管理端点
    session_timeout_seconds: 3600          # 会话超时时间（默认：1 小时）
```

### 步骤 2：启动 RL 服务

```bash
python3 examples/openclaw/train.py --config examples/openclaw/config.yaml \
    experiment_name=my-exp trial_name=trial-0 \
    rollout.backend=sglang:d1 actor.backend=fsdp:d1 \
    actor.path=Qwen/Qwen3-0.6B \
    scheduler.type=local \
    rollout.agent.admin_api_key=my-secret-admin-key
```

初始化完成后，AReaL 会打印网关地址：

```
(AReaL) RLTrainer INFO: Proxy gateway available at http://x.x.x.x:8090
```

### 步骤 3：启动会话

使用提供的辅助脚本或任何 HTTP 客户端：

```bash
curl -X POST http://<gateway>/rl/start_session \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-admin-key" \
  -d '{"task_id": "demo-task-0"}'
```

输出中应包含当前会话 ID 和该智能体会话的 API 密钥。

**为什么每个智能体会话都需要唯一的 API 密钥？** 由于可能有许多并发的智能体应用在运行，且它们调用相同的端点（例如
"/chat/completions"），我们需要一种机制来区分不同智能体的轨迹。因此，我们为每个智能体会话或轨迹分配唯一的 API
密钥，它们之间具有一一对应的关系。这样，我们就能追踪同一轨迹内的交互并设置奖励。

### 步骤 4：与模型交互

使用任何 OpenAI 兼容的客户端。例如，使用 `curl`：

```bash
curl http://<gateway>/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-sess-xxxxxxxxxxxx" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "12 * 15 + 3 等于多少？"}],
    "temperature": 0.7
  }'
```

或使用 OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<gateway>",
    api_key="sk-sess-xxxxxxxxxxxx",
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "12 * 15 + 3 等于多少？"}],
)
print(response.choices[0].message.content)
```

### 步骤 5：分配奖励并结束会话

交互完成后，分配奖励以提供 RL 训练信号：

```bash
curl http://<gateway>/rl/set_reward \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-sess-xxxxxxxxxxxx" \
  -d '{"reward": 1.0}'
```

您也可以在智能体 rollout 期间使用 completion ID 为中间步骤设置奖励。

然后，结束会话：

```bash
curl http://<gateway>/rl/end_session \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-sess-xxxxxxxxxxxx" \
  -d '{}'
```

### 步骤 6：批量采样

将步骤 3 到步骤 5 整合到单个 bash 脚本中，然后使用 `sbatch` 等工具并发运行。**每个智能体会话必须重新调用 `/rl/start_session`
以获取新的 API 密钥。**

当 AReaL 缓冲区中积累了足够的数据后，AReaL 将自动进入训练阶段。

## FAQ

> Q: 更新后的模型何时会被加载用于推理？

模型会在每个训练步骤后加载。换言之，用于推理的模型始终是最新的。 有关模型保存和检查点，请参阅 [CLI 参考](../cli_reference.md)。

> Q: 如何控制智能体脚本的提交速率？RL 服务会过载吗？

AReaL 内置了速率限制，称为**新鲜度控制（staleness control）**。 如果提交的并发请求过多，网关将向客户端返回 429。
有关新鲜度控制的详细信息，请参阅[异步 RL 指南](../algorithms/async.md)。

> Q: 我能用这种方式训练 OpenClaw 吗？

本文档中的方式与训练个性化智能体不同，因为：

- OpenClaw 假设与用户的单线程交互，即用户不能打开多个可能相互干扰的并发会话
- OpenClaw 需要使用固定 URL 和 API 密钥进行一次性设置

核心使用差异在于，OpenClaw 示例在整个交互过程中使用**固定的** API 密钥。 通过多次调用
`start_session`，旧会话会自动结束，其轨迹导出用于训练，然后使用相同的 API 密钥启动新会话。两轮之间无需重新配置您的应用程序。

有关训练 OpenClaw 智能体的详细信息，请参阅 [OpenClaw 示例](../../../examples/openclaw/README.md)。

## 认证机制

在线模式使用两层认证系统：

| 认证类型            | 令牌                          | 用途                                            |
| ------------------- | ----------------------------- | ----------------------------------------------- |
| **管理员 API 密钥** | `rollout.agent.admin_api_key` | `start_session`、`export_trajectories`          |
| **会话 API 密钥**   | 由 `start_session` 签发       | `chat/completions`、`set_reward`、`end_session` |

- **管理员 API 密钥** 在 YAML 中配置，保护管理端点。
- **会话 API 密钥** 每个会话唯一，作用域限定在该会话的交互中。

## API 参考

所有端点由代理网关提供服务。

### 管理端点（管理员认证）

#### `POST /rl/start_session`

启动新会话或刷新现有会话。

**请求体：**

```json
{
  "task_id": "my-task-0",
  "api_key": null
}
```

传入之前会话的 `api_key` 以刷新。省略或设置为 `null` 表示新会话。

**响应：**

```json
{
  "session_id": "my-task-0",
  "api_key": "sk-sess-xxxxxxxxxxxx"
}
```

#### `GET /health`

健康检查。返回后端 worker 数量。

### 会话端点（会话认证）

#### `POST /chat/completions`

OpenAI 兼容的聊天补全端点。Token 和对数概率会自动记录。

#### `POST /responses`

OpenAI Responses API 端点（聊天补全的替代方案）。

#### `POST /v1/messages`

Anthropic Messages API 端点，用于 Claude 兼容客户端。

#### `POST /rl/set_reward`

为某次交互分配奖励。

**请求体：**

```json
{
  "reward": 1.0,
  "interaction_id": null
}
```

如果 `interaction_id` 为 null，奖励将分配给最后一次交互。

#### `POST /rl/end_session`

显式结束会话并导出其轨迹。用于**批量采样**模式（每个样本使用独立 API 密钥）。使用会话刷新时不需要调用此接口。

## 错误处理

| HTTP 状态码 | 含义                     | 处理方式                       |
| ----------- | ------------------------ | ------------------------------ |
| 200         | 成功                     | -                              |
| 401         | 缺少或无效的认证         | 检查您的 API 密钥              |
| 409         | API 密钥已绑定到活跃会话 | 先结束现有会话，或使用刷新机制 |
| 429         | 没有可用容量             | 稍后重试                       |
| 502         | 后端 worker 不可达       | 检查 RL 服务是否正在运行       |

刷新时遇到 HTTP 429，表示训练流水线可能尚未完成一个周期。请在几秒后重试（默认超时为 120 秒）。

## 训练机制

训练在底层**异步**运行：

1. 外部应用程序通过网关与模型交互
1. 每个会话的交互都以 token 级别数据记录
1. 会话结束时（通过刷新或显式结束），其轨迹被导出
1. 收集到足够的轨迹后（由 `train_dataset.batch_size` 控制），AReaL 执行一次训练步骤
1. 更新后的模型权重会透明地提供给后续会话

随着收集更多轮次，模型会静默地改进。有关异步训练和新鲜度控制的详细信息，请参阅 [异步 RL 指南](../algorithms/async.md)。

## 配置参考

所有在线模式设置位于 `rollout.agent` 下：

```yaml
rollout:
  agent:
    mode: online                    # 必填：设置为 "online"
    admin_api_key: "areal-admin-key"  # 管理端点的 API 密钥
    session_timeout_seconds: 3600   # 会话超时时间（秒）
    turn_discount: 1.0              # 多轮对话的奖励折扣
    export_style: individual        # "individual" 或 "concat"
```

| 字段                      | 默认值            | 描述                                |
| ------------------------- | ----------------- | ----------------------------------- |
| `mode`                    | `inline`          | 必须设置为 `online` 以启用外部访问  |
| `admin_api_key`           | `areal-admin-key` | 管理员 API 密钥（生产环境请修改！） |
| `session_timeout_seconds` | `3600`            | 超时后自动清理过期会话              |
| `turn_discount`           | `1.0`             | 多轮奖励的几何折扣因子              |
| `export_style`            | `individual`      | 交互数据的导出方式                  |

## 限制

- **调度器兼容性**：在线模式需要 `local` 或 `slurm` 调度器，不支持 `ray` 调度器。
- **单控制器模式**：在线模式仅在单控制器模式下工作 （`scheduler.type=local` 或 `scheduler.type=slurm`）。

## 另请参阅

- [OpenClaw 示例](https://github.com/areal-project/AReaL/tree/main/examples/openclaw) - 使用
  ZeroClaw 的完整端到端示例
- [智能体 RL 教程](agentic_rl.md) - 智能体框架集成（inline/subproc 模式）
- [自定义智能体工作流](../customization/agent.md) - 创建自定义智能体工作流
- [智能体工作流参考](../reference/agent_workflow.md) - 内部架构详情
