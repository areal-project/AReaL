# AReaL CLI 指南

`areal` CLI 在 AReaL 2.0 微服务架构下提供三个顶层子命令组，每个对应一种服务：

- **`areal train`** — 解析实验 driver 与配置文件，转发 hydra 覆盖参数，将控制权交给训练脚本。
- **`areal inf`** — 启动并管理本机上的推理服务（gateway、router、model worker、data proxy）。
- **`areal agent`** — 启动并管理本机上的 agent 服务（gateway、router、N 对 worker / data-proxy）。

`areal inf` 与 `areal agent` 共用同一套生命周期模型 —— `run`、`ps`、`status`、 `logs`、`stop`，每个服务的状态存放在
`~/.areal/` 下（可通过 `AREAL_HOME` 环境变量覆盖）。`areal train` 有意保持无状态：它只是将 argv 传递给 Python
`main(args)` driver。

## 训练 CLI（`areal train`）

`areal train` 将 AReaL 训练 driver 函数与实验配置文件封装为一条命令。它并不管理训练进程的生命周期（不像 `areal inf`
会维护服务状态），只做「找到 driver、加载配置，并把 hydra 风格的覆盖参数透传下去」这件事。

### 基础概念

训练任务的最小执行单元是 **driver 函数** —— 通常是 `examples/` 下某个脚本中的 `main(args: list[str])`。CLI 只做三件事：

1. 从 `module.path:func` 解析 driver。
1. 将 `--config <path>` 解析为绝对路径并放在 argv 最前面。
1. 将其余尾部参数原封不动追加到 argv（通常是 hydra 覆盖参数）。

driver 函数的返回值会被用作退出码 —— 如果返回 `int` 则直接使用；其他情况（包括 `None`）都视为 0。

### 用法

```bash
areal train run \
  --config <path/to/experiment.yaml> \
  --driver <module.path>:<func> \
  [<hydra-override-1> <hydra-override-2> ...]
```

| flag / arg   | 是否必填 | 说明                                                        |
| ------------ | -------- | ----------------------------------------------------------- |
| `--config`   | 是       | 实验 YAML 路径；文件必须存在（CLI 会执行 `exists` 检查）    |
| `--driver`   | 是       | driver 入口，形如 `module.path:func`（冒号分隔）            |
| 尾部位置参数 | 否       | 原样转发给 driver；通常是 hydra 风格的 `key=value` 覆盖参数 |

`run_cmd` 启用了 `context_settings={"ignore_unknown_options": True}` —— 尾部位置参数 **可以包含
`--xxx` 形式的选项**，CLI 不会去解析它们，原样转发给 driver。

### 示例

运行 GSM8K GRPO（最常见的 baseline）：

```bash
areal train run \
  --config examples/math/gsm8k_grpo.yaml \
  --driver examples.math.gsm8k_rl:main \
  experiment_name=gsm8k_grpo_test \
  trial_name=t1
```

运行 SFT：

```bash
areal train run \
  --config examples/math/gsm8k_sft.yaml \
  --driver examples.math.gsm8k_sft:main
```

### Driver 函数约定

CLI 用单个参数 `argv: list[str]` 调用 driver，因此 driver 长这样：

```python
def main(args: list[str]) -> int | None:
    config, _ = load_expr_config(args, GRPOConfig)   # 或任意其他 *Config dataclass
    ...
    return 0
```

`load_expr_config` 位于 `areal.api.cli_args`，它自己消费 `args`：识别 `--config` 后面的 YAML 路径，将剩余的
`key=value` 作为 hydra 覆盖合并进 config dataclass。也就是说，hydra 解析是 **由 driver 完成**，而不是由 CLI 完成。

编写新 driver 的最小模板：

```python
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal import PPOTrainer

def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    with PPOTrainer(config, train_dataset=..., valid_dataset=...) as trainer:
        trainer.train(workflow="...", workflow_kwargs={...})
    return 0
```

### Hydra 覆盖参数

所有使用 `load_expr_config` 解析参数的 driver 都支持 hydra 风格覆盖。常见覆盖目标：

```bash
# 实验 / trial 命名
experiment_name=my_run trial_name=t1

# 集群规模
cluster.n_nodes=4 cluster.n_gpus_per_node=8

# 训练超参
actor.optimizer.lr=5e-6
total_train_epochs=20

# rollout backend
rollout.backend=sglang:d2p1t2
rollout.max_concurrent_rollouts=128

# 数据集
train_dataset.batch_size=256
```

CLI 不会校验这些 key 是否合法；driver 加载配置时 hydra 会报告未知字段。

### 退出码

| 场景                            | 退出码                           |
| ------------------------------- | -------------------------------- |
| driver 返回 `int`               | 直接使用其返回值                 |
| driver 返回 `None` / 其他       | 0                                |
| `--driver` 不包含 `:`           | UsageError（click 默认 2）       |
| `--driver` 引用的模块无法导入   | ClickException（1）              |
| `--driver` 引用的函数不在模块上 | ClickException（1）              |
| `--config` 路径不存在           | click 的 `exists=True` 捕获（2） |

driver 内部抛出的异常 **CLI 不做捕获** —— 走 Python 默认行为（打印 traceback、退出进程）。

### 尚未实现

`areal train` 目前只实现了 `run`。下列子命令是合理的未来扩展，但当前版本 **未包含**：

- `areal train ps` / `status` / `stop` —— 训练任务生命周期管理（需要先引入训练服务的状态概念）。

## 推理服务 CLI（`areal inf`）

`areal inf` 用于在本机启动并管理 AReaL 推理服务。它会启动 gateway/router、注册模型、查看服务状态、管理日志。

### 基础概念

一个推理服务通常包含以下组件：

- `gateway`：对外提供 OpenAI 兼容 API 与 RL API。
- `router`：维护「模型 → worker / data-proxy」的路由信息。
- `model worker`：真正的推理后端，例如 SGLang。
- `data proxy`：记录交互与奖励，并支持轨迹导出。

CLI 的本地状态默认存放在 `~/.areal/inf/` 下。可以通过 `AREAL_HOME` 覆盖根目录：

```bash
export AREAL_HOME=/path/to/areal-home
```

### 启动服务

启动一个空的推理服务：

```bash
areal inf run \
  --service default \
  --host 127.0.0.1 \
  --port 8080 \
  --admin-api-key areal-admin-key \
  --scheduler local \
  --detach
```

`--scheduler` 用于选择 worker / data-proxy 的调度后端。当前仅支持 `local`（也是默认值）。服务一旦启动，这个值就会固化到服务状态里
—— 后续的 `register` / `stop` / `status` 会从状态中读取，不必再传 `--scheduler`。

列出本机已知的服务：

```bash
areal inf ps
areal inf status --service default
```

`ps` 展示服务列表；`status` 深入到 gateway、router、data-proxy、worker 等各组件的状态。

列出已注册的模型：

```bash
areal inf models --service default
```

### 注册模型

`register` 让 CLI 启动一个本地推理后端并配一个 data-proxy：

```bash
areal inf register \
  --service default \
  --model-name qwen-local \
  --backend sglang:d1 \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-path Qwen/Qwen2.5-7B-Instruct \
  --engine-args "--mem-fraction-static 0.8" \
  --proxy-args "--request-timeout 120 --chat-template-type hf"
```

`--engine-args` 是一个 shell 风格字符串，原样转发给 sglang / vllm 的 worker 进程； `--proxy-args` 是
data-proxy 进程对应的参数。可用的 data-proxy 参数包括
`--request-timeout`、`--set-reward-finish-timeout`、`--tool-call-parser`、
`--reasoning-parser`、`--engine-max-tokens`、`--chat-template-type {hf|concat}`。

也可以在 `run` 时直接注册模型：

```bash
areal inf run \
  --service default \
  --port 8080 \
  --admin-api-key areal-admin-key \
  --model qwen-local \
  --backend sglang:d1 \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --engine-args "--mem-fraction-static 0.8" \
  --proxy-args "--request-timeout 120 --chat-template-type hf" \
  --detach
```

### 普通推理请求

模型注册好后，可以直接调用 gateway 的 OpenAI 兼容接口：

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer areal-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-local",
    "messages": [
      {"role": "user", "content": "Hi, give me a quick intro to AReaL."}
    ],
    "max_tokens": 128
  }'
```

### 日志与清理

查看日志：

```bash
areal inf logs --service default --component gateway -f
areal inf logs --service default --component router -f
areal inf logs --service default --component qwen-local-worker-0 -f
areal inf logs --service default --component qwen-local-data-proxy-0 -f
```

每个模型的 worker / data-proxy 日志文件名为 `<model-name>-worker-<rank>` 与
`<model-name>-data-proxy-<rank>`。如果 `--component` 写错或文件不存在，CLI 会打印可用的名字。

注销模型：

```bash
areal inf deregister --service default --model-name qwen-local
```

停止服务：

```bash
areal inf stop --service default
```

强制停止：

```bash
areal inf stop --service default --force
```

### 配置文件

`areal inf` 会从下面这个默认配置文件读取默认值：

```bash
~/.areal/inf/config.toml
```

也可以另外传入配置文件：

```bash
areal inf --config ./inf.toml run --service default --detach
```

示例：

```toml
[default]
service = "default"

[launch]
gateway_host = "127.0.0.1"
gateway_port = 8080
routing_strategy = "round_robin"

[scheduler]
type = "local"

[register.internal]
backend = "sglang:d1"
model_health_timeout = 600
engine_args = "--mem-fraction-static 0.8"
proxy_args = "--request-timeout 120 --chat-template-type hf"
```

## Agent 服务 CLI（`areal agent`）

`areal agent` 用于在本机启动一组 agent 服务进程（gateway / router + N 对 worker / data-proxy），供上层应用通过
HTTP 与 agent 交互。形态上与 `areal inf` 非常接近，但它服务的是 agent（有会话状态的多轮交互）而非无状态推理。

### 基础概念

运行中的 agent 服务包含以下组件：

- `gateway`：对外暴露会话与 agent API，每个服务一个。
- `router`：根据负载把请求路由到某个 data-proxy，每个服务一个。
- `worker[i]`：真正跑用户 agent 代码的进程。agent 类通过 `--agent` 传入的 `module.path` 形式导入。
- `data-proxy[i]`：位于 `worker[i]` 前面的会话管理 / 记账层。每个 worker 与一个 proxy 组成
  **一对**。`--num-pairs` 控制副本数。

请求流：

```
client → gateway → router → data-proxy[i] → worker[i] → agent 代码
                            └ 会话生命周期 / 记录 ┘
```

CLI 的本地状态默认存放在 `~/.areal/agent/` 下。可以通过 `AREAL_HOME` 覆盖根目录：

```bash
export AREAL_HOME=/path/to/areal-home
```

### 启动服务

最小启动 —— 一对 (worker, proxy)：

```bash
areal agent run \
  --service default \
  --agent my_package.my_agent.MyAgent \
  --num-pairs 1 \
  --admin-api-key areal-agent-admin
```

`--agent` 是必填项，是 worker 进程用来加载 agent 类的导入路径。

清除残留状态并强制启动：

```bash
areal agent run --service default --agent ... --force
```

### 查看服务状态

列出本机所有 agent 服务：

```bash
areal agent ps
areal agent ps --all          # 包含 stale 行
areal agent ps --json
```

输出列：`SERVICE / STATUS / GATEWAY / AGENT`。

查看单个服务里每个组件的健康状况：

```bash
areal agent status --service default
```

输出包含 gateway、router，以及每对 worker + proxy。`--watch` 模式按间隔刷新（默认 2 秒）：

```bash
areal agent status --service default --watch --interval 1
```

JSON 模式方便与 jq 配合：

```bash
areal agent status --service default --json | jq '.pairs[].worker'
```

### 与服务通信

CLI **不负责** 应用如何跟服务交互 —— 应用直接打 gateway 的 HTTP 接口即可。`status` 命令可以告诉你 gateway URL：

```bash
GATEWAY_URL=$(areal agent status --service default --json | jq -r '.gateway.url')
echo "gateway at $GATEWAY_URL"
```

应用带上 `--admin-api-key`（或从 gateway 拿到的 session key）向该 URL 发请求。

### 日志

每个组件都有独立日志文件：

```bash
areal agent logs --service default --component gateway -f
areal agent logs --service default --component router -f
areal agent logs --service default --component worker-0 -f
areal agent logs --service default --component proxy-0 -f
```

命名约定：

- `gateway` / `router`：服务级单例。
- `worker-<i>` / `proxy-<i>`：第 i 对 pair 的 worker / data-proxy（i 从 0 开始）。

如果 `--component` 写错，CLI 会打印可用的名字。`-f` 走 `tail -F` 语义。

### 停止服务

```bash
areal agent stop --service default
```

默认是两阶段关闭：先 SIGTERM，等 `--grace-period`（10 秒），再 SIGKILL。立即 SIGKILL：

```bash
areal agent stop --service default --force
```

`--keep-state` 保留状态文件（杀掉进程，但磁盘上的 `<svc>.json` 保留）：

```bash
areal agent stop --service default --keep-state
```

### 配置文件

`areal agent` 启动时会读取 `~/.areal/agent/config.toml` 作为默认值；也可以另外传入配置文件：

```bash
areal agent --config ./my-agent.toml run --service default --agent ...
```

示例：

```toml
[default]
service = "default"
admin_api_key = "areal-agent-admin"
log_level = "info"

[run]
agent = "my_package.my_agent.MyAgent"
num_pairs = 2
setup_timeout = 120
health_poll_interval = 5
drain_timeout = 30
session_timeout = 1800
```

优先级：**CLI 参数 > 通过 `--config` 传入的 TOML > `~/.areal/agent/config.toml` > 内置 默认值**。

### 尚未实现

当前 `areal agent` **不包含**：

- 会话级 CLI 操作（开启会话 / 设置奖励 / 导出轨迹）—— 这类操作与应用耦合很紧，由应用直接调用 gateway HTTP 处理。
- 自动故障恢复 / 心跳监控 —— `status` 是按需查询，不会持续观察组件健康。worker 死掉后需要用户运行 `status` 或看日志才能发现。
- 分布式调度 —— 只在本机启动本地进程；k8s / slurm 等超出当前 CLI 的范围。
