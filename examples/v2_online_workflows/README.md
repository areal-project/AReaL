# v2 online 工作流

`train.py` 启动原生 `PPOTrainer`；外部应用执行 Agent 请求，`client.py` 负责 benchmark 输入、评分及
session/reward 协议。原生 AReaL 负责排队、选批和版本控制。

常驻 Hermes 服务、worker、执行器和 episode 运行逻辑位于 `harnesses/`， 工具在 `environments/hermes.py`
提供的任务隔离环境中执行。

## 训练与外部请求

先准备独立运行目录、已选定的训练配置、模型、认证与任务文件。在 AReaL 仓库根目录运行：

```bash
python -m examples.v2_online_workflows.train \
  --config "$TRAIN_CONFIG" --control-dir "$RUN_ROOT/control"

python -m examples.v2_online_workflows.client \
  --benchmark "$BENCHMARK" --tasks "$TASKS" --output "$CLIENT_OUTPUT" \
  --inference-url "$AREAL_INFERENCE_URL" --model "$MODEL" \
  --mode resident --concurrency "$TASK_CONCURRENCY" \
  --executor-factory "$EXECUTOR_FACTORY" --executor-config "$EXECUTOR_CONFIG"
```

外部执行器实现 `types.TaskExecutor`；客户端保留实际请求身份，不负责重建训练张量。 需要循环供题时显式指定
`--continuous`。输出长度、Agent 轮数、训练结束条件与请求超时应依据本次实验明确配置， 不要直接把某个旧 YAML 或适配层默认值当成新实验授权。

`--mode eval` 不创建训练 session 或提交训练 reward；评测端点及模型版本须独立核对，不能仅凭客户端模式 推断后端没有缓存。训练模式需要配置
`AREAL_INFERENCE_ADMIN_KEY`，推理地址使用不含 `/v1` 的 origin。

## 目录职责

- `benchmarks/dapo.py`、`benchmarks/gsm8k.py`：冻结输入、准备公开题面及独立评分。
- `configs/hermes_dapo_ppo.yaml`、`configs/hermes_gsm8k_ppo.yaml`：两套 Hermes 在线 PPO 模板。
- `client.py`、`types.py`：外部请求、滚动供题、session/reward 协议与执行器接口。
- `train.py`：原生 Trainer 入口、进度回执与保存回调。
- `harnesses/service.py`、`worker.py`：常驻 CPU 服务及每进程串行的 Hermes worker。
- `harnesses/executor.py`：请求身份持久化、调用服务和直接读取必需产物。
- `harnesses/episode.py`、`hermes_runtime.py`：完整解题、模型采样透传和 episode 身份绑定。
- `harnesses/tools.py`、`workspace.py`：原生 terminal/file 工具绑定与工作目录。
- `environments/hermes.py`：使用现有 SIF 的隔离工具执行环境。
- `artifacts.py`、`outcome.py`、`trace.py`：原子回执、执行结果检查和调用轨迹。

Hermes 运行层复用 `hermes-agent==0.19.0`，训练使用 v2 online、无 Critic PPO、单题单采样和 `individual`
导出。每个 worker 复用 Python 运行环境，每题创建独立 AIAgent 和工具工作区。 输出长度、上下文、Agent 轮数和训练结束条件由本次运行显式配置。

旧 CC、Boba/SWE 适配、旧沙箱与 Slurm 包装位于工作区 `local/`。本地 benchmark 入口为
`local/scripts/benchmark_client.py`，仓库中的 benchmark 查找保留 `dapo_math` 作为 DAPO 别名。
