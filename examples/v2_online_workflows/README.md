# v2 online workflows

两个示例通过外部 client 向 Agent Service 发送 GSM8K／DAPO 任务，评分后提交 reward。 AReaL 的原生 Trainer
负责轨迹入批、训练和权重发布。

```text
examples/v2_online_workflows/
├── README.md
├── train.py
├── client.py
├── workload.py
├── types.py
├── artifacts.py
├── harnesses/
│   ├── __init__.py
│   └── hermes.py
├── benchmarks/
│   ├── __init__.py
│   ├── gsm8k.py
│   └── dapo.py
└── configs/
    ├── hermes_gsm8k_ppo.yaml
    └── hermes_dapo_ppo.yaml
```

## 职责

- `train.py` 复用 `examples.hermes.train` 的原生训练入口。
- `client.py` 设置 Task 并发，创建 session、调用 harness、评分、提交 reward 并保存结果。
- `harnesses/hermes.py` 调用官方 `examples.hermes.hermes.HermesAgent` 所在的 Agent
  Service，转换任务和响应。
- `types.py` 定义 `Harness` 协议和 benchmark 数据结构。其他 harness 实现 `run`、`close`，通过 client 的
  `--harness` 指定类路径。
- `benchmarks/` 加载冻结题目、构造公开题面并评分。标签只在客户端评分时使用。
- `artifacts.py` 保存私有 JSON 文件。
- `workload.py` 预留 trace 解析与回放位置，具体格式待确定。目前 client 立即提供题目，由并发空位控制发出顺序。

## 启动

准备本次运行的模型、数据、采样参数、训练结束条件和 YAML 中要求的环境变量。 Agent Service 的解释器需要能导入 `areal` 和
`hermes-agent` 提供的 `run_agent`。 Hermes 的工具集、轮数和 SDK 参数按本次运行明确配置。官方实现默认
`HERMES_MAX_TURNS=10`， 该值属于待核对的运行设置。两份训练模板要求显式提供输出长度、上下文、epoch 和更新数。

```bash
python -m examples.v2_online_workflows.train --config "$TRAIN_CONFIG"

areal agent run \
  --service "$AGENT_SERVICE" \
  --agent examples.hermes.hermes.HermesAgent \
  --num-pairs "$AGENT_PAIRS" \
  --admin-api-key "$AREAL_AGENT_ADMIN_KEY"

python -m examples.v2_online_workflows.client \
  --benchmark "$BENCHMARK" --tasks "$TASKS" --output "$CLIENT_OUTPUT" \
  --agent-url "$AGENT_GATEWAY" --inference-url "$INFERENCE_GATEWAY" \
  --model "$MODEL" --concurrency "$TASK_CONCURRENCY"
```

`BENCHMARK` 为 `gsm8k` 或 `dapo`。两个 gateway 地址使用 origin，认证从 `AREAL_AGENT_ADMIN_KEY` 和
`AREAL_INFERENCE_ADMIN_KEY` 环境变量读取。 Task 并发、Agent Service worker 容量和 rollout
模型请求容量需要分别配置。新 session 创建受原生 rollout 额度控制，额度耗尽时 client 等待后继续申请。

客户端 HTTP 超时由可选的 `--request-timeout` 显式设置。原生服务内部仍有各自的连接池和超时配置。
每次提交使用新的输出目录，`requests.json` 在发送前记录请求身份，`episodes.jsonl` 保存结果及错误阶段， `summary.json`
汇总完成、正确、执行失败和基础设施错误。客户端对失败请求保留记录，运行中的其他题继续执行。 请求恢复及 workload trace 回放待后续接入。

## 当前验证范围

已在本地完成 GSM8K 和 DAPO 训练验收，覆盖原生 PPO 更新、rollout 版本发布、HF 权重保存和退出清理。 执行失败单独记录，通过失败回执归还额度。
