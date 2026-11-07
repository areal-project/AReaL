# DAPO 与 SWE 任务工作流

在 AReaL 仓库根目录运行。`client.py` 选择 benchmark，服务端选择 Hermes `TaskAgent` 或 CC
`CCTaskAgent`；训练入口为原生 `PPOTrainer`。

Hermes／DAPO 也可通过 [Arena runner](arena/README.md) 执行，无需启动 Agent Service。 Arena 与服务端工作流共用
session／reward 客户端、独立评分和逐调用 trace；真实训练闭环仍待验收。

## 准备

Hermes 使用 `hermes-agent==0.19.0`，CC 使用锁定的 `claude-agent-sdk==0.1.71`。工具环境默认使用 Docker；SWE
还需要任务镜像及固定版本的上游评分代码。

```bash
export AREAL_TASK_TOOL_ROOT=/absolute/path/to/task-tools
export AREAL_TASK_PYTHON_IMAGE=python:3.12-slim
export AREAL_TASK_MAX_TOKENS=4096
unset HERMES_MAX_TURNS

# 已有固定 DAPO JSONL 时直接复用。
python -m examples.task_workflows.benchmarks.dapo_math \
  --output /absolute/path/to/dapo-tasks

# SWE：固定任务与镜像。
python -m examples.task_workflows.benchmarks.swe_rebench prepare \
  --output /absolute/path/to/swe-tasks --language python --count 16 --seed 1
python -m examples.task_workflows.benchmarks.swe_rebench pull-images \
  --tasks /absolute/path/to/swe-tasks/tasks.jsonl

git clone https://github.com/SWE-rebench/SWE-rebench-V2 /absolute/path/to/swe-evaluator
git -C /absolute/path/to/swe-evaluator checkout c71902a8cf8d2b725f63d51f199f4d3e56f68d2d
export AREAL_SWE_EVALUATOR_ROOT=/absolute/path/to/swe-evaluator
```

启动 v2 推理网关，设置 `AREAL_INFERENCE_URL`、`AREAL_MODEL_PATH`、`AREAL_INFERENCE_ADMIN_KEY` 和
`AREAL_AGENT_ADMIN_KEY`。推理地址填写 origin，末尾不加 `/v1`。session
管理及推理服务启动见[Hermes 示例](../hermes/README.md)。

## 运行任务

启动 Hermes 服务；CC 使用相同命令，将服务名换为 `task-workflows-cc`、类名换为
`examples.task_workflows.service.CCTaskAgent`。

```bash
CUDA_VISIBLE_DEVICES="" areal agent run --service task-workflows \
  --agent examples.task_workflows.service.TaskAgent \
  --num-pairs 1 --admin-api-key "$AREAL_AGENT_ADMIN_KEY"

export AREAL_AGENT_URL="$(areal agent status --service task-workflows --json \
  | python -c 'import json,sys; print(json.load(sys.stdin)["gateway_url"])')"

python -m examples.task_workflows.client \
  --benchmark swe_rebench --tasks /absolute/path/to/swe-tasks/tasks.jsonl \
  --output /absolute/path/to/results/swe-smoke \
  --agent-url "$AREAL_AGENT_URL" --inference-url "$AREAL_INFERENCE_URL" \
  --model "$AREAL_MODEL_PATH" --mode eval --concurrency 4
```

DAPO 将 benchmark 换为 `dapo_math`，并使用对应任务 JSONL。SWE 镜像需预先拉取；每次输出记录实际 image ID，跨运行比较使用冻结的
digest。可先运行 `swe_rebench grade-reference --tasks ... --output ...` 验证评分环境。

## 原生训练

配置模型、认证、checkpoint、`AREAL_TASK_RUN_ROOT` 和 `AREAL_TASK_STATE_ROOT` 后，在独立终端启动 Trainer：

```bash
python -m examples.task_workflows.train \
  --config examples/task_workflows/configs/dapo.yaml
```

SWE 使用 `configs/swe.yaml`。当前 Actor／Rollout 为 Megatron TP4／SGLang。客户端使用上面的任务命令，将
`--mode eval` 换为 `--mode resident`；该模式提交原生奖励并交接轨迹。通过 `elastic.enabled=true` 启用显式 GPU
池、观测和完整 checkpoint；Megatron 动态布局切换待实现。

## 评分与记录

输出为
`run.json`、`tasks-manifest.json`、`episodes.jsonl`、`summary.json`。`metadata.environment`
保存工具与产物，`metadata.inference_calls` 保存模型调用计时，`metadata.simulation_trace` 保存可导出的长度及调用依赖。

SWE 在独立评分环境应用模型产生的 patch，使用固定上游评分代码。工具工作量、推理墙钟和 episode 总耗时分别记录；逐 token 时间需来自引擎观测。

完整实现、实验与机器操作说明见外层[项目文档](../../../docs/README.md)。
