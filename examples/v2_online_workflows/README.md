# v2 online 工作流

在 AReaL 仓库根目录运行。`client.py` 选择 benchmark，服务端选择 Hermes `TaskAgent` 或 CC
`CCTaskAgent`；训练入口为原生 `PPOTrainer`。

如将 Hermes／CC 部署到 Arena，按 envarena skill 使用官方 CLI 管理任务。 Arena
专用的沙箱入口、工具适配、评分入口、打包及控制脚本均保留在开发工作区 local 中， 正式 example 只包含平台无关的工作流实现。DAPO＋Hermes
已有真实训练记录，其他组合的范围以实验记录为准。

## 准备

Hermes 使用 `hermes-agent==0.19.0`，CC 使用锁定的 `claude-agent-sdk==0.1.71`。工具环境默认使用 Docker；SWE
还需要任务镜像及固定版本的上游评分代码。

```bash
export AREAL_TASK_TOOL_ROOT=/absolute/path/to/task-tools
export AREAL_TASK_PYTHON_IMAGE=python:3.12-slim
export AREAL_TASK_MAX_TOKENS=4096
unset HERMES_MAX_TURNS

# 已有固定 DAPO JSONL 时直接复用。
python -m examples.v2_online_workflows.benchmarks.dapo_math \
  --output /absolute/path/to/dapo-tasks

# SWE：固定任务与镜像。
python -m examples.v2_online_workflows.benchmarks.swe_rebench prepare \
  --output /absolute/path/to/swe-tasks --language python --count 16 --seed 1
python -m examples.v2_online_workflows.benchmarks.swe_rebench pull-images \
  --tasks /absolute/path/to/swe-tasks/tasks.jsonl

git clone https://github.com/SWE-rebench/SWE-rebench-V2 /absolute/path/to/swe-evaluator
git -C /absolute/path/to/swe-evaluator checkout c71902a8cf8d2b725f63d51f199f4d3e56f68d2d
export AREAL_SWE_EVALUATOR_ROOT=/absolute/path/to/swe-evaluator
```

启动 v2 推理网关，设置 `AREAL_INFERENCE_URL`、`AREAL_MODEL_PATH`、`AREAL_INFERENCE_ADMIN_KEY` 和
`AREAL_AGENT_ADMIN_KEY`。推理地址填写 origin，末尾不加 `/v1`。session
管理及推理服务启动见[Hermes 示例](../hermes/README.md)。

## 运行任务

启动 Hermes 服务；CC 使用相同命令，将服务名换为 `v2-online-workflows-cc`、类名换为
`examples.v2_online_workflows.service.CCTaskAgent`。

```bash
CUDA_VISIBLE_DEVICES="" areal agent run --service v2-online-workflows \
  --agent examples.v2_online_workflows.service.TaskAgent \
  --num-pairs 1 --admin-api-key "$AREAL_AGENT_ADMIN_KEY"

export AREAL_AGENT_URL="$(areal agent status --service v2-online-workflows --json \
  | python -c 'import json,sys; print(json.load(sys.stdin)["gateway_url"])')"

python -m examples.v2_online_workflows.client \
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
python -m examples.v2_online_workflows.train \
  --config examples/v2_online_workflows/configs/online.yaml \
  --control-dir "$RUN_ROOT/control"
```

`configs/online.yaml` 为已验收的六卡配置：Megatron TP2＋四个 SGLang TP1
副本，batch32，3次更新，context256K、单次生成8K、lr1e-6。SWE 可参考
`configs/swe.yaml`，按实际资源调整。客户端使用上面的任务命令，将 `--mode eval` 换为
`--mode resident`；该模式提交原生奖励并交接轨迹。通过 `elastic.enabled=true` 启用显式 GPU 池、观测和完整
checkpoint；Megatron 动态布局切换待实现。

## 评分与记录

输出为
`run.json`、`tasks-manifest.json`、`episodes.jsonl`、`summary.json`。`metadata.environment`
保存工具与产物，`metadata.inference_calls` 保存模型调用计时，`metadata.simulation_trace` 保存可导出的长度及调用依赖。

SWE 在独立评分环境应用模型产生的 patch，使用固定上游评分代码。工具工作量、推理墙钟和 episode 总耗时分别记录；逐 token 时间需来自引擎观测。

完整实现、实验与机器操作说明见外层[项目文档](../../../docs/v2-online-workflows-consolidation.md)。

目录职责：`train.py` 为原生训练入口；`client.py` 管理外部请求的 session／reward；`service.py` 接入 AReaL Agent
Service；`harnesses/`、`benchmarks/` 和 `environments/` 分别实现 Agent 执行、任务评分和工具环境。正式入口不依赖工作区
`local/scripts`。

使用平台已有 Harness 做评测，可直接按 envarena skill 调用官方 Arena CLI，无需自有 runner。 通用 Agent 实现见
[Harness 说明](harnesses/README.md)。训练端和外部请求端独立运行， Trainer 不逐批触发任务。配置 GPU 池时使用实际 Slurm
分配；启用 elastic 时协调进程使用 `CUDA_VISIBLE_DEVICES=""`。 `--control-dir` 必须是新的私有目录，写出
`ready.json`、`training-finished.json`、`closed.json`，不含密钥。需要训练后评测时，可增加
`--post-train-timeout 600`，在保留的服务上完成评测后原子写入 `release.json`，内容为完成回执中的
`serving_versions`。等待有界，默认训练完成即关闭服务。

运行目录、机器配置、Slurm 提交参数和认证留在仓库外。Arena 实验配置模板也放在工作区 local
中。恢复请求端使用其原目录；v2不支持精确optimizer恢复，权重热启动必须记录为新实验分段。
