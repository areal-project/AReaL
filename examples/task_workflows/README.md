# DAPO 与 SWE：可组合的 v2 任务工作流

同一 `client.py` 执行两类 benchmark，服务入口选择 Hermes `TaskAgent` 或 CC `CCTaskAgent`；环境与评分不依赖
harness。训练沿用现有 online PPO，DAPO 在这里指数学数据集。

四种组合在8×H20／8×H200的评测、常驻训练与trace要求见[2×2操作文档](RUN_2X2_8GPU.md)。原八卡 Hermes
操作保留在[推理评测](EVAL_8H20.md)与[1.7B 常驻训练](TRAIN_8H20.md)。正式5轮配方为
[configs/dapo_17b_resident.yaml](configs/dapo_17b_resident.yaml)，保留原参数与实验身份，原长序列 OOM
仍待解决。手动交互和 session 管理脚本直接使用[原生 Hermes 示例](../hermes/README.md)；复用原通用实验命名时，添加覆盖参数
`experiment_name=hermes-workflows`。

执行方式并列保留：默认按[本地Docker](RUN_2X2_8GPU.md)运行；公司集群可选择[Slurm＋Singularity](RUN_SLURM_SINGULARITY.md)，通过宿主侧工具服务执行任务与评分，训练镜像内部不需要容器命令。两者复用同一任务／harness／评分契约。

| 模块                               | 职责                                                    |
| ---------------------------------- | ------------------------------------------------------- |
| `benchmarks/dapo_math.py`          | 原 DAPO 清单、prompt、最终答案评分                      |
| `benchmarks/swe_rebench.py`        | SWE-rebench V2 清单、公开任务输入、上游评分适配         |
| `environments/python.py`           | 每题隔离 Python 工具，保留原资源与执行语义              |
| `environments/repository.py`       | 每题独立仓库、终端／文件操作、完整 patch 导出           |
| `environments/singularity.py`      | 可选SIF执行后端的socket客户端                           |
| `environments/singularity_host.py` | Python3.6.8宿主服务，逐题SIF／overlay与工具计时         |
| `harnesses/hermes.py`              | Hermes 原生循环、推理身份绑定、逐调用计时与环境工具注册 |
| `harnesses/cc.py`                  | Claude Code SDK 循环、每题推理转发与共享环境 MCP 绑定   |
| `client.py` / `service.py`         | session、环境生命周期、并发、评分与原生奖励交接         |

## 准备与运行

在仓库根目录运行。Hermes 使用 `hermes-agent==0.19.0`；CC 使用项目已声明并锁定的 `claude-agent-sdk==0.1.71`，默认运行
SDK 自带 CLI。SWE 还要求 Docker 任务镜像和独立的上游评测 checkout。所有模型请求经过指定的 v2 推理网关。

```bash
export AREAL_TASK_TOOL_ROOT=/absolute/path/to/task-tools
export AREAL_TASK_PYTHON_IMAGE=python:3.12-slim
export AREAL_TASK_MAX_TOKENS=4096
unset HERMES_MAX_TURNS  # 使用 Hermes 原生默认，不增加额外轮数限制

# DAPO：可以直接使用已有的固定 manifest／JSONL，不重新选题。
python -m examples.task_workflows.benchmarks.dapo_math \
  --output /absolute/path/to/dapo-tasks

# SWE：固定 HF revision，按 seed／语言选题并保存内容摘要；默认 16 题。
python -m examples.task_workflows.benchmarks.swe_rebench prepare \
  --output /absolute/path/to/swe-tasks --language python --count 16 --seed 1
python -m examples.task_workflows.benchmarks.swe_rebench pull-images \
  --tasks /absolute/path/to/swe-tasks/tasks.jsonl

git clone https://github.com/SWE-rebench/SWE-rebench-V2 /absolute/path/to/swe-evaluator
git -C /absolute/path/to/swe-evaluator checkout c71902a8cf8d2b725f63d51f199f4d3e56f68d2d
export AREAL_SWE_EVALUATOR_ROOT=/absolute/path/to/swe-evaluator

CUDA_VISIBLE_DEVICES="" areal agent run --service task-workflows \
  --agent examples.task_workflows.service.TaskAgent \
  --num-pairs 1 --admin-api-key "$AREAL_AGENT_ADMIN_KEY"
```

镜像须预先拉取，执行期间不自动下载。Worker 的 Docker 可访问所选镜像；评分客户端也须具有同一镜像 digest。每次运行记录实际 image
ID。跨运行复用时应将选定任务的 `image_name` 冻结为不可变的 registry digest 后生成 manifest。

数学 Python 工具可先用 `python -m examples.task_workflows.environments.python` 预检：不请求模型，只检查
Docker 输出42。默认单次15秒／8KiB输出、每题16次工具调用；每次新解释器，题内文件保留，结束清理容器。
数学工具在Docker／Singularity下均分别返回`stdout`和`stderr`，兼容字段`output`等于`stdout`；运行时与程序诊断保留在`stderr`，两路输出合计沿用原大小限制，工具日志亦保存两路内容。

将下例的 `--benchmark` 与任务文件分别换为 `dapo_math`／已有 DAPO JSONL，或 `swe_rebench`／SWE JSONL：

```bash
python -m examples.task_workflows.client \
  --benchmark swe_rebench --tasks /absolute/path/to/swe-tasks/tasks.jsonl \
  --output /absolute/path/to/results/swe-smoke \
  --agent-url "$AREAL_AGENT_URL" --inference-url "$AREAL_INFERENCE_URL" \
  --model "$AREAL_MODEL_PATH" --mode eval --concurrency 4
```

客户端沿用 `AREAL_AGENT_ADMIN_KEY`、`AREAL_INFERENCE_ADMIN_KEY`。`eval` 不创建训练
session；`resident` 与 `on_demand` 使用原生奖励交接，后者还须指定 `--state-root`。训练入口为
`python -m examples.task_workflows.train --config examples/task_workflows/configs/dapo.yaml`
或 `swe.yaml`；设置 `AREAL_TASK_RUN_ROOT`、`AREAL_TASK_STATE_ROOT` 及原有模型／认证／checkpoint
变量，默认常驻，不在适配验收中自动启动。

`configs/dapo.yaml`／`swe.yaml` 是 batch4／4次更新的起始配方；1.7B正式配方是
batch40／5轮／125次更新。常驻模式核对原生奖励回执与 Trainer 更新数；按需模式须在 Trainer 主机运行客户端，并让 `--state-root`
对应同一 `AREAL_TASK_STATE_ROOT`，核对 inbox 摘要与来源 ACK。两种模式都不能仅凭客户端完成判断训练成功。

## 使用 Claude Code

保持上述任务、镜像、客户端和训练配置，选择 CC 服务入口：

```bash
CUDA_VISIBLE_DEVICES="" areal agent run --service task-workflows-cc \
  --agent examples.task_workflows.service.CCTaskAgent \
  --num-pairs 1 --admin-api-key "$AREAL_AGENT_ADMIN_KEY"

export AREAL_AGENT_URL="$(areal agent status --service task-workflows-cc --json \
  | python -c 'import json,sys; print(json.load(sys.stdin)["gateway_url"])')"
```

`AREAL_INFERENCE_URL` 须指向提供 `/v1/messages` 的 v2 Inference Gateway，填写 origin、末尾不加
`/v1`。模型由客户端 `--model` 指定；每题独立的 loopback relay 转发到该网关，绑定 session key／episode 身份。逐次推理计时从
relay 收到请求至响应转发结束，包含上游排队与网络，不包含 CLI 到 loopback 的开销。它覆盖经过该入口的模型请求，不修改 Worker 全局环境。

CC 保留 SDK 原生 agent 循环和系统提示，通过 MCP 使用相同的
`math_python`／`workspace_exec`。内置宿主文件／终端工具不启用，所有解题执行均进入对应任务环境；个人配置、插件和其他 MCP
服务不加载。没有默认附加轮数上限；可选 `AREAL_CC_MAX_TURNS`，CLI 可由 `AREAL_CC_CLI_PATH` 指定。

`metadata.transcript` 返回 SDK 可见的 assistant、tool
use／result、系统和最终结果消息；`metadata.cc_result` 保存 SDK 结束状态与 usage，`metadata.inference_calls`
保存逐请求计时。Worker 同时保存 `<tool-root>/<runtime-task-id>/cc/transcript.json`。训练
token／logprob／权重版本继续由 v2 原生推理路径采集；闭源 CLI 未暴露的内部状态不在这些记录中。

## 评分与记录

DAPO 保留原固定题目顺序、prompt、评分及 `math_python` 行为。命令与类路径统一使用 `examples.task_workflows`，工具记录位于
`metadata.environment`。

SWE agent 只收到 issue 与公开仓库初始化信息；参考 patch、测试 patch、测试列表和评测命令留在评分端。候选 patch
包括新增文件，在独立容器应用后由[固定版本的上游评测](https://github.com/SWE-rebench/SWE-rebench-V2)执行测试、解析日志并判断
`passed_match`。空候选 patch 为普通失败，绝不回退到参考 patch。上游代码有本地修改或版本不符时拒绝评分。

可在正式模型评测前用 `swe_rebench grade-reference --tasks ... --output ...` 验证参考
patch；这条命令只验收环境，不参与模型轨迹和训练。仓库／评分容器均不挂载宿主目录或凭据；私有测试仅复制进独立评分容器。资源默认每题 2 CPU／4 GiB，可用
`AREAL_TASK_REPO_CPUS`、`AREAL_TASK_REPO_MEMORY_MB`、`AREAL_TASK_REPO_TIMEOUT`、`AREAL_SWE_GRADING_TIMEOUT`
调整。

输出继续为
`run.json`、`tasks-manifest.json`、`episodes.jsonl`、`summary.json`。`metadata.environment`
保存工具与产物，`metadata.inference_calls` 保存每次推理墙钟起止；`timing` 汇总
episode／推理／工具耗时。工具并发时累计时长可能重叠，缺失测量保持 null。token／logprob／权重版本仍由 v2 原生推理链路记录，Rollout
prefill／decode 使用已有 batch 采集器。

每题新增 `task_index`、`available_at`（本次清单就绪时间，早于客户端并发等待）及 `metadata.simulation_trace`。其中
`nodes` 以 `name` 为题内调用ID，保存模型请求的实际 `input_tokens/output_tokens`、工具的 `pool/seconds` 和共同的
`predecessors`；并行工具共享模型父节点，下一模型调用等待工具汇合。Hermes／CC均从响应usage读取长度，不估算字符数；缺失保持null，检查
`lengths_complete`。仅支持原生串行模型循环＋工具fan-out，观测到模型／工具未完成就进入下一模型调用时，
`dependencies_complete=false`，不能当完整DAG使用。缓存命中未报告时也保持null；仅在确认禁用缓存后才能按Expert的none模式取0。

工具节点的 `queued` 是调用入口，`started` 是取得环境锁后的执行开始，`seconds`
排除该锁等待、包含本次所需容器初始化；每题工具池容量1只表示环境锁，不表示全机CPU容量。开启现有 `AREAL_ROLLOUT_TRACE_DIR` 后，v2轨迹导出端另写
`exported-workloads-<host>-<pid>.jsonl`： 以session／incarnation／trajectory关联，保存
`interaction_ids` 和可供Expert `WorkloadStats` 使用的真实序列长度、attention／loss token。重复导出按
`export_id` 去重；这是成功导出记录，实际Actor消费仍须对照Trainer。以上都不采逐token时间、不修改Expert模型。

## 离线验证

统一目录后 21 项正常路径检查通过，无跳过，其中 CC 3
项；[整合记录](../../../artifacts/task-workflows-consolidation-20261023/offline-validation.json)包含配置一致性与目录移除检查。[CC 验证记录](../../../artifacts/cc-task-workflows-20261023/offline-validation.json)列明真实
SDK／CLI 的验证范围。

```bash
AREAL_SWE_EVALUATOR_ROOT=/absolute/path/to/swe-evaluator \
  python -m pytest -q tests/test_cc_task_workflows.py tests/test_task_workflows.py \
  tests/test_hermes_reward_signal.py tests/test_math_verify_reward.py \
  -k 'task_workflow or hermes_math'
```

测试真实执行 OpenAI SDK、Claude Agent SDK `0.1.71`／自带 CLI `2.1.123`、MCP 工具、v2 Anthropic 请求转换、临时
Git 仓库、patch、pytest 和上游解析评分。Hermes agent、模型响应与 Docker 传输使用替身；真实 Docker／Hermes／模型／GPU
训练待实测。SDK 可见 transcript 与逐调用时间均有记录；未强制触发 CC 压缩或子 agent。

原生Actor／Rollout的三组重配置验收及切换计时见[八卡验收](RECONFIGURATION_ACCEPTANCE.md)，提供不启动服务的`--prepare-only`模式；真实GPU验收待运行。
