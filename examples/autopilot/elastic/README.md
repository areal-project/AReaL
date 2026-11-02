# Elastic training and Online reconfiguration

训练使用原生 `PPOTrainer / OnDemandActor`。`areal/elastic` 提供执行、观测、Reconfigurator 和 CPU
模拟；AutoPilot 提供通用 revision 和 Elastic 领域的评测／激活。

## Reconfiguration and revision loops

```text
任务 → Rollout → episode inbox → 完整 batch → 原生 PPO → 发布／保存
                 ↓                                  ↓
           OnlineObservation ← 已有状态／完成历史 ← 运行反馈
                 ↓
      basic 或 reference Reconfigurator
      预测 → optimizer 搜索 → 复核 → 原生 reconfigure / 模拟交接

已有测量 → 固定历史分区 → 模型 revision → 固定模型 → optimizer revision
         → 接受的完整 bundle → 模拟 / 原生 adapter 加载
```

`basic` 为使用旧 JointSimulator 的 **Basic v1**；`reference` 是当前优化版，也是默认版本。两者都包含 optimizer、模型和
Simulator。原生与模拟共用所选版本，分别提供观测和执行能力。

## 入口

- `simulate_online.py`：读取 workload、配置和冻结组件，运行 fixed／dynamic CPU
  条件模拟；`--reconfigurator basic|reference` 显式选择版本。
- `online-v1-example.json`：v1 基线配置；`online-v3-example.json`：当前示例配置。
- `workload-*`、`WORKLOAD.md`：工作量与到达过程；`RESULTS_V3.md`：已保存的历史结果。
- `../revising.py`：完整 Online 模型／optimizer 的代码修订入口。
- `revision_demo.py`：显式闭式公式的 CPU 示例，供修订流程验证使用。
- `../../hermes/train.py`：真实任务和原生 Trainer 接入。

CPU 修订示例：

```bash
python -m examples.autopilot.revising --demo --target model --output /tmp/online-model-revision
python -m examples.autopilot.revising --demo --target optimizer --output /tmp/online-optimizer-revision
```

非 demo 需提供 `--profile`、`--cases`，并通过 `--agent-command` 或 `--llm-model` 选择 proposer。模型
profile 提供 `model_settings` 和 `primary_metrics`；cases 使用 `HistoricalCase`。Optimizer
profile 提供 `assembly`、`model_source`、`model_settings`、可选 calibration cases；cases 使用
`OnlineProblem`。具体 factory 和接受规则见
[Elastic revision](../../../areal/autopilot/domains/elastic/README.md)。

## 原生接入

Hermes 入口支持 `--elastic-profile`／`--elastic-settings` 装配普通 `ReconfiguratorAdapter`，支持
`--elastic-revision`／`--elastic-bundle` 显式接入 `RevisionAdapter`；其余训练配置参数继续传给原入口。普通
adapter 不导入 AutoPilot。

Native profile 可提供所选版本的 `reconfigurator`、`replay` 或
`forecast`、`slo`、`model`、`run`、`handoff`。模型字段引用冻结 bundle、mechanism fits 和 publication
metadata。旧的 phase-only `measurements` profile 不再用于 Online 决策。原生调用仍以真实
batch／staleness、GPU 预留和执行器能力为准。Native 使用 `handoff.mode="cold"` 与显式 `phase_seconds`；当前仅支持
FSDP DP／SGLang TP1 的动态调整。

接受后的 bundle 通过模拟侧的 `load_bundle()` 或原生 `RevisionAdapter(bundle=...)` 加载。CPU 实验将其模型传入
`prediction_models`，保持外层执行成本模型固定。原生预加载、探测后在已发布、无活动更新的边界激活，保留训练进度和已有观测。后台
`RuntimeRevisionLoop` 由调用方提供 `problem_builder`，明确从已有完成历史构造 calibration／validation 和
Online 搜索输入。

当前原生真实 TTFT／最大 TBT 采集另列待办，本轮不改推理通路。模拟保留显式 hot／cold，原生使用 cold；阶段耗时由输入提供。校准迁移副本保持原系数。CPU
结果不代替 GPU 验收。

[实现](../../../areal/elastic/README.md) ·
[调整调用树](../../../areal/elastic/RECONFIGURATION.md) · [项目进度](../../../PROGRESS.md) ·
[GPU 验收准备](../../task_workflows/RECONFIGURATION_ACCEPTANCE.md)
