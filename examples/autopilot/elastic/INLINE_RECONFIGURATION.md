# Inline重配置模拟

已实现CPU条件模拟及固定／动态对照。目标是在\*\*全程GPU硬上限（包含切换）\*\*内，完成相同训练任务、batch及必要权重发布，最小化总完成时间。原生GPU状态恢复与训练质量另行验收。

- 有限任务池按固定顺序主动供给；并发与原staleness额度共同限制提交。阶段按任务序号划分。
- 每K次update发布后检查。已知任务ID、类别及初始输入长度可用于预测；未知输出、多轮与工具耗时只从已完成历史估计，真实未来留在主环境。
- 候选共用同一预测工作量，比较完成后续M个完整batch及发布的时间，尾段按剩余任务截断；同时保留视野之后的流水线供给，避免提前断粮降低候选耗时。current始终参与，超过收益门槛才切换；不支持的候选明确排除。
- 搜索／复核的CPU墙钟进入主时间线，GPU工作同时推进；候选使用显式决策延迟估计。应用前重建最新观测并复核，训练／发布边界变化时丢弃旧决定。
- 交接先停止新episode，排空在途推理和工具，完成当前update及发布；保存需迁移的Actor状态，释放全部需更换的旧部署，再申请、加载和预热目标部署。未更换部署继续占预算；保存／恢复时间、任务游标、待训练数据及权重版本均保留。

## 文件组织

| 路径                                                                              | 职责                                         |
| --------------------------------------------------------------------------------- | -------------------------------------------- |
| `areal/elastic/simulation/simulation_core.py`                                     | 共用事件执行、训练／发布、计费与状态恢复     |
| `areal/elastic/simulation/online_simulation.py`、`online_candidates.py`           | 原online供给、SLO控制、热交接与候选评分      |
| `areal/elastic/simulation/inline_simulation.py`                                   | Inline主运行、候选分支、搜索／复核时序与结果 |
| `areal/elastic/simulation/inline_inputs.py`、`inline_transitions.py`              | 任务池与全程硬预算交接                       |
| `areal/elastic/policy/basic/observations.py`                                      | 共用布局、任务、观测和预测输入；沿用原字段   |
| `areal/elastic/policy/basic/inline.py`、`inline_forecast.py`、`inline_trigger.py` | Inline决策、历史预测及轻量统计／CV触发       |
| `areal/elastic/policy/basic/online.py`、`online_forecast.py`、`history.py`        | Online决策与两种历史预测                     |
| `areal/elastic/policy/basic/forecast_workload.py`                                 | 两种模式共用的episode后续估计                |
| `areal/elastic/workload.py`                                                       | 原时间到达生成及新增按任务数／序号生成       |

校准适配、GPUAllocator、链路、运行状态及Decode推进共用现有实现，位于
`areal/elastic/simulation/`。`autopilot/domains/elastic/`负责评测、代码修订和部署适配。具体边界见[模块说明](../../../areal/elastic/simulation/README.md)。

## 运行

三个校准输入与online入口相同：`BUNDLE`为冻结组件目录，`MECHANISM_FITS`为机制拟合JSON，`PUBLICATION_METADATA`为AWEX参数元数据；不随Git携带完整校准产物。

```bash
python examples/autopilot/elastic/simulate_inline.py \
  --config examples/autopilot/elastic/inline-example.json \
  --workload examples/autopilot/elastic/inline-workloads/shape-shift.json \
  --bundle "$BUNDLE" --mechanism-fits "$MECHANISM_FITS" \
  --publication-metadata "$PUBLICATION_METADATA" --output "$OUTPUT"
```

默认实测CPU搜索／复核时间；`--decision-clock fixed`使用配置中的固定延迟，适合确定性检查，但仍执行实际候选计算。输出目录须为新目录；`--progress`打印完整搜索进度。更换为
`inline-workloads/stable.json`运行稳定案例；修改负载spec中的seed可复核另一任务样本。

原示例：16卡／2×8节点，batch40、staleness2、并发64；K=2、M=24、收益门槛3%。四候选均在同一16卡预算内，包括Actor
DP／TP和Rollout副本／TP变化。默认只运行initial fixed与dynamic；`--all-fixed`可额外运行每个固定候选，用于诊断。

输出包含冻结输入／源码身份、任务trace、逐配置摘要、事件、请求、训练记录及决策。审计任务顺序、完整batch、请求／输出token、发布版本、staleness、峰值卡数和结束资源释放。首个例子要求任务数为batch整数倍。

## Batch48与CV触发

新入口使用`inline-cv-example.json`与`inline-workloads/cv-three-phase.json`：长度变化较大→多轮长输入→16K输入／4K输出，每段3,072题／64更新，共9,216题／192更新；44候选、M=16、3%切换门槛及全程16卡上限。

补充trace为`inline-workloads/cv-long-context-shift.json`：24K输入／256输出→16K输入／4K输出→多轮长输入，阶段规模相同，单seed20261029。沿用同一配置及以下命令，只替换`--workload`；[补充结果](RESULTS_INLINE.md#batch48%E8%A1%A5%E5%85%8524k%E8%BE%93%E5%85%A5--16k%E8%BE%93%E5%85%A54k%E8%BE%93%E5%87%BA--%E5%A4%9A%E8%BD%AE%E9%95%BF%E8%BE%93%E5%85%A5)。

```bash
python examples/autopilot/elastic/simulate_inline.py \
  --config examples/autopilot/elastic/inline-cv-example.json \
  --workload examples/autopilot/elastic/inline-workloads/cv-three-phase.json \
  --bundle "$BUNDLE" --mechanism-fits "$MECHANISM_FITS" \
  --publication-metadata "$PUBLICATION_METADATA" --output "$OUTPUT" --progress
```

每2个已发布update轻量检查最近2个batch（96条episode）；首次全搜后，对照上次全搜时的统计，变化≥30%连续确认2次才重新搜索，每16个update兜底。输入、总输出、累计训练token及最大上下文统计均值／P90／CV，另记录平均轮数与多轮比例；CV比较设0.5分母下限，避免接近零的波动被放大。阶段编号与生成参数不进入检测器。

`trigger-checks.json`记录检查、距离和原因；摘要另列检查耗时与全搜次数。稳定检查只访问近期已发布样本，检查周期与全搜周期独立推进。新配置按已发布update编号确定预测采样种子，搜索与复核次数不改变同一update的采样。旧配置的触发和采样方式默认保留。

## 验证与结果

正常CPU检查覆盖共用执行、候选恢复、硬预算交接、单写者、未知未来隔离和自动切换。三条online完整轨迹逐事件／决策／训练记录与拆分前一致（排除宿主CPU秒表）；8种原负载生成结果相同，原历史验证JSON保留。

[代表结果](RESULTS_INLINE.md)包含稳定场景、形状变化场景及另一seed。完整证据与图位于仓库父目录
`artifacts/elastic-simulation-inline-20261029/`。当前展示固定实现的reconfigurator闭环；AutoPilot代码修订增益另验。

## 多阶段例子

五段、每段640题的配置入口：

```bash
python examples/autopilot/elastic/simulate_inline.py \
  --config examples/autopilot/elastic/inline-multiphase-example.json \
  --workload examples/autopilot/elastic/inline-workloads/multiturn-decode-shift.json \
  --bundle "$BUNDLE" --mechanism-fits "$MECHANISM_FITS" \
  --publication-metadata "$PUBLICATION_METADATA" --output "$OUTPUT"
```

阶段为长输出／多轮长输入交替，共3,200题；M=16，与每段更新数相同，初态为Actor DP2×TP4＋Rollout 4副本×TP2。另有
`multi-shift.json`作为原短／长输入交替控制组，配合原
`inline-example.json`运行。结果及两个seed的完整对照见[代表结果](RESULTS_INLINE.md#%E4%BA%94%E6%AE%B5%E5%8F%98%E5%8C%96%E6%AF%8F%E6%AE%B5640%E9%A2%98)。

## 原batch40持续阶段入口

长输出→多轮长输入→长输出，每段2,560题的持续阶段：

```bash
python examples/autopilot/elastic/simulate_inline.py \
  --config examples/autopilot/elastic/inline-demo-example.json \
  --workload examples/autopilot/elastic/inline-workloads/sustained-shift.json \
  --bundle "$BUNDLE" --mechanism-fits "$MECHANISM_FITS" \
  --publication-metadata "$PUBLICATION_METADATA" --output "$OUTPUT"
```

原完整对照扩大了逻辑事件／模拟时间上限，并运行四个固定候选；如今默认只比较initial
fixed与dynamic，可加`--all-fixed`复核原范围。另有`long-stages.json`为五段各1,280题的输入。[历史演示结果](RESULTS_INLINE.md)。
