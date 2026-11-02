# Online 重配置

`basic` 为使用旧 JointSimulator 的 **Basic v1**；`reference` 包含当前优化，也是默认版本。原生与模拟通过同一个 factory
选择版本，分别提供观测和执行环境。

## 决策过程

①累计历史后检查服务表现；②从当前积压、请求进度和历史构造预测；③模拟候选并按 SLO、GPU 成本选择；④用最新状态复核，再按指定方式切换。搜索期间原部署继续运行。

```text
OnlineReconfigurator.poll()                           # Reference
├── observe_history() → HistoricalReplay.observe()
├── OnlineReconfigurator.decide(view, model)
│   ├── # 1. 检查历史窗口与服务触发
│   │   _trigger(rolling_slo(...))
│   ├── # 2. 从当前状态和历史构造预测输入
│   │   HistoricalReplay.predict()
│   └── # 3. 比较候选
│       Evaluator.search() → OnlineOptimizer.search()
│       └── Evaluator.evaluate_wave() / evaluate()
│           ├── 组件模型：actor() / service() / publication()
│           └── Evaluator._from_forecast() → Evaluator._run()
└── # 4. 最新状态复核并执行
    OnlineReconfigurator.complete() / verified() / ready()
    └── environment.request_handoff(target)
        ├── 模拟：hot / cold 事件
        └── 原生：OnDemandActor.reconfigure(target=ExecutionLayout(...))
```

以上为 Reference 的 Evaluator 路径；Basic v1 保留 OnlineForecastModel／JointSimulator。

Reference 的 `bind(environment)` 返回同一个 Reconfigurator，决策与事件回调在同一类中完成；Basic v1 保留原控制实现。

默认每 10 秒检查最近 120 秒：联合达标率低于 80% 触发压力搜索；高于 90% 且完成至少 8 个请求时触发宽裕搜索。请求同时满足 TTFT≤5 秒、最大
TBT≤100 毫秒才达标；工具等待单列。

候选为 `Actor DP×TP；Rollout 副本数×TP`。逻辑 batch、staleness 与采样语义固定；先检查 GPU 放置、显存、packing
和模型覆盖，再预测服务表现。Reference 额外检查 Actor 持续处理能力覆盖外部到达率，并提供历史显存余量、分侧历史等待、排序与剪枝。

压力搜索有合格新候选时选其中成本最低者；否则包含当前配置，依次比较 SLO、Actor 供给缺口与成本。宽裕搜索包含当前配置，在合格者中选择成本最低者。Basic v1
保留服务约束与选择规则，候选预测使用粗粒度 Joint 模型。

历史预测从当前状态向前运行一个固定窗口，使用最近到达槽位和完成任务形状；未知后续轮次由历史补全，不读取主环境未来。

当前 Actor 初始化后保持驻留，不再按空闲时间自动释放或预测唤醒。配置变化所需的释放与重建仍计入候选成本。

## 切换方式

`handoff.mode` 显式选择，两种方式均保留：

- **hot**：新旧部署准备期重叠，成本为旧部署重叠 GPU·秒加目标窗口驻留成本。示例仍使用 16 卡稳态预算、4×8 准备资源池；切换峰值可超过 16。
- **cold**：排空模型调用，释放变化的旧引擎后部署；复用原生 `ReconfigurationPlanner`，按整个窗口实际 GPU
  驻留积分计费。`node_gpus` 是全程物理上限。

Cold 的 `handoff.phase_seconds` 提供
`checkpoint`、`release_actor`、`disconnect_weights`、`stop_rollout`、`rebuild_rollout`、`restore_actor`、`connect_weights`
等实际需要阶段的耗时，`provenance` 记录来源。纯主机步骤默认零；排空由请求进度决定，权重发布调用组件模型。缺少必要耗时的候选记为不支持，不套用 hot 成本。

原生当前执行 cold，支持纯 FSDP Actor DP、SGLang TP1 副本数及原 GPU 预留内的重新分配；不新增原生 hot 执行器。物理候选、当前
checkpoint 与预留由 Runtime 注入预测。现有 4B/H20 Megatron 校准可用于对应条件模拟，不能当作 FSDP 实测；原生加载时检查 Actor
backend。

原生已有轨迹可提供聚合工作量和接收时间；当前缺少真实 TTFT／最大 TBT，保持未知。本轮不改推理通路、不增加 token 流采集。

## 运行

```bash
python examples/autopilot/elastic/simulate_online.py \
  --reconfigurator reference --workload "$WORKLOAD" --config "$CONFIG" \
  --bundle "$CALIBRATION_BUNDLE" --mechanism-fits "$MECHANISM_FITS" \
  --publication-metadata "$PUBLICATION_METADATA" --output "$OUTPUT"
```

`OUTPUT` 使用新目录。默认 fixed／dynamic 都运行；`--mode` 可单选。`--search-workers 4` 使用候选进程池，设为 1
串行。默认将搜索／复核的实际 CPU 墙钟计入模拟时间；`--decision-clock fixed` 用于确定性功能对照。

`--bundle` 是冻结组件校准包。接受的 AutoPilot 组件包通过 `load_bundle()` 加载为完整 reconfigurator／模型，再传给
`OnlineSimulation.create()`；原生使用 `RevisionAdapter(bundle=...)`。

[online-v1-example.json](online-v1-example.json) 为 Basic
基线；[online-v3-example.json](online-v3-example.json) 为 Reference 示例。`identity.json`
保存源码、输入、校准和切换模式；结果包含决策、事件、请求、训练顺序与 GPU 积分。校准迁移单独保存副本，核对原系数和数值，不 refit。

[历史 v3 条件模拟结果](RESULTS_V3.md) 保留原实验身份与口径，其中 Actor
自动启停已从当前实现及示例配置移除；[工作量格式](WORKLOAD.md)；[当前进度](../../../PROGRESS.md)。GPU 动态交接与训练质量另行验收。
