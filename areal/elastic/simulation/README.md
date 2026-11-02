# Elastic CPU simulation

CPU 实验环境代替真实 training，持有实验 workload 的真实未来，产生观测并执行选定配置。Reconfigurator 的候选预测分别进入 Basic 的
JointSimulator 或 Reference 的 Evaluator，不调用本环境。

```text
simulation/
├── workload.py             实验工作负载生成、格式与读写
├── simulation_core.py      事件推进、准入、训练／发布、观测及恢复
├── online_simulation.py    Online 主环境、hot 交接、按模式创建环境
├── cold_transitions.py     cold 交接，复用原生 ReconfigurationPlanner
├── online_transitions.py   HandoffConfig、GPU 租约、链路及显式耗时
├── online_inputs.py        不可变负载与恢复模板
├── online_state.py         请求／副本的可变状态
├── online_decode.py        稳定 Decode 区间推进
└── online_parallel.py      按所选版本装配候选 CPU 进程池
```

外层 `models` 为执行成本模型，`prediction_models`
为候选预测组件；研究模型修订时固定前者，替换后者。两者未分别传入时保留同一初始组件的兼容用法，CLI 分别实例化。

请求状态记录首 token 与相邻 token 的时间；`rolling_slo()` 根据请求观测统计窗口内的 TTFT／最大 TBT 达标情况。

触发、预测、模型、候选搜索和控制时序分别位于 `reconfigurator/basic/` 或 `reference/`；模拟环境不另选算法。

`OnlineSimulation.create(..., handoff=...)` 根据 `handoff.mode` 选择执行方式：

- `hot`：准备新部署后接管，旧请求在旧副本完成；允许准备期占用额外 GPU，单独记录重叠成本。
- `cold`：等安全边界、排空模型调用，释放变化的旧引擎，再在声明的 GPU 池内部署；窗口成本使用实际占卡积分。

两种模式均保留未变化引擎、训练游标和权重版本。Cold 使用 `phase_seconds` 的显式阶段耗时；缺少必要耗时的候选不能得到有效预测。原生映射还携带实际 GPU
预留、当前布局与 checkpoint 版本。

Actor 初始化后保持驻留，不执行空闲释放或预测唤醒。`ActorView.running` 仅记录引擎是否已存在，用于首次训练启动及 Cold 布局重建。

[运行方式、决策与假设](../../../examples/autopilot/elastic/ONLINE_RECONFIGURATION.md)。
