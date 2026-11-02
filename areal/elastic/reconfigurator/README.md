# Elastic Reconfigurator

- **Basic v1**：使用旧 `JointSimulator` 的聚合模型，先作为可运行基线。
- **Reference**：使用 `Evaluator` 进行请求与 token 级预测，包含当前搜索优化。

原生与 CPU 实验通过 `build_reconfigurator("basic" | "reference", ...)` 选择同一套决策实现。候选评估使用所选版本自己的
评估组件；`OnlineSimulation` 只负责外层实验执行。

一次候选评估分为：（1）从当前观测和历史构造预测工作量；（2）通过版本自己的评估组件预测配置表现；（3）optimizer 比较结果，所选配置交回原生 Runtime
或外层实验环境执行。

```python
# 决策伪代码；make_model() 装配所选版本的候选评估器
forecasts = forecaster.predict(view, decision_index)
evaluator = reconfigurator.make_model(...)
predictions = evaluator.search(view, forecasts, candidates)
# Basic v1：OnlineForecastModel → JointSimulator
# Reference：Evaluator → 内部事件预测 + cost_model
# Optimizer 比较 CandidatePrediction；执行环境应用所选配置
```

Reference 的候选检查、事件预测与评分统一在 `evaluator/`，对外入口为 `Evaluator`。 Basic 保留 `prediction.py` 和
JointSimulator。两者均不调用外层实验环境，公共 DTO、输入模板、GPU 租约和 Decode 数学工具可复用。

外层实验的 `models` 提供冻结执行成本，`prediction_models` 提供内部预测组件；修改预测模型不替换实验成本模型。CLI
为两者分别创建组件实例并记录来源。

`handoff.mode` 保留 hot／cold。原生当前执行 cold，使用真实 GPU 预留和显式阶段耗时；服务 TTFT／最大 TBT 缺测保持未知。见
[执行环境](../simulation/README.md) 和 [原生调整](../RECONFIGURATION.md)。

[Basic v1](basic/README.md) · [Reference](reference/README.md) ·
[Revision](../../autopilot/domains/elastic/README.md)
