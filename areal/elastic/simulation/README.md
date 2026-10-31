# Elastic CPU simulation

独立的执行环境与候选分支，不依赖AutoPilot修订包。共用内核按注入的组件模型推进Rollout、工具、Actor、发布和GPU计费；真实未来留在主环境，策略只接收观测、已知任务元数据及历史预测。

| 模块                                                             | 职责                                                                                       |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `simulation_core.py`                                             | 共用事件推进、准入、训练／发布、观测与状态恢复；`SimulationConfig`／`SimulationModels`契约 |
| `online_simulation.py`／`online_candidates.py`                   | 外部到达、SLO控制与候选窗口评分；`OnlineRunConfig`为共用配置的别名                         |
| `online_models.py`                                               | 冻结组件校准、显存与发布适配，供两种模式使用                                               |
| `online_inputs.py`／`online_state.py`／`online_decode.py`        | 不可变恢复模板、运行状态及Decode区间推进，供两种模式使用                                   |
| `online_transitions.py`                                          | 共用GPU租约、链路与时间参数；online热交接允许额外准备资源                                  |
| `online_control.py`／`online_parallel.py`／`online_lifecycle.py` | online搜索时序、CPU候选进程池和可选Actor生命周期                                           |
| `serving_observer.py`                                            | 请求级服务指标观测                                                                         |

策略及预测位于 `areal/elastic/policy/basic/`；AutoPilot的评测、修订与部署适配位于
`areal/autopilot/domains/elastic/`。当前条件模拟不证明原生GPU重配置、状态数值恢复或训练质量。

[Online入口与假设](../../../examples/autopilot/elastic/ONLINE_RECONFIGURATION.md)。
