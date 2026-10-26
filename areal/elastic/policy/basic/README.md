# Basic policy

当前实现冻结为 **policy
v3.1**：正常训练／发布不再丢弃已选方案，发布完成后继续接管最新状态。v2冻结于`922ae995`，v1保留原标签`elastic-policy-v1`；容量及显存余量规则沿用v2。规则、结果与已知问题见[版本记录](VERSIONS.md)。

原生 `on_demand` 通过 `policy/adapter.py::PolicyAdapter` 统一装配和调用策略，默认使用
`IdleTimeoutPolicy`。默认空闲 60s 后请求释放， 支持零超时；最短驻留、发布、保存与进程清理由 Runtime 约束。完整 Basic 尚未成为默认策略。

| 文件                                     | 职责                                              |
| ---------------------------------------- | ------------------------------------------------- |
| `policy.py`                              | 固定超时基线、模型生命周期决策和离线联合规划入口  |
| `optimizer.py`                           | 有限视野搜索、成本／时间约束、切换门槛与查询预算  |
| `simulator.py`                           | 阶段组合、负载预测、驻留 GPU 成本及联合模拟       |
| `actor_cost.py`                          | Actor 模型到阶段预测／生命周期策略的适配          |
| `joint.py`、`serial_planning.py`         | 联合配置协议与离线严格串行模型／决策              |
| `serving.py`                             | 请求级 TTFT／TBT 统计与联合达标率约束             |
| `online.py`、`history.py`、`forecast.py` | 仅SLO压力／宽裕触发、负载阶段、历史回放与在线搜索 |
| `profile.py`                             | 实测 workflow 装配与固定布局模拟边界              |
| `serialization.py`                       | 公共规划输入的 JSON 解码                          |
| `cost_model/`                            | Actor、Rollout 与转换成本模型、校准和基础数据     |

`BasicPolicy.decide()` 处理生命周期，`JointPolicy.plan()` 返回离线联合计划；Basic 独立于 Reference。
运行历史统一来自默认 Monitor，超时基线也会积累数据；启用采集不自动切换模型搜索。 模型比较固定样本量和训练语义，缺少测量或状态时明确拒绝预测；配置转换需显式声明兼容。
原生实测模型适配目前仍要求显式 profile、固定布局、horizon=1，离线候选不代表 GPU 已支持重配置。

`JointOptimizer` 可显式传入 `serving_slo` 与 `expected_serving_requests`：先要求同一窗口全部请求有统计，
再筛选同时满足 TTFT／TBT 的请求比例，最后最小化总 GPU 秒。默认不启用 SLO；缺失统计不能判为达标。 旧联合实验使用
`ServingSLO(5.0, 0.1, 0.95)`，新在线历史模式目标为0.8。TTFT 从请求逻辑就绪开始，首轮包含任务到达后等待准入的时间； TBT
取同一次响应内最大的相邻 token 间隔，不包含两次模型调用之间的工具执行时间。保留原 staleness=2 准入规则；实际训练版本差只记录，不增加拒绝条件。模拟观测器位于
`areal/autopilot/domains/elastic/serving_observer.py`，统计模拟 token 时间，尚不包含未实测的 HTTP／网络尾延迟。

`OnlinePolicy`只接收已观察状态；历史模式在首次启动及每次切换完成（旧部署释放）后，累计满H秒运行历史才判断触发（默认120秒），达标率\<80%或>90%时搜索。采用固定H秒Serving窗口，最小化`切换GPU·秒 + H×目标总卡数`，不排空目标尾部工作。v2要求达标候选的Actor估计处理速度不低于历史外部到达率；压力下优先合格的新布局，否则包含current比较达标率、额度亏损及成本；宽裕下只在合格者中比较成本，包含current。新Actor仍须有完整batch条件性能力依据。已确认从当前状态向后回放历史模板，SLO包含整个H秒的切换过程，不回到历史窗口起点；旧随机预测入口保留供历史实验复现。
CPU持续事件环境和冻结模型适配位于 `areal/autopilot/domains/elastic/online_*.py`，尚未接入原生服务控制。
示例设置`policy.actor_memory_reserve_fraction=0.1`，按全部已完成历史batch检查90%容量预算；
不合格者也不能进入压力回退。`actor_memory_scope="evaluation"`可仅对原候选评估所用batch加余量，不额外扫旧历史；默认`all_completed`。
省略余量或设0关闭筛选，原物理显存模型及训练微批保持不变。
[运行与假设](../../../../examples/autopilot/elastic/ONLINE_RECONFIGURATION.md)。

[成本模型](cost_model/README.md)／[原生执行接口](../../../../examples/autopilot/elastic/README.md)。
离线 CLI、R10 专用回放辅助模块与原详细说明见
[本地开发归档](../../../../../artifacts/elastic-policy-development-20261021/README.md)。
