# Elastic policy

- `on_demand` 默认选择 `actor_lifecycle.policy=basic`；当前使用固定空闲超时策略，后续接完整 Basic。
- [basic](basic/README.md)：可修订的 cost model、simulator、optimizer 和调度策略；固定超时判断也由这里统一提供。
- [reference](reference/README.md)：独立 Expert 参考模型与 simulator，在实验中冻结；不含 optimizer 或调度策略。
  `actor_lifecycle.policy=reference` 已预留，但在线调度未实现，选择后在服务启动前报错，不回退 Basic。

`adapter.py::PolicyAdapter` 统一装配和调用默认超时／显式实测 Basic；不依赖 AutoPilot；观测和历史由默认启用的
`elastic/monitor/collector.py` 提供。

`reference` 表示参考实现／实验对照，不代表权威标准或真实测量。 Basic 不依赖 reference；调度动作由原生执行器验证和执行。

`native_joint.py` 显式桥接联合搜索与原生资源事务，默认不启用；只在已发布且无在途训练工作的边界规划。
调用者提供模型、预测窗口及候选物理映射；[接入与 CPU 合成验证](../RECONFIGURATION.md#%E6%98%BE%E5%BC%8F%E6%8E%A5%E5%85%A5%E8%81%94%E5%90%88%E8%A7%84%E5%88%92)。

离线 CLI、R10 回放辅助模块、设计长文与专用实验入口已移入
[本地开发归档](../../../../artifacts/elastic-policy-development-20261021/README.md)。数值模型、simulator、optimizer、
基础 profile 与必要校准输入保留；已有日志／trace 解析归入 `elastic/monitor/trace.py`。
