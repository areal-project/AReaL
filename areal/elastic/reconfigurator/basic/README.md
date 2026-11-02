# Basic v1

以原来的 `JointSimulator` 为候选预测内核，保留单一聚合 GPU 通道、均衡 token 工作量的副本加速、顺序切换阶段与显式驻留计费。Online
适配补充当前请求／工具／Actor 状态、固定窗口及 TTFT／最大 TBT 的近似预测。

- `reconfigurator.py`、`controller.py`：触发、选择、搜索复核与应用时序。
- `history.py`、`forecast.py`：历史窗口、回放与抽样预测。
- `optimizer.py`、`prediction.py`：候选查询、搜索与统一评分结果。
- `simulator.py`：原始 JointSimulator 批次内核及 Online 入口。
- `joint_online.py`：Joint 的 Online 状态、队列及窗口扩展；服务 cohort 使用聚合屏障，不模拟连续 batching。
- `joint.py`：原始联合预测数据契约。
- `scenario.py`：profile 装配与调度类型注解使用的 `ExecutionConfig`、`Request`。
- `cost_model/aggregate_rollout.py`：原始聚合速率估计；Online 适配将注入组件预测转换成速率，不重新拟合系数。
- `cost_model/joint_transition.py`：原始顺序切换阶段模型。
- `models.py`、其余 `cost_model/` 与 `profiles/`：已有组件、校准及配置。

`build_reconfigurator("basic", ...)` 装配 Basic v1。原生与 CPU 实验使用相同的候选预测入口；实验环境不参与候选计算。版本来源见
[VERSION](VERSION.md)。
