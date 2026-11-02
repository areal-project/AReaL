# AutoPilot 进度

更新：2026-11-02（香港时间）。分支 `lws/feat/elastic-reconfig`；本次提交保存链路重构前的基线。

## 当前状态

- 原生路径仍由 `OnDemandActor` 协调；下述回归 Trainer 主导的链路重构尚未实施。
- Elastic 已分为共享契约、Infra、Reconfigurator 和 Simulation。原生运行、Adapter、状态与观测归入
  Infra；进程工具并入重配置执行；布局方案保持共享，负载生成归 Simulation。见[Infra](areal/elastic/infra/README.md)。
- Reference 使用单一 Evaluator；Actor／Rollout 各自管理 Packing／Scheduling
  和成本计算，FSDP／Megatron／Archon 复用基础算子。Basic v1 保留 JointSimulator
  基线，原生与模拟共用所选决策。见[Reference](areal/elastic/reconfigurator/reference/README.md)。
- Hot／Cold 保留，原生执行 Cold；运行预算和候选范围由运行配置决定。Actor 初始化后保持驻留，首次启动、配置重建和恢复保留，空闲释放与预测唤醒已移除。
- Revision
  完整组件包接入模拟加载与原生激活。见[Revision](areal/autopilot/domains/elastic/README.md)、[重配置执行](areal/elastic/RECONFIGURATION.md)。

## 已验证

- 本轮292项正常 CPU 检查通过；14个行为案例、56个校准查询与迁移前一致。布局、事务、进程清理及观测逻辑保持原值，两版本22份 Profiles 与运行 JSON
  未改；校准副本仅迁移源码身份和导入，未
  refit。[证据](../artifacts/elastic-infra-organization-20261102/final-validation.json)。
- Basic／Reference × Hot／Cold 的 CLI／候选进程池均完成固定与动态运行，各40请求、40训练样本、1次更新。
- 前轮后端数值对照见[记录](../artifacts/reference-organization-20261102/final-validation.json)，历史
  Online v3
  条件模拟保留[原结果](examples/autopilot/elastic/RESULTS_V3.md)。原生固定布局4B＋Hermes/DAPO曾完成16题／4更新；GPU
  动态交接另验。

## 后续

- 按五步收敛链路：验证原数据管线续接 → 控制权回归 Trainer → 补齐 Controller 接口 → 接回 Elastic → 清理旧旁路。
- 首个检查点为标准 Online／`prepare_batch()` 路径下的 Actor-only 重配置；本轮假定 CPU
  主进程与数据服务持续存活，暂不处理主进程崩溃恢复。
- 随后推进 Reference optimizer 与 revision 收益，后续以 Basic 为起点验证 revision。
- Actor 驻留管理另行设计；真实 TTFT／最大 TBT、切换实测耗时、GPU 动态交接与训练质量待验证。
- H20监控与自动开跑继续暂停，本轮不跑 GPU／故障注入。Inline／双周会结果保留在
  `lws/feat/elastic-reconfig-inline`；双机分工见[DEVELOPMENT](examples/autopilot/DEVELOPMENT.md)。
