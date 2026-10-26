# AutoPilot 进度

更新：2026-10-26（香港时间）。分支 `lws/feat/elastic-reconfig`。
[开发交接](examples/autopilot/DEVELOPMENT.md)／[架构](areal/elastic/README.md)／[研究计划](examples/autopilot/PLAN.md)。

**已验证**

- 本地4B＋Hermes/DAPO真实16题／4更新通过：32次模型调用、5次工具、16次持久交接，4个版本均发布并完整保存；请求→SGLang
  batch→导出→实际消费关联无缺失，Actor／服务正常退出。[审计](../artifacts/h20-training-trace-20261025-f28c1811/smoke-001/audit.json)。
- 4B／H20组件实测审计完成：Actor 96个布局／形状，86点完整更新、10点同recipe隔离复现OOM；Rollout
  TP1／2／4／8共4,104请求，GPU迭代／长度／KV通过。[审计](../artifacts/elastic-calibration-20261025/final-audit.json)。
- 独立组件v2校准已冻结；原生AWEX发布4锚点＋1保留组测完，40次空闲发布、1次在途验证及160条rank记录通过。网关TP元数据合并修复经6项真实CPU
  planner与GPU复测。[组件](../artifacts/expert-calibrated-qwen4b-h20-20261025/README.md)／[发布](../artifacts/elastic-publication-20261025/README.md)。
- 旧完整trace固定部署回放与计费审计通过，仅属离线条件结果，不代表在线策略。当前展示50
  GB/s假设；新fit独立保存，原56文件包／11份profile未改。[旧结果](../artifacts/elastic-simulation-qwen4b-h20-nic50-20261025/README.md)／[核查](../artifacts/elastic-simulation-qwen4b-h20-20261025/audit.json)／[新fit](../artifacts/elastic-simulation-qwen4b-h20-20261025/new-fits.json)。
- 原生Actor
  DP与Rollout副本／联合重建控制已实现；联合搜索显式接入默认关闭。CPU／Gloo／DCP检查与合成控制链通过，三组真实GPU验收未执行。[验收入口](examples/task_workflows/RECONFIGURATION_ACCEPTANCE.md)／[联合规划](areal/elastic/RECONFIGURATION.md)。

**当前方向与下一步**

1. **按用户要求暂停H20监控及自动开跑**，heartbeat `h20-4b`
   已设为PAUSED，等待明确恢复。最后09:36检查8卡仍由原计算进程占用、数据盘约85G可用，未启动新实验。后续候选约1万不同题、batch40、客户端并发64，固定TP4
   Actor＋4×TP1
   Rollout；目标规模／64K／采集开销仍待验收。[计划](examples/task_workflows/TRAIN_TRACE_PLAN.md)／[监控记录](../artifacts/h20-training-trace-20261025-f28c1811/monitor/state.json)。
1. **当前实现登记为policy
   v1**（标签`elastic-policy-v1`）：固定120秒Serving评价、压力\<80%／宽裕>90%、`K＋120×目标GPU数`；首次及交接全部完成后再积累120秒。规则、对照表与已知问题已纳入[版本记录](areal/elastic/policy/basic/VERSIONS.md)，下一版先讨论optimizer的持续吞吐／积压与恢复判断，本次未改策略。
1. **v1的16组CPU实验已审计**：23,704请求、368次更新守恒，8组达到80%SLO；程序总耗时约13分钟。长输出受发布暂停影响；多轮长输入的120秒评价被剩余额度及旧Actor工作掩盖，缩容后积压，反复重配置使成本退化。[结果](../artifacts/elastic-rerun-16-20261026/README.md)／[审计](../artifacts/elastic-rerun-16-20261026/audit.json)／[精确重放诊断](../artifacts/elastic-slo-diagnosis-20261026/README.md)。保留原发布暂停、冻结fit与旧固定对照。
1. 历史负例独立保留：旧687题／17更新／尾7题单峰中，动态SLO51.94%低于固定63.65%；旧95%动态单峰未完成。旧全trace搜索保持暂停。[搜索／延迟](../artifacts/elastic-online-search-20261025/README.md)／[模拟器加速](../artifacts/elastic-deferred-state-20261025/README.md)／[原95%](../artifacts/elastic-online-campaign-qwen4b-h20-20261025/README.md)／[安全等待](../artifacts/elastic-online-safe-wait-20261025/README.md)。
1. Web trace viewer尚未实现。跨节点、mixed、HTTP尾延迟、发布中止重算及未见形状显存仍属条件模型，后续需验证。
1. 真实GPU引擎验收等待资源；不启动旧1.7B正式训练。其旧运行仅3次更新，第4次反传OOM，无checkpoint；微批／allocator修复仍未验证。
1. 公司端在 `lws/feat/elastic-envarena`
   接沙箱后，验收stdout／stderr、清理和评分，再跑16题评测／4更新训练；SWE先验证参考补丁。[2×2操作](examples/task_workflows/RUN_2X2_8GPU.md)／[Singularity](examples/task_workflows/RUN_SLURM_SINGULARITY.md)。

共同基线为 `lws/feat/elastic`；`origin`官方仓库、`fork`个人仓库，旧分支保留。本地reconfig历史提交
`7e0c7bea`／`862819fc`；当前基线以`elastic-policy-v1`保存，本轮仅本地提交，不推送。历史细节见[推进前快照](../artifacts/elastic-simulation-qwen4b-h20-20261025/progress-before-simulation.md)。
