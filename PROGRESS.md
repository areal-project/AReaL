# AutoPilot 进度

更新：2026-10-27（香港时间）。分支 `lws/feat/elastic-reconfig`。
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
1. **policy
   v3.1冻结提交**：发布期间保留候选；原冲突精确回放等待0.425秒后执行，无重复搜索／复核。长输入44联合SLO70.45%→79.11%、GPU·h6.487→6.593、额度阻塞228.02→126.78秒；2,609题／65更新／尾9题守恒，96项正常CPU检查及完整审计通过。未达80%主要因初始Actor不足、宽裕分支延迟恢复：前360秒53.64%，此后85.48%；88联合分别92.72%／83.37%。触发／选择规则未改；下一步实现独立Actor释放／拉起并验证长平台。v2冻结于`922ae995`。[版本](areal/elastic/policy/basic/VERSIONS.md)／[证据](../artifacts/elastic-policy-v3-publication-20261027/README.md)／[44与88分段对照](../artifacts/elastic-policy-v3-publication-20261027/initial-state-comparison.md)。
1. **剩余4条负载16组已审计，未达标原因已定位**：v2＋仅10%显存余量下6/16达80%；长输入88两组87.7%／85.2%、慢工具四组99.6%。长平台早期Actor被“完整batch／目标更新覆盖”检查拦住；长短平台TBT失败又受历史滞后与120秒禁搜影响。长输入44的200秒124候选已逐一复现0/183：182个TTFT失败，恢复快慢被二元SLO掩盖，打平后选TP8的8＋1。12组显存等待均0；4组原决策逐batch回放与原请求时间完全一致。仅诊断，未改策略。[完整表](../artifacts/elastic-policy-v2-remaining-20261027/table.md)／[审计](../artifacts/elastic-policy-v2-remaining-20261027/audit.json)／[诊断](../artifacts/elastic-policy-v2-remaining-diagnosis-20261027/README.md)。
1. **显存余量对照已审计**：多轮长输入88联合，原v2为66.4%／8.27GPU·h；全部历史＋10%余量、仅原候选评估加10%余量均为88.56%／5.76，显存等待和额度阻塞归零。原830秒候选batch94.82GiB超过90%预算86.02GiB，余量单独即可拦住5卡TP1。93项正常CPU检查通过；单例条件模拟，非GPU保证。原模式／示例／结果均纳入v2冻结。[对照证据](../artifacts/elastic-policy-memory-margin-only-20261026/README.md)。
1. 历史负例独立保留：旧687题／17更新／尾7题单峰中，动态SLO51.94%低于固定63.65%；旧95%动态单峰未完成。旧全trace搜索保持暂停。[搜索／延迟](../artifacts/elastic-online-search-20261025/README.md)／[模拟器加速](../artifacts/elastic-deferred-state-20261025/README.md)／[原95%](../artifacts/elastic-online-campaign-qwen4b-h20-20261025/README.md)／[安全等待](../artifacts/elastic-online-safe-wait-20261025/README.md)。
1. Web trace viewer尚未实现。跨节点、mixed、HTTP尾延迟、发布中止重算及未见形状显存仍属条件模型，后续需验证。
1. 真实GPU引擎验收等待资源；不启动旧1.7B正式训练。其旧运行仅3次更新，第4次反传OOM，无checkpoint；微批／allocator修复仍未验证。
1. 公司端在 `lws/feat/elastic-envarena`
   接沙箱后，验收stdout／stderr、清理和评分，再跑16题评测／4更新训练；SWE先验证参考补丁。[2×2操作](examples/task_workflows/RUN_2X2_8GPU.md)／[Singularity](examples/task_workflows/RUN_SLURM_SINGULARITY.md)。

共同基线为 `lws/feat/elastic`；`origin`官方仓库、`fork`个人仓库，旧分支保留。本地reconfig历史提交
`7e0c7bea`／`862819fc`；v1以`elastic-policy-v1`保存，v2在当前分支以`feat: freeze online elastic policy v2`提交固定，后续策略修改从v3开始；未推送。历史细节见[推进前快照](../artifacts/elastic-simulation-qwen4b-h20-20261025/progress-before-simulation.md)。
