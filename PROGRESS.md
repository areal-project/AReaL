# AutoPilot 进度

更新：2026-10-31（香港时间）。开发分支`lws/feat/elastic-reconfig`，远端`origin`为`areal-project/AReaL`。
[双机交接](examples/autopilot/DEVELOPMENT.md)／[架构](areal/elastic/README.md)／[策略版本](areal/elastic/policy/basic/VERSIONS.md)。

## 当前状态

- 后续优先推进online。共用simulation已从AutoPilot
  domain移入`areal/elastic/simulation/`，抽取事件内核与候选评估，拆分观测和online预测；online策略规则保持不变，115项online正常CPU检查通过。
- inline模拟、CV触发、示例及双周会结果单独保存到`lws/feat/elastic-reconfig-inline`；当前分支保留共用重构，避免混入inline专用策略与任务池。
- 共享集成分支本地及远端均在`7bab3a80`；重配置与Arena独立推进。旧分支已清理，个人GitHub仓库保留。完整实验trace、日志和校准位于仓库父目录`artifacts/`，不随Git同步。

## 已验证

- Online
  v3统一32格条件模拟完成并审计：19/32满足TTFT5秒＋最大TBT100毫秒／80%目标；同轨迹按平均TPOT50毫秒后处理32/32达标，TPOT尚未接入决策。[配置与结果](examples/autopilot/elastic/RESULTS_V3.md)。
- v3.1发布等待、Actor保存／释放／恢复、停止时仅调Rollout及分侧120秒历史等待已有正常CPU验证；冻结4B／H20组件与发布校准保留。[版本](areal/elastic/policy/basic/VERSIONS.md)。
- 真实4B＋Hermes/DAPO固定布局16题／4更新通过，32次推理、5次工具、消费／发布／checkpoint及退出审计通过。[证据](../artifacts/h20-training-trace-20261025-f28c1811/smoke-001/audit.json)。

## 下一步与约束

- 待用户整理proposal后确定online修订对象与评估任务；原生DP／副本切换GPU验收、TP切换与训练质量仍待完成，条件模拟不能代替原生验收。
- 当前SLO仍由最大TBT驱动；启动、跨节点链路与未见混合形状为条件模型。外层revision尚未验证对新闭环的额外收益。
- H20监控及自动开跑继续暂停，等待用户明确恢复；公司端Arena真实模型连通与训练验收独立推进。[采集计划](examples/task_workflows/TRAIN_TRACE_PLAN.md)。
