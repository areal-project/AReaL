# AutoPilot 进度

更新：2026-10-27（香港时间）。开发分支`lws/feat/elastic-reconfig`，远端`origin`为`areal-project/AReaL`。
[双机交接](examples/autopilot/DEVELOPMENT.md)／[架构](areal/elastic/README.md)／[版本](areal/elastic/policy/basic/VERSIONS.md)。

## 当前实现

- v3延续发布期间保留方案（v3.1：`b638e38f`），增加可选Actor保存／释放／恢复、停止时仅调Rollout和两侧独立120秒等待。历史／候选／外层共用同一生命周期规则，恢复保持DP／TP、版本和消费游标；当前仅CPU条件模拟，原生GPU重配置仍待验收。
- [当前v3配置](examples/autopilot/elastic/online-v3-example.json)及[8种人工负载参数](examples/autopilot/elastic/workload-regimes/)随代码保存；兼容示例保留旧默认。仍用TTFT5秒＋最大TBT100毫秒／80%驱动，TPOT尚未接入决策。

## 已验证

- 32格动态统一v3：补跑28、同源码及配置复用4，全部完成并审计。共28,600题／42,732请求／708更新／尾280题，任务、token版本、恢复、计费及候选选择通过；19/32达80%。同一轨迹按TPOT50ms后处理为32/32达标，部署和成本不变。[可携带结果与摘要](examples/autopilot/elastic/RESULTS_V3.md)。
- 发布前111项正常CPU检查通过，8份负载参数重生成与实验原输入逐字节一致。4B／H20组件96个Actor形状、4,104个Rollout请求及AWEX发布锚点已核查；56份冻结组件文件未改。[组件](../artifacts/expert-calibrated-qwen4b-h20-20261025/README.md)／[发布](../artifacts/elastic-publication-20261025/README.md)。
- 本地4B＋Hermes/DAPO真实16题／4更新完成，32次模型调用、5次工具、完整消费／发布／checkpoint及退出审计通过。[证据](../artifacts/h20-training-trace-20261025-f28c1811/smoke-001/audit.json)。

## 阻塞与下一步

- 最大TBT口径下，长平台／短平台／长输出仍未达80%；部分44配置达标但成本更高。启动、跨节点链路及未见混合形状仍为条件模型，GPU接管和数值恢复须独立验证；Web
  viewer未实现。
- **已按用户要求暂停H20监控及自动开跑**，heartbeat `h20-4b` 为PAUSED；10-27
  09:39最后检查8卡仍被占用、数据盘约85G可用，未启动新实验。后续候选约1万不同题、batch40、客户端并发64，固定TP4 Actor＋4×TP1
  Rollout；目标规模／64K／采集开销仍待验收。[计划](examples/task_workflows/TRAIN_TRACE_PLAN.md)／[监控记录](../artifacts/h20-training-trace-20261025-f28c1811/monitor/state.json)。
- 旧1.7B正式训练仅完成3／125次更新，第4次反传OOM且无checkpoint，修复未验收；不据此续训。公司端`lws/feat/elastic-envarena`继续验收沙箱清理／评分及16题评测／4更新训练。[2×2操作](examples/task_workflows/RUN_2X2_8GPU.md)。

共同集成分支`lws/feat/elastic`保持独立，旧分支和v1标签保留；v2冻结于`922ae995`，后续v3在当前分支演进。完整实验trace、日志及校准位于仓库父目录`artifacts/`，不随Git同步；可携带摘要与输入参数已入库。
