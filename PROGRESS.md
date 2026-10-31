# AutoPilot Inline 进度

更新：2026-10-31（香港时间）。当前分支`lws/feat/elastic-reconfig-inline`；共用重构基点`3c0e97a5`，后续online开发回到`lws/feat/elastic-reconfig`。
[运行与约束](examples/autopilot/elastic/INLINE_RECONFIGURATION.md)／[结果](examples/autopilot/elastic/RESULTS_INLINE.md)／[双机交接](examples/autopilot/DEVELOPMENT.md)。

## 当前实现与已验证结果

- Inline任务池主动补充Rollout；固定全局batch48、staleness2，Actor
  DP×TP与Rollout副本×TP联合选择，全程16卡硬上限（含交接）。候选比较相同后续工作量的完成时间，计入搜索、复核和交接成本；预测仅使用已知元数据与已完成历史。
- CV触发分离轻量检查和全候选搜索；稳定负载跳过全搜，变化须持续达到阈值，保留低频刷新。32-update配对CPU66.92→11.15秒（6倍），训练记录一致。[证据](../artifacts/elastic-inline-cv-20261029/README.md)。
- 双周会主展示均为seed20261029，9,216题／192更新；下表为组件校准的CPU条件模拟，GPU原生交接与训练质量尚未验证。

| Trace                        | 初态固定→动态       | 缩时   | 交接／峰值GPU |
| ---------------------------- | ------------------- | ------ | ------------- |
| 长度变化→多轮长输入→16K／4K  | 14,749.6→11,847.1秒 | 19.68% | 2次／16卡     |
| 24K短输出→16K／4K→多轮长输入 | 19,657.2→14,770.5秒 | 24.86% | 3次／16卡     |

- Trace
  B各阶段配置为12×1／1×4→8×1／2×4→6×2／1×4，详细交接及阶段耗时见[证据](../artifacts/elastic-inline-context-shift-20261029/README.md)。Trace
  A另有第二seed19.75%的历史复核，双周会不展示。
- Batch48筛查覆盖9类×2seed×44布局，722组完成、70组显存拒绝；仅作候选空间诊断。[筛查](../artifacts/elastic-inline-batch48-screen-20261029/README.md)。
- 共用重构已通过115项online正常CPU检查及完整pre-commit；本分支16项inline正常CPU检查通过，原46份Python／JSON改动的语义完整保留；8种online负载共7,150题生成结果与修改前一致。完整trace、日志与校准保留仓库父目录`artifacts/`，不随Git同步。

## 后续与边界

- 本分支保留inline实现与双周会结果；后续优先在`lws/feat/elastic-reconfig`推进online，待用户proposal确定修订对象与评估任务。
- 新inline闭环尚未接入外层revision并验证额外收益；原生DP／副本切换GPU验收、TP切换及训练质量待完成。
- H20监控及自动开跑继续暂停；Arena真实模型连通和训练验收独立推进。双周会已结束，幻灯片v1已撤下。
