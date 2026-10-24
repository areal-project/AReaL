# AutoPilot 进度

更新：2026-10-24。当前分支：`lws/feat/elastic-reconfig`。
[开发交接](examples/autopilot/DEVELOPMENT.md)／[架构](areal/elastic/README.md)／[研究计划](examples/autopilot/PLAN.md)。

**开发分工**

- `lws/feat/elastic`：共同基线，继续trace与运行验证；`origin`为官方仓库，个人仓库保留为`fork`，两条旧`lws/feat/autopilot*`保留。
- `lws/feat/elastic-reconfig`：本地拓扑切换；公司端由用户另建`lws/feat/elastic-envarena`接入沙箱，完整环境待验收。

**本轮 reconfig**

- 已实现显式纯FSDP Actor
  DP切换：保存完整源状态→重建Actor→DCP恢复→重新连接并发布；Rollout布局与GPU预留池保持固定。原始训练身份、PPO步数和任务消费游标不变。
- 跨DP使用显式恢复计划及确定性RNG派生；同布局仍严格恢复。持久事务在首次创建Actor前选择活动布局；每个布局重新建立测量身份，暂不接实测profile／revision自动决策。
- 19项正常路径检查通过、2项CUDA检查因无GPU跳过；其中CPU协调器覆盖切换／重启／命令入口，PyTorch2.9.1真实Gloo／DTensor／DCP完成2→4→2，参数、Adam状态、学习率及后续更新核对通过。[本地证据](../artifacts/elastic-reconfig-20261024/)。
- 尚未使用GPU；真实FSDP2／NCCL、Actor→SGLang发布及继续训练待验收，未执行故障注入。[范围与操作](areal/elastic/RECONFIGURATION.md)。

**共享基线与下一步**

1. GPU空闲后，先验收reconfig的2→4→2完整闭环；再推进Rollout副本增删及两侧GPU重新分配。Megatron／TP／CP／PP／EP切换、完整在线Reference策略仍待实现。
1. `elastic`线继续验证prefill／decode batch采集的GPU兼容性与开销、trace消费／batch关联和Expert的1.7B dense适配。
1. 1.7B正式5轮／batch40／125更新原运行第4次更新反传OOM，仅3次更新、998题正常和2题服务错误，无checkpoint，进程已清理；正式重跑未启动。须覆盖原失败规模与64K完整前后向；`n_mbs=10`及分配器配置尚非已验证修复。
1. 公司端接通Envarena后验收工具执行、stdout／stderr分流、清理及评分，再做16题评测与4更新训练；SWE先验证参考补丁。[2×2步骤](examples/task_workflows/RUN_2X2_8GPU.md)／[原Singularity方式](examples/task_workflows/RUN_SLURM_SINGULARITY.md)。

既有固定布局FSDP／Megatron恢复有GPU证据；共享工作流41项离线及真实3.6.8宿主14项检查通过，其中容器和模型为替身，详见[交接摘要](examples/autopilot/DEVELOPMENT.md)。历史见[迁移前快照](https://github.com/areal-project/AReaL/blob/86323615a3a62fc22c7371b1ceaba526ec1ada93/PROGRESS.md)。
