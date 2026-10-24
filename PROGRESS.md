# AutoPilot 进度

更新：2026-10-24。当前分支：`lws/feat/elastic-reconfig`。
[开发交接](examples/autopilot/DEVELOPMENT.md)／[架构](areal/elastic/README.md)／[研究计划](examples/autopilot/PLAN.md)。

**开发分工**

- `lws/feat/elastic`：共同基线，继续trace与运行验证；`origin`为官方仓库，个人仓库保留为`fork`，两条旧`lws/feat/autopilot*`保留。
- `lws/feat/elastic-reconfig`：本地拓扑切换；公司端由用户另建`lws/feat/elastic-envarena`接入沙箱，完整环境待验收。

**本轮 reconfig**

- Actor DP实现见`7e0c7bea`。第三阶段实现：显式启用后，可受控重建Rollout副本并在原Actor／Rollout
  GPU并集内重新分配卡；CPU服务、session、轨迹与采样序号保留，切换清空KV cache。
- 统一资源事务继承旧generation，旧Actor
  journal只读；先保存／排空，再断开AWEX、确认GPU进程释放、重建两侧、发布原版本，提交后恢复准入。训练身份、PPO步数和消费游标不变。[范围与操作](areal/elastic/RECONFIGURATION.md)。
- 第三阶段18项CPU正常路径及接口兼容性检查通过；GPU／guard使用替身，不能证明真实资源释放。此前PyTorch2.9.1真实Gloo／DTensor／DCP的2→4→2已核对参数、Adam、LR及后续更新。[证据](../artifacts/elastic-reconfig-20261024/stage3-validation.json)。
- 尚无GPU；真实FSDP2／NCCL、SGLang重建、AWEX发布及继续训练待验收。本轮未执行故障注入；实测profile／revision自动决策及热保留部分引擎尚未接入。

**共享基线与下一步**

1. GPU空闲后，验收Actor DP
   2→4→2、Rollout副本4→2→4及两侧GPU转移的完整训练闭环。Megatron／TP／CP／PP／EP切换、完整在线Reference策略仍待实现。
1. `elastic`线继续验证prefill／decode batch采集的GPU兼容性与开销、trace消费／batch关联和Expert的1.7B dense适配。
1. 1.7B正式5轮／batch40／125更新原运行第4次更新反传OOM，仅3次更新、998题正常和2题服务错误，无checkpoint，进程已清理；正式重跑未启动。须覆盖原失败规模与64K完整前后向；`n_mbs=10`及分配器配置尚非已验证修复。
1. 公司端接通Envarena后验收工具执行、stdout／stderr分流、清理及评分，再做16题评测与4更新训练；SWE先验证参考补丁。[2×2步骤](examples/task_workflows/RUN_2X2_8GPU.md)／[原Singularity方式](examples/task_workflows/RUN_SLURM_SINGULARITY.md)。

既有固定布局FSDP／Megatron恢复有GPU证据；共享工作流41项离线及真实3.6.8宿主14项检查通过，其中容器和模型为替身，详见[交接摘要](examples/autopilot/DEVELOPMENT.md)。历史见[迁移前快照](https://github.com/areal-project/AReaL/blob/86323615a3a62fc22c7371b1ceaba526ec1ada93/PROGRESS.md)。
