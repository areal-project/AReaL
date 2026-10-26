# 本地 H20 训练与 trace 采集计划

2026-10-25。Qwen3-4B-Instruct-2507＋Hermes／DAPO真实16题／4更新通过：32次推理、5次工具、16条轨迹全部消费并发布，完整checkpoint游标为4，相关进程正常退出。[运行审计](../../../artifacts/h20-training-trace-20261025-f28c1811/smoke-001/audit.json)／[Linux正常检查8项](../../../artifacts/h20-training-trace-20261025-f28c1811/validation.json)。

**2026-10-26按用户要求暂停监控及自动开跑，等待明确恢复。** Heartbeat `h20-4b`
已设为PAUSED；最后09:36检查8卡仍被占用、数据盘约85G可用，64K边界、采集开销对照与持续训练均未启动。[监控记录](../../../artifacts/h20-training-trace-20261025-f28c1811/monitor/state.json)。原始trace及模型状态保留H20隔离目录`/workspace/h20-training-trace-20261025-f28c1811/smoke-001/`；本地保留审计摘要与源码快照。

本轮采用[4B采集配方](configs/dapo_4b_trace.yaml)：Megatron
TP4＋SGLang四个TP1副本，逻辑batch4、64K上下文、每次最多4096新token。使用on_demand协调路径保留持久消费身份，idle=86400秒，本轮只有一个Actor代次且无布局切换。TP4在线发布已通过；配置64K上下文不代表64K完整前后向已通过。

## 已确认的后续规模（待实测）

- 目标约
  **10,000道不同训练题**，扩展现有1,000题清单，保持已有评估集隔离；核验固定revision的可用去重题池后冻结清单、顺序和seed。多轮重复采样另计episode。
- **逻辑batch40、客户端并发64**；SGLang起初保持每副本最多8个运行请求，四副本合计上限32。保持staleness=2，对应120个未发布／未拒绝episode的流水线额度；执行时一并核对服务端并发和准入限制。
- 保持单题单采样、`ppo_n_minibatches=1`及现有PPO其余设置；全部有效消费时，10,000题对应250次更新。失败、拒绝、未消费和尾部剩余单列。batch40作为新的固定训练基线，与人工workload的batch40口径对齐。
- 保持Actor
  TP4＋四个TP1推理副本。微批按独立token预算切分；整批padding、输入及中间张量仍可能增加显存，目标batch的真实长短混合轨迹须实测。并发128和更高引擎请求上限作为后续容量候选，由吞吐、KV、队列、显存及过期比例决定。
- 用户明确恢复后，沿用每20分钟核对8卡计算进程与显存；发现空闲后间隔至少60秒复核，启动前立即重查，利用率为0不能单独作为空闲依据。重查trace／checkpoint磁盘峰值预算，完成目标参数短测、64K容量及采集开销检查，再冻结独立run配置并持续采集。现有batch4配方保留为小规模验收身份；运行期间跟进同一run，完成后停用监控。

## 已核对的依据

- [人工 workload](../autopilot/elastic/WORKLOAD.md)
  的三份冻结文件摘要与实际计数一致：每份687题、2058次推理、1371次工具调用。它们提供到达、调用依赖、长度和工具工作量，训练还需要真实文本、token／logprob和奖励。
- [现有工作流](README.md#%E8%AF%84%E5%88%86%E4%B8%8E%E8%AE%B0%E5%BD%95) 已有
  episode／推理调用／工具计时与依赖、usage长度、奖励回执、轨迹导出工作量；[SGLang采集](../../docs/zh/best_practices/perf_profiling.md#sglang-batch-%E8%AF%B7%E6%B1%82%E6%95%B0%E4%B8%8E-token-%E6%97%B6%E5%BA%8F)
  已有引擎batch、CUDA事件与KV记录。
- 4B的Megatron／SGLang组件校准、AWEX发布及16题／4更新闭环已有实测，扩大规模待验收；[旧1.7B配方](TRAIN_8H20.md)
  正式运行第4次更新OOM。
- [历史H20只读预检](../../../artifacts/h20-training-trace-preflight-20261025/preflight.json)：当时8卡空闲、数据盘98%已用／约138G可用，200／1000／200题清单存在。旧`/workspace/AReaL`缺少入口与observer，本轮已使用隔离快照完成小规模训练；恢复时须重新核对资源和磁盘，历史空闲与空间不作为当前可用量。

## 必须记录的指标

| 层次        | 字段与口径                                                                                                                       | 现有能力／待补                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 外部任务    | run／episode／task ID、计划到达、实际就绪、客户端发出、admission、完成／评分／交接时间                                           | 已有部分时间；现client整份题单共用一个`available_at`，需明确题单模式并补逐题到达和准入时间 |
| 推理与依赖  | 节点ID、前驱、轮次；初始及每轮完整输入P、实际输出O、缓存token、起止／结束原因                                                    | 已有DAG、usage长度和调用墙钟；补跨层请求ID映射，缺失缓存值保留null                         |
| 工具        | 工具类型／池、前驱、queued／started／finished、服务秒数D、返回token数B、返回状态／截断                                           | 已有锁前后计时和服务秒数；补模型实际收到的工具返回长度，保留容器冷启动边界与资源配置       |
| Rollout引擎 | 请求→batch关联、prefill／decode／mixed、query／past／完整上下文、attention pairs、KV占用／容量、CUDA事件、rank／布局             | 复用batch observer；验证RID关联与文件完整性，mixed共享区间不重复累加                       |
| 训练工作量  | episode→session／incarnation／trajectory→export→batch／step；实际sequence数／长度、attention／loss token、接受／拒绝／丢弃及原因 | 已有export统计；补resident实际消费关联，逐项核对导出、消费、更新与发布                     |
| Actor执行   | 后端／DP／TP、每rank微批／padding、logp／advantage／PPO时间、optimizer更新成功、训练／serving版本与staleness、显存峰值           | 复用Trainer计时／perf trace；按需迁移校准脚本的微批和显存观测，独立脚本通过状态不继承      |
| 资源与质量  | Actor／Rollout驻留区间、启动／保存／恢复／发布、GPU秒及预留卡时、队列／等待、奖励／截断／错误、checkpoint身份                    | 复用生命周期事件并补resident记录；分别报告更新数、episode数、sequence uses及有效loss token |

所有原始记录带`run_id`、schema版本、进程／host／rank和配置身份。请求映射至少贯通episode、调用节点、completion／interaction、后端RID和export；缺少关联显式列为未匹配。

时间戳同时保存墙钟与单调时钟基准，耗时用单调时钟；跨进程／节点保留对齐误差。CPU
scope、HTTP调用区间和CUDA事件各用自己的口径，`timeperf/rollout`表示取batch等待与准备。Actor阶段按同一边界取各rank关键路径，避免累加嵌套scope或重复rank。

工具返回分别记录原始输出及进入下一次模型输入的实际内容／长度，冻结tokenizer与chat
template；后续完整输入直接实测，另记消息开销、压缩或截断。实际工具fan-out保留真实依赖，不强制满足合成trace的每轮一工具或固定16-token消息开销。真实数据的类型标签另存分类规则和版本。

## 实现顺序

1. 在`client.py`记录逐题就绪、发出与admission边界；若使用受控到达表，固定题单与时间表映射，产生到达不等待前题完成，容量不足的排队可追溯。
1. 在`trace.py`及harness／environment现有观测点补工具返回长度、请求身份与时间边界，保留完整性标志。
1. 在轨迹导出及Trainer实际取batch／更新／发布边界补关联记录；on-demand复用inbox／batch
   queue身份，resident单独验证。`export_id`去重，未消费和尾部不足batch的任务单列。
1. 配置`AREAL_ROLLOUT_TRACE_DIR`并开启`perf_tracer`／session记录；确认环境传到SGLang和DataProxy。独立SGLang使用AReaL启动入口。全程轻量事件，少数代表step才做深度profiling，热路径不新增强制GPU同步。

沿用原生PPOTrainer和现有采集模块；本计划不需要新增依赖或修改核心配置结构、launcher、scheduler。

## 执行阶段与通过条件

1. **冻结与预检**：记录commit及未提交源码摘要、模型revision、题单／顺序／seed、镜像digest、包版本、采样／工具预算、后端／布局／batch／导出语义／staleness／缓存及保存策略。只读核验8×H20占用、显存、磁盘与工具容器，再确定设备分配。
1. **采集正常路径检查**：集中验证到达／准入时间、工具长度、依赖、ID关联、计数去重及导出／消费对应；暂不做故障注入。
1. **目标规模验收**：16题／4更新闭环已通过。恢复后短测batch40／并发64的真实长短混合轨迹，核对完整前后向、显存、更新与发布，并独立验收64K容量。用同一小负载比较采集开／关的吞吐与时长，再冻结正式配置。
1. **首组持续采集**：按上节确认的规模运行一组固定布局训练。报告计划／实际／完成／消费／剩余任务和有效训练量，统一收尾截止，完整保留尾部。
1. **后续到达实验**：首组通过后再考虑复用单峰、周期、短突发的30分钟到达表；与真实题目固定映射，输出和工具行为由真实执行决定。首轮不承诺六组收益对照、在线重配置或质量提升。

模型选择影响后端：对齐现有4B成本模型时，优先考虑固定Megatron／SGLang布局并独立验收；使用已有FSDP4＋SGLang1起始配方则另建测量身份。当前原生联合重配置仅支持纯FSDP
DP＋SGLang
TP1／PP1，见[边界](../../areal/elastic/RECONFIGURATION.md)。采用旧1.7B配方时，必须先覆盖原失败规模及声明支持的长序列，不能直接重启正式batch40训练。

## 交付与验收

每个run独立保存manifest／完整脱敏配置、现有`episodes.jsonl`、Rollout
batch／export记录、Actor消费／阶段记录、资源事件、checkpoint及summary。结果目录放在忽略的运行产物区，文档只链接证据。

本轮16题的长度、工具返回token、completion→后端RID→GPU
batch、export→持久episode→更新／发布／checkpoint检查通过；16条均published，无未匹配ID。保存与发布版本均为4，四个rank的optimizer更新和RNG／LR状态齐全。工具服务时长包含本次容器初始化；初始化时长未单独分离。当前尚未完成分布／时间线图、采集开销对照、64K边界与持续训练，恢复后继续核验。
