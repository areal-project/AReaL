# AReaL Elastic

Elastic 是 AReaL 的执行与规划能力。唯一训练执行路径仍是 `PPOTrainer → OnDemandActor`；本包不依赖 AutoPilot。

开启 `actor_lifecycle.mode=on_demand` 即默认选择 `actor_lifecycle.policy=basic`，当前先接 Basic
的固定超时策略；Runtime 不再另写一份空闲超时判断。完整 batch 就绪才训练，默认空闲 60s 后请求释放；最短驻留
120s、权重发布确认、释放前完整保存和进程清理由 Runtime 约束。`idle_timeout=0` 表示不等待空闲超时。
`actor_lifecycle.policy=reference` 预留给独立参考策略，目前在启动服务前明确报未实现；不会退回 Basic。 完整 Basic
模型决策仍需后续接入为默认行为；显式实测 profile 已由 AReaL 自身装配和调用。

```text
elastic/
  runtime.py                  Actor 拉起、发布、周期保存、释放与故障恢复
  contracts.py                观测、动作、负载与公共契约
  identity.py                 测量适用性身份，与训练恢复／bundle 验收分开
  state/
    checkpoint_store.py       checkpoint 原子提交、latest 与保留策略
    batch_queue.py            消费锁、batch 顺序与恢复游标
    episode_inbox.py          episode 接收、staleness、组 batch 与发布确认
  control/
    boundary.py               控制接口与边界快照
    policy_hooks.py           可选修订轮询与激活恢复回调
    config_application.py     配置应用事务与固定布局 NativeConfigExecutor
    actor_reconfiguration.py  显式 Actor DP 请求、事务适配与本地命令入口
    policy_activation.py      策略激活事务
    processes.py              进程所有权与 GPU 释放确认
  monitor/
    collector.py              默认 ElasticMonitor：观测、采集、历史与测量恢复
    history.py                有界历史、去重与回退
    trace.py                  已有日志／trace 解析
  policy/
    adapter.py                默认超时与实测 Basic 的统一装配／决策入口
    basic/                    Basic cost model、simulator、optimizer 与 policy
      profile.py              实测 workflow 装配与固定布局模拟边界
      serial_planning.py      保留的离线严格串行模型与规划
    reference/                独立 Expert 参考模型／simulator；不参与 Basic 实现
```

原生执行器通过公开 `NativeControlPort` 提供串行控制边界和 admission fence。 AutoPilot 配置应用仍只接受同布局、同 GPU
池的身份确认。Runtime 另提供[显式 Actor DP 重配置](RECONFIGURATION.md)，保持原预留 GPU 池和 Rollout 布局不变；实际 GPU
验收状态见 `PROGRESS.md`。 训练后先发布权重，再按 `recover.freq_*` 周期保存完整 checkpoint／游标； 评估、清理和恢复 Rollout
后判断是否释放，释放前补齐最新恢复状态。意外故障回到最近 checkpoint 并重放未保存 batch。

发布失败时保持准入关闭；即使 serving 版本号未变，也必须重建权重连接并完整覆盖恢复权重，再确认后开放生成。 SGLang 分离部署的专用 AWEX NCCL
组有独立截止与 abort；不会中止模型默认通信组。旧操作和 watchdog 未退出、 或 abort 失败时保留 fence，不允许新代次。此机制依赖串行
scheduler RPC 与 Torch 的分组 abort 接口；不承诺处理任意 CUDA 故障。

Runtime 仅依赖 `control/boundary.py` 的接口与快照；`NativeConfigExecutor` 归属
`control/config_application.py`，边界模块不导入配置应用事务。`policy/basic/policy.py` 负责决策，
`control/policy_activation.py` 负责可靠地切换策略版本。

普通策略统一由 `policy/adapter.py::PolicyAdapter` 装配，Runtime 始终调用
`policy_adapter.decide()`；默认超时和显式实测 Basic 不依赖 AutoPilot。AutoPilot 的
`domains/elastic/native.py::RevisionAdapter` 扩展该适配器，保留修订和激活；观测与实测历史统一由 Monitor 提供。

执行进度使用三个明确名称：`applied_version` 是已执行 PPO 的版本，`published_version` 是所有 Rollout
确认的版本，`checkpoint_version` 是最新完整保存的版本。`publish_update()` 只执行前两项， 周期保存／释放前保存负责推进第三项。共享规划
schema 的 `committed` 仍指“更新已完成并发布”， `committed_samples`／`committed_tokens` 指相应工作量；SQLite
和共享规划 schema 保持不变。

`trainer/rl_trainer.py` 保留 PPO 与每轮工作顺序，通过 `elastic/runtime.py` 接入按需生命周期。
`v2/training_service/checkpoint.py` 执行完整状态读写，`runtime_state.py` 管理各 rank 的 LR／RNG；
`elastic/state/checkpoint_store.py` 只管理提交存储。模型文件摘要和原子 JSON 工具统一位于 `utils/atomic.py`。
旧导出校验与 DP 重布局辅助代码已移入本地归档，不作为新路径的验收证据。

`on_demand` 默认创建 `ElasticMonitor`。`collector.py` 汇总 Runtime／inbox 状态与 batch 负载， 记录决策前的
observation／历史／动作，以及实际阶段耗时、版本、配置和 Actor generation。 `finish_update()` 完成、Rollout
恢复后才写入完成记录；恢复中的加载／发布单独记生命周期事件。 历史默认有界，policy 读取内存窗口，revision 冻结相同数据来源的完成记录；事件顺序在恢复后保留。
当前默认 Basic 仍是超时策略，开始采集不会自动启用模型搜索。

累计样本与有效 token 由 Runtime 维护，随完整 checkpoint 的 `data_progress`／消费游标保存；旧计数缺失时从已校验的队列前缀重建。
`monitor-state.json` 在完整训练状态成功提交后单独写入；`monitor-pending.json` 补充阶段计时，同版本完成快照优先。 Monitor
I/O／数据格式故障会清空可用测量窗口、回到 Basic 超时策略并暂停 revision，直到恢复时重新打开测量；不回滚成功发布的 PPO。
模型、optimizer、rank 状态和队列保存失败仍严格报错，不能按计划释放。

三个身份各管一件事：训练恢复使用配置／模型与游标兼容性；`identity.py` 使用物理配置和相关实现的语义摘要判断测量适用性， 注释、格式和离线 trace
解析不影响它；接受 bundle 继续逐文件严格验证源码摘要。 测量历史位于 `monitor/<measurement_id>/<series_id>/`。Monitor
状态缺失／旧格式／测量身份改变时，兼容的训练 checkpoint 仍可恢复， 新建测量序列并保留训练计数，不把旧标签混入新窗口。已有 revision
根目录绑定了旧序列时暂停修订，需换新 revision 根目录；bundle 仍须重新通过验收。

数据直接来自 Runtime／inbox／发布确认、轨迹 mask 和 CPU 单调时钟；不读取 W&B。`trace.py` 按需解析已有日志。
阶段计时是调用边界耗时，缺测保留未知，不引入 GPU 同步，也不代表 kernel 时间或完整异步收益。

AutoPilot 的通用修订机制位于 [revision](../autopilot/revision/README.md)， Elastic evaluator、候选
worker、bundle 验证和原生策略桥接位于 [domains/elastic](../autopilot/domains/elastic/README.md)。接受
bundle 不能仅修改路径后继续视为有效；训练 checkpoint 另按恢复兼容性校验，Expert 校准须另做兼容审计。
