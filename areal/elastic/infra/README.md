# Elastic 原生 Infra

Infra 协调真实训练运行，接收 Reconfigurator 的目标布局并调整引擎。

```text
infra/
├── runtime.py              # 训练协调、保存恢复与权重发布
├── reconfiguration.py      # 请求、事务、执行、回执及进程清理
├── adapter.py              # 原生观测、物理映射与决策接入
├── activation.py           # Reconfigurator 版本激活与历史恢复
├── state/
│   ├── episode_inbox.py    # 轨迹接收、资格检查与 batch 组装
│   ├── batch_queue.py      # batch 队列、消费锁与游标
│   └── checkpoint_store.py # checkpoint 提交与保留
└── monitor/
    ├── collector.py        # 阶段与 Online 观测、切换计时及测量恢复
    ├── actor.py            # 训练 Worker 端的阶段采集
    ├── history.py          # 已发布更新的有界历史
    ├── identity.py         # 测量适用性身份
    └── trace.py            # 阶段日志与 trace 解析
```

一次重配置依次完成：**（1）接收目标**，Runtime 串行化请求并交给执行器；**（2）检查和规划**，核对训练边界，由共享 Planner
比较布局；**（3）保存与执行**，保存 checkpoint、记录事务并按方案调整引擎；**（4）提交与恢复运行**，核对结果、保存回执并恢复生成和准入。

```text
OnDemandActor.reconfigure(target)                         # 1. 接收目标
└── ReconfigurationExecutor.reconfigure()
    ├── OnDemandActor.control_snapshot()                  # 2. 检查和规划
    ├── ConfigApplication.preflight()
    ├── OnDemandActor.plan_reconfiguration()
    │   └── ReconfigurationPlanner.plan()
    ├── OnDemandActor.save_checkpoint()                   # 3. 保存与执行
    ├── ConfigApplication.apply()
    │   ├── atomic_json()                                # 先记录调整意图
    │   ├── ReconfigurationExecutor.apply()
    │   │   └── ReconfigurationExecutor.execute()         # 引擎调整及权重发布
    │   ├── ConfigApplication._verify()                   # 4. 提交与恢复运行
    │   └── ConfigApplication._finish()
    ├── RolloutControllerV2.continue_generation()
    └── OnlineInbox.set_admission_open(True)
```

`ReconfigurationExecutor` 组织完整执行过程；`ConfigApplication` 管理持久化事务，两者位于同一文件。进程识别、清理及 GPU
释放确认也在该文件，供 Actor 和 Rollout 共用。

State 保存训练接续所需的数据和消费进度；Monitor 提供运行表现与决策观测。`ElasticMonitor.online_observations` 按需提供原生
Online 观测，保留现有缺测口径。

`activation.py` 激活已接受的 Reconfigurator 组件版本；训练布局调整由 `reconfiguration.py` 执行。Checkpoint
对照检查位于验收工具 `examples/task_workflows/checkpoint_audit.py`。

[完整重配置链路](../RECONFIGURATION.md) · [项目进度](../../../PROGRESS.md)
