# 原生 Actor／Rollout 重配置

## 1. 调整接口

统一入口为 `OnDemandActor.reconfigure(target=ExecutionLayout(...), ...)`。 `ExecutionLayout`
描述 Actor DP、Actor GPU 池和 Rollout GPU 布局。原生 Online adapter
将目标配置映射为布局后调用同一入口；手动请求使用同一配置事务。

`OnDemandActor.plan_reconfiguration(target)` 提供无引擎操作的方案预览。
`ReconfigurationPlanner.plan()` 比较当前／目标布局和 Actor 驻留状态，返回包含有序步骤的 `ReconfigurationPlan`。
`ReconfigurationExecutor.reconfigure()` 组织保存、事务、引擎调整和恢复运行；`ConfigApplication` 持久化方案和回执。
调整方案与完成回执一起保存到 `resource-reconfiguration/state.json`。 该磁盘路径和日志身份沿用已有格式，运行目录可继续使用。

## 2. 执行过程

一次重配置分为七步：**（1）检查请求与执行边界**，核对目标布局、配置代次和权重版本；**（2）生成调整方案**，比较当前布局、目标布局与 Actor
驻留状态，确定保留和重建的引擎及执行顺序；**（3）保存状态并记录意图**，保存完整
checkpoint（已有最新保存时复用），将方案写入配置事务；**（4）按方案调整引擎**，完成必要的调用排空、资源释放、目标部署及训练状态恢复；**（5）重建权重连接并发布权重**；**（6）核对并提交结果**，保存实际布局、调整方案和完成回执；**（7）恢复运行**，开放生成和请求准入。

```text
OnDemandActor.reconfigure(target, ...)
└── ReconfigurationExecutor.reconfigure()
    ├── OnDemandActor.validate_execution_layout(target)   # 1. 检查请求与执行边界
    ├── OnDemandActor.control_snapshot()
    ├── ConfigApplication.preflight(request)
    │
    ├── OnDemandActor.plan_reconfiguration(target)        # 2. 生成调整方案
    │   └── ReconfigurationPlanner.plan(current, target, actor_running=...)
    │       └── ReconfigurationPlan(source, target, actor_running, steps)
    │
    ├── OnDemandActor.save_checkpoint(...)                # 3. 保存状态并记录意图
    │   └── CheckpointStore.commit(...)
    │       └── GatewayTrainController.full_checkpoint("save") # 经保存回调
    │
    ├── ConfigApplication.apply(request, plan=plan)
    │   ├── atomic_json(...)                             # 持久化源配置、目标及方案
    │   ├── ReconfigurationExecutor.apply(target, plan=plan)
    │   │   └── ReconfigurationExecutor.execute(plan)     # 4. 按方案调整引擎
    │   │       ├── [重建 Rollout] RolloutControllerV2.drain_model_calls()
    │   │       ├── OnlineInbox.set_admission_open(False)
    │   │       ├── [重建 Actor] OnDemandActor._release()
    │   │       ├── [保留 Actor、重建 Rollout] GatewayTrainController.disconnect_engine()
    │   │       ├── [重建 Rollout] RolloutControllerV2.stop_inference_replicas()
    │   │       ├── OnDemandActor._set_execution_layout(target)
    │   │       ├── [重建 Rollout] RolloutControllerV2.rebuild_inference_replicas(...)
    │   │       ├── ReconfiguratorAdapter.rebind_actor_layout(...)
    │   │       ├── [需要启动 Actor] OnDemandActor.ensure_actor(...)
    │   │       │   └── GatewayTrainController.full_checkpoint("load")
    │   │       ├── GatewayTrainController.connect_engine(...) # 5. 连接并发布权重
    │   │       ├── [保留 Rollout] RolloutControllerV2.pause()
    │   │       └── OnDemandActor._publish()
    │   ├── ConfigApplication._verify(...)               # 6. 核对并提交结果
    │   └── ConfigApplication._finish(...)
    │
    ├── RolloutControllerV2.continue_generation()         # 7. 恢复运行
    ├── OnlineInbox.set_admission_open(True)
    ├── RolloutControllerV2.open_model_calls()／resume()
    └── return receipt
```

同一入口根据差异生成不同方案：Actor DP 或 Actor GPU 池变化时重建 Actor；Rollout GPU 布局变化时重建
Rollout；未变化的一侧保留引擎。两侧均变化时，先释放旧引擎，再按目标布局启动。目标布局相同时返回空方案。

## 3. 运行入口与支持范围

当前支持 local／v2、纯 FSDP Actor DP、SGLang TP1／PP1 副本及原 GPU 池并集内的资源分配。Actor DP 整除有效训练
batch；GPU 设备池独占且两侧互不重叠。

只改变 Actor DP 时，传入保持两侧设备池不变的完整目标布局。改变 GPU 池或 Rollout 布局时，在首次启动前启用：

```bash
export AREAL_ELASTIC_RESOURCE_RECONFIGURATION=1
```

本地命令使用统一调整入口：

```bash
python -m areal.elastic.infra.reconfiguration --state-root "$STATE_ROOT" status
python -m areal.elastic.infra.reconfiguration --state-root "$STATE_ROOT" request \
  --request-id resources-001 --world-size 2 \
  --actor-devices 0 1 --rollout-devices 2 3 4 5 6 7 \
  --expected-generation 0 --expected-weights-version 1
```

Actor-only 保留 Rollout；Rollout-only 在 Actor 驻留时保留其 workers／训练状态，先断开旧权重连接再替换推理引擎；Actor
已释放时按需恢复以发布当前权重。 服务端 CPU router／gateway／proxy／session 和轨迹缓冲跨引擎调整保留。改变 DP 使用完整 DCP 重分片与显式
RNG 恢复计划。

原 DP 专用入口和 executor 已移除。旧 `actor-reconfiguration`
配置日志在打开运行目录时导入统一布局日志，保留代次及回执，旧文件不改写。已有资源日志继续使用。 Actor／Rollout 未变化的一侧按方案保留；当前基线仍整体重建发生变化的
Rollout 侧。切换不改变 Slurm 作业预留。

## Online 决策接入

`ReconfiguratorAdapter` 通过共享 factory 装配 `basic` 或 `reference`；`NativePlacements`
过滤执行器不支持的候选并映射物理布局，`NativeControlEnvironment.request_handoff()` 将共享版本的目标交给同一个
`reconfigure()`。旧的独立 `NativeJointPlanner` 已移除。

```text
OnDemandActor._poll_batch()
└── ReconfiguratorAdapter.poll()
    └── NativeControlEnvironment.poll()
        └── OnlineReconfigurator.poll()              # Reference：决策及回调统一
            ├── OnlineReconfigurator.decide()         # 预测、搜索与选择
            └── NativeControlEnvironment.request_handoff(target)
                ├── NativePlacements.resolve(target) # 映射到实际 GPU
                └── OnDemandActor.reconfigure(...)   # 进入上面的统一执行事务
```

原生接入复用已有观测，真实 TTFT／最大 TBT 缺测另行记录。原生释放后部署与模拟热交接的候选预测约束还需完成核对，状态见根目录 `PROGRESS.md`。

## 验证

CPU coordinator 正常路径覆盖
Actor-only／Rollout-only／联合调整、游标保留、发布及后续训练；`tests/test_elastic_native_online.py` 覆盖共享
Online 版本和接受后的组件激活。这些检查使用替身 workers。

真实 FSDP2／NCCL、SGLang 回收重启、AWEX 发布和后续 PPO 更新待 GPU
验收。准备入口与计时定义见[八卡验收](../../examples/task_workflows/RECONFIGURATION_ACCEPTANCE.md)，`--prepare-only`
不启动服务。当前 H20 自动开跑保持暂停。
