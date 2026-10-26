# 原生 Actor／Rollout 重配置

范围：local／v2／`on_demand` 生命周期入口、纯 FSDP DP、SGLang TP1／PP1、默认超时 PolicyAdapter。目标 Actor DP
必须整除有效训练 batch。此功能不改变 Slurm 作业预留；默认使用显式请求，联合规划需另行接入。

## Actor DP

原有入口保留：仅改变原 `actor_devices` 池内的 DP 数量，Rollout 不变。初始 DP 可小于池大小；具体 rank 分配由 scheduler 决定。

```bash
python -m areal.elastic.control.actor_reconfiguration --state-root "$STATE_ROOT" status
python -m areal.elastic.control.actor_reconfiguration --state-root "$STATE_ROOT" request \
  --request-id dp-grow-001 --world-size 4 \
  --expected-generation 0 --expected-weights-version 1
```

## Rollout 副本与 GPU 重新分配

首次启动训练前显式启用：

```bash
export AREAL_ELASTIC_RESOURCE_RECONFIGURATION=1
# 使用现有 PPOTrainer 训练入口和原始配置启动。
```

该模式为原 Actor／Rollout GPU 池的并集保留固定数量的 CPU guard／data proxy。GPU 引擎与这些服务分离；默认固定布局启动不受影响。重启已有
resource journal 的运行会自动恢复此模式，仍使用原始启动配置，不改其身份哈希。

先完成至少一次训练更新，再提交完整目标。例如原预留池为 Actor 0–3、Rollout 4–7，改为 Actor 0–1、Rollout 2–7：

```bash
python -m areal.elastic.control.resource_reconfiguration --state-root "$STATE_ROOT" status
python -m areal.elastic.control.resource_reconfiguration --state-root "$STATE_ROOT" request \
  --request-id resources-001 --world-size 2 \
  --actor-devices 0 1 --rollout-devices 2 3 4 5 6 7 \
  --expected-generation 0 --expected-weights-version 1
```

两侧设备必须非空、无重复、互不重叠且属于原并集。`actor_devices` 是独占预留池，可大于 DP；重建后 scheduler 从列表首项按顺序分配各
rank。`rollout_devices` 每张卡启动一个 TP1 引擎。允许暂不使用并集中的部分 GPU，但它们仍在作业预留内。

执行顺序：保存完整源 checkpoint → 写持久意图 → 并行关闭各 proxy 的模型调用准入并自然排空 → 关闭训练准入 → 断开 AWEX／释放 Actor →
确认旧 Rollout GPU 进程及其后代停止 → 启动目标 Rollout → 重绑保留的 proxy → 恢复 Actor → 重建 AWEX 并完整发布当前权重 →
提交回执 → 恢复准入。

- 这是受控重建所有 GPU 引擎；切换期间新模型调用等待，已有调用（包括流式响应）完成后才停机。KV cache 会清空。
- CPU router／gateway／proxy／callback、session 凭据、轨迹缓存和逐请求采样序号保留；工具仍可执行，新推理等待。副本数与保留 proxy
  数量独立，暂停／恢复每个推理后端只调用一次。
- 不推进 PPO step、权重版本或消费游标。首次 resource 请求将配置权威一次性交给
  `resource-reconfiguration/state.json`，沿用 Actor journal 的 generation；旧
  journal／回执只读，此后新请求统一使用完整资源入口。
- `status` 给出当前配置 generation；`expected-weights-version` 是已完成的训练更新数。协调器在取下一个 batch
  前处理请求，结果写入对应目录 `response.json`。过期请求核对后用新 ID 重提；相同 ID／参数返回原回执。

## 恢复与限制

- checkpoint placement v2 记录原并集、源 Actor 池、源 Rollout 设备及 DP。仍接受旧
  sidecar；原模型／optimizer／训练配置的恢复身份保持不变。
- 仅物理卡变化、DP 不变时恢复原 rank 的 LR／RNG；DP 变化使用显式 DCP 重分片及确定性 RNG 派生，保留固定题单顺序，不承诺跨 DP 位级一致。
- 已提交或待恢复的源配置在首次创建推理引擎前选定。未确认的 GPU 进程所有权保留日志并阻止重启，不把 RPC 超时当作进程退出。恢复不完整时保持准入关闭。
- 每个布局重建测量身份。实测 profile／revision 自动决策、Megatron／TP／CP／PP／EP 调整、保留部分引擎热运行尚不支持。

## 显式接入联合规划

`policy/native_joint.py::NativeJointPlanner` 将现有 `JointPolicy`／`JointOptimizer` 的选择交给
`reconfigure_resources()`，默认超时生命周期策略保留。调用者提供带来源的预测窗口、模型版本，以及完整
`JointPlacement(config, layout)` 候选映射；只允许改变纯 DP 和 TP1 副本数，其余模型／采样／KV／batch 配置保持一致。 Actor
lease 必须恰好等于 DP，以免预测遗漏独占空闲卡成本。源布局须在映射中。

```python
planner = NativeJointPlanner(
    joint_policy, placements,
    model_id="my-profile-v1", forecast_source="causal-history-v1",
    forecast=forecast_from_history,
)
runtime.resource_planner = planner
# 或在受控驱动中显式调用 planner.step(runtime, scenario)。
```

原生取下一批前调用 `poll()`，每个已完成版本至多接受一个预测窗口；首个更新后，只有 Actor 仍驻留、已发布、训练 episode credits
为零时才运行。先排空 proxy 模型调用、再复查 credits，最后关闭新 episode 准入；抵达中的请求仍正常获得 lease，并使本次规划延后。
保持配置或没有可行计划也恢复模型调用；有在途／排队训练工作时等待，不丢弃工作或强行模拟为空。 窗口的版本、单样本 group、batch 和 staleness
与实际状态核对，先过滤执行器不支持的候选，再搜索；配置 journal generation 用于 CAS。 预测与原生回执分别保存在
`state_root/joint-planner/`，预测不写入 Monitor 实测历史。事务回执标记 `native-joint-*` 来源。

CPU 合成链路可复跑：

```bash
python -m pytest -q tests/test_elastic_actor_reconfiguration.py::test_native_joint_planner_synthetic_trace_roundtrip
```

该检查手工构造三段双轮请求，保留 prompt／decode 长度和工具等待：Rollout 重负载→同负载→Actor 重负载，搜索得到
4+1→2+3→保持→4+1；通过真实 CPU runtime／SQLite／配置事务继续完成更新。pytest 临时目录中的 `synthetic-trace.json`
包含输入、各候选预测、模拟事件时间和实际控制回执。模型吞吐／切换成本及 GPU workers 均为合成或替身， CPU 事务耗时不代表 GPU 耗时，也没有执行数值
PPO；这不是性能收益或真实训练验收。

当前联合 simulator 仍按 batch 聚合推理，不重建逐 token continuous batching；其初态要求无在途工作。
`checkpoint_every` 等窗口参数是预测假设，不修改真实训练的保存周期。真实在线预测器、忙碌服务初态和实测切换成本尚待接入；使用专用服务，不能同时混入独立
eval 流量。

## 验证

离线联合搜索可直接复用冻结 Expert：`areal/autopilot/domains/elastic/expert_planning.py` 在外部组合
`JointOptimizer` 与 `ExpertSimulator`，Basic／Reference 内部无新增相互依赖。

[统计 workload](../../examples/autopilot/elastic/WORKLOAD.md) 将 Poisson
外部到达、类型混合和请求工作量冻结为独立 JSONL； 硬件、部署、工具池容量与模拟结果由外部实验保存。

```python
model = ExpertJointModel(expert_simulator, bindings)
# window: ExpertJointWindow，包含原始 Scenario、PlanStep 模板及预留设备。
search = JointOptimizer(candidates).optimize(window, model)
detail = model.results[model.query_id(window, search.config)]
```

显式绑定完整配置，核对 Actor profile、模型／硬件／runtime 身份与布局；每候选只替换计划配置，保留请求 DAG、长度、到达、工具池、缓存与初态。
固定成功工作窗口的总任务集合／样本／token／更新数一致；逐 batch 归属由 Expert 实际选择。`unsupported`
不回退或补零；完整请求事件、模型／NumPy 身份与来源单独保留。该离线接口不直接启用原生重配置。

```bash
python -m pytest -q tests/test_elastic_policy.py::test_basic_joint_optimizer_uses_frozen_expert_request_simulator
```

CPU 例使用 Catalog 的 Qwen3-30B-A3B／H20 分析模型、FSDP BF16 Adam/Kahan＋offload；派生 TP1／DP
布局，合成12题／24请求／3更新，比较4+4与2+6。 Host 吞吐沿用已有 prior，advantage 与转换耗时显式假设；无重拟合、无 GPU
实测，不代表1.7B当前 Adam 配方的预测精度。 pytest 临时目录的 `expert-joint-trace.json` 保留输入、profile
来源、候选预测及完整 Expert events。

CPU 正常路径覆盖资源往返切换、重启、同 DP 移卡、session／采样序号保留、流式与直接模型调用排空、4→2→4 推理副本、唯一后端暂停和本地 GPU 池更新。GPU
引擎及 guard HTTP 使用替身，不能证明真实 GPU 回收或 AWEX 成功。此前真实 Gloo／DTensor／DCP 的 2→4→2 状态恢复检查保留。

真实 FSDP2／NCCL、SGLang 进程回收与重启、AWEX 发布及后续 PPO 更新待空闲 GPU 验收；本轮未运行故障注入。结果与证据见根目录
`PROGRESS.md`。

三组真实GPU验收的准备、执行及计时定义见[八卡验收](../../examples/task_workflows/RECONFIGURATION_ACCEPTANCE.md)。`--prepare-only`不启动服务；真实验收尚未运行。
