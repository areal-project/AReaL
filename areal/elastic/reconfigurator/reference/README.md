# Reference Reconfigurator

Reference 包含当前 Online 的触发、预测、Optimizer、Evaluator 和组件成本模型。原生运行与 CPU 实验通过同一个版本 factory
使用它；Basic v1 保留原实现。

```text
reference/
├── reconfigurator.py       # 观测、决策、复核与配置应用
├── trigger.py              # SLO 触发、历史等待与搜索去重
├── forecast.py             # 历史回放与后续工作量预测
├── optimizer.py            # 候选搜索与选择
├── identity.py             # 完整 Reference 源码、配置与校准身份
└── evaluator/
    ├── __init__.py         # 按需导出 Evaluator
    ├── evaluator.py        # 候选检查、独立状态、Hot／Cold 推演与评分
    ├── contracts.py        # Request／ExecutionConfig
    └── cost_model/
        ├── contracts.py    # 硬件、模型、执行配置、估计与校准记录
        ├── loader.py       # Profiles／冻结校准加载及组件装配
        ├── actor/
        │   ├── model.py    # Logprob、Advantage、Train 及校准查询
        │   ├── operators.py # 共用算子、通信和执行依赖
        │   ├── packing.py  # DP 分发、microbatch 与观测输入
        │   ├── fsdp.py     # FSDP 执行成本
        │   ├── megatron.py # Megatron 执行成本
        │   └── archon.py   # 保留现有 Archon 执行成本
        ├── rollout/
        │   ├── model.py    # Prefill／Decode 及校准查询
        │   └── scheduling.py # 副本路由与连续组批规则
        ├── publication.py  # 权重发布与传输成本
        └── profiles/
            ├── models/
            ├── hardware/
            ├── topology/
            └── execution/{actor,rollout}/
```

一次候选评估分为：**（1）检查输入与约束**，准备共享预测工作量，检查显存和 Actor 供给；**（2）恢复预测状态**，每份 forecast 创建独立的
Evaluator 运行状态；**（3）推演执行**，按 Hot／Cold 规则切换，反复组批、计算耗时和推进事件；**（4）汇总评分**，返回候选的 SLO、GPU
成本及训练进度，交给 Optimizer 比较。

```text
Evaluator.evaluate(view, forecasts, layout)
├── prepare_context() / actor_memory_safety()        # 1. 检查输入与约束
└── _evaluate()
    ├── actor_supply()
    ├── _from_forecast() → _initialize()            # 2. 每份 forecast 恢复独立状态
    ├── _run() → tick()                            # 3. 推演 Hot／Cold 切换与工作
    │   ├── before_work() / after_work()            # 按 handoff.mode 推进交接
    │   ├── service() → _base_service()
    │   │   ├── SGLangScheduler.choose_replica() / batch()
    │   │   └── models.service()                    # 估计本轮推理耗时
    │   └── start_actor() → _base_start_actor()
    │       └── models.actor()                      # 估计训练阶段耗时
    └── CandidatePrediction(...)                    # 4. 汇总各份 forecast 的评分
```

Packing 和 Scheduling 的规则归各侧组件；Evaluator 持有请求、队列、KV、部署和事件时间。Actor 的
`ExpertActorModel.estimate()` 依次组织 `estimate_logprob()`、`estimate_advantage()` 和
`estimate_train()`，后端成本类复用基础算子。Rollout 的组批结果交给模型计算本轮耗时。

`loader.py` 负责读取 Profiles 和冻结系数，具体校准查询分别位于 Actor、Rollout 和发布模型。纯拟合与 Packing 诊断位于
`examples/profile/cost_model_tools.py`。成本模型可单独加载，包导入不启动候选推演。

Actor 初始化后保持驻留。配置变化所需的保存、释放、恢复及权重发布仍由 Evaluator 计入预测；外层 `OnlineSimulation` 独立运行实验。

## Profiles 与运行资源

Profiles 位于 `evaluator/cost_model/profiles/`。`models/` 描述模型结构，`hardware/` 描述单卡，
`topology/` 描述物理节点与互联。 `execution/actor/`、`execution/rollout/`
保存绑定模型和软件版本的执行配方；文件名统一体现模型，DP／TP、副本数和 KV 容量由装配调用传入。

`Catalog.assemble()` 显式接收 `resource_pools`，选取两侧所需设备并核对拓扑。它用于组件装配，执行配方中的
`calibration_scope` 记录校准范围；在线运行仍使用运行配置的 `handoff.node_gpus`、`search_space` 或显式
`candidates`。 当前 v3 的16卡稳态预算、32卡切换准备池和候选范围保持原样。

Qwen3-4B 使用 `execution/actor/megatron-qwen3-4b-ppo`、`execution/rollout/sglang-qwen3-4b`
和 `topology/h20-two-node8`。模型固定 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`，逻辑参数量4,022,468,096。 两侧各8卡、各自在单节点内的限制描述组件校准范围。

```python
import os
from areal.elastic.reconfigurator.reference import Catalog, ParallelLayout

# 此次组件装配的资源配置；不存入模型或拓扑 profile。
resource_pools = {
    "actor": tuple(f"node0/gpu{i}" for i in range(8)),
    "rollout": tuple(f"node1/gpu{i}" for i in range(8)),
}
parts = Catalog().assemble(
    hardware="h20", topology="h20-two-node8",
    model="qwen3-4b-instruct-2507", actor="megatron-qwen3-4b-ppo",
    rollout="sglang-qwen3-4b", actor_parallel=ParallelLayout(dp=4, tp=1),
    rollout_tp=2, rollout_replicas=4, resource_pools=resource_pools,
    kv_capacity_tokens=int(os.environ["KV_CAPACITY_TOKENS"]),
)
# parts.config / parts.actor / parts.rollout 供组件估计与候选评估使用。
```

- Actor支持当前DP／TP范围，CP／PP／EP／ETP固定1；dense FFN随TP切分，梯度走真正DP组。
  精度、Adam状态、LM-head执行选项、模型revision和运行源码摘要参与身份；未知master／remainder存储不计入静态显存下界。
- Rollout以单副本身份校准，排除副本数，保留TP、batch／KV限制等配置；prefill／decode／mixed分别绑定。 TP1／2／4／8对应BF16
  KV每token每rank 144／72／36／18KiB。容量必须传各TP服务实际
  `max_total_num_tokens`，由装配调用显式提供；forward入口槽位占用包含本批分配，不能直接当执行前逻辑KV。
- 当前仍是未校准prior；advantage缺实测模型时为unsupported，发布／重建等沿用显式transition契约。
  旧decode优先／整请求KV预留策略是模拟假设，尚未验证等同当前SGLang。GPU迭代跨度、HTTP墙钟和Trainer阶段边界不混用。
  [实测入口](../../../../examples/profile/CALIBRATION.md)所得数据须核对边界／rank去重，再单独校准并做保留集验证。

校准绑定配置、源码、NumPy、计时边界和负载范围，加载不 refit；未知组合不能套用旧系数。 目标为 AReaL v2，九条 Tau2 历史由用户确认来自
v1；版本身份不代替执行兼容核对。 当前校准、跨配置、目标硬件精度、异步收益与训练质量的验证等级分别记录。

进度统一维护在[项目进度](../../../../PROGRESS.md)。设计长文、旧进度、详细接口和绘图约定见
[本地开发归档](../../../../../artifacts/elastic-policy-development-20261021/README.md)；
原始冻结数据和历史方法见 [原始归档](../../../../../artifacts/repository-archive-20261021/README.md)。
旧30B执行参数与系数保留，profile 目录及命名统一到新结构。源码变化会更新`code_revision`；旧档案仍须显式兼容审计，不自动重绑或refit。
