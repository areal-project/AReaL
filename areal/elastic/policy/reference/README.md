# Reference model

独立冻结的 Expert 成本模型与异步 simulator，供实验对照；不含 optimizer 或在线调度策略。
`actor_lifecycle.policy=reference` 仍为待实现入口。Reference 不调用 Basic，精度需由目标数据验证。

| 文件                          | 职责                                                               |
| ----------------------------- | ------------------------------------------------------------------ |
| `cost_model/`                 | Actor、advantage、Rollout、发布、RPC 与转换成本，以及 packing 诊断 |
| `simulator.py`、`scenario.py` | 请求／组依赖、队列、版本、资源状态与异步时间／GPU 成本模拟         |
| `core.py`                     | 来源、覆盖范围、校准、可行性与数值实现身份                         |
| `catalog.py`、`trial.py`      | 基础 profile 装配、外部 trial 校验及条件模拟重建                   |
| `replay.py`                   | 已提供观测窗口的计数、版本与 allocation 记账                       |
| `profiles/`                   | 模型／硬件／系统／Actor／Rollout 的基础配置与来源                  |

主要入口：`ExpertActorModel.estimate()`、`ExpertRolloutModel.estimate()`、
`ExpertSimulator.simulate()`、`Catalog()`。默认 Catalog 提供基础 profiles；历史 trial 由调用方传外部 root。
缺失耗时保持 unsupported，零值须显式声明；同配置观测和跨配置预测分别验收。

## Qwen3-4B / Megatron

新增独立 profiles：`models/qwen3-4b-instruct-2507`、`actor/megatron-qwen3-4b-ppo`、
`rollout/sglang-qwen3-4b`、`systems/h20-actor8-rollout8`。模型固定 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`，dense SwiGLU、GQA、共享 embedding；逻辑参数量
4,022,468,096。Actor≤8卡、Rollout≤8卡，各自节点内布局，分别校准。

```python
import os
from areal.elastic.policy.reference import Catalog, ParallelLayout

parts = Catalog().assemble(
    hardware="h20", system="h20-actor8-rollout8",
    model="qwen3-4b-instruct-2507", actor="megatron-qwen3-4b-ppo",
    rollout="sglang-qwen3-4b", actor_parallel=ParallelLayout(dp=4, tp=1),
    rollout_tp=2, rollout_replicas=4,
    kv_capacity_tokens=int(os.environ["KV_CAPACITY_TOKENS"]),
)
# parts.config / parts.actor / parts.rollout 用于原有 estimate / simulator 接口。
```

- Actor支持当前DP／TP范围，CP／PP／EP／ETP固定1；dense FFN随TP切分，梯度走真正DP组。
  精度、Adam状态、LM-head执行选项、模型revision和运行源码摘要参与身份；未知master／remainder存储不计入静态显存下界。
- Rollout以单副本身份校准，排除副本数，保留TP、batch／KV限制等配置；prefill／decode／mixed分别绑定。 TP1／2／4／8对应BF16
  KV每token每rank 144／72／36／18KiB。容量必须传各TP服务实际
  `max_total_num_tokens`，profile中的65,536只是模板值；forward入口槽位占用包含本批分配，不能直接当执行前逻辑KV。
- 当前仍是未校准prior；advantage缺实测模型时为unsupported，发布／重建等沿用显式transition契约。
  旧decode优先／整请求KV预留策略是模拟假设，尚未验证等同当前SGLang。GPU迭代跨度、HTTP墙钟和Trainer阶段边界不混用。
  [实测入口](../../../../examples/profile/CALIBRATION.md)所得数据须核对边界／rank去重，再单独校准并做保留集验证。

校准绑定配置、源码、NumPy、计时边界和负载范围，加载不 refit；未知组合不能套用旧系数。 目标为 AReaL v2，九条 Tau2 历史由用户确认来自
v1；版本身份不代替执行兼容核对。 当前校准、跨配置、目标硬件精度、异步收益与训练质量的验证等级分别记录。

进度统一维护在[项目进度](../../../../PROGRESS.md)。设计长文、旧进度、详细接口和绘图约定见
[本地开发归档](../../../../../artifacts/elastic-policy-development-20261021/README.md)；
原始冻结数据和历史方法见 [原始归档](../../../../../artifacts/repository-archive-20261021/README.md)。
旧30B profiles与系数保留原样。源码变化会更新`code_revision`；旧档案仍须显式兼容审计，不自动重绑或refit。
