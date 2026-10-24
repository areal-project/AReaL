# 原生 Actor／Rollout 重配置

范围：local／v2／`on_demand` 生命周期入口、纯 FSDP DP、SGLang TP1／PP1、默认超时 PolicyAdapter。目标 Actor DP
必须整除有效训练 batch。此功能不改变 Slurm 作业预留，不自动选择目标配置。

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

## 验证

CPU 正常路径覆盖资源往返切换、重启、同 DP 移卡、session／采样序号保留、流式与直接模型调用排空、4→2→4 推理副本、唯一后端暂停和本地 GPU 池更新。GPU
引擎及 guard HTTP 使用替身，不能证明真实 GPU 回收或 AWEX 成功。此前真实 Gloo／DTensor／DCP 的 2→4→2 状态恢复检查保留。

真实 FSDP2／NCCL、SGLang 进程回收与重启、AWEX 发布及后续 PPO 更新待空闲 GPU 验收；本轮未运行故障注入。结果与证据见根目录
`PROGRESS.md`。
