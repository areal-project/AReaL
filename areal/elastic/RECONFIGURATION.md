# 原生 Actor DP 重配置

当前实现限于 local／v2／SGLang 的 `on_demand` 生命周期入口、纯 FSDP DP 和默认超时 PolicyAdapter。 Actor 可在原
`actor_devices` 预留池内改变 DP 数量；Rollout 布局、GPU 池和服务保持固定。初始 DP 可小于预留池，目标 DP 必须整除有效训练 batch。
GPU 的具体分配由现有 scheduler 决定，不保证使用池内的前 N 张卡。此功能不改变 Slurm 资源预留，也不自动选择目标配置。

## 请求切换

`STATE_ROOT` 指向当前 CPU 协调器的 `actor_lifecycle.state_root`。先完成至少一次更新，再在相同文件系统提交请求：

```bash
python -m areal.elastic.control.actor_reconfiguration --state-root "$STATE_ROOT" status

python -m areal.elastic.control.actor_reconfiguration --state-root "$STATE_ROOT" request \
  --request-id dp-grow-001 --world-size 4 \
  --expected-generation 0 --expected-weights-version 1
```

`expected-generation` 是 Actor 配置事务的 generation（首次为 0），`expected-weights-version`
是已完成的训练更新数。 协调器在下一次取 batch 前处理请求；若版本已过期则拒绝，核对状态后使用新 request ID 提交。 结果位于
`actor-reconfiguration/response.json`；历史回执保留在该目录的 `state.json`。相同 request
ID／参数重复提交返回原回执。

流程：保存完整源 checkpoint → 持久化切换意图 → 断开权重连接并释放 Actor → 按目标 DP 启动并恢复 → 重建权重连接 → 发布并确认原权重版本 →
提交配置回执。 切换不推进 PPO step、任务消费游标或权重版本；待完成事务恢复到其源边界。恢复失败时保持准入关闭。

## 状态与边界

- 原始模型／optimizer／训练配置和恢复身份保持不变；每代 checkpoint 的 `actor-layout.json` 记录实际 DP 布局与预留 GPU 池。
- 默认同布局加载仍严格匹配 rank／world size。跨 DP 使用绑定源 checkpoint、运行身份及目标数量的显式恢复计划，由 DCP 重分片参数和优化器状态。
- 跨 DP 按 checkpoint／目标布局／逻辑 rank 派生可复现的新 RNG；同布局继续恢复原随机流。固定题单与消费进度不变，不承诺跨 DP 位级一致。
- 每个 DP 布局重新建立测量身份与历史；已有固定布局实测 profile／AutoPilot revision 暂不参与切换。AutoPilot 的
  `NativeConfigExecutor` 仍保持原有限制。
- Megatron、TP／CP／PP／EP 调整、Rollout 副本增删以及跨池 GPU 转移尚未实现。

## 验证

```bash
python -m pytest -q tests/test_elastic_actor_reconfiguration.py tests/test_elastic_dp_restore.py
```

CPU 协调器测试使用 Actor／Rollout 替身；CPU 多进程测试通过真实 Gloo／DTensor／DCP 执行 2→4→2，核对参数、Adam
状态、学习率及继续更新。 CUDA 参数执行真实 FSDP2／NCCL 的同组测试，少于四张可见 CUDA 卡时跳过。GPU 测试需在隔离的空闲 GPU
池运行；实际验收状态见根目录 `PROGRESS.md`。 完整 AReaL Actor→SGLang 发布与训练闭环另需 GPU 验收。
