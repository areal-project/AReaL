# 多 Teacher On-Policy Distillation（MOPD）

MOPD 将 on-policy 强化学习与一个或多个 teacher checkpoint 的 token 级目标结合。每个
dataset source 必须选择一个 teacher group，group 可以为任意数量的 teacher 指定非负权重。
权重按配置值直接使用，不会自动归一化。

MOPD 当前运行于 single-controller 模式，使用 Megatron actor 和 teacher engine、SGLang
rollout engine 及 AWEX 共卡权重传输。actor、rollout 和常驻的 teacher companion 共用同一组
GPU，但大模型权重不会同时驻留。

## 运行时生命周期

每个训练 step 包含三个互斥阶段：

1. **Rollout：** SGLang 生成 trajectory，AReaL 将数据源的 teacher group 作为内部 task
   metadata 透传；dataset sample 和 workflow 输入都不需要路由字段。
2. **Teacher：** offload rollout 权重和 KV cache。fork 出的 Megatron teacher 进程 onload，
   依次加载当前 batch 所需的 checkpoint，并为路由到它的 sample 计算分数；actor 完成
   teacher RTensor 的物化与清理后，再次 offload teacher 权重。companion 进程保持常驻，供
   下一个 step 复用。
3. **Train：** actor 计算配置的 RL 与 distillation loss，更新权重，并通过 AWEX 向 SGLang
   发布新版本。SGLang 丢弃旧 KV cache，并在继续生成前分配新的空 cache。

actor 与 `mopd.teacher_engine` 必须使用相同的并行策略和 world size，包括相同的
pipeline parallel size。当前实现要求 actor 与 teacher controller 使用 v1；PP、TP、CP、
DP 和 EP 由所选 Megatron 模型及 allocation 校验。

## 配置

在 PPO 配置中增加 `mopd`：

```yaml
actor:
  backend: "megatron:(attn:d1p1t4c2|ffn:d1p1e8)"
  weight_update_mode: awex

rollout:
  backend: sglang:d8t1
  scheduling_strategy: {type: colocation, target: actor, fork: true}

train_dataset:
  mixture_sampling_policy: proportional
  sources:
    - {path: /data/code, type: rl, teacher_group: coding}
    - {path: /data/mixed, type: rl, teacher_group: mixed}

mopd:
  teachers:
    coder: {path: /models/teacher-coder}
    reasoning: {path: /models/teacher-reasoning}
  teacher_groups:
    coding: {coder: 1.0}
    mixed: {coder: 0.3, reasoning: 0.7}
  teacher_engine:
    backend: ${actor.backend}
    optimizer: null
    disable_dropout: true
    scheduling_strategy: {type: colocation, target: actor, fork: true}
    scheduling_spec: ${actor.scheduling_spec}
  manager:
    type: disk
    staging_root: /dev/shm/areal-mopd
  loss:
    rl_coefficient: 0.0
    distillation_coefficient: 1.0
```

启用 MOPD 时，`train_dataset.sources` 中的每个数据源都必须显式设置
`teacher_group`，且必须匹配 `mopd.teacher_groups` 中的 key；配置 valid dataset 时也遵循
相同规则。非 MOPD 场景下，`teacher_group` 默认为 `null`。sample 不能覆盖数据源的
teacher group，也不需要 `task_type` 字段。每个 teacher group 可以引用任意数量的已知
teacher ID，并且必须至少包含一个正权重。
`mixture_sampling_policy: proportional` 保持各数据源按长度占比采样；`uniform`
则通过确定性循环较短数据源，使每个数据源在一个 epoch 中贡献相同数量的样本，再由分布式
sampler 对全局索引进行 shuffle。

`manager.type: disk` 从共享存储加载 checkpoint，支持多节点运行。`local_memory`
会在 `staging_root` 下异步暂存下一个 checkpoint，使用原子发布后交给常驻 teacher
加载，并在加载完成后删除。由于该路径只在 controller 所在节点可见，
`local_memory` 要求 `scheduler.type: local` 且 actor/teacher 为单机拓扑；可通过
`min_free_bytes` 为暂存目录预留可用空间。

对 teacher 权重 $w_j$，定义 $S_T(a)=\sum_j w_j\log\pi_{T_j}(a)$ 和
$W=\sum_j w_j$。MOPD 使用 on-policy score-function surrogate 最小化未归一化的加权
reverse KL：$\sum_j w_j D_{KL}(\pi_\theta \parallel \pi_{T_j})$。

```text
rho(a) = min(exp(log pi_theta(a) - log pi_old(a)), importance_ratio_cap)
reward(a) = S_T(a) - W * stop_gradient(log pi_theta(a))
mopd_loss = -mean(rho(a) * reward(a))
loss = rl_coefficient * rl_loss + distillation_coefficient * mopd_loss
```

`importance_ratio_cap` 默认值为 `5.0`，用于限制重要性采样乘数并避免指数溢出。

该目标是多个 reverse-KL 的加权和；忽略与 student 无关的常数后，也等价于几何 teacher
ensemble。它不是 teacher cross-entropy，也不是 teacher 概率的算术混合。teacher group
权重直接生效，
不会自动归一化。

设置 `rl_coefficient: 0.0` 可执行纯 distillation；两个 coefficient 都设为正数时执行 RL 与
distillation 联合训练。

## 示例

- `examples/mopd/gsm8k_qwen3_14b_to_0_6b.py` 提供本地 GSM8K 入口和 dry-run 校验。
- `examples/mopd/gsm8k_qwen3_14b_to_0_6b_local.yaml` 在单机八卡上配置
  Qwen3-14B teacher 和 Qwen3-0.6B actor。

无需启动 worker 即可校验本地配置：

```bash
MOPD_STUDENT_MODEL_PATH=/models/Qwen3-0.6B \
MOPD_TEACHER_MODEL_PATH=/models/Qwen3-14B \
MOPD_GSM8K_PATH=/data/gsm8k \
AREAL_ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
python -m examples.mopd.gsm8k_qwen3_14b_to_0_6b \
  --config examples/mopd/gsm8k_qwen3_14b_to_0_6b_local.yaml \
  --dry-run
```

## 运维注意事项

- actor 与 teacher checkpoint 必须共享相同的 token-ID 映射，且每个模型架构都必须由所选
  Megatron adapter 支持。
- teacher companion 进程会跨 phase 常驻，但 actor 恢复显存所有权前必须 offload teacher
  权重。`DrainReceipt` 是 teacher RTensor 的 phase 资源回收边界。
- W&B 凭据及服务 endpoint 应通过环境变量传入，不要写入 YAML 或 shell 文件。
- driver 已完成并清理持久 worker 时，actor 或 rollout 的 Slurm 子作业显示 cancelled 可能是
  正常现象。应以 driver exit code 和最后一个训练 step 日志判断是否成功。
