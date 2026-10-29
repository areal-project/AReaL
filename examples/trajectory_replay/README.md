# 训练批次保存与离线回放

此例用于 single-controller SGLang RL 调试，支持 v1/v2 训练控制器。 single-controller 可以管理多个
GPU。功能默认关闭，普通训练无需改配置。 本例及验收脚本尚未在本地执行；真实 GPU 验收状态见 RFC，不以脚本存在表示通过。

在原有 RL 配置上，通过标准入口保存实际训练批次：

```bash
areal train run --config <baseline.yaml> \
  --driver examples.trajectory_replay.train:main \
  ++trajectory_debug.mode=dump \
  ++trajectory_debug.path=<snapshot-directory> \
  ++trajectory_debug.scope=rollout
```

`rollout` 保存收集后的批次，回放时重新计算辅助模型评分和 prox logp。 `full` 保存评分后、advantage 前的批次，回放时直接复用评分；仅评分的
ref/teacher 不创建，需要更新的 critic 仍创建。每步写一个 CPU 张量文件和一个版本化 manifest。
所有字段、嵌套多模态张量和逻辑分组均保留，不保存原服务引用。 `trajectory_debug.steps=[0,1,2]` 可以只保存指定步；空列表不保存任何步。
现有同名文件不可覆盖，重试捕获应使用新目录。

捕获运行退出、原推理和张量存储服务已关闭后，使用相同模型初始状态、控制器版本、 训练拓扑、batch size 和 scope 回放：

```bash
areal train run --config <baseline.yaml> \
  --driver examples.trajectory_replay.train:main \
  ++trajectory_debug.mode=replay \
  ++trajectory_debug.path=<snapshot-directory> \
  ++trajectory_debug.scope=rollout \
  valid_dataset=null ++total_train_steps=3
```

回放不加载训练数据、不启动 rollout/eval/proxy、推理 teacher 或权重传输服务。 训练 worker、训练 RPC/gateway、PPO
更新、LR、保存和统计继续使用原实现。 快照保留原始 global step；缺失文件会报错，不会循环最后一批。 要回放后续步，使用现有模型/优化器恢复机制对齐起点。
快照本身不包含模型、优化器或 RNG 状态，不保证与重新在线采样等价。

当前不支持其他控制器执行方式、MOPD、online agent、回放时推理评测， 或自动跨版本/拓扑转换。rollout-engine teacher 需要 `full`
回放。 只读取可信的自己生成的快照，仍使用 `weights_only=True` 限制反序列化。

实际 GPU 验证覆盖 v1/v2 的无 critic/teacher GRPO，以及 reference 评分的 rollout/full 路径。 上文
critic/teacher 描述的是代码接入方式，这些组合尚未完成专项 GPU 验证，不能据此承诺这些组合可用。 v1 已验证模型/优化器恢复后的离线续训；v2
原生控制器不支持 checkpoint 恢复，本功能不扩展该限制。 三步运行只证明所测短程链路，不证明长期稳定性或无内存泄漏。

## GPU 复现脚本

依赖已安装的 AReaL、SGLang、Megatron、Awex 及兼容 checkpoint/tokenizer； 不安装依赖、不创建
allocation、不操作宿主。请在已授权的 allocation 容器中运行。 v1 默认 actor `megatron:d2`、rollout
`sglang:d2`，两张 GPU 上同卡交替执行，以满足 v1 Awex CUDA IPC 的设备配对和 worker 数量要求。 v2 默认 actor
`megatron:d1`、rollout `sglang:d1`，分离占用两张 GPU。 可通过参数增加资源；后端拓扑必须与 `--gpus` 和模型相符。
临时目录须位于授权根目录下；用户已授权容器 `/tmp` 时，也可使用独立的 `/tmp/rsi-<task>` 子目录仅承载 IPC 及必要运行临时文件。路径要足够短，并支持
UNIX socket。 脚本预检失败时不启动训练，不自动改用其他目录。

以 GSM8K 和一个较小的 Qwen3 checkpoint 为最小配置；所有路径由操作员填写：

```bash
bash examples/trajectory_replay/minimal.sh \
  --config examples/math/gsm8k_grpo_awex_colocate.yaml \
  --model <read-only-model> --data <read-only-gsm8k> \
  --write-root <authorized-root> --output <new-capture-output> \
  --temp <short-authorized-temp> --version v1 --mode dump --scope rollout

bash examples/trajectory_replay/minimal.sh \
  --config examples/math/gsm8k_grpo_awex_colocate.yaml \
  --model <same-model> --data <same-data> \
  --write-root <authorized-root> --output <new-replay-output> \
  --temp <short-authorized-temp> --version v1 --mode replay --scope rollout \
  --snapshots <new-capture-output>/snapshots
```

将 `--version` 改为 `v2`，使用新的输出和捕获目录重复；两版分别捕获、分别回放。 每版对 `--scope full` 再执行一组。`minimal.sh`
固定一步真实更新； `end_to_end.sh` 使用相同参数，固定三步。端到端优先更大的兼容模型， Geometry3K 可使用其现有 Megatron
配置并传入对应模型、数据及必要 override：

```text
--config examples/vlm/qwen3_5_2b_megatron_geometry3k_grpo.yaml
--override ++sglang.enable_multimodal=true
--override ++gconfig.max_tokens=2048 --override ++sglang.context_length=2048
```

已验证的 Qwen3.5-9B 三步 full 配置使用 v1 的默认 `d2/d2` 同卡执行；v2 使用
`--actor-backend megatron:d2 --rollout-backend sglang:d1 --gpus 3` 分离执行，并设置
`--override ++rollout.max_head_offpolicyness=0`。这些结果不构成其他拓扑或模型的兼容承诺。 `--override`
可以重复，最后应用；不应在参数中传凭据或覆盖获准写入/资源边界。 脚本将模型、输出、缓存和 name-resolve 等配置通过 overrides 传给
`areal train run --config ... --driver ...`，保留基线 YAML。 driver 用标准 loader 保存脱敏后的最终有效配置。 加
`--prepare-only` 可只通过标准 loader 解析、校验并保存配置，不创建 worker； 这是配置准备，不能代替 GPU 验收。它也会占用一个新输出目录。
v2 示例每次生成临时唯一 admin key，以环境变量引用传给训练和推理服务；命令记录不包含密钥，标准配置保存会脱敏。

为避免小模型的全错答案经组归一化变为零优势，脚本关闭 reward/advantage 归一化并设置 reward bias 为 -0.5；仍调用原真实奖励与 PPO 计算。
短程脚本还显式关闭学习率 warmup，避免基于完整 dataloader 长度的 warmup 令首步学习率为零。 所有 baseline/capture/replay
对照使用相同设置，不用此短程配置声称质量提升。 直接使用 driver 时不会自动改变这些算法配置。

每次输出包含 `command.json`、`source.json`、`process.json`、`console.log`、
`result.json`，以及训练器日志、有效配置和每步 HF checkpoint。 输出目录须为新目录；保存快照目录也不能与其他捕获复用。 单次默认超时 1800
秒，可用 `--timeout` 调整。超时/中断仅终止本脚本创建的训练进程组； 不取消 allocation，不清理其他服务或旧产物。不要同时运行共享同一 GPU 的用例。

## 如何判定验收

退出码非零、缺失步文件/参数更新/关键服务日志、非有限 loss/gradient、 退化全 mask 或未观察到实际版本路径均为失败或未完成。退出零也需检查：

1. v1/v2 日志分别显示真实控制器/worker；capture 有 SGLang rollout、Megatron 更新、Awex 传输及后续
   rollout，未使用磁盘 fallback 冒充 Awex。
1. 原 capture 所属服务已销毁，replay 未创建推理、proxy、权重交换服务，也没有请求旧端点。
1. 快照值、dtype、shape、逻辑分组和多模态字段完整，loss/gradient 有限且有有效训练信号； 对比初始权重与保存的 HF
   checkpoint，确认实际参数变化，不能只数日志步骤。
1. 相同初始状态和固定批次下对比首步 loss/gradient/参数样本， BF16 初始 `rtol=1e-3, atol=1e-4`，记录最大差异及非确定性限制。
1. 每版运行相同工作量的 `--mode off` 与 dump，至少两组，区分预热， 记录保存耗时、step time、文件字节与 CPU/GPU 峰值。
   回放省去在线生成的耗时不代表在线 RL 吞吐提升。

完整标准在中文 [RFC](RFC.md)。准备的 UT 交给用户或 CI 执行：

```bash
pytest tests/test_trajectory_replay.py tests/test_recover.py tests/test_trainer_eval_before_train.py
```

本地不执行 UT，CI 专属检查保留为 PENDING_CI。真实 v1/v2、main 基线、工程检查完成后， 按任务规范由受控独立只读 Reviewer 检查完整候选，保存
Pre-CI review PASS 和 `ci_allowed=true`。每轮 verify
前先读取适用经验并记录实际采用，结果后自动进行独立只读复盘；再通过同时核对复盘回执的发布 gate 及 squash 后的实际 head 内容校验，才执行可能触发 CI
的推送、创建 PR、打标或重跑。 FAIL/INCOMPLETE 返回修复或补证，候选变化先验证并以新独立上下文重新 review。 PR 后最新 head 的必需 CI
全过才算交付，不自动合并。
