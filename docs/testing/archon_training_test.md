# ArchonEngine 训练侧测试工具

本文档介绍 `tests/experimental/archon/torchrun/` 下新增的 ArchonEngine 训练侧 对拍工具，包括三个文件：

| 文件                          | 作用                                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| `training_test_config.py`     | YAML + CLI 覆盖配置加载器，定义 `ArchonTrainingTestConfig`           |
| `run_archon_training_test.py` | `torchrun` 入口脚本，运行 N 步训练并落盘全局 loss / 显存 / `diff.pt` |
| `compare_training_dumps.py`   | 离线对拍脚本，比较两个 dump 目录的 per-step loss 与 `diff.pt` 差异   |
| `archon_training_test.yaml`   | 示例配置                                                             |

目标场景：在同一份配置上跑两次（比如 DTA 开启 vs 关闭、或两种 rollout `packing_algorithm`：`ffd`/`kk`/`dta`）， 然后用
`compare_training_dumps.py` 做 "对拍"，验证训练逻辑的等价性 / 回归性。

______________________________________________________________________

## 1. 总体流程

```
    +-------------------+       torchrun       +-------------------+
    | archon_*.yaml     | ────────────────────▶| run_archon_training_test.py |
    |   + CLI overrides |                      +-----┬-------------+
    +-------------------+                            │
                                                     ▼
                           <dump_dir>/
                           ├── stats.jsonl         (每 step 一行全局 JSON)
                           ├── diff.pt             (rank 0, 参数更新统计)

    两次跑完后 → compare_training_dumps.py --dump-a <A> --dump-b <B>
```

______________________________________________________________________

## 2. 配置：`training_test_config.py`

该工具现在只支持：**普通 AReaL YAML + `test_config`**。

```yaml
experiment_name: xxx
trial_name: xxx
cluster:
  fileroot: xxx
actor:          # AReaL 标准 actor 段（含 backend）
test_config:    # 本工具专属的测试参数
```

- 不再支持测试专用顶层 `engine`/`parallel`。
- 并行策略统一从 `actor.backend` 解析（例如 `archon:d8`）。

### 2.1 `test_config` 字段

| 字段                  | 类型   | 默认值     | 说明                                                                                        |
| --------------------- | ------ | ---------- | ------------------------------------------------------------------------------------------- |
| `step`                | `int`  | 必填（>0） | 训练迭代步数；不复用 AReaL 原生 epoch 相关配置                                              |
| `data_dir`            | `str`  | 必填       | 放一组 `.pt` 文件的目录，每个文件是 `list[1-D Tensor]` 形式的 `input_ids`；按字典序循环使用 |
| `disable_optimizer`   | `bool` | `False`    | 开启后 engine 不创建优化器、不更新参数，也不分配优化器状态显存                              |
| `save_diff`           | `bool` | `True`     | 训练结束后在 rank 0 保存 `diff.pt`（参数更新统计）                                          |
| `save_params`         | `bool` | `False`    | 兼容旧字段；低显存模式下忽略，不再导出 `params.pt`                                          |
| `save_initial_params` | `bool` | `False`    | 兼容旧字段；低显存模式下忽略，不再导出 `params_initial.pt`                                  |
| `seed`                | `int`  | `42`       | 构造 `advantages`/`logprobs` 等合成字段的随机种子基值                                       |

### 2.2 配置加载器做了什么

- 使用 OmegaConf 做 YAML 读取 + dotlist 覆盖（`test_config.step=5`、
  `actor.mb_spec.max_tokens_per_mb=8192` 等语法）。
- 并行策略固定从 `actor.backend` 读取并解析（复用 `AllocationMode.from_str`）。
- 训练 backend 当前只允许 `archon`；如果字符串解析成 `fsdp`/`megatron` 会直接报错。
- 输出目录自动对齐普通训练日志根目录：
  `<fileroot>/logs/<user>/<exp>/<trial>/<tree_training_mode>_<parallel_tag>_<model_name>`。
- `fileroot` 优先级：`stats_logger.fileroot > cluster.fileroot`。
- 手写了一个 dataclass 构造器 `_build_dataclass` / `_coerce_value`，递归把 `DictConfig` 子节点塞进
  `TrainEngineConfig` 等结构。原因是 `OmegaConf.structured(TrainEngineConfig)` 会在
  `Literal[...]` 字段 （例如 `tree_training_mode: Literal["disabled", "sparse", "dta"]`）上报
  `ValidationError`。
- `actor` 中 `TrainEngineConfig` 未使用的字段（例如 PPO/GRPO 专属字段）会自动过滤，便于直接复用普通 YAML。

### 2.3 示例 YAML

`tests/experimental/archon/torchrun/archon_training_test.yaml` 是一份可直接运行 的模板（以
Qwen2.5-0.5B-Instruct + DTA + dp=2 为例）：

```yaml
experiment_name: archon_train_test
trial_name: trial0
cluster:
  fileroot: /storage/openpsi/experiments
actor:
  backend: archon:d2
  path: /storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct/
  dtype: bfloat16
  mb_spec:
    max_tokens_per_mb: 5596
  optimizer:
    type: adam
    lr: 1.0e-5
    weight_decay: 0.01
    lr_scheduler_type: constant
    gradient_clipping: 1.0
  tree_training_mode: dta        # {disabled, sparse, dta}
  dta_block_size: 2048
  packing_algorithm: ffd         # rollout {ffd, kk, dta}；与 mb_spec.packing_algorithm 不同

test_config:
  step: 4
  data_dir: ""                   # 必须通过 CLI 覆盖
  disable_optimizer: false
  save_diff: true
  seed: 42
```

______________________________________________________________________

## 3. 数据格式

`data_dir` 下每个 `.pt` 文件对应 **一个训练步** 的候选输入。格式约定：

```python
torch.save([
    torch.tensor([id0, id1, ...], dtype=torch.long),   # 第 0 条 1-D input_ids
    torch.tensor([id0, id1, ...], dtype=torch.long),   # 第 1 条
    ...
], "step_000.pt")
```

- 每个张量必须是 1-D `input_ids`（不包含 attention_mask、label 等）。
- 文件按 **字典序** 排序，训练时 `step_idx` 取模轮转使用。
- 序列数量必须 `>= dp_world_size`，否则抛错；多出来的尾巴会按 `len // dp_world_size * dp_world_size` 截断以保证每个
  rank 供给同样多条 `redistribute_trajectories` 预期下的输入。

每条 `input_ids` 会被自动补齐成一个 GRPO 用的 trajectory dict： `attention_mask` 全 1、`loss_mask` 前 30%
token 置 0 后半置 1（当前固定比例）、 `logprobs/old_logprobs/advantages/rewards/values/prox_logp` 由
`seed + step*100003 + global_idx` 确定，**保证同 step、同条序列在所有 rank 上生成的合成字段一致**。

______________________________________________________________________

## 4. 训练入口：`run_archon_training_test.py`

### 4.1 启动命令

```bash
torchrun --nproc_per_node=$NPROC \
    tests/experimental/archon/torchrun/run_archon_training_test.py \
    --config tests/experimental/archon/torchrun/archon_training_test.yaml \
    test_config.step=4 \
    test_config.data_dir=/path/to/data_dir
```

`--config` 后面的参数为 OmegaConf dotlist 覆盖，想改任意 `actor.*` / `test_config.*` 直接写 `key=value`
即可。

直接复用普通训练 YAML 即可（只额外补 `test_config.*` 覆盖）：

```bash
torchrun --nproc_per_node=8 \
    tests/experimental/archon/torchrun/run_archon_training_test.py \
    --config examples/math/gsm8k_sft_archon_fp8.yaml \
    test_config.step=4 \
    test_config.data_dir=/path/to/data_dir
```

### 4.2 每一步做的事

1. 按字典序取 `step_files[step % len]`，`torch.load` 成 `list[Tensor]`。

1. 每个 rank 按 `dp_rank::dp_world_size` 步长选自己的本地子集，用上述合成字段 构造 trajectory dict。

1. `redistribute_trajectories(..., packing_algorithm=engine.config.packing_algorithm)`
   按配置（`ffd` / `kk` / `dta`）重新分配到各 rank。

1. `torch.cuda.reset_peak_memory_stats()` → `engine.train_batch(...)` → 计时。

1. 对每个 rank 的局部统计做 all-reduce 聚合后，记录一条 **全局** JSON 到 `<dump_dir>/stats.jsonl`：

   ```json
   {
     "step": 0, "file": "/.../000.pt", "world_size": 2, "dp_world_size": 2,
     "num_global_sequences": 8, "num_global_tokens": 30568,
     "elapsed_s_max": 1.234, "peak_mem_mib_max": 8192.5,
     "loss": 0.1234, "loss_source": "train_batch_return_global_token_weighted",
     "grad_norm_max": 0.56, "update_successful": 1.0, "lr_max": 1.0e-5
   }
   ```

### 4.3 loss 怎么抓

- 通过 `engine.train_batch(..., return_loss=True)` 直接拿每个 rank 的本地 step loss。
- 不再依赖 monkey-patch 或额外 side-channel 采样。
- 再按 token 做跨 rank 加权平均，得到单个 `global loss` 写入 `stats.jsonl`，方便直接对拍。

### 4.4 `disable_optimizer=true` 的行为

1. 进 `_create_engine` 前把
   `engine_cfg.optimizer = None`，`ArchonLMEngine._create_optimizer` 会 early-return，因此
   **完全不分配优化器状态显存**。
1. monkey-patch `engine.optimizer_step / optimizer_zero_grad` 为 no-op，但仍会：
   - 计算 `grad_norm`（对所有参数 `.grad` 做 L2）并返回；
   - 把 `param.grad` 置空，保证下一步梯度不累加。
1. 前向 + 反向正常执行，`loss` 曲线仍有意义，只是 **参数不变**。

因此 `disable_optimizer=true` 场景下，`diff.pt` 的更新指标会接近 0，主要 用来对拍 "两种配置的纯前向 / 反向 loss 是否一致"。

### 4.5 `diff.pt` 导出（低额外显存）

启动训练前，脚本按参数顺序调用 `.full_tensor()`，把每个参数的初始值转成 CPU `float32` 并只在 rank 0 保存。训练结束后，按相同顺序再次
materialize 全量参数， 在 rank 0 上与初始 CPU 快照做差并计算更新统计：

- `mean_abs_update`
- `max_abs_update`
- `l2_update`
- `rel_l2_update`（`l2_update / ||initial||_2`）

实现上是“**一次只处理一个参数**”：虽然用了 `.full_tensor()`，但峰值额外显存被限制在 单个参数 full tensor 的量级，可读性更高，也不需要对象
gather 逻辑。最终由 rank 0 写出 `diff.pt`：

参数命名上，`diff.pt["params"]` 会优先使用 `state_dict_adapter.convert_single_to_hf` 生成 HuggingFace
key；若无法映射，则回退到去掉 wrapper 前缀（如 `._orig_mod`）后的 Archon 原始 key。

```
<dump_dir>/stats.jsonl         # 每 step 全局汇总
<dump_dir>/diff.pt             # 参数更新统计（非全量参数）
```

______________________________________________________________________

## 5. 对拍入口：`compare_training_dumps.py`

用法：

```bash
python tests/experimental/archon/torchrun/compare_training_dumps.py \
    --dump-a <run_a_dump_dir> \
    --dump-b <run_b_dump_dir> \
    --loss-atol 1e-6 \
    --loss-rtol 1e-6 \
    --compare-initial        # 可选：仅对 legacy params_initial.pt 有效
```

### 5.1 loss 严格对拍

- 从两边 `<dump_dir>/stats.jsonl` 读所有记录，按 `step` 分组。
- 每 step 直接比较全局 `loss`。
- 判定公式：`|loss_a - loss_b| <= atol + rtol * |loss_b|`。
- 打印每一步的 `loss_a / loss_b / abs_gap / rel_gap / OK|MISMATCH`，再给出 整体 `PASS / FAIL`。
- 任一步 `MISMATCH` 脚本 `exit code = 1`，方便接入 CI。

### 5.2 `diff.pt` 差异（信息性，**不做强对齐**）

默认加载两边 `diff.pt`，逐参数比较更新统计 gap：

| 指标           | 含义             |
| -------------- | ---------------- |
| `max_abs_gap`  | \`               |
| `mean_abs_gap` | \`               |
| `l2_gap`       | \`               |
| `rel_l2_gap`   | \`               |
| `numel_match`  | `numel` 是否一致 |

脚本会打印全局汇总 + top-K gap 最大张量。若两边都没有 `diff.pt`，会自动回退到旧版 `params.pt` 比较逻辑。

______________________________________________________________________

## 6. 典型使用姿势

### 6.1 DTA 开关对拍

```bash
# Run A: DTA 开启
torchrun --nproc_per_node=2 \
    tests/experimental/archon/torchrun/run_archon_training_test.py \
    --config tests/experimental/archon/torchrun/archon_training_test.yaml \
    test_config.step=8 \
    test_config.data_dir=/data/token_samples \
    test_config.disable_optimizer=true

# Run B: DTA 关闭
torchrun --nproc_per_node=2 \
    tests/experimental/archon/torchrun/run_archon_training_test.py \
    --config tests/experimental/archon/torchrun/archon_training_test.yaml \
    test_config.step=8 \
    test_config.data_dir=/data/token_samples \
    test_config.disable_optimizer=true \
    actor.tree_training_mode=disabled \
    actor.packing_algorithm=ffd

# 对拍
python tests/experimental/archon/torchrun/compare_training_dumps.py \
    --dump-a /storage/openpsi/experiments/logs/$USER/archon_train_test/trial0/dta_d2_Qwen__Qwen2.5-0.5B-Instruct \
    --dump-b /storage/openpsi/experiments/logs/$USER/archon_train_test/trial0/disabled_d2_Qwen__Qwen2.5-0.5B-Instruct \
    --loss-atol 1e-4 --loss-rtol 1e-4
```

`disable_optimizer=true` 保证两次跑的初始参数完全相同，loss 差异只来自实现 差异；tolerance 按实际算子误差调整。

### 6.2 训练 benchmark（实际更新参数）

把 `disable_optimizer` 去掉并把 `save_diff=false`（仅关掉 `diff.pt` 落盘）以节省磁盘，即可拿到 `elapsed_s_max`
/ `peak_mem_mib_max` / `loss` 曲线作为性能基线。

______________________________________________________________________

## 7. 代码位置速查

```
tests/experimental/archon/torchrun/
├── archon_training_test.yaml          # 示例配置
├── training_test_config.py            # 配置加载
├── run_archon_training_test.py        # 训练入口（torchrun）
└── compare_training_dumps.py          # 离线对拍
```

相关生产代码参考：

- `areal/experimental/engine/archon_engine.py` -- `ArchonLMEngine.train_batch`。
- `areal/experimental/dta/wrapper.py` -- DTA 前向/反向包装。
- `areal/trainer/ppo/actor.py` -- `grpo_loss_fn` 签名与默认参数。
- `areal/infra/dist_rollout.py` -- `redistribute_trajectories`。
- `areal/api/cli_args.py` -- `TrainEngineConfig` 与树训练相关字段。
- `tests/experimental/archon/test_dta.py` -- 原有 DTA 单测中的对拍模式，供参考。
