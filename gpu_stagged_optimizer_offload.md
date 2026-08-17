# Megatron-Core 0.17 GPU-Staged Optimizer Offload 设计

## 1. 范围与版本边界

GPU-staged optimizer 让模型参数和梯度继续留在 GPU，同时把 FP32 master weight 与 optimizer state 保持在
pinned CPU slab。optimizer step 只把一个有界 unit 搬到 GPU，完成更新后立即异步写回 CPU。

当前实现严格绑定：

- `megatron-core==0.17.0`；
- Muon 路径绑定 `emerging-optimizers==0.3.0`；
- AdamW 使用 MCore precision-aware `DistributedOptimizer`；
- Muon 使用官方 `dist_muon`、`LayerWiseDistributedOptimizer.shard_params()` 和
  `newton_schulz_tp()`；
- 不修改 MCore、Transformer Engine 或 emerging-optimizers 的 site-packages。

这是 AReaL 内部显式能力，通过 `MegatronEngine.configure_gpu_staged_adamw()` 或
`MegatronEngine.configure_gpu_staged_muon()` 在 optimizer/DDP 构造前启用，不提供新的 CLI 参数。

## 2. CPU slab、slot 与 residency

### 2.1 AdamW

AdamW 每个 managed leaf 直接分配三个连续 pinned FP32 slab：

```text
master_param | exp_avg | exp_avg_sq
```

slab 只覆盖 MCore 最终分配给本 rank 的 optimizer shard。view 与 slab storage 保持 alias，不先在 GPU 创建完整
FP32 state。一个 staging slot 包含 `master/exp_avg/exp_avg_sq/grad` tensor、H2D/compute/D2H
stream 和对应 event。unit 可按 DP-local range 切分，slot 容量由 `bucket_size_mb` 限制。

每个本地 state 元素的基础 PCIe 流量约为 24 bytes/step：三个 FP32 tensor 各做一次 H2D 和一次 D2H。梯度已经在
GPU，不重复传输。

### 2.2 Muon 与 scalar AdamW

Muon 必须先让官方 layer-wise sharder 完成 dense/expert ownership，再调用 `bind_owned_params()`。CPU
state 为：

```text
Muon matrix:    master_param | momentum_buffer
scalar AdamW:   master_param | exp_avg | exp_avg_sq
```

一个 Muon unit 是 owner 持有的完整 TP-local 二维 parameter shard；不能按 DP、bucket 或 slot 再拆分
Newton-Schulz。非 Muon 参数沿用官方 builder 的分类并进入 scalar AdamW leaf，不能以 `ndim` 自行猜测。空
Muon/AdamW leaf 保留固定 chain 位置，但不分配 slab、slot 或伪 state。

Muon owner 元素的基础 PCIe 流量约为 16 bytes/step。slot 必须容纳本 rank 的最大完整矩阵和必要 workspace；TP 或
expert-TP 大于 1 时目前只允许 `buffer_count=1`。

### 2.3 状态机与 step 顺序

稳定训练态只有 `CPU_RESIDENT`；step 暂时进入 `STEP_ACTIVE`：

```text
grad sync / overflow / norm / clipping
  -> unit H2D
  -> GPU AdamW 或官方 Muon update
  -> 写回 GPU model parameter
  -> MCore 参数同步/all-gather
  -> state/master D2H
  -> drain
  -> CPU_RESIDENT
```

slot 只有在 `d2h_done` 后才能复用。初始化和每步 drain 后都满足：

```text
cuda_state_numel == 0
GPU optimizer peak == O(largest slot/unit workspace)
```

`offload_to_cpu()` 等价于 `drain()`；`restore_from_cpu()` 是 no-op。AWEX 的 weights/grad
生命周期不再把 managed optimizer state onload 到 GPU。

Muon 在公开 `step()` 进入 MCore chain 前执行 activity 和 metadata preflight。TP peers
对梯度存在性做固定大小共识：全无梯度共同跳过，部分 peer 有梯度则在 grad norm、clip 和 Newton-Schulz 前全组
fail-closed。owner metadata 冻结 parameter identity、shape、stride、dtype/device、storage
identity/offset、底层 data pointer 和 storage capacity；漂移时先完成显式 group status vote，再拒绝 data
collective。

## 3. 同步 managed checkpoint

### 3.1 预校验顺序

managed load 在任何 model、slab、param-group、scheduler 或 RNG mutation 前完成：

1. 所有 checkpoint rank 对 request、能力和 leaf tree 达成一致；
1. 读取 metadata-only outer schema 和 identity；
1. 用公共 validator 严格验证 AdamW/Muon hyperparameter；
1. 通过 DCP reader 的 `.metadata` 校验源 tensor key、FP32 dtype、global shape 和 chunk coverage；
1. 完成 rollback filesystem/capacity preflight；
1. 才创建磁盘 snapshot、构造完整 DCP template 并写入 CPU slab。

AdamW outer/inner validation 使用同一规则：LR/min/max/initial LR 非负且 finite，
`0 <= beta < 1`，`eps > 0`，weight decay/multiplier 合法，step 是非负且不接受 `bool` 的整数语义，flag
必须是预期 bool 类型。DCP source dtype 必须原生为 FP32，不能依赖 load 时静默 cast 后的二次检查。

### 3.2 两阶段 commit 与独立 journal

所有 fallible 本地 phase 都遵循“本地捕获结果 -> 显式 WORLD-sized Gloo control group vote -> 共同转移状态”：

```text
preflight
  -> begin/apply
  -> local validate
  -> prepare-commit
  -> unanimous global commit decision
  -> cleanup
```

commit decision 前，rollback action journal 对 master、各 moments、每个 param-group 和 runtime
metadata 分项记录 `PENDING/COMPLETED`。单项失败不阻断其余独立 action，retry 只执行 pending 项；drain action 是
slab restore 的依赖。

shared commit token 是不可逆边界。token 一经全 rank 发布：

- 新 slab 和 param-group 成为唯一权威状态；
- leaf 即使本地 enum 仍滞留 `LOAD_ACTIVE`，也按 token 投影为 `COMMIT_DECIDED/CLEANUP_PENDING`；
- abort/recovery 永久禁止；
- optimizer step 可使用新状态；
- 旧 snapshot 只进入 metadata/reference cleanup journal。

rollback journal 和 post-commit cleanup journal 使用不同类型、identity 和 API。cleanup 逐 leaf
幂等；失败只保留有界诊断和待清引用，不会恢复旧 tensor。下一次 save/load 先全 rank retry cleanup，再解析新 request。

### 3.3 POISONED 与 replacement recovery

commit decision 前 rollback 失败会进入：

```text
POISONED + retained rollback actions
  -> all-rank pending-action retry
  -> terminal receipt acknowledgement
  -> RELOAD_REQUIRED(generation)
  -> manager-authorized replacement attempt
  -> full model+optimizer+RNG load
  -> unanimous commit
  -> cleanup
  -> CLEAN
```

`RELOAD_REQUIRED` 不是普通 `CLEAN`。manager 持有稳定 recovery transaction 和 generation；每次
replacement 创建新的 attempt token。只有匹配 generation/attempt 的 leaf 才可配置新 snapshot 和
begin。snapshot 配置/materialize 失败会清理本轮 partial artifact并保留原 generation；DCP 部分写失败用本轮
snapshot 回滚，成功后仍回到同一 `RELOAD_REQUIRED`，等待下一次 replacement。

empty leaf 与有状态 leaf 使用相同 attempt、commit、cleanup/recovery action 和 terminal receipt
语义。receipt 只保留小型不可变 identity，不保留 tensor、异常 traceback 或 leaf 强引用。只有 replacement 全局 commit
且 cleanup 成功后才清除 generation、receipt、transaction 和 poison 引用。

## 4. 有界磁盘 rollback snapshot

同步 load 不再 clone 一份完整 CPU optimizer state。每个 rank、leaf、slab 使用独立 snapshot artifact；默认
chunk 为 64 MiB，支持范围为 1--512 MiB，并限制每 slab 最多 131072 chunks、header 最多 16 MiB。workspace
因而为 `O(chunk)`，RSS 不随完整 optimizer state 线性增加。

每个 slab 保存 data 和 JSON header。header 绑定 schema/version、leaf identity、slab key、FP32
dtype、numel/bytes、chunk 边界及 SHA-256 checksum。partial data/header 分别由 move journal 记录
`PLANNED/MOVED/CLEANED`；rename 或 unlink 的 after-effect 异常通过预登记名称和 inode reconcile，多轮
cleanup 不删除未知文件。

### 4.1 root 与容量契约

- `checkpoint_snapshot_root` 必须是已存在、可写、非 symlink 的真实目录；缺省为 `/tmp`；
- 从可信 root dirfd 使用 `O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`，后续 create、open、rename、unlink
  和 rmdir 尽量使用相对 dirfd；
- 每次 cleanup 重验 root、snapshot directory、owner marker 和 artifact inode；被 rename/replaced
  的目录只告警，不扫描或猜测删除；
- `f_fsid` 必须存在且非零。相同 filesystem identity 的所有 rank 汇总 required bytes，并与最小 free
  space比较，保留 64 MiB 加 5% 的安全余量；
- capacity vote 通过后实际 `posix_fallocate`/write/fsync 仍可能失败，此时走正常 all-rank abort/POISONED
  语义。

所有持久 FD 使用 detach-before-close 的 owned-FD 状态机。pre-close hook 失败会先用 `fstat()` 重验
device/inode/type；ownership 丢失或 `EBADF` 后永久 detach，绝不关闭复用的整数 FD。restore 的 data FD 由
snapshot journal持有，chunk copy完成和 FD finalize 是两个阶段；pre-close 失败重试时不会重放已完成 chunk，也不会打开第
二个 data FD。

commit 后 snapshot 永久失去 rollback authority。cleanup 只删除本事务严格证明归属的 artifact；失败保留 cleanup
pending，但新 optimizer state 仍可训练。

## 5. AdamW managed async save

Muon async checkpoint 不受支持。AdamW managed async save 使用权威 CPU slab view 作为 DCP source，不把
optimizer state 搬回 GPU。MCore 0.17 没有独立可靠的 “source staging complete”信号，因此首版 mutation
fence 保守保持到整个 async request finalize 完成：

```text
IDLE -> SAVE_STAGING -> SAVE_IN_FLIGHT -> COMPLETE
                                      \-> FAILED
```

fence 绑定 checkpoint ID/path、MCore call index、leaf identity、source storage generation 和显式
participant/control group。step、load/recovery、model-only reset、state setter 和
param-group/step mutation 在 source 安全前等待同一 request；等待不发生在 backward hook 内。首版最多一个
outstanding managed async save。

### 5.1 MCore 0.17 finalize adapter

原生 `AsyncCallsQueue.maybe_finalize_async_calls()` 和两个 torch-dist finalize callback 在
callback 前后包含隐式 default-WORLD collective；rank-local callback 异常可能造成 collective
错位。managed 路径使用严格版本/签名/source-hash/callback code identity 限定的 AReaL adapter，普通非-managed
async 仍走 MCore 原生路径。

adapter 固定执行：

```text
worker wait/reap
  -> explicit Gloo vote(worker result)
  -> local result collection
  -> explicit Gloo vote
  -> coordinator writer.finish
  -> explicit Gloo vote
  -> coordinator save_config
  -> explicit Gloo vote
  -> call-index validation and queue pop
```

每个 phase 带单调 phase ID/name。未知、伪造、缺失、额外或重排 callback 均 fail-closed。queue index、active
record index 和 manager transaction index 在 `writer.finish()`、`save_config()`、queue pop 或
lease/fence释放前必须一致。

worker/process authority 保存在持久 recovery journal。terminate/kill/join/close 任一步失败都保留同一
process、active request、call index 和逐 action进度；无法确认 worker 已退出时不释放 fence 或 AWEX lease。各
rank cleanup进度可不同，但都执行相同 phase vote；只有全 rank authority清零才共同结束 recovery publication。

### 5.2 marker 可见性

checkpoint directory 从 `/` 开始逐组件使用 `openat(O_DIRECTORY|O_NOFOLLOW)`，marker 只通过 retained
dirfd 和相对名称访问。marker payload v2绑定 path/directory identity、随机 checkpoint ID、logical call
ID、MCore call index、WORLD participant 列表、control-group约束、ordered leaf identity
digest、MCore/backend 和 `metadata.json` identity/digest。

发布协议：

```text
create incomplete (DCP可见前)
  -> write/fsync complete candidate
  -> no-replace link complete
  -> fsync directory
  -> all-rank commit decision
  -> unlink owned incomplete       # 磁盘可见性 commit point
  -> temp/FD/额外fsync cleanup
```

loader 只接受 complete 存在、incomplete 不存在且 payload 严格匹配的目录。complete+incomplete
同时存在仍拒绝。commit point后的 temp/FD cleanup错误只进入 post-commit cleanup pending，不能把已可加载
checkpoint 报告为失败或重新创建 incomplete。外部替换、symlink/FIFO/device、stale 或 unknown marker
一律拒绝且不删除。

## 6. Muon checkpoint v2 与 DP reshard

Muon checkpoint capability 为 `muon_dp_reshard_v2`。保存的逻辑 tensor key 与 source DP/global
rank和owner rank解耦，绑定：

- stable parameter name；
- dense/expert domain；
- leaf path/kind 和 state kind；
- PP/TP/EP/expert-TP local coordinate；
- global logical shape 与 FP32 dtype。

Muon parameter 恰有完整 `master_param/momentum_buffer` payload；scalar AdamW 恰有完整
`master_param/exp_avg/exp_avg_sq` payload。每个逻辑 state 在全局 manifest 中必须出现一次，且只允许一个覆盖完整
`(0, numel)` 的 DCP chunk。重复、缺失、额外、冲突或部分矩阵切分都在 snapshot/DCP/slab mutation 前拒绝。

participant metadata 不信任自报 global rank。显式 WORLD-sized Gloo gather 的接收 slot提供可信 rank；声明
rank 必须与 slot一致并完整覆盖 WORLD。每个 group 的 size/rank/members 严格使用 `type(value) is int`，拒绝
bool，且验证 self index、一致 membership 和 WORLD partition。

拓扑使用 MCore 0.17 `RankGenerator` 的 `tp-cp-ep-dp-pp` 排序验证笛卡尔闭合：dense 维度为
TP/CP/DP/PP，expert维度为 expert-TP/EP/expert-DP/PP；`dp_cp` 是 DP×CP 派生 group，不是自由维度。source
owner rank只作诊断，但必须与可信 payload 发布 rank及对应 dense/expert owner坐标一致。

load 时目标进程先由官方 sharder按新 DP/expert-DP重新计算 ownership，再让新 owner 用同一逻辑 key把完整 payload
直接加载到现有 pinned slab。source owner 变空、目标空 rank 获得新 state、或目标 owner 变空都合法，不创建伪
tensor。rollback 只恢复目标 rank load 前的本地 state。

允许变化的只有 DP/expert-DP ownership。TP、CP、PP、EP、expert-TP、parameter partition、算法配置、leaf tree
和逻辑 state schema 必须完全一致。旧 Muon schema 或缺失 identity fail-closed。

## 7. 支持矩阵与明确限制

| 能力                                | AdamW                  | Muon                            |
| ----------------------------------- | ---------------------- | ------------------------------- |
| CPU-authoritative pinned FP32 state | 支持                   | 支持                            |
| 有界 GPU staged step                | 支持                   | 支持，完整 TP-local matrix unit |
| 同步 managed checkpoint             | 支持                   | 支持                            |
| DP reshard load                     | 支持 MCore DCP reshard | 仅 DP/expert-DP owner迁移       |
| managed async save                  | 支持，单 outstanding   | 不支持                          |
| async load                          | 不支持                 | 不支持                          |
| backward-tail prefetch              | 不支持                 | 不支持                          |
| partial model/optimizer load        | 保持现有 manager 语义  | 不支持；要求完整 replacement    |
| Muon TP mode                        | 不适用                 | 仅验证过的 `duplicated`         |
| Muon TP/expert-TP multislot         | 不适用                 | 不支持；要求 `buffer_count=1`   |
| overlap param gather                | 沿 MCore AdamW约束     | 两种 overlap配置均拒绝          |

Muon 的 `blockwise`、`distributed` 或其他未完成真实数值/collective验收的 mode 在 model
枚举、builder、parameter attribute mutation 和 allocation 前拒绝。PP 拓扑变化、模型拓扑 reshard、Muon
async、CUDA graph optimizer state 和 8 卡大模型验收均不在当前边界内。

## 8. 配置、恢复与运维

内部配置示例：

```python
engine.configure_gpu_staged_adamw(
    GPUStagedAdamWConfig(
        buffer_count=2,
        bucket_size_mb=128,
        checkpoint_snapshot_root="/trusted/local/scratch",
        checkpoint_snapshot_chunk_mb=64,
    )
)

engine.configure_gpu_staged_muon(
    GPUStagedMuonConfig(
        buffer_count=1,  # TP/expert-TP > 1 时必须为1
        slot_size_mb=128,
        tp_mode="duplicated",
        checkpoint_snapshot_root="/trusted/local/scratch",
        checkpoint_snapshot_chunk_mb=64,
    )
)
```

snapshot root 应由作业预创建并为每个 rank 提供足够空间。使用共享 filesystem 时，容量按 non-zero `f_fsid` 汇总；使用
node-local root 时各 filesystem 分别计算。不要把 root 配置为 symlink，也不要在运行中 rename/替换 root 或
snapshot 目录。

故障恢复原则：

- preflight/snapshot create 失败：修复空间、权限或 schema 问题后重新发起完整 load；
- rollback action 失败：保持同一 manager，修复 I/O/注入故障后再次发起完整
  `with_model + with_optimizer + with_rng` load，manager 先重试 retained journal；
- `RELOAD_REQUIRED`：只能通过同一 manager 授权的完整 replacement load 恢复训练；
- post-commit cleanup 失败：下一次 save/load 会先统一 retry，禁止手工删除无法证明 ownership 的目录；
- orphan 目录只允许按 owner marker 做只读发现/告警，不自动删除不确定归属内容；
- async `FAILED` request 保持 mutation fence fail-closed，wait/teardown 继续收割同一 worker
  recovery journal，complete marker 保持不可见。

## 9. 资源与验收结果

资源不变量：

- 初始化、step drain 和 load 完成后 `cuda_state_numel=0`；
- GPU optimizer 峰值随 slot/最大 Muon unit 增长，不随本 rank 总 state 线性增长；
- rollback 磁盘容量约等于目标 rank 权威 FP32 state 加 header/owner metadata；
- 正常 load 额外写一遍 rollback payload；失败回滚再读取 pending chunks；commit cleanup 只删除 artifact；
- 64 MiB 默认 chunk 把 rollback 额外 RSS workspace 限制在固定 chunk 量级，避免旧式完整 optimizer clone
  造成的全量 RSS 峰值；
- managed async 没有独立 source-staging 信号，因此训练 mutation 阻塞时间保守覆盖完整 MCore async finalize。

开发验收覆盖了小模型 AdamW/Muon、混合和空 leaf、DP=1/2、TP/EP 小拓扑、DP owner 迁移、连续 step、故障注入、FD/目录/worker
泄漏和 AWEX residency。数值测试使用 `torch.testing.assert_close()` 明确 `rtol/atol`，并检查
model、master、Muon momentum、AdamW moments 及 param-group metadata。Qwen3-30B-A3B 8 卡资源复验不属于
本次开发侧验收。

关键命令：

```bash
uv run pytest -q tests/test_gpu_staged_optimizer.py
uv run pytest -q tests/test_gpu_staged_optimizer_checkpoint.py -m 'not slow'
uv run pytest -q tests/test_gpu_staged_optimizer_awex.py
uv run pytest -q tests/test_gpu_staged_muon.py
uv run pytest -q tests/test_gpu_staged_muon_checkpoint.py -m 'not slow'
uv run pytest -q tests/test_megatron_async_save.py
uv run pytest -q tests/test_megatron_optimizer_config.py
uv run pytest -q tests/v2/weight_update/test_megatron_adapter.py
```

真实拓扑入口位于 `tests/torchrun/run_gpu_staged_*.py`，包括 staged step、同步 checkpoint、managed
async、Muon topology/all-gather、fixed-DP 与 DP reshard、manager recovery 等故障矩阵。
