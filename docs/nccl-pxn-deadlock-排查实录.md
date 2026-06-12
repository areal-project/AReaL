# 一次 NCCL P2P 死锁排查实录:从 32 卡集体静默到 PXN staging 头阻塞

AWEX colocate 权重传输(train 32 rank → infer 32 rank,跨节点 NCCL P2P)在 Round 0
之后整组静默,~5 分钟后 actor 侧报 CUDA illegal memory access,job 自毁。从 2026-06-08
到 06-10 排查了十几轮,最终根因是 **NCCL PXN staging 的头阻塞死锁**,workaround 是
`NCCL_P2P_PXN_LEVEL=0`。这篇记录完整的证伪链、每一步用的工具和对应日志,
原始过程记录在 `docs/awex 调试问题记录.md` Problem 60-70。

## 现象

环境:4 节点 × 8 卡,colocate 模式(train/infer 共卡,`mem_fraction_static=0.60`),
transport 是 recursive partition + `batch_isend_irecv`(`awex/transfer/nccl_stream_batch.py`)。

每次 weight transfer 启动后:

- transfer_rank 0 一个 rank 0.36s 完成 Round 0,其余 31 rank 卡死;
- rollout.log `chunk_idx=0 ENTER=32 / EXIT=0`,之后完全静默 5 分钟;
- train 侧等 `weights_update_finished` 超时(写死 300s)→ RPC 重试第二次 convert
  → 与第一次挂起的 IPC 显存冲突 → actor 侧 CUDA IMA → teardown。

注意 **IMA 是超时重试的次生 crash,不是死锁本身**。把这两件事解耦,是后面所有
排查的前提——否则会一直追着 IMA 的栈跑偏。

## 证伪链:先排除掉的东西

排查的主线不是"猜根因然后验证",而是逐维度做"降 X 看是否还 hang"的硬证伪。
每一轮都是真实 4 节点 job(~14 分钟/轮):

| 假设 | 实验 | 结果 |
| --- | --- | --- |
| plan 数据不配平(op 数/numel/FIFO 序) | AWEX_DUMP_PLAN dump,离线逐有向边比对 472 边 | 0 mismatch,证伪 |
| 跨 round 时序错配 | 加 round-barrier | 仍 hang,证伪 |
| 单 batch 并发规模太大 | per-peer batch(一次只对一个 peer) | 仍 hang,证伪 |
| 每 batch op 数太多 | `AWEX_CHUNK_OPS` 128→8(trial-0610-1434) | 同位置同形态 hang,证伪 |
| 最低并发也不行? | `AWEX_CHUNK_OPS=1`(trial-0610-1509) | 仍 hang,并发维彻底收口 |
| 缺 warmup 建链 | 复查日志:warmup 32 次 verification successful;且后续发现卡死边此前已成功通信 7 次 | 不是首通边问题,证伪 |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` 单硬件队列 | 进程级 env dump 实测 | sglang 在 spawn scheduler 前无条件覆盖为 8,该假设无实验价值 |
| 共卡显存压力 | live transfer 时实测 driver_free=113GB/140GB | 余量充足,证伪 |

证伪链的价值:**"降 X 无效"比"加机制看能否消 hang"更有判别力**。三个并发子维度
(peer 并发、op 数、step_size=1)各降一个数量级仍 hang,说明死锁根本不在规模这一维。

## 工具一:py-spy 区分 host 层还是 GPU 层

trial-0610-1509,hang 窗口内对全部 sglang scheduler 进程抓 py-spy dump
(存档 `/storage/openpsi/users/chucai.dzq/colocate-pyspy-trial0610-1509/`):

```
synchronize (torch/cuda/__init__.py:1083)
_execute_ops_concurrent (awex/transfer/nccl_stream_batch.py:367)
execute_recursive_partition_stream_transfer (nccl_stream_batch.py:277)
_run_chunked (nccl_stream_batch.py:648)
...
```

关键判据:卡在 line 367 的 `torch.cuda.synchronize()`,**不是** line 358 的
`work.wait()`。per-peer 流程是 `nd_irecv → work.wait() → synchronize()`,
work.wait 只等"op 已 enqueue"的 CUDA event。所以:

- 卡 `work.wait` = host 端没把 op 排上去,调度层问题;
- 卡 `synchronize` = host 已 launch、**GPU 端 NCCL kernel 永不 drain**,数据层问题。

本例是后者:receiver 确实 post 了 recv,但数据没到。死锁从此锁定 GPU NCCL P2P 运行时。

副发现:这轮想开 NCCL 日志,设了 `NCCL_DEBUG="DEBUG"`——**NCCL 只认
WARN/INFO/TRACE/VERSION,非法值静默吞掉,0 行日志**。grep 不到 NCCL 行时先核对取值。

## 工具二:AWEX_P2P_TRACE + CUDA_LAUNCH_BLOCKING 画出等待图

给 transport 加了 per-op 进度日志(每 peer 打 pre-wait / post-wait / post-sync),
配合 `CUDA_LAUNCH_BLOCKING=1` 把卡点前移到具体 op 的 launch。
trial-0610-1553(job 930511-930513,`NCCL_DEBUG=TRACE,SUBSYS=INIT,P2P,NET`):

这次 transfer 先健康跑了 13 秒、几百个 per-peer batch,然后全停。每个 rank 的
最后一条 TRACE 都是 `pre-wait`,据此拼出 32 rank 的卡死指纹(rank→正在等的 peer):

- peer=3 ← rank 17,19,21,23,24-31(12 个 sender 排队 send→3)
- peer=17 ← rank 3,6,7,11,14,15
- peer=24 ← rank 5,9,12,13(24 自己还卡在 send→3)
- peer=25 ← rank 16,18,20,22(25 还困在 Round 0)
- 其余都是 Round 1 在等还困在 Round 0 的对端

**整张等待图唯一的互等对是 (3, 17)**:rank3 卡 `recv←17`,rank17 卡 `send→3`,
双方 nops=1、都停在 `batch_isend_irecv` 内部(op 已 post 进 NCCL)。其余 31 个 rank
全是 per-peer 串行走表被这一条边队头堵死的传递性等待。看 32 卡集体卡死,
**先找唯一互等对,别被 12 路 fan-in 的表象带偏**——(3↔17) 一通,全图解锁。

## 工具三:NCCL TRACE connect 行判别"建连 vs 等数据"

同一轮的 NCCL TRACE 里,wedge 边 17→3 **两个方向、两侧进程的 channel 都有 connect 记录**:

```
17[1] -> 3[3] [send] via NET/IB/3(19)/GDRDMA/Shared    (sender 侧, Channel 03/1、11/1)
17[1] -> 3[3] [receive] via NET/IB/...                  (receiver 侧同边)
```

connect 行齐全 = rendezvous 已完成,死锁在 connected 之后的数据/completion 阶段;
反之停在 connect 前才是建链死锁。这一行还埋了破案的伏笔:**`NET/IB/3(19)` 表示
这条跨节点边走的是经 rank19 的 PXN 中转**,而成功边 3→17 走的是 `NET/IB/1(1)`。

另一个重要观察:卡点位置 run-to-run 漂移(1509 卡第一个 op,1553 跑了几百 op 才卡)
——**时序敏感 = 不可能是确定性的 plan/拓扑错误**,排查方向从算法转向 NCCL/IB 运行时资源。

## 工具四:自动 watcher + proxy 线程全景

5 分钟死锁窗口人工来不及登机抓现场,写了自动 watcher(`watch_wedge.sh` /
`capture_wedge.sh`):检测 P2P-TRACE 停滞 75s 后自动对 4 节点全部进程抓
内核栈(`/proc/<pid>/task/*/stack`)+ py-spy + nvidia-smi。

trial-0610-1727(**换了 4 台完全不同的物理机**),watcher 首战产物存档
`/storage/openpsi/users/chucai.dzq/colocate-wedge-trial0610-1727/`:

- 32 rank 的 rank→peer 指纹与 1553 **一字不差**,互等对仍是 (3↔17)。跨硬件 100%
  复现 ⇒ 排除硬件/链路,死锁确定性落在同一条逻辑边。
- **proxy 线程全景(本次金矿)**:32 进程共 98 个 `NCCL Progress` 线程,97 个睡在
  `futex_wait`,**唯一活跃(busy-poll,内核栈为空)的是 rank19 进程的 "NCCL Progress 8"**
  ——正是 wedge 边 17→3 的 PXN 中转节点 `NET/IB/3(19)`。
- receiver rank3 的 proxy 在睡觉 = recv 侧网络层从未参与。

判别结论:不是"数据在网络里走丢",而是 **PXN staging 路径上的单点 proxy 空转 +
receiver proxy 未参与**。`NCCL_P2P_PXN_LEVEL=0` 从候选实验升级为最优先。

## 工具五:transfer-only replay 给 transport 无罪开释

每轮全链路 14 分钟太慢,做了 plan-dump 驱动的纯 NCCL replay
(`examples/swe/colocation/flash/replay/`,commit 89cad42):从 AWEX_DUMP_PLAN 提取
per-edge op-size 表,torchrun 32 rank 合成精确尺寸 tensor,调**真实** transport 代码,
~20 秒/轮。忠实度先验证:表算出的各 rank Round0 op 数与 live 日志七点全中。

判别矩阵(981 chunks 全量):

| 变体 | 结果 |
| --- | --- |
| plain(纯 cudaMalloc) | NO WEDGE |
| + LD_PRELOAD=torch_memory_saver | NO WEDGE |
| + recv 进 TMS VMM region | NO WEDGE |
| + TMS pause→resume 后再 recv | NO WEDGE |

⇒ transport 算法、plan、NCCL 运行时本身、TMS 显存,单独或组合都不足以死锁。
剩余差异只在 live 的进程环境(sglang 已有的 TP/EP comm、IPC 导入上下文)。

**副产物——离线核查 196784 个 op 时撞出第二个独立 bug**:`mlp.gate.weight`
(MoE router)train 侧 bf16、sglang 侧 fp32,961 个 op dtype 不配平。receiver 按
fp32 post 4N 字节、sender 只发 2N 字节,recv 永等——精确解释了 trial-1509 为何
死在 chunk 7(首个 mismatch op 进 wire 的时刻)。但 1553/1727 死在 chunk 0 且
chunk 0 全部配平 ⇒ **同一症状下面是两个独立 bug**,按 chunk 索引把死锁对齐到
plan 序 op 是最锋利的切分刀。

## 破案:GPU util 全景闭合死锁三角

trial-0610-1836(dtype 修好后):chunk0 指纹与 1553/1727 逐字相同,证实 dtype 与
chunk0 死锁无关。capture 脚本修好 `--nv` 后拿到 GPU util 全景:

- **31/32 卡 util=0%**——包括 wedge 双方 rank3/rank17,kernel 根本没 launch;
- **唯一 100% 的是 rank19 的卡**——wedge 边的 PXN 中转,也是 1727 全集群唯一活跃 proxy。

死锁三角至此闭合:

1. rank19 自己的 send→3 kernel 在 spin,等 rank3 post recv;
2. rank3 的 recv←17 的 connect/launch 需要 PXN 中转 rank19 的 staging 资源;
3. rank19 的 staging 资源被自己 spinning 的 send 占住。

机制:NCCL PXN 让跨节点 P2P 经同节点别的 GPU 中转聚合后走 NIC;在 per-peer 串行
post 的 transport 下,R0P2 时 12 个 rank 同时 send→3 而 rank3 串行只 post 了一个
recv,未配对的 in-flight send 滞留在中转 rank 的 staging 缓冲,把已配对的边也堵死。
**多对一 fan-in + per-peer 串行 + PXN = 高危组合**。这也解释了前面所有"反常":
rank0 必完成(它的 send 永远第一个被配对)、卡点随时序漂移、降并发三连无效
(fan-in 结构从未改变)。

## 验证与修复

- **trial-0610-1854(判别)**:仅加 `NCCL_P2P_PXN_LEVEL=0`,chunk0 当场解锁,
  P2P-TRACE 计数从冻死 1112 飙到 371k+,chunk 推进 434+/981,远超 dtype 死点
  chunk7——两个修复同时验证。副作用:诊断 env(TRACE+LAUNCH_BLOCKING)拖慢
  传输 ~10×,把"慢"演成 300s 超时被杀。
- **trial-0610-1914(clean run)**:PXN=0 + dtype cast + `AWEX_COLOCATE_TIMEOUT_S=1800`
  + 撤全部诊断 env。**version 1 权重传输历史首次全链路闭环**:32×
  `Finished CHUNKED weights update`(981 chunks,~9 分钟)+ 32× `Signaled
  write_finished` + PPO step 1 完成。

修复清单:

| 修复 | 位置 | commit |
| --- | --- | --- |
| `NCCL_P2P_PXN_LEVEL=0`(根因 workaround) | yaml env | 308c15ed7 / f0da6bc11 |
| gate.weight dtype cast(send clone → recv dtype) | awex fork nccl_stream_batch.py | 0877ea4 |
| train 等待可配 `AWEX_COLOCATE_TIMEOUT_S`(300→1800) | areal/engine/awex_colocate.py | f0da6bc11 |
| rollout.setup_timeout 1800 | yaml | 2cbf9ce00 |
| watcher 抓栈基建 | watch_wedge.sh / capture_wedge.sh | 多次 |

PXN=0 的代价是跨节点 P2P 不再经中转聚合、直连本 rank NIC,对这种一次性大块权重
传输影响可忽略;若要根治,transport 应改成"phase 内 receiver 先全量 post recv,
sender 再逐 peer send",消除未配对 in-flight send。

## 可复用的经验

1. **IMA 按时序排首因**。actor 的 illegal memory access 全程都是超时重试的次生
   crash;先画时间线再认凶手。
2. **py-spy 看卡在 work.wait 还是 synchronize**,一刀切开 host 调度层和 GPU 数据层。
3. **NCCL TRACE 的 connect 行是"建连 vs 等数据"判别器**;connect 行里的
   `NET/IB/x(rank)` 直接告诉你这条边走谁中转。
4. **集体卡死先找唯一互等对**,其余都是 per-peer 串行下的排队效应。
5. **GPU util 全景 + proxy 线程 futex/busy 全景**:唯一 100% 的卡、唯一 busy-poll
   的 proxy,就是死锁三角的资源占有者。
6. **卡点 run-to-run 漂移 = 时序敏感**,别再查确定性的 plan/算法,查运行时资源。
7. **复现器忠实度先验证再下结论**;"replay 不复现"只有在 op 计数与 live 全中之后
   才有资格说明是环境差异。
8. **同一症状可以是两个独立 bug**(chunk7 dtype vs chunk0 PXN),按 plan 序 op
   索引对齐死点来切分。
9. **诊断 env 有代价**(TRACE+LAUNCH_BLOCKING 慢 10×),判别完必须撤干净;
   **写死的 timeout 是排查期隐形杀手**,每处 wait 都要 env 可配。
10. **NCCL_DEBUG 只认 WARN/INFO/TRACE/VERSION**,写错静默吞掉。

## 现场存档

- py-spy dump(trial-1509):`/storage/openpsi/users/chucai.dzq/colocate-pyspy-trial0610-1509/`
- watcher 全景(trial-1727):`/storage/openpsi/users/chucai.dzq/colocate-wedge-trial0610-1727/`
  (per-pid 内核栈 + py-spy + nvidia-smi,按节点分目录)
- replay 复现器:`AReaL/examples/swe/colocation/flash/replay/`
- 原始逐轮记录:`AReaL/docs/awex 调试问题记录.md` Problem 60-70
