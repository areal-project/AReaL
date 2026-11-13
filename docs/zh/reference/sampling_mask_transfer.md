# Sampling mask transfer

此功能面向 AReaL v1 的 SGLang + Megatron 训练。唯一功能开关是顶层
`sampling_mask_transfer: true`。`gconfig.temperature`、`top_p`、`top_k` 仍然是 采样配置；开启功能不会自动增加
top-k 上限。

```yaml
sampling_mask_transfer: true
gconfig:
  temperature: 0.8
  top_p: 0.95
  top_k: 128  # 示例中的显式采样选择，不是功能默认上限
```

## 概率与训练

SGLang 保存实际用于抽样的集合和该集合归一化后的旧 logprob。当前实现采用 SGLang joint top-p/top-k 语义：top-p
的累积概率来自温度变换后的全词表分布， 再与 top-k 取交集。训练时只使用保存的集合，不重新计算 top-p/top-k。

对集合 S、动作 a、温度 τ，训练 logprob 为 `z[a]/τ - logsumexp(z[S]/τ)`。分子与分母都有梯度。PPO ratio 的分母固定为
rollout 的旧 logprob；开关自动关闭 decoupled/proximal 重算，固定 token 级 importance ratio，并覆盖
`reuse_train_logp` 模式。优势仍使用同一批 rollout。

Megatron 普通输出头和分块输出头都支持受限归一化。分块反向重算 logits，并 直接构造集合内的梯度，集合外纯 actor 梯度为零。额外 KL 等损失须另行解释。
参考模型仍使用全词表概率。若 `kl_ctl != 0`，只接受 `kl_estimator: k1`，含义是受限 rollout 策略相对于全词表参考策略的 KL
样本估计；k2/k3 启动报错。分块输出头的 entropy 是不可微统计值。

## 数据协议与生命周期

CSR 是 Compressed Sparse Row（压缩稀疏行）：把各生成位置的候选 ID 连续拼接，用 offsets 指出第 t 个位置的区间
`IDs[offsets[t]:offsets[t+1]]`。因此存储量随候选总数增长，不创建完整 `[batch, length, vocab]` mask。

- 协议版本为 1。候选 ID 使用小端 uint24，每个候选严格占 3 字节；词表不能超过 2²⁴。CPU 缓存以 CSR 存储，offsets 为 int64。
- 每个生成位置携带 source、producer nonce、row ID、协议版本和权重版本。 进入训练张量后，前三项作为 int64 列随 token
  一起拼接、重排、padding 和分片； 权重版本沿用 rollout 的 versions。
- 控制面只发送行引用、计数、生成 token、版本和校验信息；候选 ID 通过 NCCL GIN GDAKI 传输。每个连接使用两个 8 MiB 注册窗口和
  credit/ready 信号。
- 推理端在模型/KV cache 规划前预留窗口，训练 peer 就绪后注册；同一连接跨批复用。 开关自动在进程启动前设置 `NCCL_CUMEM_ENABLE=1`。
- 开关为训练和推理进程选择已安装 NCCL 包的共享库；GIN 启动时查询实际加载的 NCCL runtime，要求至少 2.31，避免 PyTorch wheel
  编译时版本与运行库不同导致误判。
- 训练端按分片拉取 CPU CSR，按输出头分块搬运并解包。批次身份忽略顺序、重复 引用和 padding，因此同一训练分片的多轮 PPO 可以复用候选。
- 共享前缀可被不同训练分片消费。源缓存按允许的策略版本窗口回收，而非在第一个 接收方确认后删除。跨分片重新分配仍需把数据传到新的接收方。
- 推理端的小 decode 块在首次读取时合并，保留原 row ID；每次合并暂存不超过 8 MiB ID 和 16k 行元数据。部分过期块在最后一行过期后释放。
- 缺失集合、重复/越界候选、动作不属于集合、版本或校验和不一致明确报错。

## 资源与指标

启动前根据最大生成位置数、显式 top-k（未限制时按词表大小）、DP 分片、保留的 策略版本和分块暂存计算保守预算。估计超过每 rank 30,000,000,000
字节时退出。 这是功能新增存储的预警，不是模型、优化器和 KV cache 总内存的精确预测。

uint24 ID 净载荷为 `3 × sum(|S_t|)` 字节。512 × 128k 个生成位置、每位置 128 候选时约为 **24 GiB**；int64
offsets 约 **512 MiB**，另有引用、缓存索引、 token/版本、传输窗口和临时重排数据。多个保留版本会增加源缓存预算。

训练统计中的 `sampling_mask/lifetime/*` 包含候选均值、P95、P99、最大值、累计接收位置数、ID 净载荷、CSR offsets、引用张量、控制面
JSON body、GIN 应用载荷和当前 CPU 缓存总量/最大 rank 占用。所有 actor rank 通过 CPU Gloo 汇总；逻辑候选统计只取最后 PP
stage 的 TP=0，实际传输按接收连接 owner 计数，避免 TP 重复计数。 位置数和载荷累计到当前训练生命周期；跨分片共享前缀的独立接收会分别计数。CPU
缓存指标为当前占用，合并暂存另计。 `gin_payload_bytes` 不含网络协议头和重传；控制面指标仅统计 JSON body。跨机测试可通过
`run_sampling_mask_gin.py --report-dir ...` 记录每端网卡计数器差值，该差值还包含同期后台通信，不能等同于进程独占的线速字节数。

`sampling_mask/http/*` 按统计导出区间汇总所有 rollout worker 和代理进程成功接收的 HTTP 生成响应（含返回 abort
的响应）：`generate_response_body_bytes` 是完整响应解压后的应用字节（含原有 token/logprob
数据），`generate_responses` 是响应数，`reference_json_value_bytes` 单列线上五整数引用数组的紧凑 JSON value
大小。后者是完整 body 的子集，不能相加；它也不同于训练侧 24N 的三个 int64 引用张量。HTTP 头、失败请求及链路重传不在这些 body 指标中，NIC
实测另列。 异步预取、暂停和续传可能使一次导出包含尚未用于当前训练批次的响应，因此响应数不等于已消费样本数； HTTP 字节指标按导出时间区间解释，GIN
接收量按实际读取的训练分片解释。

## 验证与当前限制

数值用例位于 `tests/test_sampling_mask.py`，涵盖 FP32 概率/梯度参考、整体平移、 集合外扰动、clip、单元素集合、全支持退化、uint24
和 CSR 重排。GPU 测试另有：

```bash
torchrun --standalone --nproc-per-node=2 tests/torchrun/run_sampling_mask_tp.py
torchrun --standalone --nproc-per-node=2 tests/torchrun/run_sampling_mask_pipeline.py
torchrun --standalone --nproc-per-node=2 tests/torchrun/run_sampling_mask_gin.py
```

目前不支持 repetition/frequency/presence penalty、min-p、logit bias、grammar、 自定义 logits
processor、speculative decoding、推理 PP/attention CP、蒸馏、 tree training 和 auxiliary
token/top logprobs；这些路径报错，不静默采用另一种 概率定义。训练请求必须使用配置中的 temperature/top-p。

GIN peer 必须位于不同物理 GPU。同机同 GPU 的 rollout 与 actor 需要 actor TP 组内另一张 GPU 接收后广播；TP=1
且双方只有同一张 GPU 时明确拒绝。

SGLang sampler 通过 `register_sampler_backend()` 注册，sampling-mask 接入不再改写已安装的 SGLang
源文件。未知版本会在启动前报错。

| 接口                                 | 0.5.10.post1                                           | 0.5.19                                | 0.5.20                                |
| ------------------------------------ | ------------------------------------------------------ | ------------------------------------- | ------------------------------------- |
| sampler 注册与 `forward` 调用契约    | 相同                                                   | 相同                                  | 相同                                  |
| 加载方式                             | AReaL HTTP launcher + 可序列化 scheduler spawn wrapper | 原生 `sglang.srt.plugins` entry point | 同 0.5.19                             |
| 有效配置读取                         | `get_global_server_args()`                             | `runtime_context`                     | `runtime_context`；旧 accessor 会报错 |
| deterministic 模式保留自定义 backend | 进程内包装旧方法                                       | 原生 `HookRegistry` around hook       | 原生 `register_resolution_hook`       |
| 分块 metadata 累积                   | 进程内兼容包装                                         | SGLang 原生支持                       | SGLang 原生支持                       |

新版插件入口名为 `areal_sampling_mask`，仅在 AReaL 的功能开关开启时注册。更新代码后须在 rollout 使用的 Python 环境重新安装
AReaL，以更新 entry-point 元数据。若设置了 `SGLANG_PLUGINS` 白名单，需包含该入口；缺失安装元数据或白名单排除会直接报错，不静默回退。

0.5.10 的兼容入口仅支持 HTTP、单 tokenizer worker，不支持 Ray、gRPC 和 encoder-only。旧环境若已经有上一版源码补丁，会识别原有
metadata 累积，避免重复；本次不会自动撤销共享安装中的旧补丁。重新安装干净的 SGLang 后也可使用兼容入口。

0.5.19/0.5.20 的适配验证覆盖 sampler 注册、插件加载、配置解析钩子与 metadata 接口，尚未完成这两个版本的 GPU 端到端训练验证。仓库仍锁定
0.5.10.post1；现有 AWEX 版本白名单也尚不接受正式版 0.5.19/0.5.20。本次没有升级依赖或扩充 AWEX 白名单，sampler
接口兼容不等于整个训练栈兼容。

NCCL 扩展针对运行环境的头文件和共享库编译，要求设备 API 与 GDAKI 可用。

当前基础 `uv.lock` 随 PyTorch 固定 NCCL 2.27.5；此次验证环境单独安装了 `nvidia-nccl-cu12==2.31.2`。新环境或重新执行
`uv sync` 后，需要重新选择兼容的 NCCL 包，包含头文件和共享库，例如在已激活的验证环境运行：

```bash
uv pip install --no-deps nvidia-nccl-cu12==2.31.2
```

开关会为新建 worker 设置库搜索路径，但不会自动安装或升级依赖；上述环境覆盖未并入基础锁文件。

截至 2026-11-12，单元测试、两卡 TP 数值测试、两卡 GIN 三轮 PPO 复用测试、小模型完整 AReaL 两次更新以及 35B BF16 WeftMoE 的 8
GPU / 8K / 512 样本单批训练已通过。 目标单批实际训练 3,106,912 个生成位置，候选总数 8,225,505，uint24 ID 净载荷
24,676,515 字节，GIN 应用载荷与之相等。 同后端 FP32 测试采用 logprob 绝对误差 1e-5、梯度绝对误差 1e-4；SGLang 与
Megatron BF16 的首轮 logprob 差异另行统计，不能用跨后端偏差替代上述单元测试。训练 CP/PP 的端到端组合尚未验证。 吞吐下降不超过 10%
的性能目标已在下述固定输出配置通过；多种子短跑和真实模型同批三轮 PPO 也已完成。 这些验收覆盖当前测试配置，不代表长程训练收益或全部并行布局已经验证。

## 已完成的资源实测

目标单批使用 Qwen3.5-35B-A3B-Base、DAPO、BF16 WeftMoE，8 GPU，actor TP2/DP4/EP8，推理 TP8/EP8/DP
attention8；context 8K，512 样本，显式 temperature=1、top-p=0.95、top-k=128。

| 指标                                         |                           实测 |
| -------------------------------------------- | -----------------------------: |
| 有效生成位置                                 |                      3,106,912 |
| 候选均值 / P95 / P99 / 最大值                |          2.6475 / 8 / 18 / 128 |
| ID 净载荷 / GIN 应用载荷                     |  24,676,515 / 24,676,515 bytes |
| CSR offsets                                  |               24,859,296 bytes |
| 引用张量                                     |               74,565,888 bytes |
| 控制面 JSON body                             |              165,527,761 bytes |
| 训练 CPU mask 缓存最大 rank / 所有 rank 合计 | 59,670,668 / 453,843,270 bytes |
| 整机 GPU 峰值最高卡                          |                    114,754 MiB |
| 整机 CPU 使用峰值 / 相对启动前增量           |  753,461,412 / 644,471,448 KiB |

该批实际长度和平均候选数均远小于 128k × 128 的示例，因此 ID 载荷低于 24 GiB。offsets 比 8N 多出的部分来自每个 CSR 分块的结束
offset；引用张量为 24N。CPU 缓存合计包含 TP 副本，ID 传输只统计实际 owner。 整机内存包含模型、优化器、KV cache、Ray
等，不等于功能新增内存。此批在后续控制面去重及源块合并优化之前运行，后续性能结果会单独列出。 整次运行（含初始化和退出）HCA 端口计数器合计发送 28,250,276
bytes、接收 28,250,664 bytes；包含控制流量和后台通信，同机传输同时出现在发送及接收计数中， 不能将两者相加后认作独占的 mask 网络载荷。

两机独立 GIN 测试使用两个 8 MiB slot，96 次传输及复用校验通过。应用载荷 402,653,248 bytes；主链路发送/接收计数器分别一致为
410,598,948 bytes，反向确认方向为 6,752,444 bytes。端到端测试循环约 0.096 秒，包含测试张量构造和内容校验；这不是训练整轮吞吐。

首轮 SGLang / Megatron BF16 logprob 绝对误差均值约 0.01616，超过 1e-3、1e-2、1e-1 的比例分别为
41.88%、30.14%、3.68%；ratio 绝对误差均值约 0.01614、最大 8.568。这里没有宣称跨后端初始 ratio 达到 FP32
单元测试精度，误差尾部需在长程训练中持续监测。

## 固定输出性能对照

同一节点、同一 35B BF16 WeftMoE 模型、8 GPU、context 8191；每批 128 样本，每条固定生成 4096
token，ignore_eos=true，显式 temperature=1、top-p=0.95、top-k=128。两组都使用 native RLVR
workflow，确保关闭组也实际应用 top-k。固定 seed=19、学习率和 weight decay 为 0，以保持测量期间模型权重一致。

每组运行三轮，首轮包含预热，不计入稳态平均。完整迭代使用相邻训练完成日志的墙钟间隔，包含 offload/onload、AWEX 权重同步等阶段。

| 指标                        |        关闭 |        开启 |
| --------------------------- | ----------: | ----------: |
| rollout 平均秒数            |      37.538 |      41.010 |
| PPO 更新平均秒数            |      27.862 |      29.610 |
| 完整迭代平均秒数            |     100.239 |     108.035 |
| rollout 生成 tokens/s       |    13,966.9 |    12,784.4 |
| PPO 生成 tokens/s（含加载） |    18,817.7 |    17,706.5 |
| 完整迭代生成 tokens/s       |     5,230.4 |     4,853.0 |
| 最高单卡显存 MiB            |     111,363 |     115,232 |
| 整机 CPU 使用峰值 KiB       | 753,523,008 | 753,564,140 |

最终开启组包含 HTTP 统计和非有限归一化检查，两轮稳态间隔分别为 107.307、108.762 秒。 完整迭代时间增加 **7.78%**，吞吐下降
**7.22%**，在此配置下达到 ≤10% 的目标。显存和 CPU 值是每两秒采样的整机占用，包含模型及框架，不是分配器的逐次分配峰值。512 样本、8K
上限单批已完成正确性和资源验证，但该规模尚未进行固定输出的成对性能测量。

最终开启组三轮合计接收 1,572,864 行，候选均值 3.2387、P95=9、P99=28、最大值 128。 ID/GIN 载荷均为 15,282,039
bytes；offsets 12,585,984 bytes；引用张量 37,748,736 bytes； 传输控制 JSON body 41,171,036
bytes。三个统计区间合计 384 个生成响应，完整解码 HTTP body 115,724,778 bytes，其中引用 JSON value 为 74,609,144
bytes；后者是前者的子集。

本次运行环境：Python 3.12、PyTorch 2.9.1 CUDA 12.9、Megatron-Core 0.19.0、Megatron-Bridge
0.6.0、SGLang 0.5.10.post1、WeftMoE 2.0.0、NCCL runtime 2.31.2。NCCL_DEBUG=WARN。 性能对照节点为 8 ×
NVIDIA L20X，每卡 143,771 MiB，驱动 570.148.08。

## 真实模型同批三轮 PPO

35B BF16 WeftMoE、8 GPU、8K context，32 样本，最大生成 2048 token，学习率 1e-6。
测试仅计算一次优势，在发布新权重版本前对同一份数据调用三次现有 `actor.ppo_update`， 没有增加新的公开 epoch 配置。三轮均完成，actor 与 mask
指标全部有限。

实际有效生成位置为 **59,913**；三轮训练计数 **179,739 = 3 × 59,913**；累计 mask 接收 仍为 **59,913 行**，ID 与 GIN
累计载荷均为 **518,793 bytes**。因此第二、第三轮没有重复 从 rollout 传输这些集合。默认训练循环的更新轮数保持其原有行为；本测试显式重复同批更新。

## 多随机种子短程对照

两组均为 35B BF16 WeftMoE、8 GPU、8K context，32 样本/批，最大生成 2048 token，
temperature=1、top-p=0.95、top-k=128，学习率 1e-6。每个种子完成两次更新，以下为第二次 更新的 W&B 统计。关闭组在 233、开启组在
238；表中吞吐使用当轮有效生成位置数除以完整 迭代时间，供记录运行状态使用，同节点性能判断采用前面的固定输出对照。

| 开关 | seed |  reward | entropy |  approx_kl | clip fraction | 生成 tokens/s |
| ---- | ---: | ------: | ------: | ---------: | ------------: | ------------: |
| 关闭 |   23 | 0.37500 | 0.46646 | -0.0007280 |     0.0011816 |        1059.4 |
| 关闭 |   29 | 0.68750 | 0.40000 | -0.0007752 |     0.0009455 |         997.3 |
| 关闭 |   31 | 0.43750 | 0.49902 | -0.0009310 |     0.0011437 |         942.2 |
| 开启 |   23 | 0.31250 | 0.36010 | -0.0006861 |     0.0006189 |        1026.9 |
| 开启 |   29 | 0.15625 | 0.34829 | -0.0005867 |     0.0001469 |        1053.9 |
| 开启 |   31 | 0.40625 | 0.30429 | -0.0006341 |     0.0012960 |        1005.8 |

三个种子的 reward 均值 ± 样本标准差：关闭 0.5000 ± 0.1654，开启 0.2917 ± 0.1263。 这只是各 64
个样本的短程检查，没有证明奖励改善或收敛。 开启组 entropy 对应受限策略，关闭组对应原有全词表训练策略，二者定义不同。 `approx_kl` 沿用现有 PPO
指标，是 `new_logprob - proximal_logprob` 的样本均值； 开启本功能时 proximal 固定为 rollout 的 old
logprob。该指标可能为负； 本次 `kl_ctl=0`，未将其解释成参考模型 KL。六次成功运行均完成两次更新，没有 NaN、Inf 或有效 token 丢失；开启
seed 31 首次初始化失败后的同配置重试详见环境限制。

## 测试记录与环境限制

CPU 单元与兼容性回归共 134 passed、5 skipped。五个跳过项需要 CUDA，已另行在 GPU 上验证； 实际 Megatron 关闭路径与全支持受限路径的
loss、ratio、hidden/weight 梯度对照也通过。 两卡 TP 测试覆盖普通和分块输出头，GIN 流水线测试覆盖 129 个 decode
块、分块窗口循环使用、 三轮 PPO、逆序加载与第二消费者读取共享前缀。

变更文件的 pre-commit 检查通过。全库检查中的 uv-lock hook 会用当前 uv 重写既有锁文件中的 无关依赖；这些无关改动已恢复，因此没有将全库
pre-commit 标记为通过。

多种子验证发现既有 Ray 退出清理问题：一台节点存在约 14.8 万个僵尸进程，八个并发 `psutil.children(recursive=True)` 全机扫描实测耗时
41.847 秒，超过外层 30 秒清理超时。 launcher 被终止后，RPC 子进程成为孤儿并保留模型内存，使下一次模型加载触发 CPU 内存阈值。 该失败发生在
rollout 和 mask 传输之前；失败日志保留。验证脚本增加了按本次实验目录精确匹配的 退出清理，确认内存回收后重试。本功能没有修改 Ray
launcher/scheduler，环境中的僵尸进程问题 仍应单独处理，不能将其视为已由 sampling-mask 功能修复。

## 原生插件接口验证

`tests/test_sampling_mask_sglang_plugin.py` 包含禁用路径、版本拒绝、entry-point 元数据与白名单检查。可通过
`AREAL_TEST_SGLANG_SOURCES` 指定上游源码快照，执行三版本注册与 factory、构造函数、插件加载器、deterministic
hook、metadata 累积的 CPU 契约测试。快照目录为 `v<version>/srt/`；这些测试使用真实上游相关代码并隔离 GPU runtime，不代表完整
SGLang GPU 测试。

对于 0.5.19/0.5.20，官方 `return_sampling_mask` 也是可用的数据接口；本实现继续保留自定义 sampler，使采样、保存支持集和 old
logprob 来自同一计算，并接入现有 uint24/NCCL GIN 通路。原生插件机制消除的是修改安装文件的接入方式，不是删除自定义采样逻辑。
