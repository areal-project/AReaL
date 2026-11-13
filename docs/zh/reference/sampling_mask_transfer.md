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

SGLang 适配使用带指纹校验的安装补丁，目标版本为当前锁定的 0.5.10.post1； 不兼容版本会拒绝安装。NCCL 扩展针对运行环境的头文件和共享库编译，要求设备
API 与 GDAKI 可用。

当前基础 `uv.lock` 随 PyTorch 固定 NCCL 2.27.5；此次验证环境单独安装了 `nvidia-nccl-cu12==2.31.2`。新环境或重新执行
`uv sync` 后，需要重新选择兼容的 NCCL 包，包含头文件和共享库，例如在已激活的验证环境运行：

```bash
uv pip install --no-deps nvidia-nccl-cu12==2.31.2
```

开关会为新建 worker 设置库搜索路径，但不会自动安装或升级依赖；上述环境覆盖未并入基础锁文件。

## 验证范围

本分支从 main 独立接入上述功能。CPU 数值与兼容性测试可运行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. pytest -q -m 'not slow' \
  tests/test_sampling_mask.py tests/test_http_error_diagnostics.py \
  tests/test_remote_inf_engine_r3.py tests/test_rollout_controller.py
```

此命令跳过需要 CUDA 的数值用例，并排除启动真实推理服务的慢速集成测试。 多 GPU、跨节点 GIN、完整 rollout 到训练以及固定输出性能测试需要相应硬件单独运行。
训练 CP/PP 的端到端组合尚未验证；其他集成环境下的历史性能结果不能直接作为本分支的验收结果。 同后端 FP32 验收阈值为 logprob 最大绝对误差
1e-5、梯度最大绝对误差 1e-4； 跨 SGLang/Megatron 后端的 BF16 误差需要另行报告。
