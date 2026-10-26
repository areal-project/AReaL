# 统计合成 workload

生成器：[areal/elastic/workload.py](../../../areal/elastic/workload.py)；配置：[workload-spec.json](workload-spec.json)。
全部为人工先验，不宣称拟合真实业务。当前 v2 生成器使用固定类型分布，推理与工具严格交替。

## 形式化定义

生成参数及输出为：

```math
\Theta=\left(T,s,\lambda(t),\boldsymbol{\pi},\{\theta_c\}_{c\in\mathcal C}\right),
\qquad
\mathcal W=(E_i)_{i=1}^{N(T)},\quad E_i=(\mathrm{id}_i,a_i,c_i,G_i).
```

`Θ` 是整套生成参数：`T` 是外部到达窗口，`s` 是 seed，`λ(t)` 是 episode 到达率， `π` 是固定类型概率。`C` 是任务类型集合，`θ_c`
是类型 c 的初始输入、输出、工具返回长度、 轮数和执行时长分布等参数。

`𝒲` 是生成的整条 workload trace，按外部到达时间排序，由 `N(T)` 个 episode 组成。 `N(T)` 是在窗口 `[0, T)`
内到达的任务数，不是该窗口内完成的任务数。

`E_i` 是第 i 个 episode，即一道外部任务，包含以下四项：

| 符号 | 含义                                                                           |
| ---- | ------------------------------------------------------------------------------ |
| id_i | 该任务的唯一标识                                                               |
| a_i  | 外部到达时间，对应 `available_at`；单位为相对 workload 起点的秒数              |
| c_i  | 该任务的类型，属于类型集合 C                                                   |
| G_i  | 该任务内部的调用序列，包含交替的推理／工具节点、依赖、token 长度和工具执行时长 |

任务可以在 T 之后完成。生成参数不含模型、GPU、部署、batch、并发、缓存或调度策略。

### 1. 外部到达：非齐次 Poisson 过程

给定分段常数到达率，定义累计强度：

```math
\Lambda(t)=\int_0^t\lambda(u)\,du,\qquad
N(v)-N(u)\sim\operatorname{Poisson}\!\left(\Lambda(v)-\Lambda(u)\right),
\quad 0\leq u<v\leq T.
```

不相交区间的计数独立。固定窗口内计数的均值和方差均为 `Λ(T)`。 实现先生成单位强度 Poisson 流，再反演累计强度：

```math
\xi_j\overset{\mathrm{iid}}{\sim}\operatorname{Exp}(1),\qquad
S_i=\sum_{j=1}^{i}\xi_j,\qquad
a_i=\Lambda^{-1}(S_i),\quad S_i<\Lambda(T).
```

逆函数按分段广义逆理解；零到达率区间不消耗累计强度。恒定 λ > 0 时，等价于相邻到达间隔服从 Exp(λ)。 代码对应
`arrival_times()`。默认单次高峰的窗口 T = 1800 秒：

```math
\lambda_1(t)=
\begin{cases}
0.1,&0\leq t<600,\\
1.0,&600\leq t<1200,\\
0.1,&1200\leq t<1800.
\end{cases}
\qquad N(1800)\sim\operatorname{Poisson}(720).
```

720 是期望任务数；实际数量自然采样，不因训练 batch 而补齐或删题。 30 分钟指外部到达窗口，不是生成器或模拟程序运行时长。

### 2. 任务类型：与到达时间独立

类型顺序为 `(short, long_decode, tool_heavy)`。固定权重先归一化，各题独立采样类型：

```math
\pi_c=\frac{w_c}{\sum_{d\in\mathcal C}w_d},\qquad
c_i\overset{\mathrm{iid}}{\sim}\operatorname{Categorical}(\boldsymbol{\pi}),
\qquad \{c_i\}_{i\geq1}\perp N(\cdot).
```

例如可以固定为：

```math
\boldsymbol{\pi}=(0.4,0.3,0.3).
```

类型概率在整个窗口内固定，不读取外部到达或实际 admission 时间。 后续工作量分布条件于类型，因此不同指标可以通过共同类型产生相关性。

### 3. 轮数决定工具调用次数

给定类型 c，只采样一次推理轮数。前 R_i−1 轮各调用一次工具后进入下一轮， 第 R_i 轮输出最终回答并结束：

```math
R_i\sim D_{\mathrm{turn},c},\quad R_i\in\{1,2,\ldots\},\qquad
M_i=R_i-1,\qquad \mathbb E[M_i\mid c]=\mathbb E[R_i\mid c]-1.
```

因此 R_i = 1 时没有工具调用，不再另设工具调用概率。 轮数分布可按类型选择；例如沿用现有截顶几何分布， 令
`μ_c = mean_turns`、`K_c = max_turns`：

```math
H_i\sim\operatorname{Geometric}(p_c),\quad p_c=\mu_c^{-1},
\qquad R_i=\min(H_i,K_c).
```

几何分布支持集从1开始。截顶后的分布及均值为：

```math
\Pr(R_i=r\mid c)=
\begin{cases}
p_c(1-p_c)^{r-1},&1\leq r<K_c,\\
(1-p_c)^{K_c-1},&r=K_c,
\end{cases}
\qquad
\bar R_c=\mathbb E[R_i\mid c]=\frac{1-(1-p_c)^{K_c}}{p_c}.
```

因此 `mean_turns` 是截顶前均值，p_c 是该几何分布的停止参数。 轮数确定后，工具位置完全确定。示例参数沿用现有轮数配置；`h_c` 为轮间固定消息开销：

| 类型        | μ_c | K_c | h_c（token） | 工具类型 |
| ----------- | --- | --- | ------------ | -------- |
| short       | 1.5 | 4   | 16           | python   |
| long_decode | 3   | 8   | 16           | python   |
| tool_heavy  | 6   | 12  | 16           | terminal |

截顶后平均轮数依次约为1.48、2.88、5.33。

### 4. 初始输入、每轮输出、工具返回与执行时长

给定类型 c，分别指定初始输入、每轮输出、工具返回长度和工具执行时长分布：

```math
P_{i1}\sim D_{\mathrm{prompt},c},\qquad
O_{ir}\sim D_{\mathrm{output},c},\qquad
B_{ir}\sim D_{\mathrm{return},c},\qquad
D_{ir}\sim D_{\mathrm{time},c}.
```

输出 `O_ir` 在每轮采样；工具返回 `B_ir` 与执行时长 `D_ir` 在每个非最后一轮采样，即 r \< R_i。 首版可沿用裁剪
lognormal，各分布分别给出中位数 m、对数域标准差 σ、下界 ℓ、上界 u：

```math
Z\sim\mathcal N(0,1),\qquad
L_{c,x}=\min\!\left(u_{c,x},
\max\!\left(\ell_{c,x},m_{c,x}e^{\sigma_{c,x}Z}\right)\right).
```

m／σ 均指裁剪前参数；区间外样本压到边界，边界可有概率质量。 token 数再取 `max(1, round(L))`，采用 Python 的舍入规则；执行时长为正浮点秒数。

本次三组输入共用以下中位数。工具时长以固定基准 α = 1 second/unit 表示：

| 类型        | 初始输入 | 每轮输出 | 工具返回 | 工具时长（秒） |
| ----------- | -------- | -------- | -------- | -------------- |
| short       | 512      | 128      | 256      | 0.2            |
| long_decode | 2048     | 1024     | 256      | 0.5            |
| tool_heavy  | 1024     | 256      | 1024     | 5              |

初始输入 σ=0.4、范围\[128,4096\]；输出 σ=0.5、工具返回 σ=0.6， 二者范围均为\[各自中位数/4,各自中位数×4\]。工具时长
σ=1.0、范围\[0.01,120\]秒。

| 符号 | 含义                       | 当前实现                                         |
| ---- | -------------------------- | ------------------------------------------------ |
| P_i1 | 初始输入 token 数          | `prompt` 分布                                    |
| O_ir | 第 r 次推理的输出 token 数 | `output` 分布                                    |
| B_ir | 工具返回 token 数          | `tool_output` 分布                               |
| D_ir | 工具执行时长，单位秒       | `tool_work` 采样 W_ir 后乘显式 seconds/unit 系数 |

条件于类型，各项抽样使用分开的伪随机流。只直接采样初始完整输入；后续完整输入由历史递推：

```math
P_{i,r+1}=P_{ir}+O_{ir}+h_c+B_{ir},\qquad 1\leq r<R_i.
```

后续 `P_ir` 不独立采样；`prompt` 上下界只限制初始输入， 累计上下文不重新裁剪。`P_ir` 是完整逻辑输入长度，实际 prefill
工作量还取决于实验中的缓存策略。

工具时长是在固定工具执行环境下的服务时间，不含等待空闲工具／沙箱的排队时间：

```math
f(U_{ir})=\operatorname{start}(U_{ir})+D_{ir},\qquad
\operatorname{queue}(U_{ir})
=\operatorname{start}(U_{ir})-\operatorname{ready}(U_{ir}).
```

排队由 simulator 根据工具池容量计算。推理时长由模型、硬件、长度及实际 batch 计算。

### 5. 调用序列与实际时间

每轮有一个推理节点，每两个相邻推理节点间有一次工具调用：

```math
Q_{ir}=(P_{ir},O_{ir}),\qquad
Q_{ir}\rightarrow U_{ir}\rightarrow Q_{i,r+1},
\quad 1\leq r<R_i.
```

工具节点携带工具类型、返回长度和执行时长。序列以推理开始、以推理结束，不含并行分支； JSONL 中使用节点和前驱边编码这条链。生成器只预设外部到达时间；
给定执行结果中的完成时刻 f(v)，就绪时刻满足：

```math
\operatorname{ready}(v)=
\begin{cases}
a_i,&v=Q_{i1},\\
\max_{u\in\operatorname{pred}(v)}f(u),&\text{其他节点},
\end{cases}
\qquad \operatorname{start}(v)\geq\operatorname{ready}(v).
```

实际启动／完成时刻、prefill／decode batch、工具排队和延迟，由 simulator 根据资源与调度计算。 类型／轮数／长度／依赖的生成在
`generate()`，落盘在 `write_workload()`。

### 6. 可复现性

伪随机流 seed 按 `SHA256(CanonicalJSON([version, seed, key]))` 派生。 key 区分全局 arrivals 与各
episode 的 family／turn_count／initial_prompt， 以及逐轮 output／tool_output／tool_work。新模型不需要独立的
has_tool 抽样。 到达过程共享同一组指数间隔；在固定类型分布下：

- 同 seed、同累计强度 Λ(T) 保留任务数量和 ID。
- 只改类型分布，到达时间不变。
- 只改到达曲线，既有 ID 的类型、长度、轮数和工具抽样不变。

统计规律指跨 seed 的分布；每个冻结文件是一份固定实现。 复现还需保存生成器／Python 版本与文件摘要。

## 输入与实验分离

- `spec.json`：生成参数；不含模型、GPU、DP／TP、副本、batch、并发或缓存策略。
- `workload.jsonl`：每行一个任务，包含外部到达、类型、请求 DAG、token 长度和工具服务需求；当前工具服务需求以工作量编码。
- `manifest.json`：输入／生成器摘要、Python 身份、自然任务数、各阶段计数及长度／轮数／工具工作量统计。
- 模型、初态、训练配方、硬件、准入、工具池容量、服务速率和执行 events 属于另存的实验／结果。

只有根任务外部到达固定；后续节点按前置节点完成而就绪。工具服务时长不包含排队或完整调用墙钟时间。 token 计数为合成单位，没有题目文本、token
ID、logprob、奖励或真实训练张量。

## 本次三组30分钟输入

共用 seed=20261024、固定类型概率(0.4,0.3,0.3)、相同任务内容，只有到达时刻不同。
三组累计强度均720，本次实际各687题、2058次推理、1371次工具调用。

| 目录         | 到达率，单位题/秒                        | 配置                                  |
| ------------ | ---------------------------------------- | ------------------------------------- |
| single-peak  | 三个600秒区间依次0.1／1.0／0.1           | `workload-spec.json`                  |
| periodic     | 每300秒前100秒为1.0，后200秒为0.1；共6次 | `workload-arrivals/periodic.json`     |
| short-bursts | 每300秒前30秒为3.1，后270秒为0.1；共6次  | `workload-arrivals/short-bursts.json` |

[输入目录](../../../../artifacts/elastic-workloads-30min-20261024/traces/)／[验证记录](../../../../artifacts/elastic-workloads-30min-20261024/validation.json)。
每组目录包含自己的 `spec.json`、`workload.jsonl`、`manifest.json`；完整保留自然任务数。

## 生成与格式

默认生成单次高峰；`--arrivals` 仅覆盖到达率区间，合并后的完整配置保存在输出目录：

```bash
python -m areal.elastic.workload \
  --spec examples/autopilot/elastic/workload-spec.json \
  --output /path/to/single-peak

python -m areal.elastic.workload \
  --spec examples/autopilot/elastic/workload-spec.json \
  --arrivals examples/autopilot/elastic/workload-arrivals/periodic.json \
  --output /path/to/periodic

python -m areal.elastic.workload \
  --spec examples/autopilot/elastic/workload-spec.json \
  --arrivals examples/autopilot/elastic/workload-arrivals/short-bursts.json \
  --output /path/to/short-bursts
```

v2 删除独立工具概率，要求整个窗口使用固定类型分布。旧四份 v1 trace 保留原样，读取器兼容 v1／v2； 旧配置不会被生成器静默解释为新规则。 工具服务需求仍为
`D_ir = α W_ir`：`tool_work` 采样工作量，`work_reference` 标识基准， 实验显式提供
α（seconds/unit）。本次两类基准均取1，见[换算假设](../../../../artifacts/elastic-workloads-30min-20261024/conversion-assumptions.json)；比较
GPU 部署时保持工具环境及该系数固定。JSONL 不预写排队时间或实际调用墙钟时间。

## Expert 接入与验证

7项CPU正常检查验证三个配置的严格交替、输入递推、冻结文件往返、内容一致及 Poisson 计数统计， 并以小型请求／token fixture 检查 Expert
接通；不是对完整30分钟输入的训练或GPU执行。 外部 `expert_groups_from_workload()` 转换原始序列，采用 individual
推理序列、每题1 sample、全部 accepted、全部生成 token 参与 loss 的实验假设。 缓存、工具基准速率、池容量及完整 Expert
输入与结果均由实验保存，冻结 Reference 不改动。

```bash
python -m pytest -q \
  tests/test_elastic_policy.py::test_synthetic_workload_preserves_seed_dag_and_roundtrip \
  tests/test_elastic_policy.py::test_synthetic_workload_arrival_and_composition_axes_are_independent \
  tests/test_elastic_policy.py::test_synthetic_workload_poisson_counts_match_mean_and_variance \
  tests/test_elastic_policy.py::test_basic_joint_optimizer_uses_frozen_expert_request_simulator
```

有限窗口任务数不按训练 batch 补齐、重采样或删题。当前 ExpertJointWindow 要求固定完整训练批次；接通例显式选择 batch4
的完整前缀，记录所有选中／尾部 ID。 这只评估声明的前缀，不能宣称全流处理完成；原始 JSONL 完整保留。正常路径另检查 Poisson
计数均值／方差、随机轴独立、上下文递推及输入摘要一致。

实验计时须区分：**模拟器程序运行时间**与**模型预测任务完成时间／GPU秒**；数据生成本身不产生推理起止时间。
