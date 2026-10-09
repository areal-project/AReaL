# 过程奖励

过程奖励在轨迹的结果奖励之外，为每一轮生成提供训练信号。v1 OpenAI 代理和 v2 推理数据代理会在将完整轨迹导出给 PPO/GRPO 之前进行评分。
评分器以 Python 类的形式提供；AReaL 本身不要求部署外部评判服务。

## 配置

在 Agent 训练配置中添加以下设置：

```yaml
rollout:
  _version: v1
  agent:
    chat_template_type: concat
    export_style: concat
    prm:
      enabled: true
      advantage_shaping:
        mode: process_weighted
      scorers:
        - path: examples.prm.scorers.LengthBudgetScorer
          weight: 1.0
          kwargs:
            max_output_tokens: 128
actor:
  token_rewards_as_adv: true
```

评分器列表为空或设置 `enabled: false` 时，代理评分将被禁用。目前，评分器要求使用 concat 导出、concat 对话模板、具有 token 数据的交互，以及直接过程优势。
使用 v2 推理服务时，将 `rollout._version` 设为 `v2`，其余评分器配置保持不变。不支持 individual 导出或仅包含字符串的外部模型响应。
此集成评估的是已完成的训练轨迹，而不是可选 Agent Service 重建的对话历史。

v2 外部模型模式（`rollout.api_url`）不允许启用配置了评分器的 PRM，因为该路径只保存字符串响应，不包含具有 token 数据的交互。
此限制不影响常规 SGLang/vLLM 后端，包括通过 `server_infos` 传入的现有服务。
禁用 PRM 或评分器列表为空时，仍可使用外部模型模式。

## 评分器契约

继承 `areal.reward.prm.BaseScorer`，设置唯一的类级 `name`，并实现 `async evaluate(interaction, ctx)`。
返回未经权重缩放的标量，或形状为 `[interaction.model_response.output_len]` 的张量。
标量会广播到该轮的每个输出 token；稠密张量则保留各 token 位置的奖励。
运行器将各评分器的结果乘以对应权重后相加。

`ctx["messages"]` 包含当前分支的完整对话。输入是只读的。多个并发会话可能共享评分器，因此应将请求状态保存在协程内部。
外部调用应使用异步客户端，并设置有限的超时和重试次数。异常或 `None` 会导致轨迹被拒绝；如果确实要给零分，应返回 `0.0`。

如需对完整对话联合评分，请继承 `BaseTrajectoryScorer`。其 `evaluate_trajectory(interactions, ctx)` 按父节点先于子节点的顺序接收各轮交互，
并返回从交互 ID 到未经权重缩放奖励的映射。缺失的 ID 对应零分；未知 ID 会被拒绝。

如需结构化监控，可重写 `prepare_result` 或 `evaluate_result`，返回 `PRMScorerResult(reward, observations)`。
每条 `PRMMetricObservation` 声明其作用域、目标 ID、值类型和聚合方式。布尔观测支持 `count` 和 `rate`；数值观测支持 `sum` 和 `mean`。
同一评分器的同一指标必须保持 schema 稳定。示例中的长度预算评分器演示了这一接口，无需额外依赖。

### 评分器生命周期

`BaseScorer.aclose()` 是可选的异步清理钩子，默认不执行任何操作。
评分器可以重写它，将缓冲的审计记录写完、等待后台写入任务结束，并关闭自己持有的客户端。
钩子内部的外部 I/O 应设置合理超时；运行器不额外增加清理超时或重试策略。

`PRMRunner.aclose()` 按创建的逆序关闭它根据配置创建的评分器，包括已禁用的评分器。
直接传入运行器的现成实例属于借用对象，仍由调用方负责关闭。每个自有实例最多关闭一次。
并发关闭调用会等待同一次清理完成；普通异常会记录日志，并在所有自有评分器均尝试清理后，以 `ExceptionGroup` 统一上报。
取消信号继续传播，后续关闭调用不会重试已经失败或被中断的清理。

在异步代码中创建运行器时，请使用 `await PRMRunner.create(config)`。
如果评分器解析、构造或配置校验失败，该方法会先等待此前已构造成功的自有评分器清理完毕，再重新抛出原始启动异常。
普通清理异常会记录日志，并作为异常备注附加；清理期间的外部取消信号仍会传播。借用实例不会被关闭。
若某个评分器自身的构造函数在返回前抛出异常，该构造函数内部已经获取的资源仍需由评分器自行清理。

调用方应先等待正在执行的评分结束，再关闭运行器。一旦开始关闭，新的 `run()` 调用会被拒绝。
v2 数据代理在服务启动阶段通过异步工厂创建运行器，并在生命周期结束时等待清理完成，之后再关闭推理桥接和 HTTP 客户端。
即使评分器清理失败，这些服务资源仍会被清理。PRM 启动失败时也会关闭推理桥接和 HTTP 客户端，普通服务清理异常不会覆盖原始启动异常。
同步的 `PRMRunner(config)` 接口仍保留以兼容现有调用方，但不提供异步启动回滚。
v1 代理仍使用该同步接口，暂未自动调用 `aclose()`。

## 优势塑形

评分生成与输出 token 对齐的 `token_rewards`；提示 token 的奖励为零。PPO 会将这些奖励移位到下一个 token 的预测位置。
使用直接过程优势时，塑形发生在 GAE 和优势归一化**之后**，不会改变 critic 回报：

- `additive`：将加权过程奖励加到结果优势上。
- `gvpo`：负过程奖励表示该 token 失败。若其结果优势为负，则乘以 `1 + negative_scale`；若优势接近零，则变为 `-zero_penalty`；若优势为正，则变为零。
- `process_weighted`：过程奖励必须处于 `[0, 1]`。非负结果优势乘以过程奖励。对于负优势，正过程奖励会替换该优势，零过程奖励则保留原有负优势。
  此模式不允许 `mask_no_eos_with_zero: true`。

底层 actor 也支持通过 `token_rewards_as_adv: false` 将各轮的均匀奖励纳入 GAE。
在明确稠密奖励如何转换为整轮奖励总量之前，配置式代理评分器暂不支持该模式。

## 分支、过滤与指标

评分前会克隆每条从根到叶的导出分支。因此，共享祖先可以获得各分支独立的评分，而不会修改其他分支或会话缓存。
只有评分器的所有结果通过校验后，才会应用这些结果。代理仅在全部分支评分成功时发布指标，否则会拒绝该会话的轨迹。

指标包括 `prm_turn_reward/<scorer>`、`prm_trajectory_reward/<scorer>` 和 `prm_metric/<scope>/<scorer>/<metric>/<aggregation>`。
结构化观测不乘评分器权重；奖励指标则包含权重。各 worker 的均值和比例按观测数量加权，计数和总和直接相加。
评估使用 `eval-rollout` 命名空间。

### v2 导出与指标传输

v2 控制器将现有的 `PRMConfig` 转发给各推理数据代理。评分在收集就绪轨迹之后、v2 原有的组内结果奖励归一化之前执行。
独立会话并发评分，同一会话内的各分支按顺序评分。过程 token 奖励不参与组内归一化。此集成不引入在线逐步评分或新的优势公式。

即使未启用结果奖励归一化，只要请求组内有一个需要 PRM 评分的会话缺失或评分失败，整组就会被拒绝。显式丢弃请求会跳过评分。
当 `remove_session` 为 true 时，成功、评分失败或取消后都会清理会话。这不增加对部分样本组的接纳支持。
在线工作流保留持久 HITL 会话（`__hitl__`），仅消费指定的就绪轨迹，因此导出期间新增的交互和后续就绪轨迹不会被清理。
普通会话密钥对应的导出仍会删除会话。

导出响应通过 `prm_stats` 携带现有的带类型逐轮结果和各分支评分器总分。工作流在 HTTP 重试边界之外，使用与 v1 相同的指标记录器写入这些结果。
因此，均值和比例保留对应的观测数量；并发导出不会竞争读取并清空服务端的共享统计缓冲区。
每个成功评分的会话都会贡献观测，即使组内其他成员导致整组被拒绝。这些指标反映评分行为，而非最终进入优化器的样本数量。
评分失败的会话不会发布部分分支的观测。未启用 PRM 的请求保持原有响应结构。

导出只消费一次就绪轨迹。重试已消费的轨迹不会再次调用其评分器。与现有 v2 导出一致，丢失的响应不会重放：这不保证轨迹或指标恰好交付一次。

对于分组 rollout 过滤，`examples.swe.filter_function` 提供了 `filter_mixed_or_penalized_all_wrong`：
它保留结果有对有错的样本组，以及虽然全部错误但仍含负过程信号的样本组。
`filter_mixed_or_penalized_all_wrong_mask_no_eos` 变体会遵循 actor 的截断掩码。这些过滤器应应用于完整样本组。
