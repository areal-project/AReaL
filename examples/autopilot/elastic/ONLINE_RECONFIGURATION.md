# 在线重配置模拟

当前实现冻结为[policy v3.1](../../../areal/elastic/policy/basic/VERSIONS.md)，增加发布期间保留方案；v2冻结于`922ae995`，v1保留原标签`elastic-policy-v1`。策略修订与组件校准版本独立记录。

当前历史模式使用固定Serving窗口和80%请求达标目标：`J_c = K_c + H × (Actor卡数 + Rollout卡数)`，默认`H=120`秒。切换启动／带宽仍为显式假设，旧实验结果不代表新目标的收益。

## 决策规则

1. 先累计满`replay.history_seconds`（默认120秒）的运行历史，再在每隔`policy.check_interval_seconds`（默认10秒，可设为x）的检查点判断Serving。联合达标率低于80%触发压力搜索，高于90%且至少8个请求完成时触发宽裕搜索；80%～90%含端点不搜索，不要求连续命中。首次启动及每次切换完成后，都必须重新积累H秒运行历史才恢复判断；默认H=120。切换完成指本次Actor／Rollout接管结束且旧部署已释放，不从搜索开始、新部署启动或路由接管时起算。等待期间继续服务和采集，不清除历史记录。
1. v2合格条件为窗口SLO达标且Actor估计处理速度覆盖外部到达率。压力：存在合格的新candidate时，排除current，在合格者中选`J_c`最小者；没有时包含current，先选达标率最高者，再比较额度亏损，最后按成本打平。宽裕：包含current，仅在合格者中最小化`J_c`，没有合格者则保持。成本并列时优先current（允许其参与的分支）。
1. `H`取`replay.history_seconds`，不再随阶段年龄或上一阶段时长缩短。当前实现仍从**当前观测状态**分支，把最近H秒真实到达槽位／历史形状确定性回放至H；策略不使用不足H秒的启动历史进行搜索。若已观察到输入关闭，不添加新到达，但仍按H秒驻留计费。
1. 未完成任务只使用已知进度及最近完成同类任务补全未知后缀；无同类时回退至最近完成任务。不读取主环境真实未来，不进行Poisson采样。阶段特征仍用于历史赢家优先级和应用前复核，不决定评价窗口长度。
1. 显存、packing及备用资源不支持的候选排除。示例另启用10%历史显存余量筛选，压力回退也不能绕过。新Actor须通过已知完整batch的模型能力检查，或在窗口内完成一次接管后的更新；覆盖不足记为unknown，不能称为真实GPU验收。
1. 示例关闭额外的`cooldown_seconds`及重复搜索抑制；切换完成后等待H秒的历史门槛始终适用。旧配置可显式设置更长冷却。交接、发布和搜索尚未结束时不重叠搜索。旧连续次数及TTFT／TBT分位数阈值字段仅兼容输入，不参与触发。

已确认保留**当前状态＋历史负载向后回放H秒**，保留当前积压和执行进度，不回到历史窗口起点。SLO评估覆盖整个H秒（含切换期），不只评价新部署接管后的时段；过渡期可能拉低整体达标率，压力下全部候选不合格时保留最高达标率回退。

### v2：持续补充准入额度

`lambda = 历史窗口外部到达题数 / 窗口秒数`，包含未准入任务，不以完成吞吐代替到达需求。
沿用因果历史模板，按到达顺序组batch；不足一批时循环该历史序列补满，按原余数／batch_size加权。
分别估算候选Actor完整更新阶段的加权平均耗时，加上目标布局单独运行时的一次发布耗时，得到
`mu = batch_size / (actor_seconds_per_batch + publication_seconds_per_batch)`。
已有额度、旧Actor更新、交接及等待凑批不计入该能力；未知后缀仍按现有历史规则补全，记录替代数量。

- `mu >= lambda`才通过持续能力条件。成本J与120秒SLO窗口均不变。
- 压力回退在SLO相同时最小化`max(0, lambda - mu)`，再比较J；有余量的配置不因余量更大而获额外奖励。
- 有到达但能力估计未知时，不声称合格；仍可参与压力回退，SLO打平时排在已知亏损之后。
  已观察到输入关闭时lambda设为0，有限尾部工作继续由原有batch能力检查及联合仿真处理。
- 该估计是必要检查，不保证端到端稳定：等待任务／工具完成、发布前在途GPU步排空、链路争用及负载波动仍由联合仿真处理。
- 每个历史候选记录`actor_supply`（lambda、mu、额度变化率、阶段耗时、补齐／补全证据）、
  窗口起止剩余额度和`admission_credit_blocked_seconds`。阻塞时间只累计有题等待且额度耗尽的区间，
  可与客户端并发限制重叠，不等于逐题等待时间之和。
- 搜索及应用前复核使用同一条件；仅SLO达标但亏损的incumbent不能作为成本剪枝依据。
  旧随机预测模式不追加此历史负载约束；真实训练版本差仍仅记录，staleness=2准入规则未改。

### 历史显存安全筛选

`policy.actor_memory_reserve_fraction=0.1`要求预测峰值不超过容量的90%；这是策略假设，非实测安全保证。
省略或设0保留此前行为。检查截至当前已经完成的所有到达顺序完整batch，以及当前已知batch，
不局限最近120秒、不拼接未完成任务的未来后缀。每个候选按自身DP／TP和原微批规则重新估计， 任何历史形状超过预算、命中已测OOM或缺少依据，均不进入候选评分或压力回退。

主环境仍使用原物理容量；不修改AReaL引擎、微批、耗时公式或冻结校准。
搜索、进程池和最新状态复核共用筛选；`actor_memory`记录余量、最大峰值、最坏batch、检查数量及观测时刻。
保留更早的大形状会使后续缩容更保守，也不保证未见形状安全；所有候选均被排除时保持现状，不将现状标为安全。
[聚焦实验](../../../../artifacts/elastic-policy-memory-guard-20261026/README.md)。

`policy.actor_memory_scope`默认`all_completed`。对照选`evaluation`时，余量只检查原评估实际调用的目标Actor
batch（近期代表batch、当前已知batch、窗口内开始的训练batch），不额外扫描旧历史；进行中的旧Actor仍按物理限制模拟。
报告的`scope`区分两者；`evaluation`可能含回放模板，`worst_batch`不填写真实episode ID。
同一多轮长输入88联合，单独10%余量得到88.56%／5.7576GPU·h，显存等待及额度阻塞为0，与全部历史模式基本相同；
只验证这一例，非普遍安全保证。[对照证据](../../../../artifacts/elastic-policy-memory-margin-only-20261026/README.md)。

## 成本与窗口计分

- 每秒运行成本`r_c = Actor卡数 + Rollout卡数`，窗口运行成本`R_c = H × r_c`。同卡数比较切换开销；不同卡数使用完整公式，不能假设少卡必然切换便宜。
- `K_c`为更换侧旧部署从申请新卡到释放期间的占卡积分。新目标部署驻留计入`R_c`；未更换侧不增加`K_c`，保持current时为0。启动、预热、同步、Actor更新边界及旧请求排空决定旧部署保留多久，避免重复计费。
- Serving与Actor仅共同推进H秒，冻结此时的SLO、更新数和未完成工作。**不为目标部署的尾部请求／训练继续排空计费**。如果切换本身尚未结束，仅继续估计到旧部署释放以补全`K_c`；这段不延长SLO窗口，也不增加`R_c`。无法完成交接估计则不能给出完整评分。
- TTFT≤5秒且响应内最大TBT≤100毫秒才达标；首轮含准入等待，工具间隔不计TBT。窗口末沿用现有rolling
  SLO口径：已完成请求和已明确超限的未完成请求进入分母；尚未超限的未完成请求暂不评分。没有可评分请求时不声称达标。
- 完整候选记录`switching_gpu_seconds`、`running_gpu_seconds_per_second`、`evaluation_seconds`、`running_gpu_seconds`、`score_gpu_seconds`。历史模式的`mean_gpu_seconds`兼容字段等于新评分J；`window_gpu_seconds`另外记录分支在窗口内的实际占卡积分，`tail_gpu_seconds=0`。窗口结束的完成／未完成请求、已超限未完成请求、更新数和未消费episode也单列，多样本计数累加、成本取均值。
- batch40、固定顺序、原staleness=2准入与单写者交接保持；实际token版本差只记录。候选窗口结束不代表主任务完成，主环境仍处理整条真实trace、完整训练batch和尾部数据。
- Actor显存模型拒绝时仍保留batch／游标等待可行配置；不擅自改变算法、微批或冻结fit。

## 新旧部署交接

资源池允许额外节点准备新部署，不能把初始16卡当成整个池上限。Rollout新server就绪后接新请求，旧请求与KV留在原副本直至排空，不迁移在途KV。Actor在完整update及发布边界交接权重、optimizer和消费游标，只保留一个训练写者。两侧准备／接管／释放均由同一时间线推进。

示例条件为4×8资源池、启动30秒、预热5秒、共享有效带宽50GB/s；这些是显式假设，未实测重配置成本。

### 共享16卡的条件探索

以下配置允许Actor与Rollout在**稳态合计≤16卡**内倾斜，TP仅1／2／4／8；每个TP组仍在同一8卡节点。单侧≤8卡沿用单节点放置，超过8卡时最多跨两个节点。Actor只用跨节点DP，Rollout用独立副本；batch40下新增的跨节点Actor只有DP10×TP1和DP5×TP2。全部联合候选为241组。

```json
{
  "search_space": {"steady_gpu_budget": 16, "role_gpu_limit": 16},
  "model_assumptions": {"inter_node_dp_GBps": 50}
}
```

这是补充到现有配置的字段。未指定时保留每侧8卡默认；跨节点Actor必须显式提供DP有效带宽，原校准系数不变。该值仅改变跨节点Actor的集合通信估价，发布／状态传输仍使用`handoff.effective_bandwidth_GBps`，两者不能混称同一次网络实测。DP与发布／状态链路之间的动态争用尚未建模。

稳态目标预算与准备资源池分别约束：新旧部署重叠可超过16卡，但四节点物理峰值≤32卡，所有占卡积分计费；切换期间的旧Actor＋新Rollout也可能暂时超过16卡。新Rollout按实际节点数加载模型，准备期间版本落后时同样按节点数补传，旧在途请求排空后才释放。跨节点DP、独立副本干扰和启动成本目前均为条件外推，尚未真实多节点验证。

[负载探索方案与全部结果](../../../../artifacts/elastic-workload-regimes-20261025/)将固定8＋8／4＋4、仅资源调整、仅TP调整和联合调整分别保存。每组独立父进程及worker池，避免跨实验缓存影响计入时间线的搜索墙钟；同组内部仍可复用缓存。固定和动态必须完成相同业务、全部完整训练batch和尾部数据，SLO未达80%的节省不能记为满足目标的胜出。

CLI默认按实际CPU墙钟推进决策可用时刻：搜索期间旧部署继续处理请求／训练，并更新历史，同一时刻最多一轮搜索。结果只是候选建议；搜索结束后，以最新观测重新比较当前与目标配置，复核耗时也计入主时间线。复核提前完成时等到预测中约定的应用时刻；复核超时、阶段／部署／资源归属、显存等待或输入关闭状态变化时作废。v3不再要求权重版本、消费游标或当前Actor
batch保持原值：交接层在安全边界接最新权重／优化器／游标。搜索结束时正发布则等发布完成再做首次复核；已复核方案遇发布则保存结果，发布结束直接继续应用，不重新搜索或复核。发布等待不计入复核结果的普通过期时长，其他条件仍检查。

`publication_wait_seconds`记录实际发布等待，决策分别记录`verification_publication_wait_seconds`／`application_publication_wait_seconds`及应用时状态；原预测不回写，等待和同步的实际GPU占用由主时间线计费。`application_wait_seconds`仍统计原计划等待总量，取消时可能尚未全部发生。

预测搜索耗时使用上次实际值，复核预算取配置先验与上次复核墙钟×1.25的较大值。`decision_delay_seconds`是初始估计，不代表搜索只需1秒；这是CPU条件模拟，主机负载会改变完整轨迹。`--decision-clock fixed`保留旧固定延迟对照。

## 实现与运行

| 文件                                                    | 职责                                               |
| ------------------------------------------------------- | -------------------------------------------------- |
| `areal/elastic/policy/basic/history.py`                 | 阶段特征、固定窗口、确定性回放                     |
| `areal/elastic/policy/basic/online.py`                  | 触发、80%目标与按触发类型筛选                      |
| `areal/elastic/policy/basic/search.py`                  | 固定窗口评分选择、候选优先级与成本下界             |
| `areal/autopilot/domains/elastic/online_control.py`     | 实测CPU决策事件、最新状态复核与应用时刻            |
| `areal/autopilot/domains/elastic/online_simulation.py`  | 持续主环境、状态恢复、候选联合模拟及窗口／尾部计费 |
| `areal/autopilot/domains/elastic/online_models.py`      | 冻结4B／H20组件、显存及发布模型                    |
| `areal/autopilot/domains/elastic/online_decode.py`      | 固定batch的线性Decode区间推进                      |
| `areal/autopilot/domains/elastic/online_state.py`       | 副本索引／计划失效及内部可变请求状态               |
| `areal/autopilot/domains/elastic/online_parallel.py`    | 独立候选的CPU进程池与按原索引汇总                  |
| `areal/autopilot/domains/elastic/online_inputs.py`      | 不可变负载／恢复模板与候选私有状态分离             |
| `areal/autopilot/domains/elastic/online_transitions.py` | 节点GPU租约、链路及交接假设                        |

候选模拟默认合并稳定的纯Decode区间，按冻结线性fit累计长度、时间及最大TBT；请求完成、prefill、工具、发布、交接、检查或窗口边界前停止，保留跨边界的在途GPU步。`OnlineForecastModel(..., coalesce_decode=False)`可对照逐token模式。带逐batch日志、未声明线性能力的模型及旧SLO提前剪枝模式仍逐token执行；闭式求和与逐步浮点累加存在微小数值差异。[对照与速度](../../../../artifacts/elastic-decode-coalescing-20261025/README.md)。

每副本维护请求／预留token／KV索引，Decode计划校验batch、成员revision、版本与heap事件身份；纯GPU步完成时只重启相关副本。稳定Decode可累积已完成步数、首末时间及最大TBT，延迟逐请求写回；读取请求、成员／计划／版本改变或请求完成前物化，KV与事件计费立即更新。公开观测保持独立冻结快照，全局结构事件顺序保留。[局部事件核查](../../../../artifacts/elastic-event-locality-20261025/README.md)。

CLI默认`--search-workers 4`，设为`1`串行。进程池及每进程不可变负载模板按公开输入与配置复用；每个候选重新创建可变状态，按原索引汇总。旧目标下165候选的830秒公开快照中，122组剪枝、13组完整、30组不支持，选择与全量搜索一致；单进程25.98→16.46秒，4worker复用11.29→7.09秒。这些速度数字属于旧目标快照，不能作为新评分的性能结果。[搜索证据](../../../../artifacts/elastic-online-search-20261025/README.md)／[此前模拟器加速](../../../../artifacts/elastic-deferred-state-20261025/README.md)。

配置`policy.optimize_search=false`可恢复全量评估；重复搜索抑制在示例中关闭，旧实验可显式设置`policy.deduplicate_searches=true`恢复。决策保存完整／剪枝／不支持数量、评估顺序、剪枝界限与当时最好结果；搜索、复核完成与实际应用时刻分别记录。新成本下界为固定窗口运行成本加已发生的切换重叠成本；仅有达标且可参与该分支的incumbent时才据此剪枝。历史模式关闭旧的全量请求达标率上界剪枝，因为窗口末未决请求使其分母不再固定。压力下current不能作为排除达标新candidate的成本界限。剪枝只记界限，不伪造完整K或SLO；多回放按最差达标率和平均J处理。

```bash
python examples/autopilot/elastic/simulate_online.py \
  --workload "$WORKLOAD" --config "$CONFIG" \
  --bundle "$BUNDLE" --mechanism-fits "$MECHANISM_FITS" \
  --publication-metadata "$PUBLICATION_METADATA" --output "$OUTPUT"
```

`OUTPUT`必须为新目录；默认`--mode both`，可指定`fixed`／`dynamic`。[默认配置](online-example.json)使用`replay`字段；旧`forecast`字段仅供旧随机预测实验复现。输入、源码、校准身份存入`identity.json`；各侧保存事件、请求、训练顺序、决策与summary。历史模式决策记录固定窗口来源、回放覆盖、K／R／J及相对current的评分差，`forecast_seeds`为空。

[本轮正常CPU检查](../../../../artifacts/elastic-window-cost-20261026/validation.json)／[旧目标证据](../../../../artifacts/elastic-history-optimizer-20261025/)；[原95%对照](../../../../artifacts/elastic-online-campaign-qwen4b-h20-20261025/README.md)；[安全等待诊断](../../../../artifacts/elastic-online-safe-wait-20261025/README.md)。

历史补全与阶段特征可能失准。HTTP尾延迟、原生abort重算、真实状态交接、跨节点成本和未见混合形状仍需验证。当前仅CPU模拟，不接管原生服务，不改冻结Reference源码、校准系数或旧实验。
