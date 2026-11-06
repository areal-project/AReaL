# 数据工程 recipe

`areal.v2.data_engineering` 提供有预算限制的前台数据任务：执行
recipe、清洗与去重、公开任务校准、筛选、保存来源记录并发布可校验的数据集。研究控制器选择目标和 recipe；领域 recipe
实现生成、清洗、评分、筛选和训练视图投影。这是面向任务数据的默认流程；任务项目可以按需组合多个作业或采用其他数据流程。

## 运行 TIR 示例

在仓库根目录、安装 AReaL 常规 Python 依赖后运行：

```bash
python3 -m areal.v2.data_engineering \
  --config examples/data_engineering/tir/data_generation.yaml \
  --output /tmp/tir-data-job \
  --export-hf /tmp/tir-training-data
```

首次运行时两个输出目录必须尚不存在；恢复数据作业使用 `--resume` 和原输出目录。该配置无需模型、GPU 或推理服务，直接生成带精确答案的数学任务；导入 AReaL
包仍需要常规 Python 依赖。stdout 最后一行返回 `dataset_id`、`dataset_path`，以及按需生成的
`training_dataset_path`。

TIR 包含八个数学题型、每个题型两种结构。训练视图包含 `task_id`、`messages` 和精确的 boxed `answer`。这是任务数据；训练
rollout、tokenization 和 RL/SFT/OPD 张量由下游处理。

## Python API 与执行协议

```python
import asyncio
from areal.v2.data_engineering import DataJobConfig, run_data_job

config = DataJobConfig(
    recipe="examples.data_engineering.tir.data_recipe:create_recipe",
    target_count=64,
    purpose="joint",
    seed=42,
    params={"candidate_count": 128},
)
snapshot = asyncio.run(run_data_job(config, "/tmp/tir-python-job"))
rows = snapshot.training_rows()
```

在已有异步控制器中使用 `await run_data_job(...)`。工厂采用可导入的 `module:callable`，执行器通过
`factory(**params)` 创建 `DataRecipe`。相关 Python 协议从 `areal.v2.data_engineering` 导出。

| 类型或方法                         | 约定                                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| `PublicTask`                       | 稳定 `task_id`、领域 `input` 和公开 `metadata`；公开字段中不得放入答案                           |
| `DataSample`                       | `PublicTask` 与私有 `reference`；清洗后保留的候选须可 JSON 序列化                                |
| `generate(context)`                | 生成一批候选；context 提供 seed、批次编号、目标数量、用途和输入文件路径                          |
| `clean(sample)`                    | 返回清洗后的候选，或用 `None` 排除候选                                                           |
| `score(sample, response)`          | 使用私有参考答案，返回 `[0, 1]` 内的有限分数                                                     |
| `select(samples, scores, context)` | 返回恰好 `target_count` 个已有且不同的任务 ID；候选不足时抛出 `InsufficientDataError` 请求下一批 |
| `training_row(sample)`             | 将一个选中任务投影为一条领域训练数据                                                             |
| `source_files()`                   | 可选方法，声明影响结果的辅助代码和外部输入文件                                                   |

执行器按生成 → 清洗 → 去重与排除 → 校准 → 筛选运行。各批清洗后的候选及评分累计进入候选池。仅 `InsufficientDataError`
触发下一批；其他错误在配置的瞬态重试耗尽后令任务失败。`max_batches` 和 `max_candidates` 限制总预算，领域参数保留在 `params`
中。recipe 也可以通过 `context.inputs` 读取原始数据，执行已有数据的清洗与筛选。

输入和排除项均为文件。模块入口相对配置文件解析 `inputs`、`exclude` 中的路径；插件 `params`
中的路径由插件定义，当前校准适配器相对进程工作目录解析这些路径。

`purpose` 可为 `harness`、`model`、`joint`。它进入来源记录和 recipe context；由 recipe
决定如何影响策略。执行器不会自动归因模型或 Harness 的失败。独立 dev/test 数据以及采用哪个模型或 Harness，由上层控制器管理。

相同任务 ID 必须对应相同清洗内容；重复公开输入若有冲突的参考答案则报错。通用层按精确 JSON 输入去重，语义去重和领域规范化由 `clean()`
完成。排除文件采用包含顶层 `task_id` 的 JSONL：可传入已有数据集的 `public.jsonl`，或先前任务的
`work/batch-N-public.jsonl`，排除已选择或已校准的候选。

## 并行执行与恢复

在数据任务配置中加入：

```yaml
execution:
  max_workers: 4
  batch_size: 256
  backend: process
  max_retries: 2
  retry_delay: 1.0
```

`run_data_job` 在专用协调线程中创建独立事件循环。全局协调、来源摘要、SQLite 和 snapshot I/O
均由作业线程执行，调用方事件循环可以继续响应其他任务。校准器由工厂在作业事件循环中创建， 不应复用绑定到调用方事件循环的异步资源。SQLite
连接在协调线程中创建并由同一线程使用。

取消调用方 task 会通知作业并取消待执行的异步校准；返回取消结果前会等待 worker 退出、数据库关闭和最终状态落盘。
重复取消不会中断清理。同步插件或存储调用无法被强制打断，需等待当前调用返回，再在下一阶段或数据块之前生效。 已经原子发布的 snapshot
会保留，恢复时重新校验；这个机制不能终止永不返回的插件。

默认一个 worker、每块 256 条、`thread` 后端、不自动重试。`thread` 适合阻塞 I/O；`process` 通过 spawn 启动独立 Python
进程，并行执行 CPU 生成、清洗、评分、排名和训练视图投影。进程模式要求工厂可导入、原始候选可 pickle。 Python 脚本中的 `asyncio.run(...)`
应放在 `if __name__ == "__main__":` 中。 全局去重与最终配额合并由协调线程执行；异步校准拆成最多 `max_workers` 个并发调用，每次最多
`batch_size` 个公开任务，重复执行次数仍由校准协议控制。

```bash
python3 -m areal.v2.data_engineering \
  --config examples/data_engineering/tir/data_generation.yaml \
  --output /tmp/tir-data-job --resume
```

Python 调用使用 `await run_data_job(config, output, resume=True)`。恢复要求配置、声明的输入和排除文件、
recipe/校准源码及执行器代码保持一致；身份发生变化会在执行前拒绝恢复。成功作业恢复时返回已有的、重新校验过的 snapshot。HF 导出仍要求新目录；已经导出时省略
`--export-hf`。

`work/checkpoints.sqlite3` 原子提交每个清洗、校准、投影块。候选内容、去重索引、校准结果和排名记录保存在磁盘， worker
只预取有界数据块；snapshot 写入和校验采用流式处理。筛选接收磁盘支持的 `Sequence[DataSample]` 和评分
`Mapping`；大候选池优先迭代，反复随机索引和切片可能较慢。recipe 自行转成列表仍会占用内存， `training_rows()` 与 HF
导出也会物化数据；TIR 执行器路径使用并行排名和磁盘索引筛选；直接调用示例的 `select()` 仍会构造内存排序列表。

SQLite WAL 检查点和进程锁应放在本地文件系统。当前提供单机并行，未接入 Ray/Slurm 分布式作业。 进程死亡会释放锁，SIGKILL
后无需手动删除锁文件。`work/batch-N-{candidates,public}.jsonl` 是可从检查点重新生成的诊断导出文件。

Recipe 方法应只依赖参数和声明的输入，生成需要可确定性重放；清洗和投影 worker 分别构造自己的实例，
执行器不保存实例内部状态。旧式生成中断时重放当前批次，但跳过已提交的清洗块；分片生成会复用已完成的生成文件，仅重跑不完整的分片。 启用多个 worker
时，校准器必须支持并发调用。

仅 `OSError`（包括连接故障）和 `TimeoutError` 触发重试，采用最多 60 秒的指数退避。
校验错误和取消不重试；`Attempt.failure_reason` 表示一次已完成的校准结果，不会作为异常自动重试。
已经提交的块会复用，失败或未提交的块可能重新执行。外部服务应使用任务/重复身份保证幂等， 执行器不承诺外部副作用恰好执行一次。

## 生成、评分与全局筛选并行

已有 recipe 继续可用：没有可选方法时，生成和筛选保留原协调线程实现；评分统一按数据块交给 worker。 可选的 `ShardedDataRecipe` 和
`RankedDataRecipe` 协议描述这些能力，recipe 无需显式继承它们。 并行生成需要同时实现两个方法：

| 方法                                     | 约定                                                                                           |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `generation_shards(context, shard_size)` | 生成从零开始、连续且可确定性重放的 `GenerationShard(start, stop)`；`shard_size` 是分片大小提示 |
| `generate_shard(context, shard)`         | 生成至多 `stop-start` 条原始候选；整个计划不得超过剩余 `max_candidates` 预算                   |

生成 worker 独立构造 recipe，并流式生成、清洗和写入 `work/generation-B-S.jsonl`。 相邻 JSON
元数据记录范围、行数和内容摘要。完整且通过校验的生成文件可在恢复时复用，即使此前数据库提交失败。 协调线程按分片顺序，以 `batch_size`
大小的数据块导入并完成全局去重。较大的分片减少文件数，但重试粒度也更粗； `generate(context)` 应返回相同分片流的拼接结果。

校准响应在评分前完成校验并写入检查点。评分失败时可以直接使用已保存的响应重试或恢复，无需重复已完成的模型请求。 评分 worker 独立创建
recipe，不能依赖生成或校准阶段留下的实例状态。

并行筛选需要同时实现三个方法：

| 方法                                                    | 约定                                                                     |
| ------------------------------------------------------- | ------------------------------------------------------------------------ |
| `selection_policy(context, calibrated)`                 | 返回 `SelectionPolicy(minimum_per_group, maximum_per_bucket)`            |
| `rank(sample, scores, context)`                         | 不合格时返回 `None`，合格时返回原任务 ID 对应的 `RankedCandidate`        |
| `finalize_selection(selected, eligible_count, context)` | 返回最终任务 ID 和统计；领域质量要求未满足时抛出 `InsufficientDataError` |

`RankedCandidate` 包含有限浮点 `priority`、字符串 `tie_breaker`、`group`、`bucket` 和 JSON
`metadata`。 worker 并行判断资格并按 `(priority, tie_breaker, task_id)` 排序局部数据块，SQLite 索引保存全局顺序，
已提交的排名块在恢复时复用。候选的排名应随其输入固定，不得依赖后续批次扩展后的候选池； 需要动态全局策略的 recipe 可以继续使用原 `select()`。

协调线程先按 policy 字典顺序满足每组最低配额，再按全局排名补齐，同时检查 bucket 上限并防止重复选择。
最终合并保持串行和确定性；它是明确的贪心策略，不是任意重叠约束的求解器。 recipe 直接调用 `select_ranked_candidates`
并提供内存有序迭代器时，可以复用同一合并算法。 TIR 示例使用 16 个独立 seed 的题型配方流、题型最低配额、配方上限和最终临界题数量检查。 固定 seed 下候选不受
worker 数量影响；此生成器版本与旧版逐题结果不同，已有校准报告必须匹配实际候选和源码身份。

## 校准协议

校准工厂返回异步 `Calibrator`：

```python
async def __call__(self, tasks, context):
    # tasks: Sequence[PublicTask]；context: CalibrationContext
    # 处理公开任务，并返回实现相关的来源记录。
    return CalibrationBatch(tuple(attempts), provenance={"method": "my-calibrator-v1"})
```

`Attempt` 包含 `task_id`、从零开始的 `repeat`、可 JSON 序列化的 `response` 和可选
`failure_reason`。每个任务/重复组合必须恰好返回一次。通用协议不要求模型或 Harness 引用，非空失败原因对应 `None`
分数；已完成但答错的响应使用普通评分。校准器修改公开请求对象不会改变执行器保留的候选。

校准器接收不到 `DataSample.reference`，原始响应由执行器使用 recipe 评分。该接口是数据约定，recipe
和校准代码仍是可信执行代码。校准器自行管理依赖和版本校验，可通过 provenance 返回规则版本、来源版本或实际模型/Harness 身份。执行器逐批保存这些记录；TIR
的模型/Harness 校验由其适配器完成。

TIR 要求每题两次执行。其评分器要求最后一个 boxed 答案为整数、小数、分数字符串或数值 LaTeX 分数，不求值 `1+2`
等表达式。筛选排除执行不完整和两次均正确的题，优先选一次正确的题，再补充两次均错误的题，并检查题型覆盖和配方上限。不配置校准器时，生成任务的筛选不会宣称完成模型难度校准。

### 复用已有 TIR 评估报告

```yaml
recipe: examples.data_engineering.tir.data_recipe:create_recipe
purpose: joint
seed: 42
target_count: 512
max_batches: 8
max_candidates: 8192
params:
  candidate_count: 1024
  minimum_one_pass: 128
  maximum_per_recipe: 48
calibration:
  implementation: examples.data_engineering.tir.data_calibration:recorded_responses
  repeats: 2
  params:
    model_ref: /path/to/the/evaluated/checkpoint
    harness_ref: actual-harness-id
    reports:
      - [/path/to/batch0-repeat0.json, /path/to/batch0-repeat1.json]
    backend_model_digest: actual-model-digest
    verifier_digest: actual-verifier-digest
```

为所有可能使用的批次提供报告。报告须覆盖生成的任务 ID，并匹配模型、Harness、模型摘要和 verifier 摘要。同一批重复报告须具有不同 trial ID，以及相同
task set 和 executable 身份。执行器忽略报告中保存的分数，重新对响应评分，并保存报告与源码摘要。复用时必须匹配原生成器的 seed
和候选数量；该适配器不启动 SGLang 或重新执行评估。

## 产物与训练接入

`job.json` 保存 running/succeeded/failed/cancelled 状态。`config.json`、`progress.json` 和
`work/` 下的候选与校准记录保存执行证据。成功筛选并校验后，执行器在同一文件系统中重命名临时目录，发布完整数据集。已有输出目录需要使用
`--resume`。任务前台运行，可通过中断模块取消；恢复时沿用原输出目录和配置。

数据集目录包含：

- `public.jsonl`：选中的公开任务。
- `references.jsonl`：按任务对齐的私有参考答案。
- `train.jsonl`：recipe 生成的训练视图。
- `probes.jsonl`：原始校准响应、失败原因和重新计算的分数。
- `report.json`：生成、清洗、排除和筛选统计。
- `manifest.json`：数据集身份、文件摘要、配置、配方/后端源码摘要、输入来源摘要和实现相关的校准来源记录。

目录和数据文件仅对所有者开放。`load_snapshot(path)` 和 `snapshot.training_rows()`
校验内容摘要与行对齐。数据集按约定保持不可变，并由内容确定身份；权限设置本身不会禁止所有者写入。工厂应通过 `source_files()`
声明导入的辅助代码及配置之外的文件，执行器在发布前重新检查这些来源。

使用显式适配器接入现有训练数据加载：

```python
from areal.dataset import get_custom_dataset
from areal.v2.data_engineering import export_hf_dataset, load_snapshot

snapshot = load_snapshot("/tmp/tir-data-job/dataset")
path = export_hf_dataset(snapshot, "/tmp/another-tir-training-data")
dataset = get_custom_dataset(path=str(path), type="rl")
```

导出目录的 `areal-source.json` 记录源数据集身份。训练 Workflow 必须匹配 recipe 的字段。SFT、OPD
需要各自的投影、tokenization 或教师处理适配器；TIR 的任务视图不是 SFT 对话或已经 tokenized 的 RL 轨迹。
